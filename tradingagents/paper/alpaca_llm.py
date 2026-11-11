"""Mirror the LLM paper account's suggestions to Alpaca PAPER, and report how that is going.

The agents' ratings are recorded in the simulated account (``account.py``). This module
turns the latest rating for each ticker into a paper order on Alpaca, using the same
sizing rules as the simulation (``PaperAccount.intended_shares``) but applied to what the
Alpaca account actually holds, then compares the real fills with the simulation's.

The point is to test how reliable the suggestions are with orders that really executed,
without risking money: everything goes to Alpaca's paper host only (see ``alpaca.py``).

- Long-only, whole shares, market orders. A Sell closes whatever is held; nothing is sold
  short and nothing is sold that is not held.
- Only the newest rating per ticker is acted on, and only if it is from the most recent
  completed session or later, so a stale suggestion is never sent.
- A rating whose rules call for no trade (Hold, or inside the no-trade band) sends nothing.
- A suggestion is sent once: ids are deterministic and the mirror records every order.
- Orders go in as DAY market orders. Sent while the market is open they fill at once; sent
  when it is closed they queue for the next open. The simulation fills at the next close, so
  fills differ by design and the report says by how much.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from tradingagents.paper.account import PaperAccount
from tradingagents.paper.alpaca import (
    FINAL,
    AlpacaError,
    FillRow,
    Mirror,
    MirrorOrder,
    Plan,
    Planned,
    Reconciliation,
    SendResult,
    close_on,
    refresh_orders,
)
from tradingagents.paper.runner import account_dir, fetch_all


def llm_mirror_path(config: dict, name: str) -> Path:
    return account_dir(config, name) / "alpaca_llm_mirror.json"


def _order_id(account_name: str, ticker: str, date: str, side: str) -> str:
    digest = hashlib.sha1(f"{account_name}|{ticker}|{date}".encode()).hexdigest()[:10]
    return f"ta-llm-{digest}-{side}-{ticker}"[:120]


def plan_llm_orders(account: PaperAccount, closes: dict[str, pd.Series], positions: dict[str, float],
                    mirror: Mirror, tif: str = "day", max_order_usd: float | None = None) -> Plan:
    """The paper orders that follow the newest rating of each ticker, given what Alpaca holds."""
    plan = Plan()
    cap = max_order_usd if max_order_usd is not None else account.sleeve
    for ticker in account.tickers:
        mine = [o for o in account.orders if o.ticker == ticker]
        if not mine:
            continue
        order = max(mine, key=lambda o: o.decision_date)
        key = f"{ticker}|{order.decision_date}"
        series = closes.get(ticker)
        if series is None or series.empty:
            plan.skipped.append(f"{ticker}: no price available")
            continue
        if order.decision_date < series.index[-1].strftime("%Y-%m-%d"):
            plan.skipped.append(f"{ticker} {order.rating} ({order.decision_date}): stale, run the agents again")
            continue
        if any(o.trade_key == key for o in mirror.orders):
            plan.skipped.append(f"{ticker} {order.rating} ({order.decision_date}): already sent")
            continue
        if order.rating not in account.weights:
            plan.skipped.append(f"{ticker} {order.rating}: no trade")
            continue
        price, held = float(series.iloc[-1]), float(positions.get(ticker, 0.0))
        shares = account.intended_shares(order.rating, held, price, 0.0)
        if order.rating == "Sell":
            if held <= 0:
                plan.skipped.append(f"{ticker} Sell: nothing held on Alpaca to sell")
                continue
            side, qty = "sell", held
        elif shares > 0:
            side, qty = "buy", math.floor(shares)
        elif shares < 0:
            side, qty = "sell", min(math.floor(-shares), held)
        else:
            plan.skipped.append(f"{ticker} {order.rating}: already within the no-trade band of the target")
            continue
        if qty < 1 and order.rating != "Sell":
            plan.skipped.append(f"{ticker} {order.rating}: under one whole share")
            continue
        if side == "buy" and qty * price > cap:
            plan.skipped.append(f"{ticker} {order.rating}: ${qty * price:,.0f} is over the ${cap:,.0f} order cap")
            continue
        plan.orders.append(Planned(key, ticker, order.rating, ticker, side, qty, tif,
                                   _order_id(account.name, ticker, order.decision_date, side), price))
    return plan


def send_llm_plan(broker, plan: Plan, mirror: Mirror, path: Path, now: datetime | None = None) -> SendResult:
    """Submit a plan to the paper account (sells first, to free buying power), recording each order."""
    result = SendResult()
    stamp = (now or datetime.now().astimezone()).isoformat(timespec="seconds")
    account = broker.account()
    if account.get("status") != "ACTIVE" or account.get("trading_blocked"):
        raise AlpacaError(f"the Alpaca paper account cannot trade (status {account.get('status')}, "
                          f"blocked {account.get('trading_blocked')})")
    buying_power = float(account.get("buying_power") or 0)
    if plan.orders and mirror.baseline_equity is None:
        mirror.baseline_equity, mirror.baseline_at = float(account.get("equity") or 0) or None, stamp
        mirror.baseline_positions = {p["symbol"]: float(p["qty"]) for p in broker.positions()}
    for leg in sorted(plan.orders, key=lambda o: o.side != "sell"):
        if leg.side == "buy":
            if leg.notional > buying_power:
                result.refused.append(f"{leg.symbol}: needs about ${leg.notional:,.0f}, buying power is ${buying_power:,.0f}")
                continue
            buying_power -= leg.notional
        record = MirrorOrder(client_order_id=leg.client_order_id, trade_key=leg.trade_key, pair=leg.pair,
                             action=leg.action, symbol=leg.symbol, side=leg.side, qty=leg.qty, tif=leg.tif,
                             submitted_at=stamp)
        try:
            response = broker.submit_order(leg.symbol, leg.qty, leg.side, leg.tif, leg.client_order_id)
            record.alpaca_id, record.status = response.get("id"), response.get("status", "submitted")
        except AlpacaError as exc:
            record.status, record.error = "rejected", str(exc)
            result.refused.append(f"{leg.symbol}: {exc}")
        mirror.orders.append(record)
        mirror.save(path)
        if record.status != "rejected":
            result.sent.append(record)
    return result


def reconcile_llm(account: PaperAccount, broker, mirror: Mirror, path: Path, closes: dict[str, pd.Series],
                  positions: list[dict] | None = None) -> Reconciliation:
    """Refresh the mirrored orders from Alpaca and compare their fills with the simulation's."""
    report = Reconciliation(assumed_bps=account.slippage_bps)
    refresh_orders(broker, mirror, path)
    report.open_orders = sum(o.status not in FINAL for o in mirror.orders)
    sim = {(o.ticker, o.decision_date): o for o in account.orders}
    net: dict[str, float] = {}
    for o in mirror.orders:
        if o.status != "filled" or not o.filled_avg_price:
            continue
        net[o.symbol] = net.get(o.symbol, 0.0) + (o.filled_qty if o.side == "buy" else -o.filled_qty)
        ticker, date = o.trade_key.split("|")
        simulated = sim.get((ticker, date))
        sim_price = simulated.price if simulated is not None and simulated.status == "filled" else None
        close = close_on(closes, o.symbol, o.filled_at)
        sign = 1 if o.side == "buy" else -1                   # positive bps always means worse for us
        report.rows.append(FillRow(
            ticker, o.action, o.symbol, o.side, o.filled_qty, o.filled_avg_price, sim_price,
            sign * (o.filled_avg_price - close) / close * 1e4 if close else None,
            sign * (o.filled_avg_price - sim_price) / sim_price * 1e4 if sim_price else None))
    held = {p["symbol"]: float(p["qty"]) for p in (positions if positions is not None else broker.positions())}
    before = mirror.baseline_positions
    for symbol in sorted(set(net) | set(held) | set(before)):
        expected = before.get(symbol, 0.0) + net.get(symbol, 0.0)
        if abs(expected - held.get(symbol, 0.0)) > 1e-9:
            report.drift.append(f"{symbol}: expected {expected:g} ({before.get(symbol, 0.0):g} held before the first order, "
                                f"{net.get(symbol, 0.0):+g} from orders sent), Alpaca holds {held.get(symbol, 0.0):g}"
                                " (other activity in this account will show here too)")
    return report


def _order_row(order: MirrorOrder, closes: dict[str, pd.Series]) -> dict:
    row = order.model_dump()
    close = close_on(closes, order.symbol, order.filled_at) if order.status == "filled" and order.filled_avg_price else None
    sign = 1 if order.side == "buy" else -1
    row["vs_close_bps"] = sign * (order.filled_avg_price - close) / close * 1e4 if close else None
    return row


def build_alpaca_state(account: PaperAccount, config: dict, broker, fetch, max_order_usd: float | None = None) -> dict:
    """Everything the dashboard's Alpaca section draws, as plain JSON. Reads Alpaca; sends nothing."""
    closes = fetch_all(account, fetch)
    path = llm_mirror_path(config, account.name)
    mirror = Mirror.load(path)
    info, clock, raw_positions = broker.account(), broker.clock(), broker.positions()
    held = {p["symbol"]: float(p["qty"]) for p in raw_positions}
    report = reconcile_llm(account, broker, mirror, path, closes, raw_positions) if mirror.orders else Reconciliation()
    plan = plan_llm_orders(account, closes, held, mirror, "day", max_order_usd)
    equity = float(info.get("equity") or 0)
    costs = [r.vs_close_bps for r in report.rows if r.vs_close_bps is not None]
    number = str(info.get("account_number", ""))
    return {
        "enabled": True, "error": None,
        "account": {"number": f"...{number[-4:]}", "status": info.get("status"), "equity": equity,
                    "buying_power": float(info.get("buying_power") or 0)},
        "market": {"is_open": bool(clock.get("is_open")), "next_open": clock.get("next_open"),
                   "next_close": clock.get("next_close")},
        "positions": [{"symbol": p["symbol"], "qty": float(p["qty"]),
                       "avg_entry": float(p.get("avg_entry_price") or 0), "price": float(p.get("current_price") or 0),
                       "pl": float(p.get("unrealized_pl") or 0), "pl_pct": float(p.get("unrealized_plpc") or 0)}
                      for p in raw_positions],
        "plan": [{"ticker": o.symbol, "rating": o.action, "side": o.side, "qty": o.qty, "price": o.price,
                  "notional": o.notional, "tif": o.tif} for o in plan.orders],
        "skipped": plan.skipped,
        "orders": [_order_row(o, closes) for o in reversed(mirror.orders)],
        "fills": [asdict(r) for r in report.rows],
        "drift": report.drift,
        "metrics": {
            "baseline_equity": mirror.baseline_equity, "baseline_at": mirror.baseline_at, "equity": equity,
            "return_since": (equity / mirror.baseline_equity - 1) if mirror.baseline_equity else None,
            "mean_cost_bps": (sum(costs) / len(costs)) if costs else None, "fills_measured": len(costs),
            "assumed_bps": account.slippage_bps,
            "sent": len(mirror.orders), "filled": sum(o.status == "filled" for o in mirror.orders),
            "rejected": sum(o.status in ("rejected", "canceled", "expired") for o in mirror.orders),
            "open": report.open_orders,
        },
    }
