"""Mirror a pairs account's decisions to an Alpaca PAPER account, and measure the fills.

The simulated ledger (``pairs.py``) stays the source of every decision. This module
sends the same decisions to Alpaca's paper-trading service as market-on-close orders
on the session the simulation fills them, then compares what Alpaca actually filled
with what the simulation assumed. That comparison is the point: it shows whether the
5 bps of slippage in the model is realistic.

Paper only, by construction:
- the only host this module will talk to is ``paper-api.alpaca.markets`` (a constant;
  the client takes no URL), and a live-account key is rejected by that host;
- the keys are read from ``ALPACA_PAPER_API_KEY`` / ``ALPACA_PAPER_API_SECRET``, names
  that differ from the ones Alpaca's own tools use for live keys;
- nothing is sent without an explicit ``--send``; the default is a dry run that prints
  the orders;
- orders are whole-share market orders capped in dollars, with deterministic client
  order ids, so running the command twice cannot send a decision twice.

Written from Alpaca's public REST documentation and tested against fakes only. Read a
dry run before the first ``--send``.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from pydantic import BaseModel

from tradingagents.paper.pairs import PairsAccount, Trade
from tradingagents.paper.runner import account_dir

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
ENV_KEY, ENV_SECRET = "ALPACA_PAPER_API_KEY", "ALPACA_PAPER_API_SECRET"
CLOSE_CUTOFF_MINUTES = 15          # market-on-close orders must reach Alpaca before about 3:50 pm New York
FINAL = {"filled", "canceled", "expired", "rejected"}


class AlpacaError(RuntimeError):
    """A request failed or a safety check refused. Never carries a key or a header."""


class AlpacaPaper:
    """A minimal client for Alpaca's paper-trading REST API."""

    def __init__(self, key: str | None = None, secret: str | None = None, session=None, timeout: float = 15):
        self._key = key or os.environ.get(ENV_KEY, "")
        self._secret = secret or os.environ.get(ENV_SECRET, "")
        if not self._key or not self._secret:
            raise AlpacaError(f"set {ENV_KEY} and {ENV_SECRET} to your Alpaca PAPER keys (in the environment or .env)")
        self._session = session or requests.Session()
        self._timeout = timeout

    def _call(self, method: str, path: str, **kwargs):
        response = self._session.request(
            method, PAPER_BASE_URL + path, timeout=self._timeout, **kwargs,
            headers={"APCA-API-KEY-ID": self._key, "APCA-API-SECRET-KEY": self._secret})
        if response.status_code >= 400:
            try:
                detail = response.json().get("message", "")
            except ValueError:
                detail = ""
            raise AlpacaError(f"Alpaca {method} {path}: HTTP {response.status_code} {detail}".strip())
        return response.json() if response.content else {}

    def account(self) -> dict:
        return self._call("GET", "/v2/account")

    def clock(self) -> dict:
        return self._call("GET", "/v2/clock")

    def positions(self) -> list[dict]:
        return self._call("GET", "/v2/positions")

    def asset(self, symbol: str) -> dict:
        return self._call("GET", f"/v2/assets/{symbol}")

    def submit_order(self, symbol: str, qty: int, side: str, time_in_force: str, client_order_id: str) -> dict:
        return self._call("POST", "/v2/orders", json={
            "symbol": symbol, "qty": str(qty), "side": side, "type": "market",
            "time_in_force": time_in_force, "client_order_id": client_order_id})

    def order_by_client_id(self, client_order_id: str) -> dict:
        return self._call("GET", "/v2/orders:by_client_order_id", params={"client_order_id": client_order_id})

    def cancel_order(self, order_id: str) -> None:
        self._call("DELETE", f"/v2/orders/{order_id}")


# --- the record of what was sent ---------------------------------------------------------

class MirrorOrder(BaseModel):
    client_order_id: str
    trade_key: str                      # "KO/PEP|2026-11-05": the pair and the day the trade was decided
    pair: str
    action: str                         # "enter" or "exit"
    symbol: str
    side: str                           # "buy" or "sell"
    qty: int
    tif: str
    submitted_at: str
    alpaca_id: str | None = None
    status: str = "submitted"
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    filled_at: str | None = None
    error: str | None = None


class Mirror(BaseModel):
    orders: list[MirrorOrder] = []

    def sent(self, trade_key: str, action: str) -> list[MirrorOrder]:
        return [o for o in self.orders if o.trade_key == trade_key and o.action == action]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> Mirror:
        return cls.model_validate_json(path.read_text(encoding="utf-8")) if path.is_file() else cls()


def mirror_path(config: dict, name: str) -> Path:
    return account_dir(config, name) / "alpaca_mirror.json"


# --- planning ---------------------------------------------------------------------------

@dataclass
class Planned:
    trade_key: str
    pair: str
    action: str
    symbol: str
    side: str
    qty: int
    tif: str
    client_order_id: str
    price: float                        # the last close, for the dollar caps and the printout
    trade: Trade | None = None

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass
class Plan:
    orders: list[Planned] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def trade_key(trade: Trade) -> str:
    return f"{trade.pair}|{trade.decided}"


def client_order_id(account_name: str, key: str, action: str, symbol: str) -> str:
    digest = hashlib.sha1(f"{account_name}|{key}".encode()).hexdigest()[:10]
    return f"ta-{digest}-{action}-{symbol}"[:120]


def plan_orders(account: PairsAccount, closes: dict[str, pd.Series], mirror: Mirror, tif: str = "cls",
                max_order_usd: float | None = None, max_gross_usd: float | None = None) -> Plan:
    """The paper orders that match the simulation's decisions awaiting their next close.

    A decision is acted on only if it was made at the most recent completed close,
    because that is the session the simulation fills it. Anything older is stale: the
    simulation has already moved on, and a late order would not be comparable.
    """
    plan = Plan()
    cap_order = max_order_usd if max_order_usd is not None else account.sleeve
    cap_gross = max_gross_usd if max_gross_usd is not None else account.starting_cash
    open_gross = sum(o.qty * (o.filled_avg_price or 0.0) for o in mirror.orders
                     if o.action == "enter" and o.status == "filled"
                     and not mirror.sent(o.trade_key, "exit"))
    for pair in account.pairs:
        trade = account.active(pair)
        if trade is None or trade.status not in ("pending_entry", "pending_exit"):
            continue
        a, b = pair.split("/")
        both = pd.concat([closes[a], closes[b]], axis=1, keys=["a", "b"]).dropna()
        key, action = trade_key(trade), ("enter" if trade.status == "pending_entry" else "exit")
        decided = trade.decided if action == "enter" else trade.exit_decided
        if both.empty or both.index[-1].strftime("%Y-%m-%d") != decided:
            plan.skipped.append(f"{pair} {action}: stale (decided {decided}, the simulation fills it at a close already passed)")
            continue
        if mirror.sent(key, action):
            plan.skipped.append(f"{pair} {action}: already sent")
            continue
        pa, pb = float(both["a"].iloc[-1]), float(both["b"].iloc[-1])
        if action == "enter":
            gross = account.sleeve
            qty_a = math.floor(gross / (1 + trade.beta) / pa)
            qty_b = math.floor(gross * trade.beta / (1 + trade.beta) / pb)
            if qty_a < 1 or qty_b < 1:
                plan.skipped.append(f"{pair} enter: a leg is under one whole share at ${gross:,.0f} gross (A {qty_a}, B {qty_b})")
                continue
            side_a, side_b = ("buy", "sell") if trade.side == "long_spread" else ("sell", "buy")
            legs = sorted([(a, side_a, qty_a, pa), (b, side_b, qty_b, pb)], key=lambda leg: leg[1] != "sell")   # short leg first
            cost = qty_a * pa + qty_b * pb
            if max(qty_a * pa, qty_b * pb) > cap_order or open_gross + cost > cap_gross:
                plan.skipped.append(f"{pair} enter: over the dollar cap (leg limit ${cap_order:,.0f}, gross limit ${cap_gross:,.0f})")
                continue
            open_gross += cost
        else:
            entries = [o for o in mirror.sent(key, "enter") if o.status == "filled" and o.filled_qty > 0]
            if not entries:
                plan.skipped.append(f"{pair} exit: the entry was never filled on Alpaca, so there is nothing to close")
                continue
            price = {a: pa, b: pb}
            legs = [(o.symbol, "sell" if o.side == "buy" else "buy", int(o.filled_qty), price[o.symbol]) for o in entries]
        for symbol, side, qty, price_ in legs:
            plan.orders.append(Planned(key, pair, action, symbol, side, qty, tif,
                                       client_order_id(account.name, key, action, symbol), price_, trade))
    return plan


# --- sending -----------------------------------------------------------------------------

@dataclass
class SendResult:
    sent: list[MirrorOrder] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)


def check_market(broker, tif: str) -> None:
    clock = broker.clock()
    if not clock.get("is_open"):
        raise AlpacaError("the market is closed; run this on a trading day before the close")
    if tif == "cls":
        now, close = pd.Timestamp(clock["timestamp"]), pd.Timestamp(clock["next_close"])
        if close - now < timedelta(minutes=CLOSE_CUTOFF_MINUTES):
            raise AlpacaError(f"under {CLOSE_CUTOFF_MINUTES} minutes to the close: market-on-close orders may be refused")


def send_plan(broker, plan: Plan, mirror: Mirror, path: Path, now: datetime | None = None) -> SendResult:
    """Submit a plan to the paper account, recording every order as it goes."""
    result = SendResult()
    stamp = (now or datetime.now().astimezone()).isoformat(timespec="seconds")
    account = broker.account()
    if account.get("status") != "ACTIVE" or account.get("trading_blocked"):
        raise AlpacaError(f"the Alpaca paper account cannot trade (status {account.get('status')}, "
                          f"blocked {account.get('trading_blocked')})")
    buying_power = float(account.get("buying_power") or 0)
    by_trade: dict[tuple[str, str], list[Planned]] = {}
    for order in plan.orders:
        by_trade.setdefault((order.trade_key, order.action), []).append(order)

    for (key, action), legs in by_trade.items():
        if action == "enter":
            if not account.get("shorting_enabled", False):
                result.refused.append(f"{key}: shorting is not enabled on this Alpaca account")
                continue
            need = sum(leg.notional for leg in legs)
            if need > buying_power:
                result.refused.append(f"{key}: needs about ${need:,.0f}, buying power is ${buying_power:,.0f}")
                continue
            blocked = None
            for leg in (x for x in legs if x.side == "sell"):
                asset = broker.asset(leg.symbol)
                if not (asset.get("tradable") and asset.get("shortable")):
                    blocked = f"{key}: {leg.symbol} cannot be sold short on Alpaca"
                    break
            if blocked:
                result.refused.append(blocked)
                continue
        placed: list[MirrorOrder] = []
        for leg in legs:
            record = MirrorOrder(client_order_id=leg.client_order_id, trade_key=key, pair=leg.pair, action=action,
                                 symbol=leg.symbol, side=leg.side, qty=leg.qty, tif=leg.tif, submitted_at=stamp)
            try:
                response = broker.submit_order(leg.symbol, leg.qty, leg.side, leg.tif, leg.client_order_id)
                record.alpaca_id, record.status = response.get("id"), response.get("status", "submitted")
            except AlpacaError as exc:
                record.status, record.error = "rejected", str(exc)
                result.refused.append(f"{key} {leg.symbol}: {exc}")
            mirror.orders.append(record)
            placed.append(record)
            mirror.save(path)
            if record.status == "rejected" and action == "enter":
                for other in placed:                          # never leave one leg of an entry standing
                    if other.status != "rejected" and other.alpaca_id:
                        try:
                            broker.cancel_order(other.alpaca_id)
                            other.status, other.error = "canceled", "canceled: the other leg was refused"
                        except AlpacaError as exc:
                            result.refused.append(f"{key}: could not cancel {other.symbol} ({exc}); check Alpaca")
                mirror.save(path)
                break
        result.sent += [o for o in placed if o.status != "rejected"]
    return result


# --- reconciling -------------------------------------------------------------------------

@dataclass
class FillRow:
    pair: str
    action: str
    symbol: str
    side: str
    qty: float
    alpaca_price: float
    sim_price: float | None
    vs_close_bps: float | None          # positive = worse than the official close
    vs_sim_bps: float | None


@dataclass
class Reconciliation:
    rows: list[FillRow] = field(default_factory=list)
    open_orders: int = 0
    drift: list[str] = field(default_factory=list)
    assumed_bps: float = 0.0

    def render(self) -> str:
        lines = [f"{len(self.rows)} filled order(s); {self.open_orders} not final yet"]
        if self.rows:
            lines += ["", f"{'Pair':9}{'Action':7}{'Symbol':8}{'Side':6}{'Qty':>6}{'Alpaca':>10}{'Sim':>10}{'vs close':>10}{'vs sim':>9}"]
            for r in self.rows:
                close = f"{r.vs_close_bps:+.1f}" if r.vs_close_bps is not None else "–"
                sim_p = f"{r.sim_price:.2f}" if r.sim_price is not None else "–"
                vs_sim = f"{r.vs_sim_bps:+.1f}" if r.vs_sim_bps is not None else "–"
                lines.append(f"{r.pair:9}{r.action:7}{r.symbol:8}{r.side:6}{r.qty:6.0f}{r.alpaca_price:10.2f}{sim_p:>10}{close:>10}{vs_sim:>9}")
            measured = [r.vs_close_bps for r in self.rows if r.vs_close_bps is not None]
            if measured:
                lines += ["", f"Mean cost vs the official close: {sum(measured) / len(measured):+.1f} bps per fill "
                              f"(the simulation assumes {self.assumed_bps:+.1f}). Positive means worse than the close.",
                          "Basis points are per fill; with few fills this is an anecdote, not an estimate."]
        if self.drift:
            lines += ["", "Positions on Alpaca that do not match what was sent:"] + [f"  {d}" for d in self.drift]
        return "\n".join(lines)


def reconcile(account: PairsAccount, broker, mirror: Mirror, path: Path, closes: dict[str, pd.Series]) -> Reconciliation:
    """Refresh every mirrored order from Alpaca, then compare fills with the simulation."""
    report = Reconciliation(assumed_bps=account.slippage_bps)
    for order in mirror.orders:
        if order.status in FINAL:
            continue
        try:
            data = broker.order_by_client_id(order.client_order_id)
        except AlpacaError as exc:
            order.error = str(exc)
            continue
        order.alpaca_id = data.get("id", order.alpaca_id)
        order.status = data.get("status", order.status)
        order.filled_qty = float(data.get("filled_qty") or 0)
        order.filled_avg_price = float(data["filled_avg_price"]) if data.get("filled_avg_price") else None
        order.filled_at = data.get("filled_at")
    mirror.save(path)
    report.open_orders = sum(o.status not in FINAL for o in mirror.orders)

    trades = {trade_key(t): t for t in account.trades}
    net: dict[str, float] = {}
    for o in mirror.orders:
        if o.status != "filled" or not o.filled_avg_price:
            continue
        net[o.symbol] = net.get(o.symbol, 0.0) + (o.filled_qty if o.side == "buy" else -o.filled_qty)
        trade = trades.get(o.trade_key)
        sim = None
        if trade is not None:
            a, _ = o.pair.split("/")
            if o.action == "enter":
                sim = trade.a_entry if o.symbol == a else trade.b_entry
            else:
                sim = trade.a_exit if o.symbol == a else trade.b_exit
        close = None
        if o.filled_at and o.symbol in closes:
            stamp = pd.Timestamp(o.filled_at)
            day = (stamp.tz_convert("America/New_York").tz_localize(None) if stamp.tzinfo else stamp).normalize()
            series = closes[o.symbol]
            same = series[series.index == day]
            close = float(same.iloc[0]) if len(same) else None
        sign = 1 if o.side == "buy" else -1               # positive bps always means worse for us
        report.rows.append(FillRow(
            o.pair, o.action, o.symbol, o.side, o.filled_qty, o.filled_avg_price, sim,
            sign * (o.filled_avg_price - close) / close * 1e4 if close else None,
            sign * (o.filled_avg_price - sim) / sim * 1e4 if sim else None))
    held = {p["symbol"]: float(p["qty"]) for p in broker.positions()}
    for symbol in sorted(set(net) | set(held)):
        if abs(net.get(symbol, 0.0) - held.get(symbol, 0.0)) > 1e-9:
            report.drift.append(f"{symbol}: sent orders net {net.get(symbol, 0.0):+.0f}, Alpaca holds {held.get(symbol, 0.0):+.0f}"
                                " (other activity in this account will show here too)")
    return report
