"""Run and report a pairs account: the only part of the pairs mode that touches prices."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.files import locked
from tradingagents.dataflows.vendors.yahoo.market import get_closes
from tradingagents.paper.account import max_drawdown, total_return
from tradingagents.paper.pairs import PairsAccount
from tradingagents.paper.runner import BENCHMARK, RunInProgress, account_dir, completed_closes

MIN_TRADES_FOR_A_VIEW = 30
TRADES_PER_PAIR_YEAR = 3          # the 1990-2022 average of the rules, for the sample-size note
HISTORY_BUFFER_DAYS = 460         # ~ one 252-session window plus slack, before the first session handled


def pairs_path(config: dict, name: str):
    return account_dir(config, name) / "pairs.json"


def create_pairs_account(config: dict, name: str, pairs: list[str], cash: float,
                         slippage_bps: float = 5.0, borrow_annual: float = 0.005,
                         created: str | None = None) -> PairsAccount:
    folder = account_dir(config, name)
    if (folder / "account.json").exists() or pairs_path(config, name).exists():
        raise ValueError(f"an account named {name!r} already exists in {folder}")
    account = PairsAccount(name=name, created=created or get_current_date(), starting_cash=cash,
                           pairs=pairs, slippage_bps=slippage_bps, borrow_annual=borrow_annual)
    account.save(pairs_path(config, name))
    return account


def fetch_pairs_closes(account: PairsAccount, fetch=None) -> dict[str, pd.Series]:
    """Completed closes for every ticker and SPY, from before the earliest session to handle."""
    fetch = fetch or get_closes
    earliest = min([account.created, *account.cursors.values()])
    start = (datetime.strptime(earliest, "%Y-%m-%d") - timedelta(days=HISTORY_BUFFER_DAYS)).strftime("%Y-%m-%d")
    end = (datetime.strptime(get_current_date(), "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    return {t: completed_closes(t, start, end, fetch) for t in dict.fromkeys([*account.tickers, BENCHMARK])}


@dataclass
class PairsRun:
    events: list[str] = field(default_factory=list)
    short_history: list[str] = field(default_factory=list)   # pairs without a full window yet


def run_pairs(account: PairsAccount, config: dict, fetch=None) -> PairsRun:
    """Replay every completed session since the last run, then save.

    One run per account at a time. The account is re-read under the lock so a run
    that just finished elsewhere is not overwritten with a stale copy.
    """
    path = pairs_path(config, account.name)
    with locked(path, wait=False) as held:
        if not held:
            raise RunInProgress(f"a run for pairs account {account.name!r} is already in progress")
        fresh = PairsAccount.load(path)
        for name in PairsAccount.model_fields:
            setattr(account, name, getattr(fresh, name))
        closes = fetch_pairs_closes(account, fetch)
        result = PairsRun(events=account.process(closes))
        for pair in account.pairs:
            a, b = pair.split("/")
            if a in closes and b in closes:
                both = pd.concat([closes[a], closes[b]], axis=1).dropna()
                if len(both) <= account.rules.lookback:
                    result.short_history.append(f"{pair} ({len(both)} sessions of {account.rules.lookback})")
            else:
                result.short_history.append(f"{pair} (no prices)")
        account.save(path)
    return result


@dataclass
class PairsStatus:
    account: PairsAccount
    rows: list[dict]
    equity: pd.Series
    spy: pd.Series
    beta_to_spy: tuple[float, int] | None

    def render(self) -> str:
        a, r = self.account, self.account.rules
        lines = [
            f"Pairs account {a.name!r}: {', '.join(a.pairs)} · started {a.created} with ${a.starting_cash:,.0f} "
            f"(${a.sleeve:,.0f} gross per pair) · slippage {a.slippage_bps:g} bps · borrow {a.borrow_annual:.1%}/yr",
            f"Fixed rules: enter |z|>{r.entry:g}, exit |z|<{r.exit:g}, stop |z|>{r.stop:g}, "
            f"{r.max_hold}-session limit, {r.lookback}-session window; trades at the next close", "",
            f"{'Pair':11}{'State':15}{'Side':14}{'Held':>5}{'z now':>8}{'beta':>7}{'P&L':>9}",
        ]
        for row in self.rows:
            z = f"{row['z']:+.2f}" if row["z"] is not None else "–"
            beta = f"{row['beta']:.2f}" if row["beta"] is not None else "–"
            pnl = f"{row['pnl_pct']:+.2%}" if row["pnl_pct"] is not None else "–"
            lines.append(f"{row['pair']:11}{row['state']:15}{row['side'] or '–':14}{row['held'] or '–':>5}"
                         f"{z:>8}{beta:>7}{pnl:>9}")
        lines.append("")
        if len(self.equity):
            ret, dd = total_return(self.equity, a.starting_cash), max_drawdown(self.equity)
            spy = total_return(self.spy, a.starting_cash) if len(self.spy) else None
            lines.append(f"Account value ${self.equity.iloc[-1]:,.2f} ({ret:+.2%}) · max drawdown {dd:.2%}"
                         + (f" · SPY buy-and-hold {spy:+.2%}" if spy is not None else ""))
            if self.beta_to_spy:
                beta, n = self.beta_to_spy
                lines.append(f"Beta to SPY {beta:+.2f} over {n} days: a market-neutral book should sit near 0.")
        else:
            lines.append("No sessions after the start date yet; run again after the next close.")
        done = a.closed()
        if done:
            pnls = [(t.a_shares * (t.a_exit - t.a_entry) + t.b_shares * (t.b_exit - t.b_entry) - t.borrow) / a.sleeve
                    for t in done]
            stops = sum(t.exit_reason == "stop" for t in done) / len(done)
            lines.append(f"Closed trades: {len(done)} · won {sum(p > 0 for p in pnls) / len(done):.0%} · "
                         f"mean {np.mean(pnls):+.2%} of the pair's capital · worst {min(pnls):+.2%} · stopped out {stops:.0%}")
        else:
            lines.append("Closed trades: 0")
        if len(done) < MIN_TRADES_FOR_A_VIEW:
            per_year = TRADES_PER_PAIR_YEAR * len(a.pairs)
            lines += ["", f"Under {MIN_TRADES_FOR_A_VIEW} closed trades this is noise, not evidence. At about "
                          f"{TRADES_PER_PAIR_YEAR} trades per pair a year (the 1990-2022 average) {len(a.pairs)} pair(s) "
                          f"give about {per_year} a year, so expect roughly {MIN_TRADES_FOR_A_VIEW / per_year:.1f} years. "
                          "Judge the result against cash, not against SPY."]
        return "\n".join(lines)


def pairs_status(account: PairsAccount, fetch=None) -> PairsStatus:
    """The account as of the latest completed close. Reporting never writes the ledger."""
    closes = fetch_pairs_closes(account, fetch)
    account.process(closes)
    equity = account.equity_curve(closes)
    spy_closes = closes.get(BENCHMARK, pd.Series(dtype=float))
    days = equity.index
    spy = pd.Series(dtype=float)
    if len(days) and len(spy_closes):
        entry = float(spy_closes[spy_closes.index >= days[0]].iloc[0])
        spy = pd.Series([account.starting_cash / entry * float(spy_closes[spy_closes.index <= d].iloc[-1])
                         if (spy_closes.index <= d).any() else np.nan for d in days], index=days).dropna()
    beta = None
    if len(equity) and len(spy):
        joined = pd.concat([equity.pct_change(), spy.pct_change()], axis=1, keys=["e", "s"]).dropna()
        if len(joined) >= 30 and joined["s"].var() > 0:
            beta = (float(joined["e"].cov(joined["s"]) / joined["s"].var()), len(joined))

    rows = []
    for pair in account.pairs:
        trade, now = account.active(pair), account.current_z(pair, closes)
        pnl = account.trade_pnl(trade, closes) if trade is not None and trade.entered else None
        held = None
        if trade is not None and trade.entered:
            a, b = pair.split("/")
            sessions = pd.concat([closes[a], closes[b]], axis=1).dropna().index
            held = int((sessions >= pd.Timestamp(trade.entered)).sum())
        rows.append({
            "pair": pair, "state": trade.status.replace("_", " ") if trade else "flat",
            "side": trade.side.replace("_", " ") if trade else None, "held": held,
            "z": now[0] if now else None, "beta": now[1] if now else None,
            "pnl_pct": pnl / account.sleeve if pnl is not None else None,
        })
    return PairsStatus(account, rows, equity, spy, beta)
