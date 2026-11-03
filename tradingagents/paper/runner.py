"""Run the graph day by day against a paper account, and report how it is doing.

The account (``account.py``) knows nothing about the graph or the data vendors;
this module is the only place the two meet.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from tradingagents.agents.rating import is_review
from tradingagents.backtest import BacktestSummary, summarize
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.files import locked
from tradingagents.dataflows.symbols import safe_ticker_component
from tradingagents.dataflows.vendors.yahoo.market import get_closes
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.paper.account import PaperAccount, max_drawdown, total_return
from tradingagents.portfolio import PortfolioContext, Position

logger = logging.getLogger(__name__)

BENCHMARK = "SPY"
# Below this many scored decisions the figures say almost nothing about skill.
MIN_DECISIONS_FOR_A_VIEW = 30

ClosesFn = Callable[[str, str, str], pd.Series]


def account_dir(config: dict, name: str) -> Path:
    return Path(config["results_dir"]) / "paper" / safe_ticker_component(name)


def account_path(config: dict, name: str) -> Path:
    return account_dir(config, name) / "account.json"


def create_account(config: dict, name: str, tickers: list[str], cash: float,
                   slippage_bps: float = 5.0, created: str | None = None) -> PaperAccount:
    path = account_path(config, name)
    if path.exists():
        raise ValueError(f"paper account {name!r} already exists at {path}")
    account = PaperAccount(name=name, created=created or get_current_date(),
                           starting_cash=cash, tickers=tickers, slippage_bps=slippage_bps)
    account.save(path)
    return account


def completed_closes(ticker: str, start: str, end: str, fetch: ClosesFn = get_closes) -> pd.Series:
    """Daily closes that are final: positive, tz-naive, and not today's open session."""
    closes = fetch(ticker, start, end)
    closes = closes[closes > 0].dropna()
    tz = getattr(closes.index, "tz", None)
    if not closes.empty:
        today = pd.Timestamp.now(tz="UTC").tz_convert(tz).date() if tz else pd.Timestamp.now().date()
        closes = closes[closes.index.date < today]
    if tz is not None:
        closes = closes.tz_localize(None)
    closes.index = closes.index.normalize()
    return closes


def fetch_all(account: PaperAccount, fetch: ClosesFn = get_closes) -> dict[str, pd.Series]:
    """Closes from the day before creation through today, for the tickers and SPY."""
    start = (datetime.strptime(account.created, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    end = (datetime.strptime(get_current_date(), "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    return {t: completed_closes(t, start, end, fetch) for t in dict.fromkeys([*account.tickers, BENCHMARK])}


class RunInProgress(RuntimeError):
    """Another process or thread is already running this account."""


def market_traded(date: str, fetch: ClosesFn = get_closes) -> bool:
    """Whether the US market held a session on ``date`` (SPY printed a bar for it).

    Meant for a run after the close: during the session the bar exists but is
    partial. A failed fetch raises, so an outage is not mistaken for a holiday.
    """
    end = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    bars = fetch(BENCHMARK, date, end)
    return bool(len(bars)) and any(ts.strftime("%Y-%m-%d") == date for ts in bars.index)


def new_york_today() -> str:
    """Today's date in New York, the market's own calendar day, wherever this machine is."""
    import pytz

    return datetime.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d")


def log_run(config: dict, name: str, message: str) -> None:
    """Append one line to the account's ``runs.log``: the answer to "did it run?"."""
    path = account_dir(config, name) / "runs.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}\n")


def book_for(account: PaperAccount) -> PortfolioContext:
    """The account as the decision agents see it: real fills only, never a pending order."""
    return PortfolioContext(
        cash=round(account.cash, 2), currency="USD",
        positions=[Position(ticker=t, quantity=shares, average_price=cost)
                   for t, (shares, cost) in account.holdings().items()],
    )


@dataclass
class DayResult:
    date: str
    decided: list[tuple[str, str]] = field(default_factory=list)   # (ticker, rating)
    skipped: list[str] = field(default_factory=list)               # already decided that day
    failures: list[tuple[str, str]] = field(default_factory=list)
    filled: int = 0


def run_day(account: PaperAccount, config: dict, *args, **kwargs) -> DayResult:
    """Run one day for the account, one run at a time per account.

    The account is re-read under the lock, so a run that just finished elsewhere
    (the scheduler, the dashboard) is not overwritten with a stale copy.
    """
    path = account_path(config, account.name)
    with locked(path, wait=False) as held:
        if not held:
            raise RunInProgress(f"a run for paper account {account.name!r} is already in progress")
        fresh = PaperAccount.load(path)
        for field_name in PaperAccount.model_fields:
            setattr(account, field_name, getattr(fresh, field_name))
        return _run_day(account, config, *args, **kwargs)


def _run_day(
    account: PaperAccount,
    config: dict,
    date: str | None = None,
    selected_analysts=("market", "social", "news", "fundamentals"),
    graph: TradingAgentsGraph | None = None,
    fetch: ClosesFn = get_closes,
    progress: Callable[[int, int, str], None] | None = None,
) -> DayResult:
    """Fill what can be filled, then decide for every ticker on ``date`` (default today).

    The account is saved after each ticker, so an interrupted day resumes by being
    run again: a ticker already decided for the date is skipped.
    """
    date = date or get_current_date()
    path = account_path(config, account.name)
    result = DayResult(date=date)

    result.filled = len(account.fill_pending(fetch_all(account, fetch)))
    account.save(path)

    if graph is None:
        run_dir = account_dir(config, account.name)
        graph = TradingAgentsGraph(selected_analysts, config={
            **config, "results_dir": str(run_dir), "memory_log_path": str(run_dir / "trading_memory.md")})

    todo = [t for t in account.tickers if not account.has_decision(t, date)]
    result.skipped = [t for t in account.tickers if t not in todo]
    for index, ticker in enumerate(todo, 1):
        if progress:
            progress(index, len(todo), ticker)
        try:
            _, rating = graph.propagate(ticker, date, "stock", portfolio=book_for(account))
        except Exception as exc:  # one unreachable vendor must not end the day
            logger.warning("Paper decision %s %s failed: %s", ticker, date, exc)
            result.failures.append((ticker, str(exc)))
            continue
        account.record(ticker, date, "REVIEW" if is_review(rating) else rating)
        account.save(path)
        result.decided.append((ticker, rating))

    for ticker in account.tickers:  # settle matured decisions so `status` can score them
        try:
            graph.settle_pending(ticker)
        except Exception as exc:
            logger.warning("Settling %s failed: %s", ticker, exc)
    return result


@dataclass
class Status:
    account: PaperAccount
    curves: dict[str, pd.Series]
    decisions: BacktestSummary | None
    pending: int

    def render(self) -> str:
        a = self.account
        filled = sum(o.status == "filled" for o in a.orders)
        lines = [f"Paper account {a.name!r}: {', '.join(a.tickers)} · started {a.created} "
                 f"with ${a.starting_cash:,.0f} · slippage {a.slippage_bps:g} bps",
                 f"Orders: {len(a.orders)} decisions, {filled} filled, {self.pending} awaiting their entry close", ""]
        if all(not len(c) for c in self.curves.values()):
            lines.append("No closes after the start date yet; run again after the next session.")
        else:
            lines.append(f"{'':28}{'return':>9}{'max drawdown':>14}")
            for label, curve in self.curves.items():
                ret, dd = total_return(curve, a.starting_cash), max_drawdown(curve)
                if ret is None:
                    continue
                lines.append(f"{label:28}{ret:>+9.2%}{dd:>+14.2%}")
            holdings = a.holdings()
            if holdings:
                lines += ["", "Holding: " + ", ".join(f"{t} {s:,.4g} @ {c:,.2f}" for t, (s, c) in holdings.items())
                          + f" · cash ${a.cash:,.2f}"]
        if self.decisions is not None:
            lines += ["", self.decisions.render()]
            if self.decisions.resolved < MIN_DECISIONS_FOR_A_VIEW:
                lines += ["", f"Only {self.decisions.resolved} scored decisions (under {MIN_DECISIONS_FOR_A_VIEW}): "
                              "this is noise, not evidence. Judge it after many more, and against the "
                              "buy-and-hold line above, not against zero."]
        return "\n".join(lines)


def status(account: PaperAccount, config: dict, fetch: ClosesFn = get_closes) -> Status:
    closes = fetch_all(account, fetch)
    account.fill_pending(closes)        # in memory only: reporting never writes the ledger
    curves = {
        "Strategy": account.equity_curve(closes),
        f"Buy & hold {'/'.join(account.tickers)}": account.buy_and_hold(closes),
        f"Buy & hold {BENCHMARK}": account.buy_and_hold(closes, [BENCHMARK]),
    }
    log = account_dir(config, account.name) / "trading_memory.md"
    return Status(account, curves, summarize(log) if log.is_file() else None,
                  pending=sum(o.status == "pending" for o in account.orders))
