"""Market-neutral pairs paper trading: rules fixed in advance, filled at the next close,
costs charged, and a missed day replayed exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from cli.main import app
from tradingagents.dataflows.files import locked
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.paper import pairs_runner
from tradingagents.paper.pairs import PairsAccount, Rules
from tradingagents.paper.runner import RunInProgress

CLI = CliRunner()
LOOKBACK = 60


def make_pair(n=100, jump_at=None, jump=0.03, back_at=None, after=None, beta=1.0):
    """Two price series whose log spread zigzags +-0.01, with an optional jump."""
    k = np.arange(n)
    log_b = 4 + 0.0008 * k + 0.02 * np.sin(2 * np.pi * k / 17)
    spread = 0.01 * np.where(k % 2 == 0, 1.0, -1.0)
    if jump_at is not None:
        end = back_at if back_at is not None else n
        spread[jump_at:end] += jump
        if after is not None:                 # the spread widens further once the trade is on
            spread[jump_at + 1:end] += after
    index = pd.bdate_range("2024-01-02", periods=n)
    return {"A": pd.Series(np.exp(beta * log_b + spread), index=index),
            "B": pd.Series(np.exp(log_b), index=index)}


def account(**rules):
    return PairsAccount(name="p", created="2024-01-01", starting_cash=1000.0, pairs=["A/B"],
                        slippage_bps=5.0, borrow_annual=0.005, rules=Rules(lookback=LOOKBACK, **rules))


def reference_first_entry(closes, rules):
    """The first session the rules should act on, computed with pandas rolling windows."""
    la, lb = np.log(closes["A"]), np.log(closes["B"])
    n = rules.lookback
    cov, var_b = la.rolling(n).cov(lb), lb.rolling(n).var()
    beta = cov / var_b
    spread = la - beta * lb
    mu = la.rolling(n).mean() - beta * lb.rolling(n).mean()
    sd = np.sqrt(la.rolling(n).var() - 2 * beta * cov + beta ** 2 * var_b)
    z = (spread - mu) / sd
    hits = z[(z.abs() > rules.entry) & beta.between(rules.beta_min, rules.beta_max)]
    hits = hits[hits.index > pd.Timestamp("2024-01-01")]
    hits = hits[[i >= n for i in range(len(z)) if z.index[i] in hits.index]]
    return hits.index[0], float(hits.iloc[0])


@pytest.mark.unit
def test_decides_at_the_close_and_fills_at_the_next_close_with_slippage():
    closes = make_pair(jump_at=70)
    acct = account()
    acct.process(closes)
    when, z = reference_first_entry(closes, acct.rules)
    trade = acct.trades[0]
    index = closes["A"].index
    assert trade.decided == when.strftime("%Y-%m-%d") and trade.side == "short_spread"   # z > 0: A ran ahead
    assert pd.Timestamp(trade.entered) == index[index.get_loc(when) + 1]
    day = pd.Timestamp(trade.entered)
    assert trade.a_entry == pytest.approx(closes["A"][day] * (1 - 5e-4))     # selling A short: worse price
    assert trade.b_entry == pytest.approx(closes["B"][day] * (1 + 5e-4))     # buying B: worse price
    assert trade.a_shares < 0 < trade.b_shares


@pytest.mark.unit
def test_legs_are_sized_by_beta_and_the_sleeve_is_the_gross():
    closes = make_pair(jump_at=70)
    acct = account()
    acct.process(closes)
    t = acct.trades[0]
    notional_a, notional_b = abs(t.a_shares) * t.a_entry, abs(t.b_shares) * t.b_entry
    assert notional_a + notional_b == pytest.approx(acct.sleeve)
    assert notional_b / notional_a == pytest.approx(t.beta)


@pytest.mark.unit
def test_convergence_closes_the_trade_and_the_pnl_matches_a_hand_calculation():
    closes = make_pair(n=100, jump_at=70, back_at=76)
    acct = account()
    acct.process(closes)
    t = acct.trades[0]
    assert t.status == "closed" and t.exit_reason == "converged"
    held_closes = closes["A"][(closes["A"].index > pd.Timestamp(t.entered)) & (closes["A"].index <= pd.Timestamp(t.exited))]
    short_series = closes["A"] if t.side == "short_spread" else closes["B"]
    short_shares = t.a_shares if t.side == "short_spread" else t.b_shares
    borrow = 0.005 / 252 * abs(short_shares) * short_series[held_closes.index].sum()
    by_hand = t.a_shares * (t.a_exit - t.a_entry) + t.b_shares * (t.b_exit - t.b_entry) - borrow
    assert acct.trade_pnl(t, closes) == pytest.approx(by_hand)
    assert by_hand > 0                                                       # the spread came back
    assert acct.equity_curve(closes).iloc[-1] == pytest.approx(1000.0 + by_hand)


@pytest.mark.unit
def test_a_spread_that_keeps_widening_is_stopped_out():
    closes = make_pair(n=100, jump_at=70, after=0.15)
    acct = account()
    acct.process(closes)
    t = acct.trades[0]
    assert t.exit_reason == "stop" and t.status == "closed"
    assert acct.trade_pnl(t, closes) < 0                                    # short the leg that kept running


@pytest.mark.unit
def test_time_stop_leaves_after_max_hold_sessions_and_costs_money_when_nothing_moves():
    closes = make_pair(n=110, jump_at=70)                                   # the gap never closes
    acct = account(max_hold=10)
    acct.process(closes)
    t = acct.trades[0]
    index = closes["A"].index
    assert t.exit_reason == "time"
    assert index.get_loc(pd.Timestamp(t.exit_decided)) - index.get_loc(pd.Timestamp(t.entered)) + 1 == 10
    pnl = acct.trade_pnl(t, closes)
    assert -0.01 * acct.sleeve < pnl < 0                                    # slippage and borrow, nothing else


@pytest.mark.unit
def test_replaying_missed_days_equals_running_every_day():
    closes = make_pair(n=100, jump_at=70, back_at=76)
    once = account()
    once.process(closes)
    daily = account()
    for end in range(LOOKBACK - 5, 101):
        daily.process({t: s.iloc[:end] for t, s in closes.items()})
    assert once.trades == daily.trades and once.cursors == daily.cursors


@pytest.mark.unit
def test_processing_twice_does_nothing_new():
    closes = make_pair(jump_at=70, back_at=76)
    acct = account()
    acct.process(closes)
    snapshot = acct.model_dump()
    assert acct.process(closes) == [] and acct.model_dump() == snapshot


@pytest.mark.unit
def test_a_beta_outside_the_band_never_trades():
    closes = make_pair(jump_at=70, beta=-0.5)                               # A moves against B
    acct = account()
    acct.process(closes)
    assert acct.trades == []


@pytest.mark.unit
def test_too_little_history_means_no_signals():
    acct = account()
    acct.process(make_pair(n=40, jump_at=30))
    assert acct.trades == [] and acct.current_z("A/B", make_pair(n=40)) is None


@pytest.mark.unit
@pytest.mark.parametrize("bad", [["KO/KO"], ["KO"], ["KO/PEP", "KO/PEP"], ["/PEP"]])
def test_a_pair_is_two_different_tickers_listed_once(bad):
    with pytest.raises(ValueError):
        PairsAccount(name="p", created="2024-01-01", starting_cash=1000.0, pairs=bad)


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setitem(DEFAULT_CONFIG, "results_dir", str(tmp_path))
    monkeypatch.setattr(pairs_runner, "get_current_date", lambda: "2024-01-01")
    closes = make_pair(n=330, jump_at=300, back_at=306)
    monkeypatch.setattr(pairs_runner, "get_closes", lambda t, s, e: {"KO": closes["A"], "PEP": closes["B"],
                                                                      "SPY": closes["B"]}[t])
    return tmp_path


@pytest.mark.unit
def test_the_cli_creates_runs_and_reports(results):
    assert CLI.invoke(app, ["paper", "pairs", "init", "pp", "ko/pep", "--cash", "2000"]).exit_code == 0
    assert CLI.invoke(app, ["paper", "pairs", "init", "pp", "KO/PEP"]).exit_code == 1      # name taken
    run = CLI.invoke(app, ["paper", "pairs", "run", "pp"])
    assert run.exit_code == 0 and "will enter short spread" in run.output and "exited (converged)" in run.output
    assert "no new sessions" in CLI.invoke(app, ["paper", "pairs", "run", "pp"]).output
    status = CLI.invoke(app, ["paper", "pairs", "status", "pp"])
    assert status.exit_code == 0 and "KO/PEP" in status.output and "Closed trades: 1" in status.output
    assert "noise, not evidence" in status.output and "Beta to SPY" in status.output
    assert "pairs:" in (results / "paper" / "pp" / "runs.log").read_text()


@pytest.mark.unit
def test_status_does_not_write_the_ledger(results):
    CLI.invoke(app, ["paper", "pairs", "init", "pp", "KO/PEP"])
    path = pairs_runner.pairs_path(DEFAULT_CONFIG, "pp")
    before = path.read_text()
    assert CLI.invoke(app, ["paper", "pairs", "status", "pp"]).exit_code == 0
    assert path.read_text() == before


@pytest.mark.unit
def test_one_run_per_account_at_a_time(results):
    CLI.invoke(app, ["paper", "pairs", "init", "pp", "KO/PEP"])
    path = pairs_runner.pairs_path(DEFAULT_CONFIG, "pp")
    acct = PairsAccount.load(path)
    with locked(path, wait=False) as held:
        assert held
        with pytest.raises(RunInProgress):
            pairs_runner.run_pairs(acct, DEFAULT_CONFIG)
        busy = CLI.invoke(app, ["paper", "pairs", "run", "pp"])
    assert busy.exit_code == 0 and "already in progress" in busy.output


@pytest.mark.unit
def test_the_cli_refuses_an_unknown_account_and_an_llm_account_name(results):
    assert CLI.invoke(app, ["paper", "pairs", "run", "nope"]).exit_code == 1
    CLI.invoke(app, ["paper", "init", "llm", "NVDA", "--cash", "1000"])
    assert CLI.invoke(app, ["paper", "pairs", "init", "llm", "KO/PEP"]).exit_code == 1
