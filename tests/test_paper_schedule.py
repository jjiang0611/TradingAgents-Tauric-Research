"""What a scheduler needs from `paper run`: skip closed days, retry failures, never collide."""

from __future__ import annotations

import pandas as pd
import pytest
from typer.testing import CliRunner

from cli.main import app
from tradingagents.dataflows.files import locked
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.paper import runner
from tradingagents.paper.account import PaperAccount

CLI = CliRunner()


def bars(*days):
    return lambda ticker, start, end: pd.Series([100.0] * len(days), index=pd.to_datetime(list(days)))


@pytest.mark.unit
def test_market_traded_means_spy_printed_a_bar_that_day():
    assert runner.market_traded("2026-03-03", bars("2026-03-03"))
    assert not runner.market_traded("2026-03-07", bars())                  # nothing back: weekend
    assert not runner.market_traded("2026-03-07", bars("2026-03-06"))      # only an earlier day


@pytest.mark.unit
def test_an_outage_is_not_mistaken_for_a_holiday():
    def down(*a):
        raise OSError("vendor down")

    with pytest.raises(OSError):
        runner.market_traded("2026-03-03", down)


@pytest.fixture
def account(tmp_path, monkeypatch):
    monkeypatch.setitem(DEFAULT_CONFIG, "results_dir", str(tmp_path))
    runner.create_account(DEFAULT_CONFIG, "main", ["AAA"], 1000.0, created="2026-03-01")
    return tmp_path


def log_text(root):
    return (root / "paper" / "main" / "runs.log").read_text()


@pytest.mark.unit
def test_closed_market_exits_zero_without_running_the_graph(account, monkeypatch):
    monkeypatch.setattr(runner, "market_traded", lambda date, *a: False)
    monkeypatch.setattr(runner, "run_day", lambda *a, **k: pytest.fail("must not run"))
    result = CLI.invoke(app, ["paper", "run", "main", "--trading-days-only", "--date", "2026-03-07"])
    assert result.exit_code == 0 and "no US market session" in result.output
    assert "no US market session" in log_text(account)


@pytest.mark.unit
def test_a_failed_ticker_exits_nonzero_so_the_next_trigger_retries(account, monkeypatch):
    monkeypatch.setattr(runner, "market_traded", lambda date, *a: True)
    monkeypatch.setattr(runner, "run_day", lambda *a, **k: runner.DayResult(
        date="2026-03-03", failures=[("AAA", "vendor down")]))
    result = CLI.invoke(app, ["paper", "run", "main", "--trading-days-only", "--date", "2026-03-03"])
    assert result.exit_code == 1 and "FAILED AAA" in log_text(account)


@pytest.mark.unit
def test_a_run_already_in_progress_is_skipped_not_failed(account, monkeypatch):
    def busy(*a, **k):
        raise runner.RunInProgress("a run for paper account 'main' is already in progress")

    monkeypatch.setattr(runner, "market_traded", lambda date, *a: True)
    monkeypatch.setattr(runner, "run_day", busy)
    result = CLI.invoke(app, ["paper", "run", "main", "--trading-days-only", "--date", "2026-03-03"])
    assert result.exit_code == 0 and "skipped" in log_text(account)


@pytest.mark.unit
def test_only_one_run_per_account_at_a_time(account):
    acct = PaperAccount.load(runner.account_path(DEFAULT_CONFIG, "main"))
    with locked(runner.account_path(DEFAULT_CONFIG, "main"), wait=False) as held:
        assert held
        with pytest.raises(runner.RunInProgress):
            runner.run_day(acct, DEFAULT_CONFIG, "2026-03-03", graph=object())
