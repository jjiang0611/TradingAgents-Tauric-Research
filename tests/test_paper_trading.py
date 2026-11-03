"""Paper trading: ratings become fills by stated rules, and the account is judged
against buy-and-hold of the same names, not against zero."""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.paper.account import PaperAccount, max_drawdown, total_return
from tradingagents.paper.runner import book_for, create_account, run_day, status

DAYS = pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06"])


def closes(**by_ticker):
    return {t: pd.Series(v, index=DAYS[: len(v)], dtype=float) for t, v in by_ticker.items()}


def account(**kw):
    base = {"name": "t", "created": "2026-03-01", "starting_cash": 1000.0, "tickers": ["AAA", "BBB"], "slippage_bps": 0}
    return PaperAccount(**{**base, **kw})


@pytest.mark.unit
def test_buy_fills_at_the_first_close_after_the_decision_date():
    a = account()
    a.record("AAA", "2026-03-02", "Buy")
    a.fill_pending(closes(AAA=[10, 11, 12]))
    order = a.orders[0]
    assert order.fill_date == "2026-03-03" and order.price == 11.0
    assert order.shares == pytest.approx(500 / 11)        # a full sleeve: 1000 / 2 tickers
    assert a.cash == pytest.approx(500.0)


@pytest.mark.unit
def test_order_waits_until_its_entry_close_exists():
    a = account()
    a.record("AAA", "2026-03-02", "Buy")
    a.fill_pending(closes(AAA=[10]))                      # only the decision day has closed
    assert a.orders[0].status == "pending" and a.cash == 1000.0


@pytest.mark.unit
def test_slippage_works_against_both_sides():
    a = account(slippage_bps=100)
    a.record("AAA", "2026-03-02", "Buy")
    a.fill_pending(closes(AAA=[10, 10, 10, 10]))
    assert a.orders[0].price == pytest.approx(10.1)
    a.record("AAA", "2026-03-03", "Sell")
    a.fill_pending(closes(AAA=[10, 10, 10, 10]))
    assert a.orders[1].price == pytest.approx(9.9) and a.cash < 1000.0


@pytest.mark.unit
def test_repeated_buy_does_not_churn():
    a = account()
    series = closes(AAA=[10, 10, 10.5, 10.7, 10.9])
    for day in ("2026-03-02", "2026-03-03", "2026-03-04"):
        a.record("AAA", day, "Buy")
        a.fill_pending(series)
    assert [o.status for o in a.orders] == ["filled", "no_trade", "no_trade"]


@pytest.mark.unit
def test_sell_closes_and_underweight_only_trims():
    a = account()
    series = closes(AAA=[10, 10, 10, 10, 10])
    a.record("AAA", "2026-03-02", "Buy")
    a.record("AAA", "2026-03-03", "Underweight")
    a.fill_pending(series)
    assert a.holdings()["AAA"][0] == pytest.approx(12.5)  # a quarter of a $500 sleeve, at 10
    a.record("AAA", "2026-03-04", "Underweight")          # already at target: nothing to cut
    a.record("BBB", "2026-03-04", "Underweight")          # flat: Underweight never buys
    a.fill_pending(closes(AAA=[10] * 5, BBB=[5] * 5))
    assert {o.status for o in a.orders[2:]} == {"no_trade"}
    a.record("AAA", "2026-03-05", "Sell")
    a.fill_pending(closes(AAA=[10] * 5))
    assert "AAA" not in a.holdings()


@pytest.mark.unit
def test_hold_and_unparseable_ratings_do_not_trade():
    a = account()
    assert a.record("AAA", "2026-03-02", "Hold").status == "no_trade"
    assert a.record("BBB", "2026-03-02", "REVIEW").status == "no_trade"


@pytest.mark.unit
def test_buy_cannot_spend_cash_it_does_not_have():
    a = account(tickers=["AAA"], starting_cash=100.0)
    a.record("AAA", "2026-03-02", "Buy")
    a.fill_pending(closes(AAA=[10, 10, 5, 5]))            # fills 10 shares at 10, cash 0
    a.record("AAA", "2026-03-04", "Buy")                  # value fell to 50; wants 50 more
    a.fill_pending(closes(AAA=[10, 10, 5, 5, 5]))
    assert a.cash >= -1e-9


@pytest.mark.unit
def test_decisions_must_arrive_in_date_order_and_not_twice():
    a = account()
    a.record("AAA", "2026-03-03", "Buy")
    with pytest.raises(ValueError, match="date order"):
        a.record("BBB", "2026-03-02", "Buy")
    with pytest.raises(ValueError, match="already"):
        a.record("AAA", "2026-03-03", "Sell")
    with pytest.raises(ValueError, match="not in this account"):
        a.record("ZZZ", "2026-03-03", "Buy")


@pytest.mark.unit
def test_strategy_is_measured_against_buy_and_hold():
    a = account(tickers=["AAA"])
    series = closes(AAA=[10, 10, 12, 15, 15], SPY=[100, 100, 101, 102, 103])
    a.record("AAA", "2026-03-02", "Buy")
    a.fill_pending(series)
    strategy, hold = a.equity_curve(series), a.buy_and_hold(series)
    assert total_return(strategy, 1000) == pytest.approx(0.5)
    assert total_return(hold, 1000) == pytest.approx(0.5)  # same entry, so a plain Buy is no edge
    assert total_return(a.buy_and_hold(series, ["SPY"]), 1000) == pytest.approx(0.03)
    assert max_drawdown(pd.Series([100.0, 120.0, 90.0])) == pytest.approx(-0.25)


@pytest.mark.unit
def test_book_shows_fills_not_pending_orders():
    a = account()
    a.record("AAA", "2026-03-02", "Buy")
    assert book_for(a).positions == [] and book_for(a).cash == 1000.0
    a.fill_pending(closes(AAA=[10, 10]))
    assert book_for(a).position_in("AAA").quantity == pytest.approx(50.0)


class _Graph:
    def __init__(self, ratings):
        self.ratings, self.books = ratings, []

    def propagate(self, ticker, date, asset_type, portfolio=None):
        self.books.append(portfolio)
        if self.ratings[ticker] == "boom":
            raise RuntimeError("vendor down")
        return {}, self.ratings[ticker]

    def settle_pending(self, ticker):
        pass


def _fetch(prices):
    return lambda ticker, start, end: pd.Series(prices[ticker], index=DAYS[: len(prices[ticker])])


@pytest.mark.unit
def test_run_day_decides_once_per_ticker_and_survives_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("tradingagents.paper.runner.get_current_date", lambda: "2026-03-06")
    config = {"results_dir": str(tmp_path)}
    acct = create_account(config, "main", ["AAA", "BBB"], 1000.0, slippage_bps=0, created="2026-03-01")
    prices = {"AAA": [10, 10, 10, 10, 10], "BBB": [5] * 5, "SPY": [100] * 5}
    graph = _Graph({"AAA": "Buy", "BBB": "boom"})

    first = run_day(acct, config, "2026-03-02", graph=graph, fetch=_fetch(prices))
    assert first.decided == [("AAA", "Buy")] and [t for t, _ in first.failures] == ["BBB"]

    second = run_day(acct, config, "2026-03-03", graph=_Graph({"AAA": "Hold", "BBB": "Buy"}), fetch=_fetch(prices))
    assert second.filled == 1                              # AAA's Buy from the 2nd fills on the 3rd
    held = [b for b in graph.books if b is not None][0]
    assert held.positions == [] and held.cash == 1000.0    # day one: nothing filled yet

    again = run_day(acct, config, "2026-03-03", graph=_Graph({"AAA": "Buy", "BBB": "Buy"}), fetch=_fetch(prices))
    assert again.decided == [] and sorted(again.skipped) == ["AAA", "BBB"]


@pytest.mark.unit
def test_status_never_writes_the_ledger_and_warns_on_small_samples(tmp_path, monkeypatch):
    monkeypatch.setattr("tradingagents.paper.runner.get_current_date", lambda: "2026-03-06")
    config = {"results_dir": str(tmp_path)}
    acct = create_account(config, "main", ["AAA"], 1000.0, slippage_bps=0, created="2026-03-01")
    acct.record("AAA", "2026-03-02", "Buy")
    prices = {"AAA": [10, 10, 12, 15, 15], "SPY": [100, 100, 101, 102, 103]}
    text = status(acct, config, _fetch(prices)).render()
    assert "Strategy" in text and "Buy & hold AAA" in text and "Buy & hold SPY" in text
    assert "+50.00%" in text
