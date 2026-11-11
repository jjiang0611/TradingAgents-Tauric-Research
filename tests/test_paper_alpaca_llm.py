"""The LLM paper account's suggestions mirrored to Alpaca PAPER: sized by the simulation's own
rules against what Alpaca really holds, long-only, sent once, never stale, and measured."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.test_paper_alpaca import FakeBroker
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.paper import alpaca_llm, runner
from tradingagents.paper.account import PaperAccount
from tradingagents.paper.alpaca import AlpacaError, Mirror
from tradingagents.paper.alpaca_llm import (
    build_alpaca_state,
    plan_llm_orders,
    reconcile_llm,
    send_llm_plan,
)

DAYS = pd.bdate_range("2024-03-01", periods=10)            # the last completed close is 2024-03-14
LAST = DAYS[-1].strftime("%Y-%m-%d")


def closes(a=50.0, b=20.0):
    return {"AAA": pd.Series([a] * len(DAYS), index=DAYS), "BBB": pd.Series([b] * len(DAYS), index=DAYS),
            "SPY": pd.Series([100.0] * len(DAYS), index=DAYS)}


def account(cash=10_000.0):
    return PaperAccount(name="main", created="2024-01-01", starting_cash=cash, tickers=["AAA", "BBB"], slippage_bps=5.0)


def plan(acct, held=None, mirror=None, **kw):
    return plan_llm_orders(acct, closes(), held or {}, mirror or Mirror(), **kw)


@pytest.mark.unit
def test_a_buy_is_sized_to_the_sleeve_in_whole_shares_and_sent_as_a_day_order():
    acct = account()
    acct.record("AAA", LAST, "Buy")
    result = plan(acct)
    (order,) = result.orders
    assert (order.symbol, order.side, order.qty, order.tif) == ("AAA", "buy", 100, "day")      # $5,000 sleeve at $50
    assert order.client_order_id == plan(acct).orders[0].client_order_id
    assert order.notional <= acct.sleeve


@pytest.mark.unit
def test_hold_and_an_unparseable_rating_send_nothing():
    acct = account()
    acct.record("AAA", LAST, "Hold")
    acct.record("BBB", LAST, "REVIEW")
    result = plan(acct)
    assert result.orders == [] and all("no trade" in line for line in result.skipped)


@pytest.mark.unit
def test_a_suggestion_older_than_the_latest_close_is_stale_but_a_newer_one_is_fine():
    stale = account()
    stale.record("AAA", "2024-03-13", "Buy")
    assert plan(stale).orders == [] and "stale" in plan(stale).skipped[0]
    later = account()
    later.record("AAA", "2024-03-18", "Buy")                  # decided today, after the last completed close
    assert len(plan(later).orders) == 1


@pytest.mark.unit
def test_only_the_newest_rating_per_ticker_counts_and_a_suggestion_is_sent_once():
    acct = account()
    acct.record("AAA", "2024-03-13", "Buy")
    acct.record("AAA", LAST, "Sell")
    assert plan(acct).orders == [] and "nothing held" in plan(acct).skipped[0]
    fresh = account()
    fresh.record("AAA", LAST, "Buy")
    first = plan(fresh)
    mirror = Mirror()
    mirror.orders.append(alpaca_llm.MirrorOrder(
        client_order_id=first.orders[0].client_order_id, trade_key=first.orders[0].trade_key, pair="AAA",
        action="Buy", symbol="AAA", side="buy", qty=100, tif="day", submitted_at="t"))
    assert plan(fresh, mirror=mirror).orders == [] and "already sent" in plan(fresh, mirror=mirror).skipped[0]


@pytest.mark.unit
def test_the_rules_are_applied_to_what_alpaca_holds_not_to_the_simulation():
    acct = account()
    acct.record("AAA", LAST, "Overweight")                    # target half the $5,000 sleeve = $2,500
    assert plan(acct, held={"AAA": 60}).orders == []          # $3,000 is inside the $500 band
    (trim,) = plan(acct, held={"AAA": 80}).orders             # $4,000 is above it: sell down to $2,500
    assert (trim.side, trim.qty) == ("sell", 30)
    (add,) = plan(acct, held={}).orders
    assert (add.side, add.qty) == ("buy", 50)


@pytest.mark.unit
def test_underweight_trims_and_sell_closes_exactly_what_is_held_and_never_goes_short():
    acct = account()
    acct.record("AAA", LAST, "Underweight")
    assert [(o.side, o.qty) for o in plan(acct, held={"AAA": 100}).orders] == [("sell", 75)]   # down to $1,250
    assert plan(acct, held={}).orders == []                   # nothing held: never short
    acct2 = account()
    acct2.record("AAA", LAST, "Sell")
    (close_out,) = plan(acct2, held={"AAA": 12.5}).orders
    assert (close_out.side, close_out.qty) == ("sell", 12.5)


@pytest.mark.unit
def test_caps_and_tiny_accounts_refuse_instead_of_sending():
    acct = account()
    acct.record("AAA", LAST, "Buy")
    assert "order cap" in plan(acct, max_order_usd=100).skipped[0]
    tiny = account(cash=40.0)
    tiny.record("AAA", LAST, "Buy")
    assert "under one whole share" in plan(tiny).skipped[0]


@pytest.mark.unit
def test_sending_puts_sells_first_checks_buying_power_and_records_the_baseline(tmp_path):
    acct = account()
    acct.record("AAA", LAST, "Buy")
    acct.record("BBB", LAST, "Underweight")
    broker, mirror, path = FakeBroker(), Mirror(), tmp_path / "m.json"
    result = send_llm_plan(broker, plan_llm_orders(acct, closes(), {"BBB": 250}, mirror), mirror, path)
    assert [c[2] for c in broker.calls] == ["sell", "buy"] and result.refused == []
    assert mirror.baseline_equity == 100_000.0 and Mirror.load(path).orders == mirror.orders
    again = plan_llm_orders(acct, closes(), {"BBB": 250}, Mirror.load(path))
    assert again.orders == []

    poor = FakeBroker()
    poor.acct["buying_power"] = "10"
    low = send_llm_plan(poor, plan_llm_orders(acct, closes(), {}, Mirror()), Mirror(), tmp_path / "p.json")
    assert poor.calls == [] and "buying power" in low.refused[0]

    broken = FakeBroker()
    broken.acct["status"] = "ACCOUNT_UPDATED"
    with pytest.raises(AlpacaError, match="cannot trade"):
        send_llm_plan(broken, plan_llm_orders(acct, closes(), {}, Mirror()), Mirror(), tmp_path / "q.json")


@pytest.mark.unit
def test_a_rejected_order_is_recorded_not_raised(tmp_path):
    acct = account()
    acct.record("AAA", LAST, "Buy")
    broker = FakeBroker()
    broker.reject.add("AAA")
    mirror = Mirror()
    result = send_llm_plan(broker, plan_llm_orders(acct, closes(), {}, mirror), mirror, tmp_path / "m.json")
    assert [o.status for o in mirror.orders] == ["rejected"] and "insufficient buying power" in result.refused[0]


@pytest.mark.unit
def test_reconcile_measures_real_fills_against_the_official_close_and_the_simulation(tmp_path):
    acct = account()
    acct.record("AAA", LAST, "Buy")
    broker, mirror, path = FakeBroker(), Mirror(), tmp_path / "m.json"
    send_llm_plan(broker, plan_llm_orders(acct, closes(), {}, mirror), mirror, path)
    later = {t: pd.concat([s, pd.Series([50.0], index=[DAYS[-1] + pd.offsets.BDay(1)])]) for t, s in closes().items()}
    acct.fill_pending(later)                                  # the simulation fills at the next close, $50.025
    order = mirror.orders[0]
    fill_day = (DAYS[-1] + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    broker.fills[order.client_order_id] = {"id": order.alpaca_id, "status": "filled", "filled_qty": "100",
                                           "filled_avg_price": "50.04", "filled_at": f"{fill_day}T15:00:00Z"}
    broker.pos = [{"symbol": "AAA", "qty": "100"}]
    report = reconcile_llm(acct, broker, mirror, path, later)
    (row,) = report.rows
    assert row.vs_close_bps == pytest.approx(8.0, abs=0.01)                 # $50.04 vs a $50.00 close
    assert row.vs_sim_bps == pytest.approx((50.04 - 50.025) / 50.025 * 1e4, abs=0.01)
    assert report.drift == [] and report.open_orders == 0


@pytest.mark.unit
def test_the_dashboard_state_has_the_plan_the_account_and_no_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "get_current_date", lambda: "2024-03-20")
    config = {**DEFAULT_CONFIG, "results_dir": str(tmp_path)}
    acct = runner.create_account(config, "main", ["AAA", "BBB"], 10_000.0, created="2024-01-01")
    acct.record("AAA", LAST, "Buy")
    acct.save(runner.account_path(config, "main"))
    series = closes()
    state = build_alpaca_state(acct, config, FakeBroker(), lambda t, s, e: series[t])
    assert state["enabled"] and state["account"]["number"] == "...0001" and state["market"]["is_open"]
    assert [(o["ticker"], o["side"], o["qty"]) for o in state["plan"]] == [("AAA", "buy", 100)]
    assert state["metrics"]["sent"] == 0 and state["positions"] == [] and state["error"] is None
    assert "SECRET" not in str(state) and "KEY" not in str(state)


@pytest.mark.unit
def test_a_position_held_before_the_first_order_is_not_reported_as_drift(tmp_path):
    acct = account()
    acct.record("AAA", LAST, "Sell")
    broker, mirror, path = FakeBroker(), Mirror(), tmp_path / "m.json"
    broker.pos = [{"symbol": "AAA", "qty": "12.5"}]                      # held before the mirror ever sent anything
    send_llm_plan(broker, plan_llm_orders(acct, closes(), {"AAA": 12.5}, mirror), mirror, path)
    assert mirror.baseline_positions == {"AAA": 12.5}
    order = mirror.orders[0]
    broker.fills[order.client_order_id] = {"id": order.alpaca_id, "status": "filled", "filled_qty": "12.5",
                                           "filled_avg_price": "50", "filled_at": f"{LAST}T20:00:00Z"}
    broker.pos = []                                                      # sold it all, as asked
    assert reconcile_llm(acct, broker, mirror, path, closes()).drift == []
    broker.pos = [{"symbol": "AAA", "qty": "3"}]                         # something else is now in the account
    assert "AAA: expected 0" in reconcile_llm(acct, broker, mirror, path, closes()).drift[0]
