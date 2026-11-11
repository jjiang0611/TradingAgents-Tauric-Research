"""Mirroring a pairs account to Alpaca PAPER: only ever the paper host, nothing sent without
--send, no decision sent twice, and the real fills measured against the simulation's."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from cli.main import app
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.paper import alpaca, pairs_runner
from tradingagents.paper.alpaca import (
    AlpacaError,
    AlpacaPaper,
    Mirror,
    plan_orders,
    reconcile,
    send_plan,
)
from tradingagents.paper.pairs import PairsAccount, Rules

CLI = CliRunner()
LOOKBACK = 60


def make_pair(n, jump_at=None, back_at=None, jump=0.03):
    k = np.arange(n)
    log_b = 4 + 0.0008 * k + 0.02 * np.sin(2 * np.pi * k / 17)
    spread = 0.01 * np.where(k % 2 == 0, 1.0, -1.0)
    if jump_at is not None:
        spread[jump_at:back_at if back_at is not None else n] += jump
    index = pd.bdate_range("2024-01-02", periods=n)
    return {"A": pd.Series(np.exp(log_b + spread), index=index), "B": pd.Series(np.exp(log_b), index=index)}


def account(cash=10_000.0, **kw):
    return PairsAccount(name="p", created="2024-01-01", starting_cash=cash, pairs=["A/B"], slippage_bps=5.0,
                        rules=Rules(lookback=LOOKBACK), **kw)


def upto(closes, n):
    return {t: s.iloc[:n] for t, s in closes.items()}


def pending_entry(cash=10_000.0):
    full = make_pair(100, jump_at=70, back_at=80)
    acct = account(cash)
    acct.process(upto(full, 71))                         # the decision is made at the 71st close
    assert acct.trades[0].status == "pending_entry"
    return acct, full


class FakeBroker:
    def __init__(self):
        self.calls, self.canceled, self.fills, self.pos = [], [], {}, []
        self.acct = {"status": "ACTIVE", "trading_blocked": False, "buying_power": "1000000",
                     "shorting_enabled": True, "account_number": "PA0001", "equity": "100000"}
        self.unshortable, self.reject = set(), set()
        self.clock_data = {"is_open": True, "timestamp": "2026-11-05T19:00:00Z", "next_close": "2026-11-05T21:00:00Z"}

    def account(self):
        return self.acct

    def clock(self):
        return self.clock_data

    def positions(self):
        return self.pos

    def asset(self, symbol):
        return {"tradable": True, "shortable": symbol not in self.unshortable}

    def submit_order(self, symbol, qty, side, tif, cid):
        if symbol in self.reject:
            raise AlpacaError("Alpaca POST /v2/orders: HTTP 403 insufficient buying power")
        self.calls.append((symbol, qty, side, tif, cid))
        return {"id": f"id-{len(self.calls)}", "status": "accepted"}

    def order_by_client_id(self, cid):
        return self.fills.get(cid, {"status": "accepted"})

    def cancel_order(self, oid):
        self.canceled.append(oid)


# ---------- the HTTP client ----------

class FakeSession:
    def __init__(self, status=200, body=None):
        self.requests, self.status, self.body = [], status, body if body is not None else {"ok": True}

    def request(self, method, url, headers=None, timeout=None, **kw):
        self.requests.append((method, url, headers, kw))
        outer = self

        class Response:
            status_code = outer.status
            content = b"x"

            def json(self):
                return outer.body
        return Response()


@pytest.mark.unit
def test_the_client_talks_only_to_the_paper_host_and_has_no_url_to_change():
    assert "base_url" not in inspect.signature(AlpacaPaper.__init__).parameters
    session = FakeSession()
    client = AlpacaPaper("KEYID", "SECRETVALUE", session=session)
    client.account()
    client.clock()
    client.positions()
    client.asset("KO")
    client.submit_order("KO", 5, "buy", "cls", "cid")
    client.order_by_client_id("cid")
    client.cancel_order("x")
    assert all(url.startswith("https://paper-api.alpaca.markets/") for _, url, _, _ in session.requests)
    method, url, headers, kw = session.requests[4]
    assert (method, url) == ("POST", "https://paper-api.alpaca.markets/v2/orders")
    assert kw["json"] == {"symbol": "KO", "qty": "5", "side": "buy", "type": "market",
                          "time_in_force": "cls", "client_order_id": "cid"}
    assert headers["APCA-API-KEY-ID"] == "KEYID"


@pytest.mark.unit
def test_keys_are_required_and_never_leak_into_an_error(monkeypatch):
    monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_API_SECRET", raising=False)
    with pytest.raises(AlpacaError, match="ALPACA_PAPER_API_KEY"):
        AlpacaPaper()
    client = AlpacaPaper("KEYID", "SECRETVALUE", session=FakeSession(403, {"message": "forbidden"}))
    with pytest.raises(AlpacaError) as caught:
        client.account()
    text = str(caught.value)
    assert "403" in text and "forbidden" in text and "KEYID" not in text and "SECRETVALUE" not in text


# ---------- planning ----------

@pytest.mark.unit
def test_entry_orders_are_whole_share_market_on_close_with_the_short_leg_first():
    acct, full = pending_entry()
    plan = plan_orders(acct, upto(full, 71), Mirror())
    assert [o.action for o in plan.orders] == ["enter", "enter"] and plan.skipped == []
    short, long_ = plan.orders
    assert (short.side, long_.side) == ("sell", "buy") and short.symbol == "A" and long_.symbol == "B"   # A ran ahead
    assert all(isinstance(o.qty, int) and o.qty >= 1 and o.tif == "cls" for o in plan.orders)
    assert short.notional + long_.notional <= acct.sleeve
    assert short.client_order_id != long_.client_order_id
    assert plan_orders(acct, upto(full, 71), Mirror()).orders[0].client_order_id == short.client_order_id   # deterministic


@pytest.mark.unit
def test_a_decision_the_simulation_has_already_filled_is_stale_and_not_sent():
    acct, full = pending_entry()
    plan = plan_orders(acct, upto(full, 72), Mirror())               # a later close has printed
    assert plan.orders == [] and "stale" in plan.skipped[0]


@pytest.mark.unit
def test_dollar_caps_and_tiny_accounts_refuse_instead_of_sending():
    acct, full = pending_entry()
    assert "over the dollar cap" in plan_orders(acct, upto(full, 71), Mirror(), max_order_usd=100).skipped[0]
    assert "over the dollar cap" in plan_orders(acct, upto(full, 71), Mirror(), max_gross_usd=100).skipped[0]
    small, full = pending_entry(cash=20.0)
    assert "under one whole share" in plan_orders(small, upto(full, 71), Mirror()).skipped[0]


# ---------- sending ----------

def send(acct, full, broker, tmp_path, **plan_kw):
    mirror = Mirror()
    path = tmp_path / "mirror.json"
    plan = plan_orders(acct, upto(full, 71), mirror, **plan_kw)
    return send_plan(broker, plan, mirror, path), mirror, path


@pytest.mark.unit
def test_a_send_submits_the_short_leg_first_and_records_everything(tmp_path):
    acct, full = pending_entry()
    broker = FakeBroker()
    result, mirror, path = send(acct, full, broker, tmp_path)
    assert [c[2] for c in broker.calls] == ["sell", "buy"] and result.refused == []
    assert [o.alpaca_id for o in mirror.orders] == ["id-1", "id-2"]
    assert Mirror.load(path).orders == mirror.orders
    again = plan_orders(acct, upto(full, 71), Mirror.load(path))        # running it twice sends nothing new
    assert again.orders == [] and "already sent" in again.skipped[0]


@pytest.mark.unit
@pytest.mark.parametrize("break_it, expect", [
    (lambda b: b.acct.update(shorting_enabled=False), "shorting is not enabled"),
    (lambda b: b.acct.update(buying_power="10"), "buying power"),
    (lambda b: b.acct.update(status="ACCOUNT_UPDATED"), "cannot trade"),
    (lambda b: b.unshortable.add("A"), "cannot be sold short"),
])
def test_safety_checks_refuse_before_any_order_is_sent(tmp_path, break_it, expect):
    acct, full = pending_entry()
    broker = FakeBroker()
    break_it(broker)
    if expect == "cannot trade":
        with pytest.raises(AlpacaError, match=expect):
            send(acct, full, broker, tmp_path)
    else:
        result, mirror, _ = send(acct, full, broker, tmp_path)
        assert expect in result.refused[0] and broker.calls == [] and mirror.orders == []


@pytest.mark.unit
def test_if_the_second_leg_is_refused_the_first_is_cancelled(tmp_path):
    acct, full = pending_entry()
    broker = FakeBroker()
    broker.reject.add("B")
    result, mirror, _ = send(acct, full, broker, tmp_path)
    assert broker.canceled == ["id-1"]
    assert [o.status for o in mirror.orders] == ["canceled", "rejected"]
    assert any("insufficient buying power" in r for r in result.refused)


@pytest.mark.unit
def test_the_market_must_be_open_and_not_about_to_close():
    broker = FakeBroker()
    alpaca.check_market(broker, "cls")
    broker.clock_data["next_close"] = "2026-11-05T19:05:00Z"
    with pytest.raises(AlpacaError, match="minutes to the close"):
        alpaca.check_market(broker, "cls")
    broker.clock_data["is_open"] = False
    with pytest.raises(AlpacaError, match="closed"):
        alpaca.check_market(broker, "day")


# ---------- reconciling ----------

def filled(broker, order, price, day):
    broker.fills[order.client_order_id] = {
        "id": order.alpaca_id, "status": "filled", "filled_qty": str(order.qty),
        "filled_avg_price": str(price), "filled_at": f"{day}T21:00:00.5Z"}


@pytest.mark.unit
def test_reconcile_measures_the_real_cost_against_the_official_close(tmp_path):
    acct, full = pending_entry()
    broker = FakeBroker()
    _, mirror, path = send(acct, full, broker, tmp_path)
    acct.process(upto(full, 72))                                    # the simulation fills at the next close
    fill_day = full["A"].index[71]
    close = {s: float(full[s].iloc[71]) for s in "AB"}
    sell, buy = mirror.orders                                       # A sold short, B bought
    filled(broker, sell, close["A"] * (1 - 3e-4), fill_day.strftime("%Y-%m-%d"))    # 3 bps worse than the close
    filled(broker, buy, close["B"] * (1 + 8e-4), fill_day.strftime("%Y-%m-%d"))     # 8 bps worse than the close
    broker.pos = [{"symbol": "A", "qty": f"-{sell.qty}"}, {"symbol": "B", "qty": str(buy.qty)}]
    report = reconcile(acct, broker, mirror, path, upto(full, 72))
    by_symbol = {r.symbol: r for r in report.rows}
    assert by_symbol["A"].vs_close_bps == pytest.approx(3.0, abs=0.01) and by_symbol["B"].vs_close_bps == pytest.approx(8.0, abs=0.01)
    assert by_symbol["A"].vs_sim_bps == pytest.approx(-2.0, abs=0.01)           # the simulation assumed 5 bps
    assert report.drift == [] and report.open_orders == 0
    assert "+5.5 bps per fill" in report.render()


@pytest.mark.unit
def test_reconcile_flags_positions_that_do_not_match_what_was_sent(tmp_path):
    acct, full = pending_entry()
    broker = FakeBroker()
    _, mirror, path = send(acct, full, broker, tmp_path)
    for o in mirror.orders:
        filled(broker, o, 50.0, "2024-04-10")
    broker.pos = [{"symbol": "A", "qty": "5"}]
    report = reconcile(acct, broker, mirror, path, upto(full, 71))
    assert any("A:" in d for d in report.drift) and any("B:" in d for d in report.drift)


@pytest.mark.unit
def test_an_exit_closes_exactly_what_filled_and_nothing_if_the_entry_never_filled(tmp_path):
    full = make_pair(100, jump_at=70, back_at=76)
    acct = account()
    broker, mirror, path = FakeBroker(), Mirror(), tmp_path / "m.json"
    acct.process(upto(full, 71))
    send_plan(broker, plan_orders(acct, upto(full, 71), mirror), mirror, path)
    for n in range(72, 101):                                        # run until the simulation decides to exit
        acct.process(upto(full, n))
        if acct.trades[0].status == "pending_exit":
            break
    assert acct.trades[0].status == "pending_exit"
    closes = upto(full, n)
    assert plan_orders(acct, closes, mirror).orders == []           # entries are not marked filled yet
    for o in mirror.orders:
        filled(broker, o, 50.0, "2024-04-10")
    for o in mirror.orders:
        o.status, o.filled_qty, o.filled_avg_price = "filled", float(o.qty), 50.0
    exits = plan_orders(acct, closes, mirror).orders
    entered = {o.symbol: (o.side, o.qty) for o in mirror.orders}
    assert {o.symbol: o.qty for o in exits} == {s: q for s, (_, q) in entered.items()}
    assert all(o.action == "exit" and o.side != entered[o.symbol][0] for o in exits)


# ---------- the command line ----------

@pytest.fixture
def cli_setup(tmp_path, monkeypatch):
    monkeypatch.setitem(DEFAULT_CONFIG, "results_dir", str(tmp_path))
    monkeypatch.setattr(pairs_runner, "get_current_date", lambda: "2024-01-01")
    full = make_pair(331, jump_at=330)                              # the decision lands on the last close
    monkeypatch.setattr(pairs_runner, "get_closes", lambda t, s, e: {"KO": full["A"], "PEP": full["B"], "SPY": full["B"]}[t])
    broker = FakeBroker()
    monkeypatch.setattr(alpaca, "AlpacaPaper", lambda *a, **k: broker)
    assert CLI.invoke(app, ["paper", "pairs", "init", "pp", "KO/PEP", "--cash", "10000"]).exit_code == 0
    return broker, tmp_path


@pytest.mark.unit
def test_the_cli_is_a_dry_run_unless_told_to_send(cli_setup):
    broker, root = cli_setup
    dry = CLI.invoke(app, ["paper", "pairs", "alpaca", "send", "pp"])
    assert dry.exit_code == 0 and "DRY RUN" in dry.output and "sell" in dry.output and "Add --send" in dry.output
    assert broker.calls == [] and not (root / "paper" / "pp" / "alpaca_mirror.json").exists()
    sent = CLI.invoke(app, ["paper", "pairs", "alpaca", "send", "pp", "--send"])
    assert sent.exit_code == 0 and "SENDING" in sent.output and len(broker.calls) == 2
    assert "Nothing to send." in CLI.invoke(app, ["paper", "pairs", "alpaca", "send", "pp", "--send"]).output
    assert len(broker.calls) == 2


@pytest.mark.unit
def test_the_cli_refuses_to_send_when_the_market_is_closed_and_without_keys(cli_setup, monkeypatch):
    broker, _ = cli_setup
    broker.clock_data["is_open"] = False
    closed = CLI.invoke(app, ["paper", "pairs", "alpaca", "send", "pp", "--send"])
    assert closed.exit_code == 1 and "closed" in closed.output and broker.calls == []

    def no_keys(*a, **k):
        raise AlpacaError("set ALPACA_PAPER_API_KEY and ALPACA_PAPER_API_SECRET to your Alpaca PAPER keys")

    monkeypatch.setattr(alpaca, "AlpacaPaper", no_keys)
    assert CLI.invoke(app, ["paper", "pairs", "alpaca", "send", "pp"]).exit_code == 1


# ---------- the one-shot plumbing check (test-order) ----------

class AutoBroker(FakeBroker):
    """Fills each order at once, at a price per side, unless told a side never fills."""

    def __init__(self, buy=100.0, sell=99.95, fills=("buy", "sell")):
        super().__init__()
        self.prices, self.fills_for, self.sides = {"buy": buy, "sell": sell}, set(fills), {}

    def submit_order(self, symbol, qty, side, tif, cid):
        response = super().submit_order(symbol, qty, side, tif, cid)
        self.sides[cid] = (side, qty)
        return response

    def order_by_client_id(self, cid):
        side, qty = self.sides[cid]
        if side in self.fills_for:
            return {"id": f"id-{cid}", "status": "filled", "filled_qty": str(qty), "filled_avg_price": str(self.prices[side])}
        return {"id": f"id-{cid}", "status": "accepted"}


def ticking():
    count = iter(range(0, 10_000, 10))
    return {"sleep": lambda s: None, "clock": lambda: next(count)}


@pytest.mark.unit
def test_a_test_order_is_a_dry_run_unless_sent():
    broker = AutoBroker()
    result = alpaca.run_test_order(broker, "spy", 2)
    assert result.ok and broker.calls == [] and "DRY RUN" in result.lines[-1] and "sell 2 SPY" in result.lines[-1]


@pytest.mark.unit
def test_a_sent_test_order_buys_then_sells_back_and_reports_the_round_trip_cost():
    broker = AutoBroker(buy=100.0, sell=99.90)
    result = alpaca.run_test_order(broker, "SPY", 2, send=True, stamp="T1", **ticking())
    assert result.ok
    assert [(c[0], c[1], c[2], c[3], c[4]) for c in broker.calls] == [
        ("SPY", 2, "buy", "day", "ta-test-T1-SPY-buy"), ("SPY", 2, "sell", "day", "ta-test-T1-SPY-sell")]
    assert "Round trip cost $+0.20 (+10.0 bps)" in result.lines[-1]


@pytest.mark.unit
def test_keep_leaves_the_position_open_and_says_so():
    broker = AutoBroker()
    result = alpaca.run_test_order(broker, "SPY", 1, send=True, keep=True, stamp="T2", **ticking())
    assert result.ok and [c[2] for c in broker.calls] == ["buy"] and "left open" in result.lines[-1]


@pytest.mark.unit
def test_a_buy_that_never_fills_is_cancelled_and_nothing_is_sold():
    broker = AutoBroker(fills=())
    result = alpaca.run_test_order(broker, "SPY", 1, send=True, stamp="T3", **ticking())
    assert not result.ok and [c[2] for c in broker.calls] == ["buy"] and len(broker.canceled) == 1
    assert "nothing was sold" in result.lines[-1]


@pytest.mark.unit
def test_a_sell_that_never_fills_tells_you_the_position_is_still_open():
    broker = AutoBroker(fills=("buy",))
    result = alpaca.run_test_order(broker, "SPY", 3, send=True, stamp="T4", **ticking())
    assert not result.ok and "you still hold 3 SPY" in result.lines[-1]


@pytest.mark.unit
@pytest.mark.parametrize("kwargs, message", [
    ({"qty": 0}, "1 to 10"), ({"qty": 11}, "1 to 10"), ({"symbol": "../x"}, "does not look like a ticker"),
])
def test_a_test_order_refuses_silly_inputs(kwargs, message):
    with pytest.raises(AlpacaError, match=message):
        alpaca.run_test_order(AutoBroker(), **{"symbol": "SPY", "qty": 1, **kwargs})


@pytest.mark.unit
def test_a_test_order_refuses_a_closed_market_a_blocked_account_and_an_untradable_symbol():
    closed = AutoBroker()
    closed.clock_data["is_open"] = False
    with pytest.raises(AlpacaError, match="closed"):
        alpaca.run_test_order(closed, "SPY", 1, send=True)
    assert closed.calls == []
    blocked = AutoBroker()
    blocked.acct["trading_blocked"] = True
    with pytest.raises(AlpacaError, match="cannot trade"):
        alpaca.run_test_order(blocked, "SPY", 1)
    untradable = AutoBroker()
    untradable.asset = lambda symbol: {"tradable": False}
    with pytest.raises(AlpacaError, match="not tradable"):
        alpaca.run_test_order(untradable, "SPY", 1)


@pytest.mark.unit
def test_the_cli_test_order_dry_run_then_send(monkeypatch):
    broker = AutoBroker(buy=50.0, sell=49.99)
    monkeypatch.setattr(alpaca, "AlpacaPaper", lambda *a, **k: broker)
    dry = CLI.invoke(app, ["paper", "pairs", "alpaca", "test-order", "KO"])
    assert dry.exit_code == 0 and "DRY RUN" in dry.output and broker.calls == []
    sent = CLI.invoke(app, ["paper", "pairs", "alpaca", "test-order", "KO", "--qty", "2", "--send"])
    assert sent.exit_code == 0 and "Round trip cost" in sent.output and len(broker.calls) == 2
    broker.fills_for = {"buy"}
    broker.sides.clear()
    stuck = CLI.invoke(app, ["paper", "pairs", "alpaca", "test-order", "KO", "--send", "--timeout", "0.3"])
    assert stuck.exit_code == 1
