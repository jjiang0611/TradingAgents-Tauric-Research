"""The paper dashboard: its numbers, its research view, and the guard on the Run route.

Run spends LLM tokens, so a page on another site must not be able to start one.
"""

from __future__ import annotations

import http.client
import json
import re
import socket
import threading
import time

import pandas as pd
import pytest

from tradingagents.paper import dashboard as dash, runner
from tradingagents.paper.account import PaperAccount

# The suite refuses every connection. These tests talk to a server they start on
# 127.0.0.1, so they allow that address and nothing else.
_REAL_CONNECT, _REAL_GETADDRINFO = socket.socket.connect, socket.getaddrinfo
_LOOPBACK = ("127.0.0.1", "localhost")


def _allow_loopback_only(monkeypatch):
    def connect(self, address):
        if address[0] not in _LOOPBACK:
            raise OSError(f"test tried to reach the network: {address}")
        return _REAL_CONNECT(self, address)

    def getaddrinfo(host, *args, **kwargs):
        if host not in _LOOPBACK:
            raise OSError(f"test tried to reach the network: {host}")
        return _REAL_GETADDRINFO(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


DAYS = pd.bdate_range("2026-03-02", periods=6)
PRICES = {"AAA": [10, 10, 11, 12, 12, 13], "SPY": [100, 100, 101, 102, 103, 104]}


def fetch(ticker, start, end):
    return pd.Series(PRICES[ticker], index=DAYS, dtype=float)


@pytest.fixture
def served(tmp_path, monkeypatch):
    _allow_loopback_only(monkeypatch)
    monkeypatch.setattr(runner, "get_current_date", lambda: "2026-03-10")
    monkeypatch.setattr(dash, "get_current_date", lambda: "2026-03-10")
    config = {"results_dir": str(tmp_path)}
    account = runner.create_account(config, "main", ["AAA"], 1000.0, slippage_bps=0, created="2026-03-01")
    account.record("AAA", "2026-03-02", "Buy")
    account.fill_pending(runner.fetch_all(account, fetch))
    account.save(runner.account_path(config, "main"))
    state = {
        "company_of_interest": "AAA", "trade_date": "2026-03-02", "final_rating": "Buy",
        "market_report": "Trend up.\n\n<script>alert(1)</script>", "investment_debate_state": {"bull_history": "bull"},
        "final_trade_decision": "**Rating**: Buy\n\n**Executive Summary**: Add on dips.\n\n**Price Target**: 14",
        "run_settings": {"deep_think_llm": "m"},
    }
    path = dash.research_path(config, "main", "AAA", "2026-03-02")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(state))

    release = threading.Event()

    def slow_runner(acct, cfg, date=None, progress=None, **kw):
        progress(1, 1, "AAA")
        release.wait(5)
        return runner.DayResult(date=date or "2026-03-10", decided=[("AAA", "Hold")])

    server = dash.make_server(config, "main", port=0, fetch=fetch, runner=slow_runner)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, config, release
    release.set()
    server.shutdown()
    server.server_close()


def call(server, method, path, body=None, headers=None, host=None):
    port = server.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    h = {"Host": host or f"127.0.0.1:{port}", **(headers or {})}
    conn.request(method, path, body=body, headers=h)
    res = conn.getresponse()
    data = res.read()
    conn.close()
    return res.status, data


def token_of(server):
    _, page = call(server, "GET", "/")
    return re.search(r'const TOKEN = "([^"]+)"', page.decode()).group(1)


@pytest.mark.unit
def test_state_reports_lines_and_the_latest_call_without_writing_the_ledger(served):
    server, config, _ = served
    before = runner.account_path(config, "main").read_text()
    status, data = call(server, "GET", "/api/state")
    body = json.loads(data)
    assert status == 200
    assert body["lines"]["strategy"]["return"] == pytest.approx(0.30, abs=0.01)
    assert body["latest"]["AAA"]["summary"] == "Add on dips."
    assert body["curve"]["dates"] and set(body["curve"]) >= {"strategy", "hold", "spy"}
    assert runner.account_path(config, "main").read_text() == before


@pytest.mark.unit
def test_research_is_rendered_without_raw_html_and_confined_to_account_tickers(served):
    server, _, _ = served
    status, data = call(server, "GET", "/api/research?ticker=AAA&date=2026-03-02")
    assert status == 200
    body = json.loads(data)
    html = json.dumps(body["sections"])
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert body["rating"] == "Buy"
    assert call(server, "GET", "/api/research?ticker=../../etc&date=2026-03-02")[0] == 400
    assert call(server, "GET", "/api/research?ticker=AAA&date=2026-3-2")[0] == 400
    assert call(server, "GET", "/api/research?ticker=AAA&date=2026-03-09")[0] == 404


@pytest.mark.unit
def test_a_foreign_host_header_is_refused(served):
    server, _, _ = served
    assert call(server, "GET", "/api/state", host="evil.example")[0] == 403
    assert call(server, "POST", "/api/run", body="{}", host="evil.example")[0] == 403


@pytest.mark.unit
def test_run_needs_the_page_token_and_a_json_body(served):
    server, _, _ = served
    json_type = {"Content-Type": "application/json"}
    assert call(server, "POST", "/api/run", "{}", json_type)[0] == 403
    assert call(server, "POST", "/api/run", "{}", {**json_type, "X-Dashboard-Token": "wrong"})[0] == 403
    token = token_of(server)
    assert call(server, "POST", "/api/run", "{}", {"Content-Type": "text/plain", "X-Dashboard-Token": token})[0] == 415
    bad_date = json.dumps({"date": "tomorrow"})
    assert call(server, "POST", "/api/run", bad_date, {**json_type, "X-Dashboard-Token": token})[0] == 400


@pytest.mark.unit
def test_run_starts_once_reports_progress_and_finishes(served):
    server, _, release = served
    headers = {"Content-Type": "application/json", "X-Dashboard-Token": token_of(server)}
    assert call(server, "POST", "/api/run", json.dumps({"date": "2026-03-10"}), headers)[0] == 202
    assert call(server, "POST", "/api/run", "{}", headers)[0] == 409        # one run at a time
    running = json.loads(call(server, "GET", "/api/run")[1])
    assert running["state"] == "running" and running["log"] == ["[1/1] analyzing AAA"]
    release.set()
    for _ in range(50):
        snap = json.loads(call(server, "GET", "/api/run")[1])
        if snap["state"] != "running":
            break
        time.sleep(0.1)
    assert snap["state"] == "done" and snap["result"]["decided"] == [["AAA", "Hold"]]


@pytest.mark.unit
def test_a_failed_run_is_reported_not_raised(tmp_path):
    config = {"results_dir": str(tmp_path)}
    runner.create_account(config, "main", ["AAA"], 1000.0, created="2026-03-01")

    def boom(*a, **k):
        raise RuntimeError("no API key")

    manager = dash.RunManager(config, "main", runner=boom, fetch=fetch)
    assert manager.start(None)
    for _ in range(50):
        if manager.snapshot()["state"] != "running":
            break
        time.sleep(0.05)
    assert manager.snapshot()["state"] == "error" and "no API key" in manager.snapshot()["error"]
    assert isinstance(PaperAccount.load(runner.account_path(config, "main")), PaperAccount)
