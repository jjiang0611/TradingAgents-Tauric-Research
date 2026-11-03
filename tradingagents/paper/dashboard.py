"""A local web dashboard for a paper account: the numbers, the research, and a Run button.

Serves one page and a small JSON API from the standard library, bound to
127.0.0.1. Everything it shows is already on disk (the account ledger and the
state log every run writes), plus closing prices fetched the way ``status`` does.

Because "Run" spends LLM tokens, the one state-changing route is guarded: it needs
a per-process token that only the served page carries, a JSON content type (which
a cross-site form cannot send), and a Host header naming this server (which stops
DNS rebinding). Agent text is rendered with raw HTML disabled.
"""

from __future__ import annotations

import json
import re
import secrets
import socketserver
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from tradingagents.backtest import summarize
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.symbols import safe_ticker_component
from tradingagents.dataflows.vendors.yahoo.market import get_closes
from tradingagents.paper.account import PaperAccount, max_drawdown, total_return
from tradingagents.paper.runner import (
    BENCHMARK,
    MIN_DECISIONS_FOR_A_VIEW,
    DayResult,
    account_dir,
    account_path,
    fetch_all,
    run_day,
)
from tradingagents.report_html import _markdown, _render_agent_text
from tradingagents.reporting import _report_parts

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SUMMARY = re.compile(r"\*\*Executive Summary\*\*:\s*(.+?)(?:\n\s*\n|$)", re.S)
STATE_TTL_SECONDS = 60


def _round(values, places=2) -> list[float | None]:
    return [None if pd.isna(v) else round(float(v), places) for v in values]


def research_path(config: dict, name: str, ticker: str, date: str) -> Path:
    return (account_dir(config, name) / safe_ticker_component(ticker)
            / "TradingAgentsStrategy_logs" / f"full_states_log_{date}.json")


def build_state(account: PaperAccount, config: dict, fetch=get_closes) -> dict:
    """Everything the page draws, as plain JSON. Never writes the ledger."""
    closes = fetch_all(account, fetch)
    account.fill_pending(closes)
    strategy = account.equity_curve(closes)
    hold = account.buy_and_hold(closes)
    spy = account.buy_and_hold(closes, [BENCHMARK])
    start = account.starting_cash

    def line(curve):
        return {"return": total_return(curve, start), "drawdown": max_drawdown(curve),
                "value": float(curve.iloc[-1]) if len(curve) else start}

    holdings = []
    for ticker, (shares, cost) in account.holdings().items():
        last = float(closes[ticker].iloc[-1]) if len(closes.get(ticker, [])) else None
        value = shares * last if last is not None else None
        holdings.append({
            "ticker": ticker, "shares": shares, "avg_cost": cost, "last": last, "value": value,
            "pnl": (last - cost) * shares if last is not None and cost else None,
            "pnl_pct": (last / cost - 1) if last is not None and cost else None,
        })

    orders = [{**o.model_dump(), "has_research": research_path(
                  config, account.name, o.ticker, o.decision_date).is_file()}
              for o in sorted(account.orders, key=lambda o: (o.decision_date, o.ticker), reverse=True)]

    latest = {}
    for o in orders:
        latest.setdefault(o["ticker"], o)
    for ticker, o in latest.items():
        o["summary"] = _summary(research_path(config, account.name, ticker, o["decision_date"]))

    log = account_dir(config, account.name) / "trading_memory.md"
    scored = None
    if log.is_file():
        s = summarize(log)
        scored = {"resolved": s.resolved, "pending": s.pending, "unscored": s.unscored, "holding": s.holding,
                  "by_rating": {r: {"count": v.count, "hit_rate": v.hit_rate, "mean_alpha": v.mean_alpha}
                                for r, v in s.by_rating.items()}}

    today = get_current_date()
    return {
        "name": account.name, "created": account.created, "today": today,
        "tickers": account.tickers, "starting_cash": start, "slippage_bps": account.slippage_bps,
        "cash": account.cash, "as_of": strategy.index[-1].strftime("%Y-%m-%d") if len(strategy) else None,
        "lines": {"strategy": line(strategy), "hold": line(hold), "spy": line(spy)},
        "curve": {"dates": [d.strftime("%Y-%m-%d") for d in strategy.index],
                  "strategy": _round(strategy), "hold": _round(hold), "spy": _round(spy)},
        "holdings": holdings, "orders": orders, "latest": latest,
        "decided_today": sorted({o.ticker for o in account.orders if o.decision_date == today}),
        "pending": sum(o.status == "pending" for o in account.orders),
        "scored": scored, "min_decisions": MIN_DECISIONS_FOR_A_VIEW,
    }


def _summary(path: Path) -> str | None:
    try:
        text = json.loads(path.read_text(encoding="utf-8")).get("final_trade_decision") or ""
    except (OSError, ValueError):
        return None
    found = _SUMMARY.search(text)
    return found.group(1).strip() if found else None


def load_research(config: dict, name: str, ticker: str, date: str) -> dict:
    """One decision's research, sectioned like the report tree, rendered to safe HTML."""
    path = research_path(config, name, ticker, date)
    if not path.is_file():
        raise FileNotFoundError(f"no research saved for {ticker} on {date}")
    state = json.loads(path.read_text(encoding="utf-8"))
    md = _markdown()
    sections = [
        {"heading": heading,
         "items": [{"agent": agent, "html": _render_agent_text(md, text)} for agent, _file, text in written]}
        for heading, _folder, written in _report_parts(state)
    ]
    return {"ticker": ticker, "date": date, "rating": state.get("final_rating"),
            "settings": state.get("run_settings") or {}, "sections": sections}


class RunManager:
    """Runs one day's analysis in a thread and reports its progress; one run at a time."""

    def __init__(self, config: dict, name: str, runner: Callable = run_day, fetch=get_closes):
        self.config, self.name, self.runner, self.fetch = config, name, runner, fetch
        self._lock = threading.Lock()
        self._snap = {"state": "idle", "log": []}

    def snapshot(self) -> dict:
        with self._lock:
            return {**self._snap, "log": list(self._snap["log"])}

    def start(self, date: str | None) -> bool:
        with self._lock:
            if self._snap["state"] == "running":
                return False
            self._snap = {"state": "running", "date": date or get_current_date(),
                          "started": time.time(), "log": []}
        threading.Thread(target=self._work, args=(date,), daemon=True).start()
        return True

    def _note(self, line: str) -> None:
        with self._lock:
            self._snap["log"].append(line)

    def _work(self, date: str | None) -> None:
        try:
            account = PaperAccount.load(account_path(self.config, self.name))
            result: DayResult = self.runner(
                account, self.config, date, fetch=self.fetch,
                progress=lambda i, n, t: self._note(f"[{i}/{n}] analyzing {t}"))
            outcome = {"state": "done", "result": {
                "date": result.date, "filled": result.filled, "decided": result.decided,
                "skipped": result.skipped, "failures": result.failures}}
        except Exception as exc:  # shown on the page; the run is the user's to retry
            outcome = {"state": "error", "error": str(exc)}
        with self._lock:
            self._snap = {**self._snap, **outcome, "finished": time.time()}


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self) -> None:
        # HTTPServer.server_bind asks DNS for the host's name, which can stall start-up.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[0], self.server_address[1]


def _page() -> str:
    return resources.files("tradingagents.paper").joinpath("dashboard.html").read_text(encoding="utf-8")


def make_server(config: dict, name: str, host: str = "127.0.0.1", port: int = 8765,
                fetch=get_closes, runner: Callable = run_day) -> ThreadingHTTPServer:
    """Build (not start) the dashboard server. ``port=0`` picks a free port."""
    if not account_path(config, name).is_file():
        raise FileNotFoundError(f"no paper account {name!r}")
    token = secrets.token_urlsafe(24)
    runs = RunManager(config, name, runner, fetch)
    cache: dict = {"at": 0.0, "state": None}
    cache_lock = threading.Lock()

    def state() -> dict:
        with cache_lock:
            fresh = time.time() - cache["at"] < STATE_TTL_SECONDS
            finished_since = runs.snapshot().get("finished", 0) > cache["at"]
            if not fresh or finished_since:
                cache["state"] = build_state(PaperAccount.load(account_path(config, name)), config, fetch)
                cache["at"] = time.time()
            return cache["state"]

    class Handler(BaseHTTPRequestHandler):
        server_version = "PaperDashboard"

        def log_message(self, *args):  # quiet: the CLI owns the terminal
            pass

        def _allowed_host(self) -> bool:
            _, bound = self.server.server_address[:2]
            return (self.headers.get("Host") or "") in {f"127.0.0.1:{bound}", f"localhost:{bound}"}

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                             "connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status: int = 200) -> None:
            self._send(status, json.dumps(payload, default=str).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            if not self._allowed_host():
                return self._json({"error": "unexpected Host header"}, 403)
            url = urlparse(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == "/":
                    page = _page().replace("__TOKEN__", token)
                    return self._send(200, page.encode(), "text/html; charset=utf-8")
                if url.path == "/api/state":
                    if query.get("refresh"):
                        with cache_lock:
                            cache["at"] = 0.0
                    return self._json(state())
                if url.path == "/api/run":
                    return self._json(runs.snapshot())
                if url.path == "/api/research":
                    ticker, date = (query.get("ticker") or [""])[0], (query.get("date") or [""])[0]
                    account = PaperAccount.load(account_path(config, name))
                    if ticker not in account.tickers or not _DATE.match(date):
                        return self._json({"error": "unknown ticker or malformed date"}, 400)
                    return self._json(load_research(config, name, ticker, date))
            except FileNotFoundError as exc:
                return self._json({"error": str(exc)}, 404)
            except Exception as exc:
                return self._json({"error": str(exc)}, 500)
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            if not self._allowed_host():
                return self._json({"error": "unexpected Host header"}, 403)
            if urlparse(self.path).path != "/api/run":
                return self._json({"error": "not found"}, 404)
            if not secrets.compare_digest(self.headers.get("X-Dashboard-Token") or "", token):
                return self._json({"error": "missing or wrong token; reload the page"}, 403)
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._json({"error": "send application/json"}, 415)
            try:
                length = min(int(self.headers.get("Content-Length") or 0), 4096)
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._json({"error": "body is not JSON"}, 400)
            date = body.get("date") if isinstance(body, dict) else None
            if date is not None and not (isinstance(date, str) and _DATE.match(date)):
                return self._json({"error": "date must be YYYY-MM-DD"}, 400)
            if not runs.start(date):
                return self._json({"error": "a run is already in progress"}, 409)
            with cache_lock:
                cache["at"] = 0.0
            self._json(runs.snapshot(), 202)

    return _Server((host, port), Handler)
