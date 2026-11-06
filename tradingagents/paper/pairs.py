"""Market-neutral pairs paper trading: short the leg that ran ahead, buy the laggard.

A separate account type from ``account.py``. That one acts on the LLM agents'
ratings and is long-only; this one is systematic (no LLM), holds a long and a
short leg per pair, and exists to test the rules below forward in time.

The rules are fixed when the account is created and stored in it. They are the
ones used in the 1990-2022 test that preceded this module, where they earned about
nothing after costs: this is a forward test of that result, not a promise of one.

- Signal: z-score of the log spread ``logA - beta*logB`` against its own trailing
  ``lookback``-session mean and standard deviation (beta is the trailing OLS slope
  of logA on logB). Beta, mean and deviation are frozen when a trade is decided.
- Enter when ``|z| > entry``: short the leg that ran ahead, long the laggard. Exit
  when the frozen z is back inside ``exit``; stop at ``|z| > stop``; time stop at
  ``max_hold`` sessions.
- A decision made at a close is executed at the NEXT completed close.
- Each pair gets an equal sleeve of ``starting_cash / len(pairs)`` as *gross*
  exposure: ``1/(1+beta)`` of it long in A and ``beta/(1+beta)`` short in B (or the
  reverse), fractional shares, fixed at entry (no rebalancing).
- Costs: ``slippage_bps`` against every fill, plus ``borrow_annual`` on the short
  leg's value per session held. Short proceeds earn nothing and no leg is ever
  unavailable to borrow, which flatters the result a little.

Because nothing here depends on news or an LLM, a session that was missed is not
lost: ``process`` replays every completed session after the last one it handled,
in order, and reaches the decisions a daily run would have.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tradingagents.paper.account import _asof, _trading_days

_FLOOR = 1e-6   # a spread deviation below this is a degenerate window, not a signal


class Rules(BaseModel):
    model_config = ConfigDict(frozen=True)

    lookback: int = 252
    entry: float = 2.0
    exit: float = 0.5
    stop: float = 4.0
    max_hold: int = 60
    beta_min: float = 0.3
    beta_max: float = 3.0


Status = Literal["pending_entry", "open", "pending_exit", "closed"]
Reason = Literal["converged", "stop", "time"]


class Trade(BaseModel):
    pair: str
    side: Literal["long_spread", "short_spread"]   # long_spread: long A, short B
    beta: float                                     # frozen at the decision
    mu: float
    sd: float
    decided: str
    status: Status = "pending_entry"
    entered: str | None = None
    a_shares: float = 0.0                           # signed: positive long, negative short
    b_shares: float = 0.0
    a_entry: float | None = None                    # executed price, slippage included
    b_entry: float | None = None
    exit_decided: str | None = None
    exit_reason: Reason | None = None
    exited: str | None = None
    a_exit: float | None = None
    b_exit: float | None = None
    borrow: float = 0.0                             # dollars of borrow fee, settled at exit


class PairsAccount(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    name: str
    created: str
    starting_cash: float = Field(gt=0)
    pairs: list[str] = Field(min_length=1)          # "KO/PEP": A is the first, B the second
    slippage_bps: float = Field(default=5.0, ge=0)
    borrow_annual: float = Field(default=0.005, ge=0)
    rules: Rules = Field(default_factory=Rules)
    cursors: dict[str, str] = Field(default_factory=dict)   # pair -> last session handled
    trades: list[Trade] = Field(default_factory=list)

    @model_validator(mode="after")
    def _valid_pairs(self):
        seen = set()
        for pair in self.pairs:
            parts = pair.split("/")
            if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
                raise ValueError(f"a pair is two different tickers, like KO/PEP; got {pair!r}")
            if pair in seen:
                raise ValueError(f"pair {pair} is listed twice")
            seen.add(pair)
        return self

    # --- derived ---------------------------------------------------------

    @property
    def sleeve(self) -> float:
        return self.starting_cash / len(self.pairs)

    @property
    def tickers(self) -> list[str]:
        return list(dict.fromkeys(t for pair in self.pairs for t in pair.split("/")))

    def active(self, pair: str) -> Trade | None:
        live = [t for t in self.trades if t.pair == pair and t.status != "closed"]
        return live[-1] if live else None

    def closed(self) -> list[Trade]:
        return [t for t in self.trades if t.status == "closed"]

    # --- the engine ------------------------------------------------------

    def process(self, closes: dict[str, pd.Series]) -> list[str]:
        """Handle every completed session after each pair's cursor, in order.

        Returns a line per decision or fill. ``closes`` maps ticker to completed
        daily closes (ascending, naive dates).
        """
        events: list[str] = []
        for pair in self.pairs:
            events += self._process_pair(pair, closes)
        return events

    def _window(self, la, lb, i: int):
        n = self.rules.lookback
        wa, wb = la[i - n + 1: i + 1], lb[i - n + 1: i + 1]
        var_b = np.var(wb, ddof=1)
        if var_b <= 0:
            return None
        beta = np.cov(wa, wb, ddof=1)[0, 1] / var_b
        spread = wa - beta * wb
        return float(beta), float(spread.mean()), max(float(spread.std(ddof=1)), _FLOOR)

    def _process_pair(self, pair: str, closes: dict[str, pd.Series]) -> list[str]:
        a, b = pair.split("/")
        if a not in closes or b not in closes:
            return []
        both = pd.concat([closes[a], closes[b]], axis=1, keys=["a", "b"]).dropna()
        if both.empty:
            return []
        idx, pa, pb = both.index, both["a"].to_numpy(float), both["b"].to_numpy(float)
        la, lb = np.log(pa), np.log(pb)
        rules, events = self.rules, []
        first = int(idx.searchsorted(pd.Timestamp(self.cursors.get(pair, self.created)), side="right"))
        for i in range(first, len(idx)):
            d = idx[i].strftime("%Y-%m-%d")
            trade = self.active(pair)
            if trade is not None and trade.status == "pending_entry":
                self._execute_entry(trade, d, float(pa[i]), float(pb[i]))
                events.append(f"{d} {pair}: entered {trade.side.replace('_', ' ')}")
            elif trade is not None and trade.status == "pending_exit":
                self._execute_exit(trade, d, float(pa[i]), float(pb[i]), idx, pa, pb, i)
                events.append(f"{d} {pair}: exited ({trade.exit_reason})")
                trade = None
            if trade is None:
                if i < rules.lookback:
                    continue
                window = self._window(la, lb, i)
                if window is None:
                    continue
                beta, mu, sd = window
                z = (la[i] - beta * lb[i] - mu) / sd
                if abs(z) > rules.entry and rules.beta_min <= beta <= rules.beta_max:
                    side = "short_spread" if z > 0 else "long_spread"
                    self.trades.append(Trade(pair=pair, side=side, beta=beta, mu=mu, sd=sd, decided=d))
                    events.append(f"{d} {pair}: z {z:+.2f} -> will enter {side.replace('_', ' ')} at the next close")
            elif trade.status == "open":
                held = i - int(idx.get_loc(pd.Timestamp(trade.entered))) + 1
                zf = (la[i] - trade.beta * lb[i] - trade.mu) / trade.sd
                reason: Reason | None = None
                if abs(zf) > rules.stop:
                    reason = "stop"
                elif held >= rules.max_hold:
                    reason = "time"
                elif (trade.side == "short_spread" and zf <= rules.exit) or \
                        (trade.side == "long_spread" and zf >= -rules.exit):
                    reason = "converged"
                if reason:
                    trade.status, trade.exit_decided, trade.exit_reason = "pending_exit", d, reason
                    events.append(f"{d} {pair}: z {zf:+.2f} -> will exit ({reason}) at the next close")
            self.cursors[pair] = d
        return events

    def _execute_entry(self, trade: Trade, day: str, pa: float, pb: float) -> None:
        slip = self.slippage_bps / 1e4
        gross = self.sleeve
        notional_a, notional_b = gross / (1 + trade.beta), gross * trade.beta / (1 + trade.beta)
        if trade.side == "long_spread":                       # buy A, sell B short
            trade.a_entry, trade.b_entry = pa * (1 + slip), pb * (1 - slip)
            trade.a_shares, trade.b_shares = notional_a / trade.a_entry, -notional_b / trade.b_entry
        else:                                                 # sell A short, buy B
            trade.a_entry, trade.b_entry = pa * (1 - slip), pb * (1 + slip)
            trade.a_shares, trade.b_shares = -notional_a / trade.a_entry, notional_b / trade.b_entry
        trade.status, trade.entered = "open", day

    def _execute_exit(self, trade: Trade, day: str, pa: float, pb: float, idx, pas, pbs, i: int) -> None:
        slip = self.slippage_bps / 1e4
        if trade.side == "long_spread":                       # sell A, buy B back
            trade.a_exit, trade.b_exit = pa * (1 - slip), pb * (1 + slip)
        else:
            trade.a_exit, trade.b_exit = pa * (1 + slip), pb * (1 - slip)
        i_entry = int(idx.get_loc(pd.Timestamp(trade.entered)))
        short_px, short_sh = (pbs, trade.b_shares) if trade.side == "long_spread" else (pas, trade.a_shares)
        trade.borrow = float(self.borrow_annual / 252 * abs(short_sh) * short_px[i_entry + 1: i + 1].sum())
        trade.status, trade.exited = "closed", day

    # --- measuring -------------------------------------------------------

    def _borrow_accrued(self, trade: Trade, closes: dict[str, pd.Series], day: pd.Timestamp) -> float:
        a, b = trade.pair.split("/")
        series, shares = (closes[b], trade.b_shares) if trade.side == "long_spread" else (closes[a], trade.a_shares)
        window = series[(series.index > pd.Timestamp(trade.entered)) & (series.index <= day)]
        return float(self.borrow_annual / 252 * abs(shares) * window.sum())

    def trade_pnl(self, trade: Trade, closes: dict[str, pd.Series], day: pd.Timestamp | None = None) -> float | None:
        """Dollar P&L of a trade: realized when closed by ``day``, else marked at ``day``."""
        if trade.entered is None or (day is not None and pd.Timestamp(trade.entered) > day):
            return None
        a, b = trade.pair.split("/")
        if trade.status == "closed" and (day is None or pd.Timestamp(trade.exited) <= day):
            return (trade.a_shares * (trade.a_exit - trade.a_entry)
                    + trade.b_shares * (trade.b_exit - trade.b_entry) - trade.borrow)
        day = day if day is not None else max(closes[a].index[-1], closes[b].index[-1])
        return (trade.a_shares * (_asof(closes[a], day) - trade.a_entry)
                + trade.b_shares * (_asof(closes[b], day) - trade.b_entry)
                - self._borrow_accrued(trade, closes, day))

    def equity_curve(self, closes: dict[str, pd.Series]) -> pd.Series:
        """Account value on each session after creation: starting cash plus every trade's P&L."""
        days = _trading_days({t: closes[t] for t in self.tickers if t in closes}, self.created)
        if days.empty:
            return pd.Series(dtype=float)
        values = []
        for day in days:
            total = self.starting_cash
            for trade in self.trades:
                pnl = self.trade_pnl(trade, closes, day)
                total += pnl or 0.0
            values.append(total)
        return pd.Series(values, index=days)

    def current_z(self, pair: str, closes: dict[str, pd.Series]) -> tuple[float, float] | None:
        """(z, beta) of the pair at its latest completed close, or None without enough history."""
        a, b = pair.split("/")
        if a not in closes or b not in closes:
            return None
        both = pd.concat([closes[a], closes[b]], axis=1, keys=["a", "b"]).dropna()
        if len(both) <= self.rules.lookback:
            return None
        la, lb = np.log(both["a"].to_numpy(float)), np.log(both["b"].to_numpy(float))
        window = self._window(la, lb, len(both) - 1)
        if window is None:
            return None
        beta, mu, sd = window
        return (la[-1] - beta * lb[-1] - mu) / sd, beta

    # --- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> PairsAccount:
        try:
            return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"pairs account {path} is not usable: {exc}") from exc
