"""The paper account: an order ledger and the rules that turn ratings into fills.

State is only the list of orders. Cash, holdings and the equity curve are all
derived from the filled ones, so the account cannot drift from its own history.

Execution model (all of it, so nothing is implicit):

- Long-only, fractional shares, no commissions.
- A decision dated D fills at the first completed close after D, the same entry
  convention the memory log uses to score a decision, because a run dated D reads
  news published after D's close.
- Slippage of ``slippage_bps`` is applied against every fill.
- Each ticker gets an equal sleeve of ``starting_cash / len(tickers)``, fixed at
  creation. A rating sets the sleeve fraction held (``weights``):

  Buy         add up to the full sleeve; never sells
  Overweight  rebalance to half the sleeve, up or down
  Hold        no trade
  Underweight cut to a quarter of the sleeve; never buys
  Sell        close the position
  REVIEW      no trade (the decision had no parseable rating)

  Buy and Underweight act in one direction only, and a move smaller than
  ``band`` of a sleeve is ignored, so a rating repeated day after day does not
  churn the account through slippage.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Fraction of a ticker's sleeve each rating aims to hold.
DEFAULT_WEIGHTS = {"Buy": 1.0, "Overweight": 0.5, "Underweight": 0.25, "Sell": 0.0}
_NO_TRADE = {"Hold"}


class Order(BaseModel):
    ticker: str
    decision_date: str
    rating: str
    status: Literal["pending", "filled", "no_trade"] = "pending"
    fill_date: str | None = None
    price: float | None = None          # executed price, slippage included
    shares: float = 0.0                 # signed: positive buys, negative sells
    note: str = ""


class PaperAccount(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    name: str
    created: str                        # YYYY-MM-DD; fills start with the next close
    starting_cash: float = Field(gt=0)
    tickers: list[str] = Field(min_length=1)
    slippage_bps: float = Field(default=5.0, ge=0)
    band: float = Field(default=0.10, ge=0, lt=1)
    min_trade_value: float = Field(default=1.0, ge=0)
    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    orders: list[Order] = Field(default_factory=list)

    @model_validator(mode="after")
    def _distinct_tickers(self):
        if len({t.upper() for t in self.tickers}) != len(self.tickers):
            raise ValueError("tickers must be distinct")
        return self

    # --- derived state -------------------------------------------------

    @property
    def sleeve(self) -> float:
        return self.starting_cash / len(self.tickers)

    def _filled(self) -> list[Order]:
        return sorted((o for o in self.orders if o.status == "filled"),
                      key=lambda o: (o.fill_date, o.ticker))

    @property
    def cash(self) -> float:
        return self.starting_cash - sum(o.shares * o.price for o in self._filled())

    def holdings(self) -> dict[str, tuple[float, float | None]]:
        """ticker -> (shares, average cost). Selling keeps the average cost."""
        shares: dict[str, float] = {}
        cost: dict[str, float] = {}
        for o in self._filled():
            held = shares.get(o.ticker, 0.0)
            if o.shares > 0:
                cost[o.ticker] = cost.get(o.ticker, 0.0) + o.shares * o.price
            elif held > 0:
                cost[o.ticker] = cost.get(o.ticker, 0.0) * (1 + o.shares / held)
            shares[o.ticker] = held + o.shares
        return {t: (s, cost[t] / s if s > 1e-12 else None)
                for t, s in shares.items() if s > 1e-12}

    # --- recording decisions -------------------------------------------

    def has_decision(self, ticker: str, date: str) -> bool:
        return any(o.ticker == ticker and o.decision_date == date for o in self.orders)

    def record(self, ticker: str, date: str, rating: str) -> Order:
        """Add a decision. Dates may not move backwards: a fill is priced from what
        the account held when it filled, so history cannot be inserted before it."""
        if ticker not in self.tickers:
            raise ValueError(f"{ticker} is not in this account ({', '.join(self.tickers)})")
        if self.has_decision(ticker, date):
            raise ValueError(f"{ticker} already has a decision dated {date}")
        if date < self.created:
            raise ValueError(f"{date} is before the account was created ({self.created})")
        if self.orders and date < max(o.decision_date for o in self.orders):
            raise ValueError("decisions must be recorded in date order")
        order = Order(ticker=ticker, decision_date=date, rating=rating)
        if rating in _NO_TRADE or rating not in self.weights:
            order.status = "no_trade"
            order.note = "Hold: no trade" if rating in _NO_TRADE else f"{rating}: no tradeable rating"
        self.orders.append(order)
        return order

    # --- filling -------------------------------------------------------

    def fill_pending(self, closes: dict[str, pd.Series]) -> list[Order]:
        """Fill every pending order whose entry close has printed; return those filled.

        ``closes`` maps ticker to completed daily closes (ascending, naive dates).
        An order with no close after its decision date stays pending.
        """
        plan = []
        for order in self.orders:
            if order.status != "pending":
                continue
            series = closes.get(order.ticker)
            if series is None:
                continue
            later = series[series.index > pd.Timestamp(order.decision_date)]
            if later.empty:
                continue
            plan.append((later.index[0], order.ticker, order, float(later.iloc[0])))
        filled = []
        for fill_day, _, order, close in sorted(plan, key=lambda p: (p[0], p[1])):
            self._fill(order, fill_day.strftime("%Y-%m-%d"), close)
            filled.append(order)
        return filled

    def _fill(self, order: Order, fill_date: str, close: float) -> None:
        slip = self.slippage_bps / 10_000
        held = self.holdings().get(order.ticker, (0.0, None))[0]
        value = held * close
        target = self.weights[order.rating] * self.sleeve
        tolerance = self.band * self.sleeve
        rating = order.rating

        shares = 0.0
        if rating == "Sell":
            shares = -held
        elif rating == "Buy" and target - value > tolerance:
            shares = (target - value) / (close * (1 + slip))
        elif rating == "Overweight" and abs(target - value) > tolerance:
            shares = (target - value) / (close * (1 + slip if target > value else 1 - slip))
        elif rating == "Underweight" and value - target > tolerance:
            shares = -(value - target) / (close * (1 - slip))

        order.fill_date = fill_date
        if shares > 0:
            price = close * (1 + slip)
            affordable = max(self.cash, 0.0) / price
            if shares > affordable:
                shares, order.note = affordable, "limited by available cash"
        elif shares < 0:
            price = close * (1 - slip)
            shares = max(shares, -held)
        else:
            price = close
        if abs(shares * price) < self.min_trade_value or abs(shares) < 1e-12:
            order.status, order.note = "no_trade", order.note or "within the no-trade band"
            return
        order.status, order.shares, order.price = "filled", shares, price

    # --- measuring -----------------------------------------------------

    def equity_curve(self, closes: dict[str, pd.Series]) -> pd.Series:
        """Account value on every date any ticker closed, from creation onward."""
        days = _trading_days(closes, self.created)
        if days.empty:
            return pd.Series(dtype=float)
        filled = self._filled()
        values = []
        for day in days:
            cash, value = self.starting_cash, 0.0
            held: dict[str, float] = {}
            for o in filled:
                if pd.Timestamp(o.fill_date) <= day:
                    cash -= o.shares * o.price
                    held[o.ticker] = held.get(o.ticker, 0.0) + o.shares
            for ticker, shares in held.items():
                if abs(shares) > 1e-12 and ticker in closes:
                    value += shares * _asof(closes[ticker], day)
            values.append(cash + value)
        return pd.Series(values, index=days)

    def buy_and_hold(self, closes: dict[str, pd.Series], tickers: list[str] | None = None) -> pd.Series:
        """The same capital, split equally over ``tickers`` (default: this account's)
        and held from the first close after creation: the baseline to beat."""
        names = tickers or self.tickers
        days = _trading_days(closes, self.created)
        if days.empty:
            return pd.Series(dtype=float)
        entry = days[0]
        shares = {t: (self.starting_cash / len(names)) / _asof(closes[t], entry) for t in names}
        return pd.Series(
            [sum(shares[t] * _asof(closes[t], d) for t in names) for d in days], index=days)

    # --- persistence ---------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> PaperAccount:
        try:
            return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"paper account {path} is not usable: {exc}") from exc


def _trading_days(closes: dict[str, pd.Series], created: str) -> pd.DatetimeIndex:
    """Dates after creation on which any series closed; the first is the entry day."""
    if not closes:
        return pd.DatetimeIndex([])
    union = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in closes.values()])))
    return union[union > pd.Timestamp(created)]


def _asof(series: pd.Series, day: pd.Timestamp) -> float:
    prior = series[series.index <= day]
    if prior.empty:
        raise ValueError(f"no close on or before {day.date()}")
    return float(prior.iloc[-1])


def total_return(curve: pd.Series, start: float) -> float | None:
    return float(curve.iloc[-1] / start - 1) if len(curve) else None


def max_drawdown(curve: pd.Series) -> float | None:
    if not len(curve):
        return None
    return float((curve / curve.cummax() - 1).min())
