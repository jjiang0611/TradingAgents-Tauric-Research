"""Robinhood account data -> PortfolioContext (read-only).

Takes the payloads returned by the Robinhood tools ``get_portfolio`` and
``get_equity_positions`` (with or without their ``{"data": ...}`` envelope) and
builds the book the decision agents read. It never talks to Robinhood and never
places an order; fetching the payloads is the caller's job.

Only equities and cash are mapped. If the account holds options, crypto or any
other asset class, the conversion fails rather than return a book that looks
complete but is missing holdings: a silent omission would let the agents treat
a held instrument as "no current position".
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tradingagents.portfolio import PortfolioContext, Position

# Value fields on get_portfolio that this adapter does not translate into positions.
_UNMAPPED_VALUE_FIELDS = (
    "options_value",
    "futures_value",
    "event_contracts_value",
    "crypto_value",
    "mutual_funds_value",
    "fixed_income_value",
)


def _data(payload: dict[str, Any]) -> dict[str, Any]:
    inner = payload.get("data", payload)
    if not isinstance(inner, dict):
        raise ValueError("Robinhood payload has no usable 'data' object")
    return inner


def _number(value: Any, field: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Robinhood field {field!r} is not a number: {value!r}") from exc


def portfolio_from_robinhood(
    portfolio: dict[str, Any],
    positions_pages: Iterable[dict[str, Any]],
) -> PortfolioContext:
    """Build a PortfolioContext from ``get_portfolio`` and ``get_equity_positions`` output.

    ``positions_pages`` is every page of ``get_equity_positions``; pass a one-item
    list when the account fits on one page. Cash is the unleveraged buying power
    (falling back to ``cash``), so margin is never presented to the agents as
    free cash.
    """
    summary = _data(portfolio)

    for field in _UNMAPPED_VALUE_FIELDS:
        value = _number(summary.get(field), field)
        if value:
            raise ValueError(
                f"account holds {field}={value:g}, which this adapter does not map to "
                "positions; refusing to build a book that omits it"
            )

    buying_power = summary.get("buying_power") or {}
    cash = _number(buying_power.get("unleveraged_buying_power"), "unleveraged_buying_power")
    if cash is None:
        cash = _number(summary.get("cash"), "cash")

    positions: list[Position] = []
    for page in positions_pages:
        for row in _data(page).get("positions", []):
            quantity = _number(row.get("quantity"), "quantity") or 0.0
            if quantity == 0:
                continue
            if row.get("type") == "short":
                quantity = -abs(quantity)
            positions.append(
                Position(
                    ticker=row["symbol"],
                    quantity=quantity,
                    average_price=_number(row.get("average_buy_price"), "average_buy_price"),
                )
            )

    return PortfolioContext(
        cash=cash,
        currency=summary.get("currency"),
        positions=positions,
    )
