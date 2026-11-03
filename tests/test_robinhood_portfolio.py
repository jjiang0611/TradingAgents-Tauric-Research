"""Robinhood payloads -> PortfolioContext: faithful, and loud about what it cannot map."""

from __future__ import annotations

import pytest

from tradingagents.brokers.robinhood import portfolio_from_robinhood

PORTFOLIO = {
    "data": {
        "total_value": "191.90",
        "equity_value": "191.90",
        "options_value": "0",
        "crypto_value": "0",
        "cash": "5",
        "currency": "USD",
        "buying_power": {"buying_power": "50.0000", "unleveraged_buying_power": "5.0000"},
    }
}
POSITIONS = {
    "data": {
        "positions": [
            {"symbol": "VOO", "quantity": "0.276247", "average_buy_price": "712.15", "type": "long"},
            {"symbol": "OLD", "quantity": "0.000000", "type": "long"},
        ]
    }
}


@pytest.mark.unit
def test_maps_positions_and_uses_unleveraged_cash():
    book = portfolio_from_robinhood(PORTFOLIO, [POSITIONS])
    assert book.cash == 5.0 and book.currency == "USD"
    assert [p.ticker for p in book.positions] == ["VOO"]
    assert book.position_in("VOO").average_price == pytest.approx(712.15)


@pytest.mark.unit
def test_empty_account_is_a_flat_book_not_missing_context():
    book = portfolio_from_robinhood(PORTFOLIO, [{"data": {"positions": []}}])
    assert book.positions == []
    assert "No current position in VOO" in book.render("VOO")


@pytest.mark.unit
def test_short_positions_are_negative():
    page = {"positions": [{"symbol": "XYZ", "quantity": "3", "type": "short"}]}
    assert portfolio_from_robinhood(PORTFOLIO, [page]).positions[0].quantity == -3.0


@pytest.mark.unit
def test_unmapped_asset_class_fails_instead_of_dropping_holdings():
    held = {"data": {**PORTFOLIO["data"], "options_value": "120.50"}}
    with pytest.raises(ValueError, match="options_value"):
        portfolio_from_robinhood(held, [POSITIONS])


@pytest.mark.unit
def test_malformed_number_is_rejected():
    page = {"positions": [{"symbol": "VOO", "quantity": "lots", "type": "long"}]}
    with pytest.raises(ValueError, match="quantity"):
        portfolio_from_robinhood(PORTFOLIO, [page])
