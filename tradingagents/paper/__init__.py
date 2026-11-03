"""Forward paper trading: a simulated account that acts on the system's ratings.

Kept apart from ``tradingagents.backtest``, which scores decision quality and
deliberately has no execution model. Everything the system leaves open (how big,
at what price, with what friction) is decided here, explicitly, in one place.
"""

from tradingagents.paper.account import DEFAULT_WEIGHTS, Order, PaperAccount

__all__ = ["PaperAccount", "Order", "DEFAULT_WEIGHTS"]
