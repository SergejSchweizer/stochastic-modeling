"""Shared market conventions."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MarketConfig:
    """Market inputs shared by pricing and simulation services."""

    spot: float = 232.90
    rate: float = 0.015
    trading_days: int = 250

    def year_fraction(self, days: int | float) -> float:
        """Convert trading days to a year fraction."""
        if days < 0:
            raise ValueError("days must be non-negative")
        return float(days) / self.trading_days


DEFAULT_MARKET = MarketConfig()
