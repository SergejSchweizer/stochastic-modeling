"""Risk-neutral Monte Carlo engines and summary values."""

from dataclasses import dataclass

import numpy as np

from stochastic_modeling.config import DEFAULT_MARKET, MarketConfig
from stochastic_modeling.models import CirParameters, HestonParameters


@dataclass(frozen=True, slots=True)
class MonteCarloEstimate:
    fair_value: float
    standard_error: float
    confidence_low: float
    confidence_high: float
    paths: int

    def client_price(self, fee: float = 0.04) -> float:
        if fee < 0:
            raise ValueError("fee must be non-negative")
        return self.fair_value * (1 + fee)


def simulate_heston_asian(
    model: HestonParameters,
    days: int = 20,
    paths: int = 200_000,
    seed: int = 624,
    market: MarketConfig = DEFAULT_MARKET,
) -> np.ndarray:
    """Simulate discounted arithmetic-average ATM Asian call payoffs."""
    if days <= 0 or paths <= 1:
        raise ValueError("days must be positive and paths must exceed one")
    rng = np.random.default_rng(seed)
    dt = 1 / market.trading_days
    log_spot = np.full(paths, np.log(market.spot))
    variance = np.full(paths, model.v0)
    running_sum = np.full(paths, market.spot)
    orthogonal_scale = np.sqrt(1 - model.rho**2)
    for _ in range(days):
        stock_shock, independent_shock = rng.standard_normal((2, paths))
        variance_shock = model.rho * stock_shock + orthogonal_scale * independent_shock
        positive_variance = np.maximum(variance, 0)
        log_spot += (market.rate - 0.5 * positive_variance) * dt + np.sqrt(
            positive_variance * dt
        ) * stock_shock
        variance += (
            model.kappa * (model.theta - positive_variance) * dt
            + model.sigma * np.sqrt(positive_variance * dt) * variance_shock
        )
        running_sum += np.exp(log_spot)
    payoff = np.maximum(running_sum / (days + 1) - market.spot, 0)
    return np.exp(-market.rate * market.year_fraction(days)) * payoff


def estimate_asian_call(samples: np.ndarray, confidence: float = 0.95) -> MonteCarloEstimate:
    """Summarize Monte Carlo samples using the normal estimator interval."""
    if samples.size < 2 or confidence != 0.95:
        raise ValueError("at least two samples and 95% confidence are required")
    fair_value = float(np.mean(samples))
    standard_error = float(np.std(samples, ddof=1) / np.sqrt(samples.size))
    return MonteCarloEstimate(
        fair_value=fair_value,
        standard_error=standard_error,
        confidence_low=fair_value - 1.96 * standard_error,
        confidence_high=fair_value + 1.96 * standard_error,
        paths=int(samples.size),
    )


def simulate_cir_terminal(
    model: CirParameters,
    initial_rate: float,
    paths: int = 100_000,
    days: int = 250,
    seed: int = 628,
    market: MarketConfig = DEFAULT_MARKET,
) -> np.ndarray:
    """Simulate terminal CIR rates with daily full truncation."""
    if initial_rate < 0 or paths <= 1 or days <= 0:
        raise ValueError("invalid CIR simulation inputs")
    rng = np.random.default_rng(seed)
    dt = 1 / market.trading_days
    rates = np.full(paths, initial_rate)
    for _ in range(days):
        positive_rates = np.maximum(rates, 0)
        rates += model.mean_reversion * (
            model.long_run_rate - positive_rates
        ) * dt + model.volatility * np.sqrt(positive_rates * dt) * rng.standard_normal(paths)
    return np.maximum(rates, 0)
