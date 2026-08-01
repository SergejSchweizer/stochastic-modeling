"""Euribor curve construction and CIR calibration."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution

from stochastic_modeling.models import CirParameters


@dataclass(frozen=True, slots=True)
class EuriborCurve:
    tenor_days: np.ndarray
    annual_rates: np.ndarray

    @classmethod
    def supplied(cls) -> "EuriborCurve":
        return cls(
            tenor_days=np.array([7, 30, 90, 180, 365], dtype=float),
            annual_rates=np.array([0.00648, 0.00679, 0.01173, 0.01809, 0.02556]),
        )

    def weekly(self) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate non-negative weekly rates for one year."""
        weekly_days = np.arange(7, 366, 7, dtype=float)
        spline = CubicSpline(self.tenor_days, self.annual_rates, bc_type="natural")
        return weekly_days, np.maximum(spline(weekly_days), 0.0)


def cir_zero_rate(
    maturity: np.ndarray,
    model: CirParameters,
    initial_rate: float,
) -> np.ndarray:
    """Continuously compounded CIR zero rate."""
    a, b, eta = model.mean_reversion, model.long_run_rate, model.volatility
    gamma = np.sqrt(a**2 + 2 * eta**2)
    exponential = np.exp(gamma * maturity)
    denominator = (gamma + a) * (exponential - 1) + 2 * gamma
    numerator = 2 * gamma * np.exp((a + gamma) * maturity / 2)
    coefficient_a = (numerator / denominator) ** (2 * a * b / eta**2)
    coefficient_b = 2 * (exponential - 1) / denominator
    bond_price = coefficient_a * np.exp(-coefficient_b * initial_rate)
    return -np.log(bond_price) / maturity


def calibrate_cir(
    maturities: np.ndarray,
    market_rates: np.ndarray,
    initial_rate: float,
    seed: int = 627,
) -> tuple[CirParameters, float]:
    """Fit CIR parameters to continuously compounded market rates."""

    def objective(values: np.ndarray) -> float:
        model = CirParameters(*values)
        return float(np.mean((cir_zero_rate(maturities, model, initial_rate) - market_rates) ** 2))

    result = differential_evolution(
        objective,
        [(0.01, 20), (0.0001, 0.2), (0.001, 1.5)],
        seed=seed,
        polish=True,
    )
    model = CirParameters(*result.x)
    return model, objective(result.x)
