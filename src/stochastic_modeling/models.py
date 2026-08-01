"""Immutable parameter objects for the project models."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HestonParameters:
    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float

    @property
    def feller_diagnostic(self) -> float:
        """Return 2*kappa*theta-sigma^2; non-negative satisfies Feller."""
        return 2 * self.kappa * self.theta - self.sigma**2


@dataclass(frozen=True, slots=True)
class BatesParameters(HestonParameters):
    jump_intensity: float
    jump_mean: float
    jump_vol: float


@dataclass(frozen=True, slots=True)
class CirParameters:
    mean_reversion: float
    long_run_rate: float
    volatility: float

    @property
    def feller_diagnostic(self) -> float:
        """Return 2*a*b-eta^2; non-negative satisfies Feller."""
        return 2 * self.mean_reversion * self.long_run_rate - self.volatility**2
