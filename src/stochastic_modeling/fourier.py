"""Characteristic functions and interchangeable Fourier pricing strategies."""

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd
from scipy.special import roots_legendre

from stochastic_modeling.config import DEFAULT_MARKET, MarketConfig
from stochastic_modeling.models import BatesParameters, HestonParameters

CharacteristicFunction = Callable[[np.ndarray | complex, float], np.ndarray | complex]


@dataclass(frozen=True, slots=True)
class Quadrature:
    """Fixed Gauss-Legendre quadrature grid used during calibration."""

    nodes: np.ndarray
    weights: np.ndarray

    @classmethod
    def gauss_legendre(cls, count: int = 192, upper: float = 160.0) -> "Quadrature":
        if count < 2 or upper <= 0:
            raise ValueError("count must be >=2 and upper must be positive")
        raw_nodes, raw_weights = roots_legendre(count)
        return cls(
            nodes=0.5 * (raw_nodes + 1.0) * upper,
            weights=0.5 * upper * raw_weights,
        )


DEFAULT_QUADRATURE = Quadrature.gauss_legendre()


def heston_cf(
    z: np.ndarray | complex,
    maturity: float,
    model: HestonParameters,
    rate: float = DEFAULT_MARKET.rate,
) -> np.ndarray | complex:
    """Stable characteristic function of log(S_T / S_0)."""
    kappa, theta, sigma, rho, v0 = (
        model.kappa,
        model.theta,
        model.sigma,
        model.rho,
        model.v0,
    )
    d = np.sqrt((rho * sigma * 1j * z - kappa) ** 2 + sigma**2 * (1j * z + z**2))
    g = (kappa - rho * sigma * 1j * z - d) / (kappa - rho * sigma * 1j * z + d)
    c = rate * 1j * z * maturity + (kappa * theta / sigma**2) * (
        (kappa - rho * sigma * 1j * z - d) * maturity
        - 2 * np.log((1 - g * np.exp(-d * maturity)) / (1 - g))
    )
    d_coefficient = ((kappa - rho * sigma * 1j * z - d) / sigma**2) * (
        (1 - np.exp(-d * maturity)) / (1 - g * np.exp(-d * maturity))
    )
    return np.exp(c + d_coefficient * v0)


def bates_cf(
    z: np.ndarray | complex,
    maturity: float,
    model: BatesParameters,
    market: MarketConfig = DEFAULT_MARKET,
) -> np.ndarray | complex:
    """Risk-neutral Bates characteristic function of log(S_T / S_0)."""
    compensator = np.exp(model.jump_mean + 0.5 * model.jump_vol**2) - 1
    base = heston_cf(
        z,
        maturity,
        model,
        rate=market.rate - model.jump_intensity * compensator,
    )
    jump = np.exp(
        model.jump_intensity
        * maturity
        * (np.exp(1j * z * model.jump_mean - 0.5 * model.jump_vol**2 * z**2) - 1)
    )
    return base * jump


class PricingStrategy(Protocol):
    """Interface shared by Fourier pricing strategies."""

    def calls(
        self,
        strikes: np.ndarray,
        maturity: float,
        characteristic_function: CharacteristicFunction,
        market: MarketConfig = DEFAULT_MARKET,
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class LewisPricer:
    quadrature: Quadrature = DEFAULT_QUADRATURE

    def calls(
        self,
        strikes: np.ndarray,
        maturity: float,
        characteristic_function: CharacteristicFunction,
        market: MarketConfig = DEFAULT_MARKET,
    ) -> np.ndarray:
        """Price calls with the Lewis (2001) representation."""
        strikes = np.asarray(strikes, dtype=float)
        u, weights = self.quadrature.nodes, self.quadrature.weights
        phi = characteristic_function(u - 0.5j, maturity)
        prices = []
        for strike in strikes:
            integrand = np.real(np.exp(1j * u * np.log(market.spot / strike)) * phi / (u**2 + 0.25))
            integral = np.sum(weights * integrand)
            prices.append(
                market.spot
                - np.exp(-market.rate * maturity) * np.sqrt(market.spot * strike) * integral / np.pi
            )
        return np.maximum(np.asarray(prices), 0.0)


@dataclass(frozen=True, slots=True)
class CarrMadanPricer:
    quadrature: Quadrature = DEFAULT_QUADRATURE
    alpha: float = 1.5

    def calls(
        self,
        strikes: np.ndarray,
        maturity: float,
        characteristic_function: CharacteristicFunction,
        market: MarketConfig = DEFAULT_MARKET,
    ) -> np.ndarray:
        """Price calls with the damped Carr-Madan (1999) transform."""
        strikes = np.asarray(strikes, dtype=float)
        u, weights = self.quadrature.nodes, self.quadrature.weights
        shifted = u - 1j * (self.alpha + 1)
        phi_log_spot = np.exp(1j * shifted * np.log(market.spot)) * characteristic_function(
            shifted, maturity
        )
        denominator = self.alpha**2 + self.alpha - u**2 + 1j * (2 * self.alpha + 1) * u
        psi = np.exp(-market.rate * maturity) * phi_log_spot / denominator
        return np.asarray(
            [
                np.exp(-self.alpha * np.log(strike))
                * np.sum(weights * np.real(np.exp(-1j * u * np.log(strike)) * psi))
                / np.pi
                for strike in strikes
            ]
        )


def model_prices(
    frame: pd.DataFrame,
    model: HestonParameters | BatesParameters,
    pricer: PricingStrategy,
    market: MarketConfig = DEFAULT_MARKET,
) -> np.ndarray:
    """Price a single-maturity frame of calls and puts with a strategy."""
    if frame.empty or frame["T"].nunique() != 1:
        raise ValueError("frame must contain one non-empty maturity")
    maturity = float(frame["T"].iloc[0])
    strikes = frame["strike"].to_numpy(float)
    if isinstance(model, BatesParameters):

        def characteristic_function(z, t):
            return bates_cf(z, t, model, market)
    else:

        def characteristic_function(z, t):
            return heston_cf(z, t, model, market.rate)

    calls = pricer.calls(strikes, maturity, characteristic_function, market)
    puts = calls - market.spot + strikes * np.exp(-market.rate * maturity)
    return np.where(frame["type"].to_numpy() == "C", calls, puts)
