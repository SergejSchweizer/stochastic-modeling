"""Calibration application service."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares

from stochastic_modeling.config import DEFAULT_MARKET, MarketConfig
from stochastic_modeling.fourier import PricingStrategy, model_prices
from stochastic_modeling.models import BatesParameters, HestonParameters

ModelKind = Literal["heston", "bates"]


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    parameters: HestonParameters | BatesParameters
    mse: float
    residuals: np.ndarray

    @property
    def rmse(self) -> float:
        return float(np.sqrt(self.mse))


class CalibrationService:
    """Two-stage global/local calibration with injected pricing strategy."""

    HESTON_BOUNDS = [(0.001, 15), (0.001, 1), (0.01, 3), (-0.995, 0.995), (0.001, 1)]
    BATES_BOUNDS = HESTON_BOUNDS + [(0.001, 5), (-0.5, 0.25), (0.01, 1)]

    def __init__(
        self,
        pricer: PricingStrategy,
        market: MarketConfig = DEFAULT_MARKET,
        max_iterations: int = 90,
        population_size: int = 10,
    ) -> None:
        self.pricer = pricer
        self.market = market
        self.max_iterations = max_iterations
        self.population_size = population_size

    @staticmethod
    def _build(kind: ModelKind, values: np.ndarray):
        return HestonParameters(*values) if kind == "heston" else BatesParameters(*values)

    def calibrate(
        self,
        frame: pd.DataFrame,
        kind: ModelKind = "heston",
        seed: int = 622,
    ) -> CalibrationResult:
        """Minimize unweighted dollar-price MSE for one maturity."""
        if kind not in {"heston", "bates"}:
            raise ValueError("kind must be heston or bates")
        bounds = self.HESTON_BOUNDS if kind == "heston" else self.BATES_BOUNDS
        market_prices = frame["price"].to_numpy(float)

        def residuals(values: np.ndarray) -> np.ndarray:
            model = self._build(kind, values)
            return model_prices(frame, model, self.pricer, self.market) - market_prices

        def objective(values: np.ndarray) -> float:
            errors = residuals(values)
            return float(np.mean(errors**2)) if np.isfinite(errors).all() else 1e12

        global_fit = differential_evolution(
            objective,
            bounds,
            seed=seed,
            popsize=self.population_size,
            maxiter=self.max_iterations,
            tol=1e-7,
            polish=False,
            workers=1,
        )
        local_fit = least_squares(
            residuals,
            global_fit.x,
            bounds=np.asarray(bounds).T,
            max_nfev=1500,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        errors = local_fit.fun
        return CalibrationResult(
            parameters=self._build(kind, local_fit.x),
            mse=float(np.mean(errors**2)),
            residuals=errors,
        )
