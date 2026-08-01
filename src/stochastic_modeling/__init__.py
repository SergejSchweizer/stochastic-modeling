"""Public API for the stochastic modelling coursework package."""

from stochastic_modeling.calibration import CalibrationResult, CalibrationService
from stochastic_modeling.config import DEFAULT_MARKET, MarketConfig
from stochastic_modeling.data import load_option_data, parity_analysis
from stochastic_modeling.fourier import (
    CarrMadanPricer,
    LewisPricer,
    Quadrature,
    bates_cf,
    heston_cf,
    model_prices,
)
from stochastic_modeling.models import BatesParameters, CirParameters, HestonParameters
from stochastic_modeling.rates import EuriborCurve, calibrate_cir, cir_zero_rate
from stochastic_modeling.simulation import (
    MonteCarloEstimate,
    estimate_asian_call,
    simulate_cir_terminal,
    simulate_heston_asian,
)

__all__ = [
    "DEFAULT_MARKET",
    "BatesParameters",
    "CalibrationResult",
    "CalibrationService",
    "CarrMadanPricer",
    "CirParameters",
    "EuriborCurve",
    "HestonParameters",
    "LewisPricer",
    "MarketConfig",
    "MonteCarloEstimate",
    "Quadrature",
    "bates_cf",
    "calibrate_cir",
    "cir_zero_rate",
    "estimate_asian_call",
    "heston_cf",
    "load_option_data",
    "model_prices",
    "parity_analysis",
    "simulate_cir_terminal",
    "simulate_heston_asian",
]
