import numpy as np
import pandas as pd
import pytest

from stochastic_modeling.fourier import (
    CarrMadanPricer,
    LewisPricer,
    Quadrature,
    bates_cf,
    heston_cf,
    model_prices,
)
from stochastic_modeling.models import BatesParameters, HestonParameters

HESTON = HestonParameters(2.0, 0.04, 0.35, -0.7, 0.04)
BATES = BatesParameters(2.0, 0.04, 0.35, -0.7, 0.04, 0.2, -0.08, 0.15)


def test_quadrature_validation():
    grid = Quadrature.gauss_legendre(32, 80)
    assert len(grid.nodes) == len(grid.weights) == 32
    with pytest.raises(ValueError):
        Quadrature.gauss_legendre(1, 80)
    with pytest.raises(ValueError):
        Quadrature.gauss_legendre(32, 0)


def test_characteristic_functions_equal_one_at_origin():
    assert heston_cf(0j, 0.25, HESTON) == pytest.approx(1 + 0j)
    assert bates_cf(0j, 0.25, BATES) == pytest.approx(1 + 0j)


def test_pricing_strategies_agree_and_puts_follow_parity():
    frame = pd.DataFrame(
        {
            "strike": [225.0, 232.9, 240.0, 225.0],
            "type": ["C", "C", "C", "P"],
            "T": [0.1] * 4,
        }
    )
    grid = Quadrature.gauss_legendre(128, 140)
    lewis = model_prices(frame, HESTON, LewisPricer(grid))
    carr_madan = model_prices(frame, HESTON, CarrMadanPricer(grid))
    assert np.all(lewis > 0)
    assert np.max(np.abs(lewis - carr_madan)) < 0.02
    expected_put = lewis[0] - 232.90 + 225 * np.exp(-0.015 * 0.1)
    assert lewis[3] == pytest.approx(expected_put)


def test_bates_pricing_and_frame_validation():
    frame = pd.DataFrame({"strike": [232.9], "type": ["C"], "T": [0.24]})
    assert model_prices(frame, BATES, LewisPricer())[0] > 0
    with pytest.raises(ValueError, match="one non-empty maturity"):
        model_prices(frame.iloc[0:0], HESTON, LewisPricer())
    with pytest.raises(ValueError, match="one non-empty maturity"):
        model_prices(pd.concat([frame, frame.assign(T=0.5)]), HESTON, LewisPricer())
