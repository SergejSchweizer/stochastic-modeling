import numpy as np
import pytest

from stochastic_modeling.models import CirParameters, HestonParameters
from stochastic_modeling.simulation import (
    estimate_asian_call,
    simulate_cir_terminal,
    simulate_heston_asian,
)


def test_asian_simulation_is_reproducible_and_summarized():
    model = HestonParameters(2.0, 0.04, 0.3, -0.7, 0.04)
    first = simulate_heston_asian(model, days=3, paths=500, seed=1)
    second = simulate_heston_asian(model, days=3, paths=500, seed=1)
    assert np.array_equal(first, second)
    estimate = estimate_asian_call(first)
    assert estimate.paths == 500
    assert estimate.confidence_low < estimate.fair_value < estimate.confidence_high
    assert estimate.client_price() == pytest.approx(estimate.fair_value * 1.04)
    with pytest.raises(ValueError, match="non-negative"):
        estimate.client_price(-0.1)


def test_asian_simulation_can_return_all_paths():
    model = HestonParameters(2.0, 0.04, 0.3, -0.7, 0.04)
    payoffs, spot_paths = simulate_heston_asian(model, days=3, paths=500, seed=1, return_paths=True)
    assert spot_paths.shape == (4, 500)
    assert np.all(spot_paths[0] == 232.90)
    assert np.array_equal(payoffs, simulate_heston_asian(model, days=3, paths=500, seed=1))


def test_asian_validation():
    model = HestonParameters(2.0, 0.04, 0.3, -0.7, 0.04)
    with pytest.raises(ValueError, match="days must be positive"):
        simulate_heston_asian(model, days=0)
    with pytest.raises(ValueError, match="95%"):
        estimate_asian_call(np.array([1.0, 2.0]), confidence=0.9)
    with pytest.raises(ValueError, match="two samples"):
        estimate_asian_call(np.array([1.0]))


def test_cir_simulation_and_validation():
    model = CirParameters(1.0, 0.03, 0.1)
    rates = simulate_cir_terminal(model, 0.01, paths=500, days=5, seed=2)
    assert rates.shape == (500,)
    assert np.all(rates >= 0)
    with pytest.raises(ValueError, match="invalid CIR"):
        simulate_cir_terminal(model, -0.01)
