from types import SimpleNamespace

import numpy as np

import stochastic_modeling.rates as rates_module
from stochastic_modeling.models import CirParameters
from stochastic_modeling.rates import EuriborCurve, calibrate_cir, cir_zero_rate


def test_supplied_curve_and_weekly_interpolation():
    curve = EuriborCurve.supplied()
    days, rates = curve.weekly()
    assert len(days) == len(rates) == 52
    assert np.all(rates >= 0)
    assert rates[0] == curve.annual_rates[0]


def test_cir_zero_rates_and_calibration(monkeypatch):
    model = CirParameters(1.0, 0.03, 0.1)
    maturities = np.array([0.1, 0.5, 1.0])
    rates = cir_zero_rate(maturities, model, 0.01)
    assert np.all(np.isfinite(rates))
    assert np.all(rates > 0)

    values = np.array([1.0, 0.03, 0.1])

    def fake_de(objective, bounds, **kwargs):
        del bounds, kwargs
        objective(values)
        return SimpleNamespace(x=values)

    monkeypatch.setattr(rates_module, "differential_evolution", fake_de)
    calibrated, mse = calibrate_cir(maturities, rates, 0.01)
    assert calibrated == model
    assert mse < 1e-20
