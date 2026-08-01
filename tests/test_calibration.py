from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import stochastic_modeling.calibration as calibration_module
from stochastic_modeling.calibration import CalibrationService


class DeterministicPricer:
    def calls(self, strikes, maturity, characteristic_function, market):
        del maturity, characteristic_function
        return np.full_like(strikes, market.spot * 0.05, dtype=float)


@pytest.fixture
def frame():
    return pd.DataFrame(
        {"strike": [230.0, 230.0], "type": ["C", "P"], "T": [0.06, 0.06], "price": [11.5, 8.3]}
    )


def _stub_optimizers(monkeypatch):
    heston_x = np.array([2.0, 0.04, 0.3, -0.7, 0.04])

    def fake_de(objective, bounds, **kwargs):
        del bounds, kwargs
        objective(heston_x)
        return SimpleNamespace(x=heston_x)

    def fake_ls(residuals, x, **kwargs):
        del kwargs
        return SimpleNamespace(x=x, fun=residuals(x))

    monkeypatch.setattr(calibration_module, "differential_evolution", fake_de)
    monkeypatch.setattr(calibration_module, "least_squares", fake_ls)


def test_heston_calibration_service(monkeypatch, frame):
    _stub_optimizers(monkeypatch)
    result = CalibrationService(DeterministicPricer()).calibrate(frame)
    assert result.mse >= 0
    assert result.rmse == pytest.approx(np.sqrt(result.mse))
    assert result.parameters.kappa == 2.0


def test_bates_calibration_and_invalid_kind(monkeypatch, frame):
    bates_x = np.array([2.0, 0.04, 0.3, -0.7, 0.04, 0.2, -0.1, 0.15])

    def fake_de(objective, bounds, **kwargs):
        del objective, bounds, kwargs
        return SimpleNamespace(x=bates_x)

    monkeypatch.setattr(calibration_module, "differential_evolution", fake_de)
    monkeypatch.setattr(
        calibration_module,
        "least_squares",
        lambda residuals, x, **kwargs: SimpleNamespace(x=x, fun=residuals(x)),
    )
    service = CalibrationService(DeterministicPricer())
    assert service.calibrate(frame, "bates").parameters.jump_intensity == 0.2
    with pytest.raises(ValueError, match="heston or bates"):
        service.calibrate(frame, "invalid")
