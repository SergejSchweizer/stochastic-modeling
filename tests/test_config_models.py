import pytest

from stochastic_modeling.config import MarketConfig
from stochastic_modeling.models import BatesParameters, CirParameters, HestonParameters


def test_market_year_fraction_and_validation():
    market = MarketConfig(trading_days=250)
    assert market.year_fraction(15) == pytest.approx(0.06)
    with pytest.raises(ValueError, match="non-negative"):
        market.year_fraction(-1)


def test_parameter_diagnostics():
    heston = HestonParameters(2.0, 0.04, 0.3, -0.7, 0.04)
    bates = BatesParameters(2.0, 0.04, 0.3, -0.7, 0.04, 0.2, -0.1, 0.15)
    cir = CirParameters(1.0, 0.03, 0.1)
    assert heston.feller_diagnostic == pytest.approx(0.07)
    assert bates.feller_diagnostic == pytest.approx(0.07)
    assert cir.feller_diagnostic == pytest.approx(0.05)
