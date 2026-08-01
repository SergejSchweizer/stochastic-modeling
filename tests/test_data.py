import pandas as pd
import pytest

from stochastic_modeling.data import load_option_data, parity_analysis


def _write_csv(tmp_path, contents):
    path = tmp_path / "options.csv"
    path.write_text(contents, encoding="utf-8")
    return path


def test_load_and_parity_analysis(tmp_path):
    path = _write_csv(
        tmp_path,
        "Days to maturity,Strike,Price,Type\n15,230,5,C\n15,230,2,p\n",
    )
    frame = load_option_data(path)
    assert list(frame.columns) == ["days", "strike", "price", "type", "T"]
    assert frame["type"].tolist() == ["C", "P"]
    result = parity_analysis(frame)
    assert result.shape == (1, 4)
    assert result.loc[0, "C - P"] == 3


def test_load_rejects_missing_columns_and_bad_types(tmp_path):
    missing = _write_csv(tmp_path, "Strike,Price,Type\n230,5,C\n")
    with pytest.raises(ValueError, match="missing required"):
        load_option_data(missing)
    bad = _write_csv(tmp_path, "Days to maturity,Strike,Price,Type\n15,230,5,X\n")
    with pytest.raises(ValueError, match="C or P"):
        load_option_data(bad)


def test_parity_rejects_multiple_maturities_and_unpaired_strikes():
    multiple = pd.DataFrame(
        {"strike": [230, 230], "price": [5, 2], "type": ["C", "P"], "T": [0.06, 0.24]}
    )
    with pytest.raises(ValueError, match="one maturity"):
        parity_analysis(multiple)
    unpaired = multiple.assign(T=0.06, strike=[230, 235])
    with pytest.raises(ValueError, match="strikes must match"):
        parity_analysis(unpaired)
