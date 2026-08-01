"""Input normalization and no-arbitrage diagnostics."""

from pathlib import Path

import numpy as np
import pandas as pd

from stochastic_modeling.config import DEFAULT_MARKET, MarketConfig


def load_option_data(path: str | Path, market: MarketConfig = DEFAULT_MARKET) -> pd.DataFrame:
    """Load the supplied option quotes into a canonical schema."""
    frame = pd.read_csv(path)
    frame.columns = [column.strip() for column in frame.columns]
    frame = frame.rename(
        columns={
            "Days to maturity": "days",
            "Strike": "strike",
            "Price": "price",
            "Type": "type",
        }
    )
    required = {"days", "strike", "price", "type"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame = frame.loc[:, ["days", "strike", "price", "type"]].copy()
    frame["type"] = frame["type"].astype(str).str.strip().str.upper()
    if not frame["type"].isin(["C", "P"]).all():
        raise ValueError("type values must be C or P")
    frame["T"] = frame["days"].map(market.year_fraction)
    return frame


def parity_analysis(frame: pd.DataFrame, market: MarketConfig = DEFAULT_MARKET) -> pd.DataFrame:
    """Compare paired call/put quotes with European put-call parity."""
    if frame["T"].nunique() != 1:
        raise ValueError("parity analysis requires exactly one maturity")
    calls = frame.query("type == 'C'").set_index("strike")["price"].sort_index()
    puts = frame.query("type == 'P'").set_index("strike")["price"].sort_index()
    if not calls.index.equals(puts.index):
        raise ValueError("call and put strikes must match")
    maturity = float(frame["T"].iloc[0])
    strikes = calls.index.to_numpy(float)
    theoretical = market.spot - strikes * np.exp(-market.rate * maturity)
    observed = calls.to_numpy() - puts.to_numpy()
    return pd.DataFrame(
        {
            "strike": strikes,
            "C - P": observed,
            "parity value": theoretical,
            "gap": observed - theoretical,
        }
    )
