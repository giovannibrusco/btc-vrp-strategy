"""
Data loading and preprocessing for BTC Options RND Trading.

Handles:
- Loading options CSV with standardized column names
- Loading BTC perpetual 5-minute data
- IS/OOS temporal split
- Basic cleaning and deduplication
"""

import pandas as pd
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | None = None) -> dict:
    """Load YAML configuration."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "params.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_options(config: dict | None = None) -> pd.DataFrame:
    """
    Load and standardize the options dataset.

    Returns DataFrame with columns:
        date, option_type, strike, expiry, mark_iv, underlying_price,
        dte, open_interest, bid_price, ask_price, delta, gamma, vega, theta,
        instrument_name, volume_usd, ...
    """
    if config is None:
        config = load_config()

    csv_path = PROJECT_ROOT / config["data"]["options_csv"]
    df = pd.read_csv(csv_path, low_memory=False)

    # Standardize column names to lowercase with underscores
    df.columns = [c.strip().lower() for c in df.columns]

    # Parse dates
    df["date"] = pd.to_datetime(df["date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"]) if "expiry_date" in df.columns else None

    # Clean option_type
    df["option_type"] = df["option_type"].str.strip().str.lower()

    # Compute time to maturity
    df["T"] = df["dte"] / 365.0

    # Compute moneyness
    df["moneyness"] = df["strike"] / df["underlying_price"]

    # Sort and deduplicate
    sort_cols = ["date", "option_type", "strike"]
    if "expiry_date" in df.columns:
        sort_cols = ["date", "expiry_date", "strike", "option_type"]
    df = df.drop_duplicates().sort_values(sort_cols).reset_index(drop=True)

    return df


def load_btc_5min(config: dict | None = None) -> pd.DataFrame:
    """Load BTC perpetual 5-minute OHLCV data."""
    if config is None:
        config = load_config()

    csv_path = PROJECT_ROOT / config["data"]["btc_5min_csv"]
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    return df


def split_is_oos(df: pd.DataFrame, config: dict | None = None,
                 date_col: str = "date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into In-Sample and Out-of-Sample based on config dates.

    Returns (df_is, df_oos).
    """
    if config is None:
        config = load_config()

    is_start = pd.Timestamp(config["split"]["is_start"])
    is_end = pd.Timestamp(config["split"]["is_end"])
    oos_start = pd.Timestamp(config["split"]["oos_start"])
    oos_end = pd.Timestamp(config["split"]["oos_end"])

    dates = pd.to_datetime(df[date_col])
    df_is = df[(dates >= is_start) & (dates <= is_end)].copy()
    df_oos = df[(dates >= oos_start) & (dates <= oos_end)].copy()

    return df_is, df_oos


def load_funding_rates(config: dict | None = None) -> pd.DataFrame:
    """Load daily funding rate data for BTC perpetual."""
    if config is None:
        config = load_config()

    csv_path = PROJECT_ROOT / config["data"]["funding_daily_csv"]
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_btc_spot_daily(config: dict | None = None) -> pd.DataFrame:
    """Load Binance BTCUSDT daily spot prices."""
    if config is None:
        config = load_config()
    csv_path = PROJECT_ROOT / config["data"]["btc_spot_daily_csv"]
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_btc_daily(config: dict | None = None) -> pd.DataFrame:
    """Build daily OHLC from 5-minute data."""
    df_5m = load_btc_5min(config)
    df_daily = df_5m.set_index("timestamp").resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna().reset_index()
    df_daily = df_daily.rename(columns={"timestamp": "date"})
    return df_daily
