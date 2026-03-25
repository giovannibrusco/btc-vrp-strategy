"""
Walk-forward Signal Generation using RND engine.

For each trading day:
1. Extract RND for all valid expiries
2. Select best expiry (composite score or max gap)
3. Compute gap = (fair_value - spot) / spot
4. Generate LONG/SHORT/NEUTRAL signal
5. Save enriched features (RND moments, coverage, OI, etc.)
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.rnd_engine import extract_rnd_for_expiry, select_best_expiry, select_expiry_max_gap


def generate_signals(df_options, config, verbose=False):
    """
    Walk-forward signal generation across all trading days.

    Parameters:
        df_options: full options DataFrame (from data_loader.load_options)
        config: configuration dict
        verbose: print progress details

    Returns:
        DataFrame with columns:
            Date, Spot, Fair_Value, Gap, Expiry, DTE,
            RND_Std, RND_Skew, RND_Kurt, Smile_Coverage,
            OI_Total, Score_Composito, T
    """
    method = config["expiry_selection"]["method"]
    unique_days = sorted(df_options["date"].unique())

    signals = []

    for day in tqdm(unique_days, desc="Generating signals", disable=not verbose):
        df_day = df_options[df_options["date"] == day].copy()

        if df_day.empty:
            continue

        # Get all unique expiries for this day
        expiry_col = "expiry_date" if "expiry_date" in df_day.columns else "expiry"
        expiries = df_day[expiry_col].dropna().unique()

        # Extract RND for each expiry
        rnd_results = []
        for expiry in expiries:
            result = extract_rnd_for_expiry(df_day, expiry, config, use_otm_stitching=True)
            if result is not None:
                rnd_results.append(result)

        if not rnd_results:
            continue

        # Select best expiry
        if method == "composite_score":
            best_expiry = select_best_expiry(df_day, config, rnd_results)
        else:
            best_expiry = select_expiry_max_gap(rnd_results, config)

        if best_expiry is None:
            # Fallback: try max_gap without DTE constraint
            best_expiry = select_expiry_max_gap(rnd_results, config)

        if best_expiry is None:
            continue

        # Get the RND result for the best expiry
        best_result = None
        for r in rnd_results:
            if r["expiry"] == best_expiry:
                best_result = r
                break

        if best_result is None:
            continue

        spot = best_result["spot"]
        fair_value = best_result["fair_value"]
        gap = (fair_value - spot) / spot

        signals.append({
            "Date": day,
            "Spot": spot,
            "Fair_Value": fair_value,
            "Gap": gap,
            "Expiry": best_expiry,
            "DTE": best_result["dte"],
            "RND_Std": best_result["std"],
            "RND_Skew": best_result["skewness"],
            "RND_Kurt": best_result["kurtosis"],
            "Smile_Coverage": best_result["coverage"],
            "OI_Total": best_result["oi_total"],
            "T": best_result["T"],
        })

    df_signals = pd.DataFrame(signals)

    if df_signals.empty:
        return df_signals

    df_signals["Date"] = pd.to_datetime(df_signals["Date"])
    df_signals["Expiry"] = pd.to_datetime(df_signals["Expiry"])

    return df_signals


def label_signals(df_signals, gap_threshold=0.03):
    """
    Add direction labels based on gap threshold.

    Adds column 'Signal': 'LONG', 'SHORT', or 'NEUTRAL'
    """
    df = df_signals.copy()
    df["Signal"] = "NEUTRAL"
    df.loc[df["Gap"] > gap_threshold, "Signal"] = "LONG"
    df.loc[df["Gap"] < -gap_threshold, "Signal"] = "SHORT"
    return df
