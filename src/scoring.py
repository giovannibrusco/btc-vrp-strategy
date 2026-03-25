"""
Option Scoring Engine — Select the best option to sell.

Implements the VRP scoring function:
    Score(K, T) = gap_probability(K, T) × vega(K, T)

Two modes:
- score_all_expiries(): score a single day (used internally)
- precompute_scores(): score ALL days, save to CSV (Phase 1)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.rnd_engine import extract_rnd_for_expiry
from src.historical_distribution import build_p_distribution, compute_gap_probability


def score_option_chain(df_day: pd.DataFrame, expiry, config: dict,
                       daily_prices: pd.DataFrame, current_date,
                       lookback_days: int = 365) -> list[dict]:
    """
    Score all options for a given expiry on a given day.

    Returns a list of dicts (one per scored option), not just the best.
    This way Phase 1 saves everything and Phase 2 can pick.
    """
    rnd_result = extract_rnd_for_expiry(df_day, expiry, config)
    if rnd_result is None:
        return []

    K_rnd = rnd_result["K_valid"]
    rnd_density = rnd_result["density"]
    spot = rnd_result["spot"]
    dte = rnd_result["dte"]
    T = rnd_result["T"]

    if dte < 2:
        return []

    p_result = build_p_distribution(
        daily_prices=daily_prices,
        current_date=current_date,
        t_days=int(round(dte)),
        spot=spot,
        strikes=K_rnd,
        lookback_days=lookback_days,
    )
    if p_result is None:
        return []

    p_density = p_result["density_strike"]
    gap = compute_gap_probability(rnd_density, p_density)

    # Get options for this expiry
    expiry_col = "expiry_date" if "expiry_date" in df_day.columns else "expiry"
    df_exp = df_day[df_day[expiry_col] == expiry].copy()

    if df_exp.empty or "vega" not in df_exp.columns:
        return []

    # Only tradeable options with positive vega
    df_exp = df_exp[
        (df_exp["vega"].notna()) &
        (df_exp["vega"] > 0) &
        (df_exp["bid_price"] > 0)
    ].copy()

    if df_exp.empty:
        return []

    # Interpolate gap at each option strike
    df_exp["gap_prob"] = np.interp(df_exp["strike"].values, K_rnd, gap)

    # Only where Q > P
    df_exp = df_exp[df_exp["gap_prob"] > 0].copy()
    if df_exp.empty:
        return []

    df_exp["score"] = df_exp["gap_prob"] * df_exp["vega"]

    results = []
    for _, row in df_exp.iterrows():
        results.append({
            "expiry": expiry,
            "strike": row["strike"],
            "option_type": row["option_type"],
            "score": row["score"],
            "gap_prob": row["gap_prob"],
            "vega": row["vega"],
            "delta": row.get("delta", np.nan),
            "gamma": row.get("gamma", np.nan),
            "theta": row.get("theta", np.nan),
            "mark_iv": row.get("mark_iv", np.nan),
            "bid_price": row["bid_price"],
            "ask_price": row.get("ask_price", np.nan),
            "open_interest": row.get("open_interest", 0),
            "spot": spot,
            "dte": dte,
            "T": T,
            "rnd_fair_value": rnd_result["fair_value"],
            "rnd_std": rnd_result["std"],
            "rnd_skewness": rnd_result["skewness"],
            "kde_bandwidth": p_result["bandwidth"],
            "kde_n_returns": p_result["n_returns"],
        })

    return results


def score_day(df_day: pd.DataFrame, config: dict,
              daily_prices: pd.DataFrame, current_date,
              lookback_days: int = 365) -> list[dict]:
    """
    Score all valid options across all expiries for one day.

    Returns list of scored options (all of them, not just the best).
    """
    exp_cfg = config["expiry_selection"]
    min_dte = exp_cfg["min_dte"]
    max_dte = exp_cfg["max_dte"]

    expiry_col = "expiry_date" if "expiry_date" in df_day.columns else "expiry"
    expiries = df_day[expiry_col].unique()

    valid_expiries = []
    for exp in expiries:
        df_exp = df_day[df_day[expiry_col] == exp]
        if df_exp.empty:
            continue
        dte = df_exp["dte"].iloc[0]
        if min_dte <= dte <= max_dte:
            valid_expiries.append(exp)

    if not valid_expiries:
        return []

    all_scored = []
    for expiry in valid_expiries:
        scored = score_option_chain(
            df_day=df_day,
            expiry=expiry,
            config=config,
            daily_prices=daily_prices,
            current_date=current_date,
            lookback_days=lookback_days,
        )
        all_scored.extend(scored)

    return all_scored


def precompute_scores(df_options: pd.DataFrame, daily_prices: pd.DataFrame,
                      config: dict, lookback_days: int = 365,
                      output_path: str | None = None,
                      verbose: bool = True) -> pd.DataFrame:
    """
    PHASE 1: Precompute scores for all options on all trading days.

    This is the heavy computation — run once, save to CSV.
    For each day, scores all valid options and saves:
    date, expiry, strike, option_type, score, gap_prob, vega, delta, ...

    The backtest (Phase 2) reads this CSV and runs instantly.
    """
    # Prepare dates
    df_options["date"] = pd.to_datetime(df_options["date"])
    trading_dates = sorted(df_options["date"].unique())

    # Index options by date
    options_by_date = {date: group for date, group in df_options.groupby("date")}

    all_rows = []
    n_skipped = 0

    iterator = tqdm(trading_dates, desc="Phase 1: Scoring") if verbose else trading_dates

    for date in iterator:
        date_ts = pd.Timestamp(date)
        df_day = options_by_date.get(date_ts)
        if df_day is None:
            n_skipped += 1
            continue

        scored = score_day(
            df_day=df_day,
            config=config,
            daily_prices=daily_prices,
            current_date=date_ts,
            lookback_days=lookback_days,
        )

        for row in scored:
            row["date"] = date_ts
            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    df_scores = pd.DataFrame(all_rows)

    # Reorder columns
    first_cols = ["date", "expiry", "strike", "option_type", "score",
                  "gap_prob", "vega", "delta", "mark_iv", "bid_price",
                  "ask_price", "spot", "dte"]
    other_cols = [c for c in df_scores.columns if c not in first_cols]
    df_scores = df_scores[first_cols + other_cols]

    df_scores = df_scores.sort_values(["date", "score"], ascending=[True, False])

    if output_path:
        df_scores.to_csv(output_path, index=False)
        if verbose:
            print(f"\nSaved {len(df_scores)} scored options to {output_path}")
            print(f"Trading days: {len(trading_dates)} (skipped {n_skipped})")
            print(f"Days with scores: {df_scores['date'].nunique()}")

    return df_scores


def check_anomalous_gap(current_gap: float, historical_gaps: list[float],
                        threshold_z: float = 2.0) -> bool:
    """
    Check if the current gap is anomalously high vs historical.
    """
    if len(historical_gaps) < 10:
        return False

    gaps = np.array(historical_gaps)
    mean = gaps.mean()
    std = gaps.std()

    if std == 0:
        return False

    z_score = (current_gap - mean) / std
    return z_score > threshold_z
