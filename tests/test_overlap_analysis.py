"""
Overlap analysis: Premium/DTE vs Top Vega.

Checks:
1. How often the two scoring methods select the same option
2. Characteristics of selected options (DTE, moneyness, type)
3. PnL correlation on overlapping trade periods
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_config, load_options, load_funding_rates
from src.backtest_engine_v2 import (
    run_backtest_v2, compute_metrics_v2, load_precomputed_scores,
    _build_option_lookup, _build_quarterly_lookup,
)

SCORES_PATH = "Data/scores_v2_dte3.csv"
FUTURES_PATH = "Data/deribit_futures_daily.csv"

EXIT_DTE = 5
PROFIT_TARGET = 0.50


def main():
    config = load_config()

    print("Loading data...", flush=True)
    df_scores_raw = load_precomputed_scores(SCORES_PATH)
    df_options = load_options(config)
    df_funding = load_funding_rates(config)

    df_futures = pd.read_csv(FUTURES_PATH)
    df_futures["date"] = pd.to_datetime(df_futures["date"])
    df_futures["expiry"] = pd.to_datetime(df_futures["expiry"])

    print("Building lookups...", flush=True)
    option_lookup = _build_option_lookup(df_options)
    qtr_price_lookup, qtr_avail_lookup = _build_quarterly_lookup(df_futures)

    # ================================================================
    # 1. SIGNAL-LEVEL OVERLAP — Which option does each scoring pick?
    # ================================================================
    print("\n" + "=" * 65)
    print("SIGNAL-LEVEL OVERLAP ANALYSIS")
    print("=" * 65)

    # Premium/DTE scores
    df_pdt = df_scores_raw.copy()
    df_pdt["score_pdt"] = df_pdt["bid_price"] * df_pdt["spot"] / df_pdt["dte"].clip(lower=1)

    # Top Vega scores
    df_vega = df_scores_raw.copy()
    df_vega["score_vega"] = df_vega["vega"].abs()

    # Best per day for each
    best_pdt = (df_pdt.sort_values("score_pdt", ascending=False)
                .drop_duplicates("date")
                .set_index("date")[["strike", "expiry", "option_type", "dte", "bid_price", "spot", "mark_iv", "score_pdt"]])
    best_pdt.columns = [f"{c}_pdt" for c in best_pdt.columns]

    best_vega = (df_vega.sort_values("score_vega", ascending=False)
                 .drop_duplicates("date")
                 .set_index("date")[["strike", "expiry", "option_type", "dte", "bid_price", "spot", "mark_iv", "score_vega"]])
    best_vega.columns = [f"{c}_vega" for c in best_vega.columns]

    # Merge on date
    merged = best_pdt.join(best_vega, how="inner")

    # Check exact match (same strike + expiry + type)
    merged["same_option"] = (
        (merged["strike_pdt"] == merged["strike_vega"]) &
        (merged["expiry_pdt"] == merged["expiry_vega"]) &
        (merged["option_type_pdt"] == merged["option_type_vega"])
    )
    merged["same_expiry"] = merged["expiry_pdt"] == merged["expiry_vega"]

    n_days = len(merged)
    n_same_option = merged["same_option"].sum()
    n_same_expiry = merged["same_expiry"].sum()

    print(f"\n  Total trading days with both signals: {n_days}")
    print(f"  Same EXACT option:  {n_same_option} ({n_same_option/n_days*100:.1f}%)")
    print(f"  Same expiry:        {n_same_expiry} ({n_same_expiry/n_days*100:.1f}%)")
    print(f"  Different option:   {n_days - n_same_option} ({(n_days - n_same_option)/n_days*100:.1f}%)")

    # ================================================================
    # 2. CHARACTERISTICS — What does each scoring prefer?
    # ================================================================
    print("\n" + "=" * 65)
    print("OPTION CHARACTERISTICS — What each scoring selects")
    print("=" * 65)

    # Moneyness
    merged["moneyness_pdt"] = merged["strike_pdt"] / merged["spot_pdt"]
    merged["moneyness_vega"] = merged["strike_vega"] / merged["spot_vega"]

    print(f"\n  {'Metric':<25s} {'Premium/DTE':>12s} {'Top Vega':>12s}")
    print("  " + "-" * 50)
    print(f"  {'Mean DTE':<25s} {merged['dte_pdt'].mean():>12.1f} {merged['dte_vega'].mean():>12.1f}")
    print(f"  {'Median DTE':<25s} {merged['dte_pdt'].median():>12.1f} {merged['dte_vega'].median():>12.1f}")
    print(f"  {'Std DTE':<25s} {merged['dte_pdt'].std():>12.1f} {merged['dte_vega'].std():>12.1f}")
    print(f"  {'Mean Moneyness':<25s} {merged['moneyness_pdt'].mean():>12.3f} {merged['moneyness_vega'].mean():>12.3f}")
    print(f"  {'Mean IV':<25s} {merged['mark_iv_pdt'].mean():>12.1f} {merged['mark_iv_vega'].mean():>12.1f}")
    print(f"  {'Mean Bid (BTC)':<25s} {merged['bid_price_pdt'].mean():>12.4f} {merged['bid_price_vega'].mean():>12.4f}")

    # Option type breakdown
    pdt_calls = (merged["option_type_pdt"] == "call").sum()
    vega_calls = (merged["option_type_vega"] == "call").sum()
    print(f"  {'% Calls':<25s} {pdt_calls/n_days*100:>12.1f} {vega_calls/n_days*100:>12.1f}")
    print(f"  {'% Puts':<25s} {(n_days-pdt_calls)/n_days*100:>12.1f} {(n_days-vega_calls)/n_days*100:>12.1f}")

    # DTE distribution
    print(f"\n  DTE Distribution:")
    for label, col in [("Premium/DTE", "dte_pdt"), ("Top Vega", "dte_vega")]:
        dte_vals = merged[col]
        pct_short = (dte_vals < 14).sum() / n_days * 100
        pct_mid = ((dte_vals >= 14) & (dte_vals < 45)).sum() / n_days * 100
        pct_long = (dte_vals >= 45).sum() / n_days * 100
        print(f"    {label}: <14d={pct_short:.0f}%  14-45d={pct_mid:.0f}%  >45d={pct_long:.0f}%")

    # ================================================================
    # 3. TRADE-LEVEL OVERLAP — PnL correlation
    # ================================================================
    print("\n" + "=" * 65)
    print("TRADE-LEVEL ANALYSIS")
    print("=" * 65)

    bt = config["backtest"]
    split = config["split"]
    common = dict(
        df_options=df_options,
        df_funding=df_funding,
        df_futures=df_futures,
        exit_dte=EXIT_DTE,
        profit_target_pct=PROFIT_TARGET,
        initial_equity=bt["initial_equity"],
        tail_strike_pct=bt["tail_strike_pct"],
        gap_anomaly_z=bt["gap_anomaly_z_threshold"],
        verbose=False,
        _option_lookup=option_lookup,
        _quarterly_price_lookup=qtr_price_lookup,
        _quarterly_avail_lookup=qtr_avail_lookup,
    )

    # Run backtests for both periods
    for period_label, start, end in [
        ("IS", split["is_start"], split["is_end"]),
        ("OOS", split["oos_start"], split["oos_end"]),
    ]:
        # Premium/DTE
        df_pdt_scores = df_scores_raw.copy()
        df_pdt_scores["score"] = df_pdt_scores["bid_price"] * df_pdt_scores["spot"] / df_pdt_scores["dte"].clip(lower=1)
        trades_pdt = run_backtest_v2(df_scores=df_pdt_scores, start_date=start, end_date=end, **common)

        # Top Vega
        df_vega_scores = df_scores_raw.copy()
        df_vega_scores["score"] = df_vega_scores["vega"].abs()
        trades_vega = run_backtest_v2(df_scores=df_vega_scores, start_date=start, end_date=end, **common)

        if trades_pdt.empty or trades_vega.empty:
            print(f"\n  {period_label}: insufficient trades for comparison")
            continue

        m_pdt = compute_metrics_v2(trades_pdt)
        m_vega = compute_metrics_v2(trades_vega)

        print(f"\n  {period_label}:")
        print(f"    Premium/DTE: {m_pdt['n_trades']} trades, Sharpe={m_pdt['sharpe']:.2f}, mean DTE={trades_pdt['entry_dte'].mean():.0f}d")
        print(f"    Top Vega:    {m_vega['n_trades']} trades, Sharpe={m_vega['sharpe']:.2f}, mean DTE={trades_vega['entry_dte'].mean():.0f}d")

        # Check overlap in entry dates
        pdt_entries = set(pd.to_datetime(trades_pdt["entry_date"]).dt.normalize())
        vega_entries = set(pd.to_datetime(trades_vega["entry_date"]).dt.normalize())
        overlap_entries = pdt_entries & vega_entries
        print(f"    Overlapping entry dates: {len(overlap_entries)} / PDT={len(pdt_entries)}, Vega={len(vega_entries)}")

        # Check if same option was traded on overlapping dates
        if overlap_entries:
            n_same = 0
            for d in overlap_entries:
                pdt_row = trades_pdt[pd.to_datetime(trades_pdt["entry_date"]).dt.normalize() == d].iloc[0]
                vega_row = trades_vega[pd.to_datetime(trades_vega["entry_date"]).dt.normalize() == d].iloc[0]
                if (pdt_row["strike"] == vega_row["strike"] and
                    pdt_row["expiry"] == vega_row["expiry"]):
                    n_same += 1
            print(f"    Same option on overlap dates: {n_same}/{len(overlap_entries)}")

        # Build daily equity curves for correlation
        trades_pdt["entry_date"] = pd.to_datetime(trades_pdt["entry_date"])
        trades_pdt["exit_date"] = pd.to_datetime(trades_pdt["exit_date"])
        trades_vega["entry_date"] = pd.to_datetime(trades_vega["entry_date"])
        trades_vega["exit_date"] = pd.to_datetime(trades_vega["exit_date"])

        # Simple: correlate trade PnLs on overlapping entry dates
        if len(overlap_entries) >= 3:
            pdt_pnl = {}
            for _, row in trades_pdt.iterrows():
                d = row["entry_date"].normalize()
                if d in overlap_entries:
                    pdt_pnl[d] = row["total_pnl"]
            vega_pnl = {}
            for _, row in trades_vega.iterrows():
                d = row["entry_date"].normalize()
                if d in overlap_entries:
                    vega_pnl[d] = row["total_pnl"]

            common_dates = sorted(set(pdt_pnl.keys()) & set(vega_pnl.keys()))
            if len(common_dates) >= 3:
                pdt_arr = [pdt_pnl[d] for d in common_dates]
                vega_arr = [vega_pnl[d] for d in common_dates]
                corr, p_corr = __import__("scipy").stats.pearsonr(pdt_arr, vega_arr)
                print(f"    PnL correlation on overlap: {corr:.3f} (p={p_corr:.3f})")

    # ================================================================
    # 4. Summary
    # ================================================================
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  - Same option selected: {n_same_option/n_days*100:.0f}% of days")
    print(f"  - Premium/DTE prefers: short DTE ({merged['dte_pdt'].mean():.0f}d avg), higher premium")
    print(f"  - Top Vega prefers: long DTE ({merged['dte_vega'].mean():.0f}d avg), ATM options")
    if n_same_option / n_days < 0.3:
        print(f"  → Le due strategie sono COMPLEMENTARI (basso overlap)")
    elif n_same_option / n_days < 0.6:
        print(f"  → Overlap moderato — parzialmente complementari")
    else:
        print(f"  → Alto overlap — essenzialmente la stessa strategia")


if __name__ == "__main__":
    main()
