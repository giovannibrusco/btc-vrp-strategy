"""
Sensitivity Analysis V2 — Grid search on profit_target × exit_dte.

Tests all combinations on IS, then validates best on OOS.
Hedging: Deribit quarterly futures (no funding, fixed basis cost).
"""

import sys
import time
import itertools
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_config, load_options, load_funding_rates
from src.backtest_engine_v2 import (
    run_backtest_v2, compute_metrics_v2, load_precomputed_scores,
    _build_option_lookup, _build_quarterly_lookup,
)

SCORES_PATH = "Data/scores_v2.csv"
FUTURES_PATH = "Data/deribit_futures_daily.csv"

# Parameter grid
PROFIT_TARGETS = [0.25, 0.35, 0.50, 0.65, 0.75, 0.85]
EXIT_DTES = [2, 3, 5, 7, 10]


def run_grid():
    config = load_config()
    split = config["split"]
    bt = config["backtest"]

    print("Loading data...", flush=True)
    df_scores = load_precomputed_scores(SCORES_PATH)
    df_options = load_options(config)
    df_funding = load_funding_rates(config)

    df_futures = pd.read_csv(FUTURES_PATH)
    df_futures["date"] = pd.to_datetime(df_futures["date"])
    df_futures["expiry"] = pd.to_datetime(df_futures["expiry"])
    print(f"  Quarterly futures: {len(df_futures)} rows", flush=True)

    # Build lookups ONCE
    print("Building option lookup (once)...", flush=True)
    option_lookup = _build_option_lookup(df_options)
    print(f"  {len(option_lookup)} entries", flush=True)

    print("Building quarterly futures lookup...", flush=True)
    qtr_price_lookup, qtr_avail_lookup = _build_quarterly_lookup(df_futures)
    print(f"  {len(qtr_price_lookup)} price entries", flush=True)

    results = []
    combos = list(itertools.product(PROFIT_TARGETS, EXIT_DTES))
    print(f"\nTesting {len(combos)} combinations on IS ({split['is_start']} → {split['is_end']})\n", flush=True)

    t0 = time.time()

    for i, (pt, edte) in enumerate(combos):
        df_trades = run_backtest_v2(
            df_scores=df_scores,
            df_options=df_options,
            df_funding=df_funding,
            df_futures=df_futures,
            exit_dte=edte,
            profit_target_pct=pt,
            initial_equity=bt["initial_equity"],
            tail_strike_pct=bt["tail_strike_pct"],
            gap_anomaly_z=bt["gap_anomaly_z_threshold"],
            start_date=split["is_start"],
            end_date=split["is_end"],
            verbose=False,
            _option_lookup=option_lookup,
            _quarterly_price_lookup=qtr_price_lookup,
            _quarterly_avail_lookup=qtr_avail_lookup,
        )

        if df_trades.empty:
            continue

        m = compute_metrics_v2(df_trades)
        m["profit_target"] = pt
        m["exit_dte"] = edte
        results.append(m)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(combos)} done...", flush=True)

    elapsed = time.time() - t0
    print(f"\nGrid search completed in {elapsed:.0f}s", flush=True)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("sharpe", ascending=False)
    df_results.to_csv("Data/sensitivity_v2_IS.csv", index=False)

    # Print best results
    print(f"\n{'='*70}")
    print(f"TOP 10 — IS (Quarterly Futures Hedge)")
    print(f"{'='*70}")
    print(f"{'PT':>5s} {'DTE':>4s} {'N':>4s} {'WR%':>5s} {'Sharpe':>7s} {'Ret%':>7s} {'DD%':>7s} {'$/trade':>8s} {'OptPnL':>8s} {'HedgePnL':>9s} {'TxnCost':>8s}")
    print("-" * 78)
    for _, r in df_results.head(10).iterrows():
        avg_opt = r['option_pnl_total'] / r['n_trades']
        avg_hedge = r['hedge_pnl_total'] / r['n_trades']
        avg_txn = r['hedge_txn_cost_total'] / r['n_trades']
        print(f"{r['profit_target']:>5.0%} {r['exit_dte']:>4.0f} {r['n_trades']:>4.0f} {r['win_rate']:>5.1f} {r['sharpe']:>7.2f} {r['total_return_pct']:>7.1f} {r['max_drawdown_pct']:>7.1f} {r['mean_pnl']:>8,.0f} {avg_opt:>8,.0f} {avg_hedge:>9,.0f} {avg_txn:>8,.0f}")
    sys.stdout.flush()

    # OOS validation of top 5
    print(f"\n{'='*70}")
    print(f"OOS VALIDATION ({split['oos_start']} → {split['oos_end']})")
    print(f"{'='*70}")
    print(f"{'PT':>5s} {'DTE':>4s} {'IS_N':>5s} {'IS_Sh':>6s} {'IS_Ret':>7s} {'OOS_N':>5s} {'OOS_Sh':>7s} {'OOS_Ret':>8s} {'OOS_DD':>7s} {'OOS_$/tr':>9s}")
    print("-" * 70)

    for _, r_is in df_results.head(5).iterrows():
        df_oos = run_backtest_v2(
            df_scores=df_scores,
            df_options=df_options,
            df_funding=df_funding,
            df_futures=df_futures,
            exit_dte=int(r_is["exit_dte"]),
            profit_target_pct=r_is["profit_target"],
            initial_equity=bt["initial_equity"],
            tail_strike_pct=bt["tail_strike_pct"],
            gap_anomaly_z=bt["gap_anomaly_z_threshold"],
            start_date=split["oos_start"],
            end_date=split["oos_end"],
            verbose=False,
            _option_lookup=option_lookup,
            _quarterly_price_lookup=qtr_price_lookup,
            _quarterly_avail_lookup=qtr_avail_lookup,
        )
        if df_oos.empty:
            print(f"{r_is['profit_target']:>5.0%} {r_is['exit_dte']:>4.0f} {r_is['n_trades']:>5.0f} {r_is['sharpe']:>6.2f} {r_is['total_return_pct']:>7.1f}   NO OOS TRADES")
            continue

        m_oos = compute_metrics_v2(df_oos)
        print(f"{r_is['profit_target']:>5.0%} {r_is['exit_dte']:>4.0f} {r_is['n_trades']:>5.0f} {r_is['sharpe']:>6.2f} {r_is['total_return_pct']:>7.1f} {m_oos['n_trades']:>5.0f} {m_oos['sharpe']:>7.2f} {m_oos['total_return_pct']:>8.1f} {m_oos['max_drawdown_pct']:>7.1f} {m_oos['mean_pnl']:>9,.0f}")
    sys.stdout.flush()

    # Full OOS table for all combos
    print(f"\n{'='*70}")
    print(f"FULL OOS VALIDATION — All {len(df_results)} combos")
    print(f"{'='*70}")
    print(f"{'PT':>5s} {'DTE':>4s} {'IS_Sh':>6s} {'IS_Ret':>7s} {'OOS_Sh':>7s} {'OOS_Ret':>8s} {'OOS_DD':>7s}")
    print("-" * 50)

    for _, r_is in df_results.iterrows():
        df_oos = run_backtest_v2(
            df_scores=df_scores,
            df_options=df_options,
            df_funding=df_funding,
            df_futures=df_futures,
            exit_dte=int(r_is["exit_dte"]),
            profit_target_pct=r_is["profit_target"],
            initial_equity=bt["initial_equity"],
            tail_strike_pct=bt["tail_strike_pct"],
            gap_anomaly_z=bt["gap_anomaly_z_threshold"],
            start_date=split["oos_start"],
            end_date=split["oos_end"],
            verbose=False,
            _option_lookup=option_lookup,
            _quarterly_price_lookup=qtr_price_lookup,
            _quarterly_avail_lookup=qtr_avail_lookup,
        )
        if df_oos.empty:
            print(f"{r_is['profit_target']:>5.0%} {r_is['exit_dte']:>4.0f} {r_is['sharpe']:>6.2f} {r_is['total_return_pct']:>7.1f}     N/A")
            continue

        m_oos = compute_metrics_v2(df_oos)
        print(f"{r_is['profit_target']:>5.0%} {r_is['exit_dte']:>4.0f} {r_is['sharpe']:>6.2f} {r_is['total_return_pct']:>7.1f} {m_oos['sharpe']:>7.2f} {m_oos['total_return_pct']:>8.1f} {m_oos['max_drawdown_pct']:>7.1f}")
    sys.stdout.flush()


if __name__ == "__main__":
    run_grid()
