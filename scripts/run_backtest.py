"""
Run V2 Backtest — Two-phase pipeline.

Phase 1: Precompute scores (slow, run once)
Phase 2: Simulate trading on IS and OOS separately (fast)

Hedging: Deribit quarterly futures (no funding, fixed basis cost).

Usage:
    python scripts/run_backtest.py              # Run both phases
    python scripts/run_backtest.py --phase2     # Skip Phase 1 (use cached scores)
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_config, load_options, load_btc_daily, load_funding_rates
from src.backtest_engine_v2 import run_backtest_v2, compute_metrics_v2, load_precomputed_scores

import pandas as pd

SCORES_PATH = "Data/scores_v2.csv"
FUTURES_PATH = "Data/deribit_futures_daily.csv"


def run_phase1(config):
    """Phase 1: Precompute scores for all trading days."""
    from src.scoring import precompute_scores

    print("=" * 60)
    print("PHASE 1: Precomputing scores (RND + KDE + scoring)")
    print("=" * 60)

    t0 = time.time()

    print("Loading options data...")
    df_options = load_options(config)
    print(f"  {len(df_options)} options loaded")

    print("Loading daily prices...")
    daily_prices = load_btc_daily(config)
    print(f"  {len(daily_prices)} daily prices loaded")

    lookback = config["backtest"]["lookback_days"]
    print(f"  KDE lookback: {lookback} days\n")

    df_scores = precompute_scores(
        df_options=df_options,
        daily_prices=daily_prices,
        config=config,
        lookback_days=lookback,
        output_path=SCORES_PATH,
        verbose=True,
    )

    elapsed = time.time() - t0
    print(f"\nPhase 1 completed in {elapsed/60:.1f} minutes")
    return df_scores


def run_phase2(config, df_scores=None):
    """Phase 2: Run backtest on IS and OOS separately."""
    t0 = time.time()

    if df_scores is None:
        print(f"Loading precomputed scores from {SCORES_PATH}...")
        df_scores = load_precomputed_scores(SCORES_PATH)
        print(f"  {len(df_scores)} scored options loaded")

    print("Loading options data...")
    df_options = load_options(config)

    print("Loading funding rates...")
    df_funding = load_funding_rates(config)

    print("Loading quarterly futures data...")
    df_futures = pd.read_csv(FUTURES_PATH)
    df_futures["date"] = pd.to_datetime(df_futures["date"])
    df_futures["expiry"] = pd.to_datetime(df_futures["expiry"])
    print(f"  {len(df_futures)} rows loaded")

    bt = config["backtest"]
    split = config["split"]

    common_params = dict(
        df_scores=df_scores,
        df_options=df_options,
        df_funding=df_funding,
        df_futures=df_futures,
        exit_dte=bt["exit_dte"],
        profit_target_pct=bt["profit_target_pct"],
        initial_equity=bt["initial_equity"],
        tail_strike_pct=bt["tail_strike_pct"],
        gap_anomaly_z=bt["gap_anomaly_z_threshold"],
    )

    # ==================== IN-SAMPLE ====================
    print("\n" + "=" * 60)
    print(f"IN-SAMPLE: {split['is_start']} → {split['is_end']}")
    print("=" * 60)

    df_is = run_backtest_v2(
        **common_params,
        start_date=split["is_start"],
        end_date=split["is_end"],
        verbose=True,
    )

    if not df_is.empty:
        df_is.to_csv("Data/trades_v2_IS.csv", index=False)
        m_is = compute_metrics_v2(df_is)

    # ==================== OUT-OF-SAMPLE ====================
    print("\n" + "=" * 60)
    print(f"OUT-OF-SAMPLE: {split['oos_start']} → {split['oos_end']}")
    print("=" * 60)

    df_oos = run_backtest_v2(
        **common_params,
        start_date=split["oos_start"],
        end_date=split["oos_end"],
        verbose=True,
    )

    if not df_oos.empty:
        df_oos.to_csv("Data/trades_v2_OOS.csv", index=False)
        m_oos = compute_metrics_v2(df_oos)

    # ==================== COMPARISON ====================
    if not df_is.empty and not df_oos.empty:
        print("\n" + "=" * 60)
        print("IS vs OOS COMPARISON")
        print("=" * 60)
        print(f"{'Metric':<25s} {'IS':>12s} {'OOS':>12s}")
        print("-" * 50)
        print(f"{'Trades':<25s} {m_is['n_trades']:>12d} {m_oos['n_trades']:>12d}")
        print(f"{'Win Rate %':<25s} {m_is['win_rate']:>12.1f} {m_oos['win_rate']:>12.1f}")
        print(f"{'Total Return %':<25s} {m_is['total_return_pct']:>12.1f} {m_oos['total_return_pct']:>12.1f}")
        print(f"{'Sharpe':<25s} {m_is['sharpe']:>12.2f} {m_oos['sharpe']:>12.2f}")
        print(f"{'Max Drawdown %':<25s} {m_is['max_drawdown_pct']:>12.1f} {m_oos['max_drawdown_pct']:>12.1f}")
        print(f"{'Mean P&L / trade':<25s} {m_is['mean_pnl']:>12,.0f} {m_oos['mean_pnl']:>12,.0f}")
        print(f"{'Trades / year':<25s} {m_is['trades_per_year']:>12.1f} {m_oos['trades_per_year']:>12.1f}")
        print(f"{'Avg Holding Days':<25s} {m_is['avg_holding_days']:>12.1f} {m_oos['avg_holding_days']:>12.1f}")
        print(f"\n{'P&L Breakdown / trade':}")
        print(f"{'  Option':<25s} {m_is['option_pnl_total']/m_is['n_trades']:>12,.0f} {m_oos['option_pnl_total']/m_oos['n_trades']:>12,.0f}")
        print(f"{'  Hedge':<25s} {m_is['hedge_pnl_total']/m_is['n_trades']:>12,.0f} {m_oos['hedge_pnl_total']/m_oos['n_trades']:>12,.0f}")
        print(f"{'  Hedge txn':<25s} {-m_is['hedge_txn_cost_total']/m_is['n_trades']:>12,.0f} {-m_oos['hedge_txn_cost_total']/m_oos['n_trades']:>12,.0f}")

    elapsed = time.time() - t0
    print(f"\nPhase 2 completed in {elapsed:.1f} seconds")
    return df_is, df_oos


if __name__ == "__main__":
    config = load_config()

    skip_phase1 = "--phase2" in sys.argv

    if skip_phase1:
        scores_file = Path(SCORES_PATH)
        if not scores_file.exists():
            print(f"ERROR: {SCORES_PATH} not found. Run without --phase2 first.")
            sys.exit(1)
        run_phase2(config)
    else:
        df_scores = run_phase1(config)
        run_phase2(config, df_scores)
