"""
Predictive significance test for Premium/DTE scoring.

1. Demeaned regression: PnL ~ β × (score - expanding_mean(score))
   Tests if HIGHER Premium/DTE score predicts HIGHER PnL (timing value).
2. Random benchmark: 100 iterations of random option selection,
   compared to Premium/DTE performance.
"""

import numpy as np
import pandas as pd
from scipy import stats
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

# Premium/DTE params (from sensitivity analysis)
EXIT_DTE = 5
PROFIT_TARGET = 0.50  # PT doesn't matter much for this scoring


def rescore_premium_dte(df_scores):
    """Replace gap×vega score with Premium/DTE = bid_price × spot / DTE."""
    df = df_scores.copy()
    df["score"] = df["bid_price"] * df["spot"] / df["dte"].clip(lower=1)
    return df


def run_with_scoring(df_scores, df_options, df_funding, df_futures,
                     option_lookup, qtr_price_lookup, qtr_avail_lookup,
                     config, period="is"):
    """Run backtest for a given period with pre-scored data."""
    split = config["split"]
    bt = config["backtest"]

    start = split["is_start"] if period == "is" else split["oos_start"]
    end = split["is_end"] if period == "is" else split["oos_end"]

    df_trades = run_backtest_v2(
        df_scores=df_scores,
        df_options=df_options,
        df_funding=df_funding,
        df_futures=df_futures,
        exit_dte=EXIT_DTE,
        profit_target_pct=PROFIT_TARGET,
        initial_equity=bt["initial_equity"],
        tail_strike_pct=bt["tail_strike_pct"],
        gap_anomaly_z=bt["gap_anomaly_z_threshold"],
        start_date=start,
        end_date=end,
        verbose=False,
        _option_lookup=option_lookup,
        _quarterly_price_lookup=qtr_price_lookup,
        _quarterly_avail_lookup=qtr_avail_lookup,
    )
    return df_trades


def demeaned_regression(df_trades):
    """
    Regress PnL on demeaned score (expanding mean).
    If β > 0 and significant → score discriminates good from bad moments.
    """
    scores = df_trades["score"].values
    pnls = df_trades["total_pnl"].values

    # Expanding mean of score
    expanding_mean = pd.Series(scores).expanding().mean().values
    demeaned = scores - expanding_mean

    # OLS regression: PnL = α + β × demeaned_score + ε
    slope, intercept, r_value, p_value, std_err = stats.linregress(demeaned, pnls)

    return {
        "beta": slope,
        "intercept": intercept,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "std_err": std_err,
        "n": len(pnls),
        "mean_pnl": pnls.mean(),
        "scores_mean": scores.mean(),
        "scores_std": scores.std(),
    }


def random_benchmark(df_scores_raw, df_options, df_funding, df_futures,
                     option_lookup, qtr_price_lookup, qtr_avail_lookup,
                     config, n_iter=100, period="is"):
    """
    Run backtest with random option selection (n_iter times).
    Returns list of Sharpe ratios for comparison.
    """
    split = config["split"]
    bt = config["backtest"]
    start = split["is_start"] if period == "is" else split["oos_start"]
    end = split["is_end"] if period == "is" else split["oos_end"]

    sharpes = []
    mean_pnls = []
    n_trades_list = []

    for i in range(n_iter):
        # Random score assignment
        df_rand = df_scores_raw.copy()
        df_rand["score"] = np.random.random(len(df_rand))

        df_trades = run_backtest_v2(
            df_scores=df_rand,
            df_options=df_options,
            df_funding=df_funding,
            df_futures=df_futures,
            exit_dte=EXIT_DTE,
            profit_target_pct=PROFIT_TARGET,
            initial_equity=bt["initial_equity"],
            tail_strike_pct=bt["tail_strike_pct"],
            gap_anomaly_z=99.0,  # disable gap filter for random
            start_date=start,
            end_date=end,
            verbose=False,
            _option_lookup=option_lookup,
            _quarterly_price_lookup=qtr_price_lookup,
            _quarterly_avail_lookup=qtr_avail_lookup,
        )

        if not df_trades.empty:
            m = compute_metrics_v2(df_trades)
            sharpes.append(m["sharpe"])
            mean_pnls.append(m["mean_pnl"])
            n_trades_list.append(m["n_trades"])

        if (i + 1) % 25 == 0:
            print(f"  Random iter {i+1}/{n_iter}...", flush=True)

    return sharpes, mean_pnls, n_trades_list


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

    # Rescore with Premium/DTE
    df_scores_pdt = rescore_premium_dte(df_scores_raw)

    # ================================================================
    # 1. Run Premium/DTE backtest (IS + OOS)
    # ================================================================
    print("\n" + "=" * 65)
    print("PREMIUM/DTE — Backtest Results")
    print("=" * 65)

    df_is = run_with_scoring(df_scores_pdt, df_options, df_funding, df_futures,
                             option_lookup, qtr_price_lookup, qtr_avail_lookup,
                             config, "is")
    df_oos = run_with_scoring(df_scores_pdt, df_options, df_funding, df_futures,
                              option_lookup, qtr_price_lookup, qtr_avail_lookup,
                              config, "oos")

    m_is = compute_metrics_v2(df_is) if not df_is.empty else {}
    m_oos = compute_metrics_v2(df_oos) if not df_oos.empty else {}

    if m_is:
        print(f"  IS:  {m_is['n_trades']} trades, Sharpe={m_is['sharpe']:.2f}, "
              f"Ret={m_is['total_return_pct']:.1f}%, DD={m_is['max_drawdown_pct']:.1f}%, "
              f"$/trade=${m_is['mean_pnl']:,.0f}")
    if m_oos:
        print(f"  OOS: {m_oos['n_trades']} trades, Sharpe={m_oos['sharpe']:.2f}, "
              f"Ret={m_oos['total_return_pct']:.1f}%, DD={m_oos['max_drawdown_pct']:.1f}%, "
              f"$/trade=${m_oos['mean_pnl']:,.0f}")

    # ================================================================
    # 2. Demeaned Regression — Does higher score → higher PnL?
    # ================================================================
    print("\n" + "=" * 65)
    print("DEMEANED REGRESSION — Score predictive power")
    print("=" * 65)

    df_all = pd.concat([df_is, df_oos], ignore_index=True)

    for label, df_sub in [("IS", df_is), ("OOS", df_oos), ("IS+OOS", df_all)]:
        if df_sub.empty or len(df_sub) < 5:
            print(f"  {label}: insufficient trades")
            continue
        reg = demeaned_regression(df_sub)
        sign = "+" if reg["beta"] > 0 else ""
        sig = "***" if reg["p_value"] < 0.01 else "**" if reg["p_value"] < 0.05 else "*" if reg["p_value"] < 0.10 else ""
        print(f"\n  {label} (N={reg['n']}):")
        print(f"    β = {sign}{reg['beta']:.2f}  (p={reg['p_value']:.4f}) {sig}")
        print(f"    R² = {reg['r_squared']:.4f}")
        print(f"    Intercept (base PnL) = ${reg['intercept']:,.0f}")
        print(f"    Interpretation: {'Score DOES discriminate' if reg['p_value'] < 0.05 and reg['beta'] > 0 else 'Score does NOT add timing value'}")

    # ================================================================
    # 3. Random Benchmark — Premium/DTE vs Random Selection
    # ================================================================
    print("\n" + "=" * 65)
    print("RANDOM BENCHMARK — 100 iterations")
    print("=" * 65)

    for label, period, m_strat in [("IS", "is", m_is), ("OOS", "oos", m_oos)]:
        if not m_strat:
            continue

        print(f"\n  Running {label} random benchmark...", flush=True)
        rand_sharpes, rand_pnls, rand_ntrades = random_benchmark(
            df_scores_raw, df_options, df_funding, df_futures,
            option_lookup, qtr_price_lookup, qtr_avail_lookup,
            config, n_iter=100, period=period
        )

        strat_sharpe = m_strat["sharpe"]
        strat_pnl = m_strat["mean_pnl"]
        pct_rank_sharpe = np.mean([s < strat_sharpe for s in rand_sharpes]) * 100
        pct_rank_pnl = np.mean([p < strat_pnl for p in rand_pnls]) * 100

        print(f"\n  {label} Results:")
        print(f"    Premium/DTE Sharpe: {strat_sharpe:.2f}  |  Random mean: {np.mean(rand_sharpes):.2f} ± {np.std(rand_sharpes):.2f}")
        print(f"    Premium/DTE $/trade: ${strat_pnl:,.0f}  |  Random mean: ${np.mean(rand_pnls):,.0f} ± ${np.std(rand_pnls):,.0f}")
        print(f"    Random trades/run: {np.mean(rand_ntrades):.0f}")
        print(f"    Percentile rank (Sharpe): {pct_rank_sharpe:.0f}th")
        print(f"    Percentile rank ($/trade): {pct_rank_pnl:.0f}th")
        beats_pct = np.mean([s < strat_sharpe for s in rand_sharpes]) * 100
        print(f"    Beats random: {beats_pct:.0f}% of iterations")

    # ================================================================
    # 4. Additional: Top Vega benchmark comparison
    # ================================================================
    print("\n" + "=" * 65)
    print("BENCHMARK — Premium/DTE vs Top Vega vs Random")
    print("=" * 65)

    # Top Vega scoring
    df_scores_vega = df_scores_raw.copy()
    df_scores_vega["score"] = df_scores_vega["vega"].abs()

    df_vega_is = run_with_scoring(df_scores_vega, df_options, df_funding, df_futures,
                                  option_lookup, qtr_price_lookup, qtr_avail_lookup,
                                  config, "is")
    df_vega_oos = run_with_scoring(df_scores_vega, df_options, df_funding, df_futures,
                                   option_lookup, qtr_price_lookup, qtr_avail_lookup,
                                   config, "oos")

    m_vega_is = compute_metrics_v2(df_vega_is) if not df_vega_is.empty else {}
    m_vega_oos = compute_metrics_v2(df_vega_oos) if not df_vega_oos.empty else {}

    print(f"\n  {'Strategy':<18s} {'IS Sharpe':>10s} {'IS $/tr':>10s} {'IS N':>5s} {'OOS Sharpe':>11s} {'OOS $/tr':>10s} {'OOS N':>6s}")
    print("  " + "-" * 72)

    if m_is and m_oos:
        print(f"  {'Premium/DTE':<18s} {m_is['sharpe']:>10.2f} {m_is['mean_pnl']:>10,.0f} {m_is['n_trades']:>5.0f} {m_oos['sharpe']:>11.2f} {m_oos['mean_pnl']:>10,.0f} {m_oos['n_trades']:>6.0f}")
    if m_vega_is and m_vega_oos:
        print(f"  {'Top Vega':<18s} {m_vega_is['sharpe']:>10.2f} {m_vega_is['mean_pnl']:>10,.0f} {m_vega_is['n_trades']:>5.0f} {m_vega_oos['sharpe']:>11.2f} {m_vega_oos['mean_pnl']:>10,.0f} {m_vega_oos['n_trades']:>6.0f}")

    # Show random stats
    print(f"  {'Random (mean)':<18s} {np.mean(rand_sharpes) if 'rand_sharpes' in dir() else 0:>10.2f} {np.mean(rand_pnls) if 'rand_pnls' in dir() else 0:>10,.0f}")

    print("\n" + "=" * 65)
    print("CONCLUSION")
    print("=" * 65)


if __name__ == "__main__":
    main()
