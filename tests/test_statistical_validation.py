"""
Statistical Validation — Premium/DTE & Top Vega.

Tests:
1. t-test: mean PnL significantly > 0?
2. Bootstrap CI: Sharpe ratio confidence intervals (10,000 resamples)
3. Regime analysis: performance in low/mid/high volatility regimes
4. Outlier dependence: what happens if we remove best N trades?
5. Cost stress test: how much can costs increase before Sharpe < 0?
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
EXIT_DTE = 5
PROFIT_TARGET = 0.50
N_BOOTSTRAP = 10_000


def run_strategy(scoring_fn, df_scores_raw, df_options, df_funding, df_futures,
                 option_lookup, qtr_price_lookup, qtr_avail_lookup, config, period):
    """Run a backtest with a given scoring function."""
    split = config["split"]
    bt = config["backtest"]
    start = split["is_start"] if period == "is" else split["oos_start"]
    end = split["is_end"] if period == "is" else split["oos_end"]

    df_scores = df_scores_raw.copy()
    df_scores["score"] = scoring_fn(df_scores)

    return run_backtest_v2(
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


def test_ttest(pnls, label):
    """One-sample t-test: H0: mean PnL = 0."""
    t_stat, p_value = stats.ttest_1samp(pnls, 0)
    p_one_sided = p_value / 2 if t_stat > 0 else 1 - p_value / 2
    sig = "***" if p_one_sided < 0.01 else "**" if p_one_sided < 0.05 else "*" if p_one_sided < 0.10 else ""
    print(f"    {label}:")
    print(f"      N={len(pnls)}, mean=${np.mean(pnls):,.0f}, std=${np.std(pnls, ddof=1):,.0f}")
    print(f"      t-stat={t_stat:.3f}, p(one-sided)={p_one_sided:.4f} {sig}")
    return {"t_stat": t_stat, "p_value": p_one_sided, "n": len(pnls)}


def test_bootstrap_sharpe(pnls, trades_per_year, label, n_boot=N_BOOTSTRAP):
    """Bootstrap CI for Sharpe ratio."""
    rng = np.random.default_rng(42)
    n = len(pnls)
    sharpes = []

    for _ in range(n_boot):
        sample = rng.choice(pnls, size=n, replace=True)
        if sample.std() > 0:
            s = sample.mean() / sample.std() * np.sqrt(trades_per_year)
            sharpes.append(s)

    sharpes = np.array(sharpes)
    ci_lo, ci_hi = np.percentile(sharpes, [2.5, 97.5])
    ci_90_lo, ci_90_hi = np.percentile(sharpes, [5, 95])
    original_sharpe = np.mean(pnls) / np.std(pnls, ddof=1) * np.sqrt(trades_per_year)

    print(f"    {label}:")
    print(f"      Sharpe = {original_sharpe:.2f}")
    print(f"      95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"      90% CI: [{ci_90_lo:.2f}, {ci_90_hi:.2f}]")
    print(f"      P(Sharpe > 0): {(sharpes > 0).mean()*100:.1f}%")
    print(f"      P(Sharpe > 1): {(sharpes > 1).mean()*100:.1f}%")
    return {"sharpe": original_sharpe, "ci_95": (ci_lo, ci_hi), "ci_90": (ci_90_lo, ci_90_hi),
            "p_gt_0": (sharpes > 0).mean(), "p_gt_1": (sharpes > 1).mean()}


def test_regime_analysis(df_trades, df_options, label):
    """Split trades by BTC volatility regime at entry."""
    df_trades = df_trades.copy()
    df_trades["entry_date"] = pd.to_datetime(df_trades["entry_date"])

    # Compute 30-day rolling realized vol from options underlying prices
    df_opt = df_options.copy()
    df_opt["date"] = pd.to_datetime(df_opt["date"])
    daily_prices = df_opt.groupby("date")["underlying_price"].first().sort_index()
    log_rets = np.log(daily_prices / daily_prices.shift(1)).dropna()
    rv30 = log_rets.rolling(30).std() * np.sqrt(365) * 100  # annualized %

    # Map each trade to its entry-date RV
    rv_dict = rv30.to_dict()
    df_trades["rv30"] = df_trades["entry_date"].map(lambda d: rv_dict.get(d, np.nan))
    df_trades = df_trades.dropna(subset=["rv30"])

    if len(df_trades) < 5:
        print(f"    {label}: insufficient trades with RV data")
        return

    # Define regimes
    q33 = df_trades["rv30"].quantile(0.33)
    q67 = df_trades["rv30"].quantile(0.67)

    regimes = {
        f"Low (<{q33:.0f}%)": df_trades[df_trades["rv30"] < q33],
        f"Mid ({q33:.0f}-{q67:.0f}%)": df_trades[(df_trades["rv30"] >= q33) & (df_trades["rv30"] < q67)],
        f"High (>{q67:.0f}%)": df_trades[df_trades["rv30"] >= q67],
    }

    print(f"    {label}:")
    print(f"      {'Regime':<22s} {'N':>4s} {'WR%':>5s} {'Mean$':>8s} {'Sharpe':>7s}")
    print(f"      " + "-" * 48)

    for regime_name, df_r in regimes.items():
        n = len(df_r)
        if n < 2:
            print(f"      {regime_name:<22s} {n:>4d}   insufficient data")
            continue
        wr = (df_r["total_pnl"] > 0).sum() / n * 100
        mean_pnl = df_r["total_pnl"].mean()
        pnls = df_r["total_pnl"].values
        ret_pct = df_r["return_pct"].values
        sh = ret_pct.mean() / ret_pct.std() * np.sqrt(12) if ret_pct.std() > 0 else 0  # rough
        print(f"      {regime_name:<22s} {n:>4d} {wr:>5.1f} {mean_pnl:>8,.0f} {sh:>7.2f}")


def test_outlier_dependence(df_trades, label):
    """What happens if we remove the top-N best trades?"""
    pnls = df_trades["total_pnl"].values
    returns_pct = df_trades["return_pct"].values
    n = len(pnls)

    # Compute base metrics
    total_days = (pd.to_datetime(df_trades["exit_date"]).max() -
                  pd.to_datetime(df_trades["entry_date"]).min()).days
    tpy = n / (total_days / 365) if total_days > 0 else 12

    print(f"    {label} (N={n}):")
    print(f"      {'Remove':>10s} {'N':>4s} {'Mean$':>8s} {'WR%':>5s} {'Sharpe':>7s} {'Ret%':>7s}")
    print(f"      " + "-" * 44)

    for remove in [0, 1, 2, 3, 5]:
        if remove >= n:
            break
        # Sort by PnL descending, remove top `remove`
        idx = np.argsort(pnls)[::-1][remove:]
        sub_pnls = pnls[idx]
        sub_rets = returns_pct[idx]
        wr = (sub_pnls > 0).sum() / len(sub_pnls) * 100
        sh = sub_rets.mean() / sub_rets.std() * np.sqrt(tpy) if sub_rets.std() > 0 else 0
        cum_ret = sub_rets.sum()
        lbl = "None" if remove == 0 else f"Top {remove}"
        print(f"      {lbl:>10s} {len(sub_pnls):>4d} {sub_pnls.mean():>8,.0f} {wr:>5.1f} {sh:>7.2f} {cum_ret:>7.1f}")


def test_cost_stress(df_trades, label):
    """How much can hedge costs increase before strategy breaks?"""
    pnls = df_trades["total_pnl"].values
    hedge_costs = df_trades["hedge_txn_cost_total"].values
    returns_pct = df_trades["return_pct"].values
    n = len(pnls)

    total_days = (pd.to_datetime(df_trades["exit_date"]).max() -
                  pd.to_datetime(df_trades["entry_date"]).min()).days
    tpy = n / (total_days / 365) if total_days > 0 else 12

    print(f"    {label} (N={n}):")
    print(f"      {'Cost mult':>10s} {'Mean$':>8s} {'Sharpe':>7s}")
    print(f"      " + "-" * 28)

    for mult in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        extra_cost = hedge_costs * (mult - 1)
        adj_pnls = pnls - extra_cost
        adj_rets = returns_pct - (extra_cost / df_trades["equity_before"].values * 100)
        sh = adj_rets.mean() / adj_rets.std() * np.sqrt(tpy) if adj_rets.std() > 0 else 0
        print(f"      {mult:>10.1f}x {adj_pnls.mean():>8,.0f} {sh:>7.2f}")
        if sh < 0:
            print(f"      → Strategy breaks at ~{mult:.1f}x current costs")
            break


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

    # Scoring functions
    score_pdt = lambda df: df["bid_price"] * df["spot"] / df["dte"].clip(lower=1)
    score_vega = lambda df: df["vega"].abs()

    # Run all backtests
    strategies = {}
    for name, score_fn in [("Premium/DTE", score_pdt), ("Top Vega", score_vega)]:
        for period in ["is", "oos"]:
            key = f"{name}_{period}"
            print(f"  Running {key}...", flush=True)
            df_trades = run_strategy(
                score_fn, df_scores_raw, df_options, df_funding, df_futures,
                option_lookup, qtr_price_lookup, qtr_avail_lookup, config, period
            )
            strategies[key] = df_trades

    # ================================================================
    # 1. T-TEST — Mean PnL > 0?
    # ================================================================
    print("\n" + "=" * 65)
    print("1. T-TEST — H0: mean PnL = 0 (one-sided)")
    print("=" * 65)

    for key, df_t in strategies.items():
        if df_t.empty:
            continue
        print()
        test_ttest(df_t["total_pnl"].values, key)

    # ================================================================
    # 2. BOOTSTRAP SHARPE CI
    # ================================================================
    print("\n" + "=" * 65)
    print("2. BOOTSTRAP SHARPE — 10,000 resamples")
    print("=" * 65)

    for key, df_t in strategies.items():
        if df_t.empty or len(df_t) < 5:
            continue
        m = compute_metrics_v2(df_t)
        print()
        test_bootstrap_sharpe(df_t["total_pnl"].values, m["trades_per_year"], key)

    # ================================================================
    # 3. REGIME ANALYSIS
    # ================================================================
    print("\n" + "=" * 65)
    print("3. REGIME ANALYSIS — Performance by RV30 tercile")
    print("=" * 65)

    for key, df_t in strategies.items():
        if df_t.empty:
            continue
        print()
        test_regime_analysis(df_t, df_options, key)

    # ================================================================
    # 4. OUTLIER DEPENDENCE
    # ================================================================
    print("\n" + "=" * 65)
    print("4. OUTLIER DEPENDENCE — Remove best N trades")
    print("=" * 65)

    for key, df_t in strategies.items():
        if df_t.empty:
            continue
        print()
        test_outlier_dependence(df_t, key)

    # ================================================================
    # 5. COST STRESS TEST
    # ================================================================
    print("\n" + "=" * 65)
    print("5. COST STRESS TEST — Hedge cost multiplier")
    print("=" * 65)

    for key, df_t in strategies.items():
        if df_t.empty:
            continue
        print()
        test_cost_stress(df_t, key)

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    print("\n" + "=" * 65)
    print("SUMMARY TABLE")
    print("=" * 65)
    print(f"\n  {'Test':<30s} {'PDT IS':>8s} {'PDT OOS':>8s} {'Vega IS':>8s} {'Vega OOS':>9s}")
    print("  " + "-" * 65)

    for name, score_fn in [("Premium/DTE", score_pdt), ("Top Vega", score_vega)]:
        for period in ["is", "oos"]:
            key = f"{name}_{period}"

    # Simplified summary
    for key in ["Premium/DTE_is", "Premium/DTE_oos", "Top Vega_is", "Top Vega_oos"]:
        df_t = strategies[key]
        if df_t.empty:
            continue
        m = compute_metrics_v2(df_t)
        pnls = df_t["total_pnl"].values
        t_stat, p_val = stats.ttest_1samp(pnls, 0)
        p_one = p_val / 2 if t_stat > 0 else 1
        print(f"  {key:<22s}  N={m['n_trades']:>3d}  Sharpe={m['sharpe']:>5.2f}  "
              f"t={t_stat:>5.2f}  p={p_one:.4f}  {'✓' if p_one < 0.05 else '✗'}")

    print()


if __name__ == "__main__":
    main()
