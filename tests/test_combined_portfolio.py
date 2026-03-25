"""
Combined Portfolio 50/50 — Premium/DTE + Top Vega baseline.

1. Merge trade streams, compute combined equity curve
2. Metrics: Sharpe, DD, bootstrap CI
3. OOS period context: what was BTC doing?
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


def run_strategy(scoring_fn, df_scores_raw, common, config, period):
    split = config["split"]
    start = split["is_start"] if period == "is" else split["oos_start"]
    end = split["is_end"] if period == "is" else split["oos_end"]
    df_scores = df_scores_raw.copy()
    df_scores["score"] = scoring_fn(df_scores)
    return run_backtest_v2(df_scores=df_scores, start_date=start, end_date=end, **common)


def combined_metrics(df_pdt, df_vega, initial_equity, label):
    """
    Combine two trade streams into one portfolio.
    Each strategy gets 50% of capital (50k each).
    Trades are interleaved chronologically.
    """
    half_eq = initial_equity / 2

    # Scale PnLs: each strategy runs on half capital
    # Since both were backtested with 100k, scale returns proportionally
    # With 1 contract each, PnL doesn't change — but returns are on half equity
    df_p = df_pdt.copy()
    df_v = df_vega.copy()
    df_p["strategy"] = "Premium/DTE"
    df_v["strategy"] = "Top Vega"
    df_p["entry_date"] = pd.to_datetime(df_p["entry_date"])
    df_p["exit_date"] = pd.to_datetime(df_p["exit_date"])
    df_v["entry_date"] = pd.to_datetime(df_v["entry_date"])
    df_v["exit_date"] = pd.to_datetime(df_v["exit_date"])

    # Merge and sort by exit date
    df_all = pd.concat([df_p, df_v], ignore_index=True).sort_values("exit_date").reset_index(drop=True)

    # Recompute equity curve on combined capital
    equity = initial_equity
    equities = []
    returns_pct = []
    for _, row in df_all.iterrows():
        ret = row["total_pnl"] / equity * 100
        equity += row["total_pnl"]
        equities.append(equity)
        returns_pct.append(ret)

    df_all["equity_combined"] = equities
    df_all["return_pct_combined"] = returns_pct

    pnls = df_all["total_pnl"].values
    rets = np.array(returns_pct)
    n = len(df_all)
    total_days = (df_all["exit_date"].max() - df_all["entry_date"].min()).days
    tpy = n / (total_days / 365) if total_days > 0 else 12

    # Sharpe
    sharpe = rets.mean() / rets.std() * np.sqrt(tpy) if rets.std() > 0 else 0

    # Max drawdown
    eq_arr = np.array(equities)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak
    max_dd = dd.min() * 100

    # Win rate
    wr = (pnls > 0).sum() / n * 100

    print(f"\n  {label}:")
    print(f"    Trades: {n} (PDT={len(df_p)}, Vega={len(df_v)})")
    print(f"    Total Return: {(equity / initial_equity - 1) * 100:.1f}%")
    print(f"    Sharpe: {sharpe:.2f}")
    print(f"    Max DD: {max_dd:.1f}%")
    print(f"    Win Rate: {wr:.1f}%")
    print(f"    Mean $/trade: ${pnls.mean():,.0f}")
    print(f"    Trades/year: {tpy:.1f}")

    # Bootstrap Sharpe CI
    rng = np.random.default_rng(42)
    boot_sharpes = []
    for _ in range(N_BOOTSTRAP):
        sample = rng.choice(pnls, size=n, replace=True)
        if sample.std() > 0:
            s = sample.mean() / sample.std() * np.sqrt(tpy)
            boot_sharpes.append(s)
    boot_sharpes = np.array(boot_sharpes)
    ci_lo, ci_hi = np.percentile(boot_sharpes, [2.5, 97.5])

    print(f"\n    Bootstrap Sharpe 95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"    P(Sharpe > 0): {(boot_sharpes > 0).mean()*100:.1f}%")
    print(f"    P(Sharpe > 1): {(boot_sharpes > 1).mean()*100:.1f}%")

    # Correlation between strategies (on overlapping exit-date windows)
    # Simple: just report the diversification benefit
    pdt_mean_ret = df_p["return_pct"].mean()
    vega_mean_ret = df_v["return_pct"].mean()
    pdt_std = df_p["return_pct"].std()
    vega_std = df_v["return_pct"].std()

    print(f"\n    Individual strategy stats:")
    print(f"      PDT:  mean_ret={pdt_mean_ret:.2f}%, std={pdt_std:.2f}%")
    print(f"      Vega: mean_ret={vega_mean_ret:.2f}%, std={vega_std:.2f}%")
    print(f"      Combined std: {rets.std():.2f}% (diversification effect)")

    return df_all, sharpe, max_dd


def btc_context(df_options, config):
    """Show what BTC was doing during IS and OOS periods."""
    split = config["split"]
    df = df_options.copy()
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date")["underlying_price"].first().sort_index()

    print("\n" + "=" * 65)
    print("BTC MARKET CONTEXT")
    print("=" * 65)

    for label, start, end in [
        ("IS", split["is_start"], split["is_end"]),
        ("OOS", split["oos_start"], split["oos_end"]),
    ]:
        sd = pd.Timestamp(start)
        ed = pd.Timestamp(end)
        prices = daily[(daily.index >= sd) & (daily.index <= ed)]

        if prices.empty:
            continue

        p_start = prices.iloc[0]
        p_end = prices.iloc[-1]
        p_min = prices.min()
        p_max = prices.max()
        ret = (p_end / p_start - 1) * 100

        # 30-day rolling vol
        log_rets = np.log(prices / prices.shift(1)).dropna()
        rv30 = log_rets.rolling(30).std() * np.sqrt(365) * 100
        mean_rv = rv30.mean()
        min_rv = rv30.min()
        max_rv = rv30.max()

        # Identify sub-periods
        mid = prices.index[len(prices)//2]
        p_mid = prices.loc[mid]

        print(f"\n  {label}: {start} → {end}")
        print(f"    BTC: ${p_start:,.0f} → ${p_end:,.0f} ({ret:+.1f}%)")
        print(f"    Range: ${p_min:,.0f} — ${p_max:,.0f}")
        print(f"    RV30 mean: {mean_rv:.0f}%, range: [{min_rv:.0f}%, {max_rv:.0f}%]")

        # Characterize the period
        if abs(ret) < 20:
            trend = "sideways/range-bound"
        elif ret > 50:
            trend = "strong uptrend"
        elif ret > 20:
            trend = "moderate uptrend"
        elif ret < -50:
            trend = "strong downtrend"
        elif ret < -20:
            trend = "moderate downtrend"
        else:
            trend = "mixed"

        if mean_rv > 70:
            vol_regime = "high volatility"
        elif mean_rv > 45:
            vol_regime = "moderate volatility"
        else:
            vol_regime = "low volatility"

        print(f"    Character: {trend}, {vol_regime}")

        # Quarterly breakdown
        quarters = prices.resample("QE").agg(["first", "last"])
        quarters["ret"] = (quarters["last"] / quarters["first"] - 1) * 100
        print(f"    Quarterly returns:")
        for idx, qrow in quarters.iterrows():
            q_label = f"      {idx.strftime('%Y-Q')}{(idx.month-1)//3+1}"
            print(f"{q_label}: {qrow['ret']:+.1f}%  (${qrow['first']:,.0f} → ${qrow['last']:,.0f})")


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

    score_pdt = lambda df: df["bid_price"] * df["spot"] / df["dte"].clip(lower=1)
    score_vega = lambda df: df["vega"].abs()

    common = dict(
        df_options=df_options,
        df_funding=df_funding,
        df_futures=df_futures,
        exit_dte=EXIT_DTE,
        profit_target_pct=PROFIT_TARGET,
        initial_equity=config["backtest"]["initial_equity"],
        tail_strike_pct=config["backtest"]["tail_strike_pct"],
        gap_anomaly_z=config["backtest"]["gap_anomaly_z_threshold"],
        verbose=False,
        _option_lookup=option_lookup,
        _quarterly_price_lookup=qtr_price_lookup,
        _quarterly_avail_lookup=qtr_avail_lookup,
    )

    # Run all 4 backtests
    print("\nRunning backtests...", flush=True)
    results = {}
    for name, fn in [("PDT", score_pdt), ("Vega", score_vega)]:
        for period in ["is", "oos"]:
            results[f"{name}_{period}"] = run_strategy(fn, df_scores_raw, common, config, period)

    initial_eq = config["backtest"]["initial_equity"]

    # ================================================================
    # INDIVIDUAL STRATEGY METRICS (for comparison)
    # ================================================================
    print("\n" + "=" * 65)
    print("INDIVIDUAL STRATEGIES — Baseline")
    print("=" * 65)

    for key, df_t in results.items():
        if df_t.empty:
            continue
        m = compute_metrics_v2(df_t)
        print(f"  {key:<12s}  N={m['n_trades']:>3d}  Sharpe={m['sharpe']:.2f}  "
              f"Ret={m['total_return_pct']:.1f}%  DD={m['max_drawdown_pct']:.1f}%  "
              f"$/tr=${m['mean_pnl']:,.0f}")

    # ================================================================
    # COMBINED PORTFOLIO 50/50
    # ================================================================
    print("\n" + "=" * 65)
    print("COMBINED PORTFOLIO 50/50 — Premium/DTE + Top Vega")
    print("=" * 65)

    for period in ["is", "oos"]:
        df_pdt = results[f"PDT_{period}"]
        df_vega = results[f"Vega_{period}"]
        if df_pdt.empty or df_vega.empty:
            continue
        label = f"{'IS' if period == 'is' else 'OOS'} Combined"
        combined_metrics(df_pdt, df_vega, initial_eq, label)

    # ================================================================
    # FULL PERIOD (IS + OOS combined)
    # ================================================================
    print("\n" + "=" * 65)
    print("FULL PERIOD COMBINED")
    print("=" * 65)

    df_pdt_full = pd.concat([results["PDT_is"], results["PDT_oos"]], ignore_index=True)
    df_vega_full = pd.concat([results["Vega_is"], results["Vega_oos"]], ignore_index=True)
    combined_metrics(df_pdt_full, df_vega_full, initial_eq, "Full Period (IS+OOS) Combined")

    # ================================================================
    # BTC CONTEXT
    # ================================================================
    btc_context(df_options, config)


if __name__ == "__main__":
    main()
