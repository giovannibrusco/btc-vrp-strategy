"""
Final Combined Portfolio — Mixed sizing.

- Top Vega: vega-normalized (target=50)
- Premium/DTE: fixed 1 contract
- Combined: merge both streams, full metrics with MTM drawdown
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
VEGA_TARGET = 50  # for Top Vega


def run_strat(scoring_fn, df_scores_raw, common_base, config, period,
              sizing="fixed", target_vega=0):
    split = config["split"]
    start = split["is_start"] if period == "is" else split["oos_start"]
    end = split["is_end"] if period == "is" else split["oos_end"]
    df_scores = df_scores_raw.copy()
    df_scores["score"] = scoring_fn(df_scores)
    return run_backtest_v2(
        df_scores=df_scores, start_date=start, end_date=end,
        sizing=sizing, target_vega_usd=target_vega,
        **common_base,
    )


def full_metrics(df_trades, label, initial_equity):
    """Print comprehensive metrics including MTM drawdown."""
    if df_trades.empty:
        print(f"  {label}: no trades")
        return {}

    m = compute_metrics_v2(df_trades)
    pnls = df_trades["total_pnl"].values
    n = m["n_trades"]

    # Trade-level DD for comparison
    eq_trade = df_trades["equity_after"].values
    dd_trade = ((eq_trade - np.maximum.accumulate(eq_trade)) / np.maximum.accumulate(eq_trade)).min() * 100

    print(f"\n  {label}:")
    print(f"    Trades:          {n}")
    print(f"    Sharpe:          {m['sharpe']:.2f}")
    print(f"    Total Return:    {m['total_return_pct']:.1f}%")
    print(f"    MTM Max DD:      {m['max_drawdown_pct']:.1f}%")
    print(f"    Trade-level DD:  {dd_trade:.1f}%")
    print(f"    Win Rate:        {m['win_rate']:.1f}%")
    print(f"    Mean $/trade:    ${m['mean_pnl']:,.0f}")
    print(f"    Trades/year:     {m['trades_per_year']:.1f}")
    if "n_contracts" in df_trades.columns:
        print(f"    Contracts:       {df_trades['n_contracts'].mean():.2f} ± {df_trades['n_contracts'].std():.2f}")
    if "entry_vega" in df_trades.columns:
        tot_vega = df_trades["entry_vega"] * df_trades["n_contracts"]
        print(f"    Tot Vega/trade:  {tot_vega.mean():.1f} ± {tot_vega.std():.1f}")

    return m


def combined_mtm_metrics(df_pdt, df_vega, initial_equity, label):
    """
    Merge two strategy streams with proper daily MTM drawdown.

    Uses daily equity from both strategies to build a combined
    mark-to-market equity curve.
    """
    # Get daily equity from each strategy
    daily_pdt = df_pdt.attrs.get("daily_equity", pd.DataFrame())
    daily_vega = df_vega.attrs.get("daily_equity", pd.DataFrame())

    # Build combined daily MTM equity
    # Each strategy contributes its unrealized P&L to the total
    if not daily_pdt.empty and not daily_vega.empty:
        # Merge on date — both have columns: date, equity
        # The "equity" for each already includes unrealized P&L
        # Combined equity = initial_equity + (pdt_equity - initial_equity) + (vega_equity - initial_equity)
        # = pdt_equity + vega_equity - initial_equity

        dp = daily_pdt.set_index("date")["equity"].rename("eq_pdt")
        dv = daily_vega.set_index("date")["equity"].rename("eq_vega")

        daily_combined = pd.DataFrame({"eq_pdt": dp, "eq_vega": dv})
        # Forward fill for dates where one strategy has data but not the other
        daily_combined = daily_combined.sort_index().ffill()
        # Fill initial values for any NaN at start
        daily_combined["eq_pdt"] = daily_combined["eq_pdt"].fillna(initial_equity)
        daily_combined["eq_vega"] = daily_combined["eq_vega"].fillna(initial_equity)

        daily_combined["equity"] = daily_combined["eq_pdt"] + daily_combined["eq_vega"] - initial_equity

        eq_mtm = daily_combined["equity"].values
        peak_mtm = np.maximum.accumulate(eq_mtm)
        dd_mtm = ((eq_mtm - peak_mtm) / peak_mtm).min() * 100
    else:
        dd_mtm = None

    # Trade-level combined equity
    df_p = df_pdt.copy()
    df_v = df_vega.copy()
    df_p.attrs = {}
    df_v.attrs = {}
    df_p["strategy"] = "PDT"
    df_v["strategy"] = "Vega"
    df_p["entry_date"] = pd.to_datetime(df_p["entry_date"])
    df_p["exit_date"] = pd.to_datetime(df_p["exit_date"])
    df_v["entry_date"] = pd.to_datetime(df_v["entry_date"])
    df_v["exit_date"] = pd.to_datetime(df_v["exit_date"])

    df_all = pd.concat([df_p, df_v], ignore_index=True).sort_values("exit_date").reset_index(drop=True)

    equity = initial_equity
    equities = []
    returns_pct = []
    for _, row in df_all.iterrows():
        ret = row["total_pnl"] / equity * 100
        equity += row["total_pnl"]
        equities.append(equity)
        returns_pct.append(ret)

    pnls = df_all["total_pnl"].values
    rets = np.array(returns_pct)
    n = len(df_all)
    total_days = (df_all["exit_date"].max() - df_all["entry_date"].min()).days
    tpy = n / (total_days / 365) if total_days > 0 else 12
    sharpe = rets.mean() / rets.std() * np.sqrt(tpy) if rets.std() > 0 else 0

    # Trade-level DD
    eq_arr = np.array(equities)
    peak = np.maximum.accumulate(eq_arr)
    dd_trade = ((eq_arr - peak) / peak).min() * 100

    wr = (pnls > 0).sum() / n * 100

    print(f"\n  {label}:")
    print(f"    Trades:          {n} (PDT={len(df_p)}, Vega={len(df_v)})")
    print(f"    Sharpe:          {sharpe:.2f}")
    print(f"    Total Return:    {(equity/initial_equity-1)*100:.1f}%")
    if dd_mtm is not None:
        print(f"    MTM Max DD:      {dd_mtm:.1f}%")
    print(f"    Trade-level DD:  {dd_trade:.1f}%")
    print(f"    Win Rate:        {wr:.1f}%")
    print(f"    Mean $/trade:    ${pnls.mean():,.0f}")
    print(f"    Trades/year:     {tpy:.1f}")

    # P&L breakdown by strategy
    pdt_pnl = df_p["total_pnl"].sum()
    vega_pnl = df_v["total_pnl"].sum()
    print(f"    PDT contribution:  ${pdt_pnl:,.0f} ({pdt_pnl/(pdt_pnl+vega_pnl)*100:.0f}%)")
    print(f"    Vega contribution: ${vega_pnl:,.0f} ({vega_pnl/(pdt_pnl+vega_pnl)*100:.0f}%)")

    # Bootstrap Sharpe CI
    rng = np.random.default_rng(42)
    boot_sharpes = []
    for _ in range(N_BOOTSTRAP):
        sample = rng.choice(pnls, size=n, replace=True)
        if sample.std() > 0:
            boot_sharpes.append(sample.mean() / sample.std() * np.sqrt(tpy))
    boot_sharpes = np.array(boot_sharpes)
    ci_lo, ci_hi = np.percentile(boot_sharpes, [2.5, 97.5])
    ci90_lo, ci90_hi = np.percentile(boot_sharpes, [5, 95])

    print(f"\n    Bootstrap Sharpe (10,000 resamples):")
    print(f"      95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"      90% CI: [{ci90_lo:.2f}, {ci90_hi:.2f}]")
    print(f"      P(Sharpe > 0): {(boot_sharpes > 0).mean()*100:.1f}%")
    print(f"      P(Sharpe > 1): {(boot_sharpes > 1).mean()*100:.1f}%")
    print(f"      P(Sharpe > 2): {(boot_sharpes > 2).mean()*100:.1f}%")

    # t-test
    t_stat, p_val = stats.ttest_1samp(pnls, 0)
    p_one = p_val / 2 if t_stat > 0 else 1
    sig = "***" if p_one < 0.01 else "**" if p_one < 0.05 else "*" if p_one < 0.10 else ""
    print(f"\n    t-test (H0: mean PnL=0):")
    print(f"      t={t_stat:.3f}  p(one-sided)={p_one:.4f} {sig}")

    return sharpe, dd_mtm if dd_mtm else dd_trade, df_all


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

    bt = config["backtest"]
    initial_eq = bt["initial_equity"]
    common_base = dict(
        df_options=df_options,
        df_funding=df_funding,
        df_futures=df_futures,
        exit_dte=EXIT_DTE,
        profit_target_pct=PROFIT_TARGET,
        initial_equity=initial_eq,
        tail_strike_pct=bt["tail_strike_pct"],
        gap_anomaly_z=bt["gap_anomaly_z_threshold"],
        verbose=False,
        _option_lookup=option_lookup,
        _quarterly_price_lookup=qtr_price_lookup,
        _quarterly_avail_lookup=qtr_avail_lookup,
    )

    score_pdt = lambda df: df["bid_price"] * df["spot"] / df["dte"].clip(lower=1)
    score_vega = lambda df: df["vega"].abs()

    # ================================================================
    # 1. INDIVIDUAL STRATEGIES — With optimal sizing
    # ================================================================
    print("\n" + "=" * 70)
    print("INDIVIDUAL STRATEGIES — Optimal Sizing")
    print("=" * 70)

    results = {}
    for period in ["is", "oos"]:
        # Premium/DTE: fixed 1 contract
        results[f"pdt_{period}"] = run_strat(
            score_pdt, df_scores_raw, common_base, config, period,
            sizing="fixed"
        )
        full_metrics(results[f"pdt_{period}"], f"Premium/DTE {period.upper()} (fixed)", initial_eq)

        # Top Vega: vega-normalized target=50
        results[f"vega_{period}"] = run_strat(
            score_vega, df_scores_raw, common_base, config, period,
            sizing="vega", target_vega=VEGA_TARGET
        )
        full_metrics(results[f"vega_{period}"], f"Top Vega {period.upper()} (vega={VEGA_TARGET})", initial_eq)

    # ================================================================
    # 2. COMBINED PORTFOLIO — Mixed sizing
    # ================================================================
    print("\n" + "=" * 70)
    print("COMBINED PORTFOLIO — Mixed Sizing (PDT fixed + Vega normalized)")
    print("=" * 70)

    for period in ["is", "oos"]:
        df_pdt = results[f"pdt_{period}"]
        df_vega = results[f"vega_{period}"]
        if df_pdt.empty or df_vega.empty:
            continue
        label = f"{period.upper()} Combined (mixed sizing)"
        combined_mtm_metrics(df_pdt, df_vega, initial_eq, label)

    # ================================================================
    # 3. FULL PERIOD — IS + OOS combined
    # ================================================================
    print("\n" + "=" * 70)
    print("FULL PERIOD (IS + OOS)")
    print("=" * 70)

    # Merge IS and OOS for each strategy — clear attrs first to avoid concat issue
    pdt_is_copy = results["pdt_is"].copy()
    pdt_oos_copy = results["pdt_oos"].copy()
    vega_is_copy = results["vega_is"].copy()
    vega_oos_copy = results["vega_oos"].copy()

    # Save daily equity before clearing
    daily_pdt_is = results["pdt_is"].attrs.get("daily_equity", pd.DataFrame())
    daily_pdt_oos = results["pdt_oos"].attrs.get("daily_equity", pd.DataFrame())
    daily_vega_is = results["vega_is"].attrs.get("daily_equity", pd.DataFrame())
    daily_vega_oos = results["vega_oos"].attrs.get("daily_equity", pd.DataFrame())

    pdt_is_copy.attrs = {}
    pdt_oos_copy.attrs = {}
    vega_is_copy.attrs = {}
    vega_oos_copy.attrs = {}

    df_pdt_full = pd.concat([pdt_is_copy, pdt_oos_copy], ignore_index=True)
    df_vega_full = pd.concat([vega_is_copy, vega_oos_copy], ignore_index=True)

    # For full period MTM, merge daily equity from IS and OOS
    daily_pdt_is = results["pdt_is"].attrs.get("daily_equity", pd.DataFrame())
    daily_pdt_oos = results["pdt_oos"].attrs.get("daily_equity", pd.DataFrame())
    daily_vega_is = results["vega_is"].attrs.get("daily_equity", pd.DataFrame())
    daily_vega_oos = results["vega_oos"].attrs.get("daily_equity", pd.DataFrame())

    # Stitch daily equity: OOS equity continues from IS final value
    if not daily_pdt_is.empty and not daily_pdt_oos.empty:
        pdt_final_is_eq = daily_pdt_is["equity"].iloc[-1]
        daily_pdt_oos_adj = daily_pdt_oos.copy()
        oos_start_eq = daily_pdt_oos_adj["equity"].iloc[0]
        daily_pdt_oos_adj["equity"] = daily_pdt_oos_adj["equity"] + (pdt_final_is_eq - oos_start_eq)
        daily_pdt_full_eq = pd.concat([daily_pdt_is, daily_pdt_oos_adj], ignore_index=True)
        df_pdt_full.attrs["daily_equity"] = daily_pdt_full_eq

    if not daily_vega_is.empty and not daily_vega_oos.empty:
        vega_final_is_eq = daily_vega_is["equity"].iloc[-1]
        daily_vega_oos_adj = daily_vega_oos.copy()
        oos_start_eq_v = daily_vega_oos_adj["equity"].iloc[0]
        daily_vega_oos_adj["equity"] = daily_vega_oos_adj["equity"] + (vega_final_is_eq - oos_start_eq_v)
        daily_vega_full_eq = pd.concat([daily_vega_is, daily_vega_oos_adj], ignore_index=True)
        df_vega_full.attrs["daily_equity"] = daily_vega_full_eq

    combined_mtm_metrics(df_pdt_full, df_vega_full, initial_eq, "Full Period Combined (mixed sizing)")

    # ================================================================
    # 4. COMPARISON TABLE
    # ================================================================
    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE")
    print("=" * 70)

    print(f"\n  {'Configuration':<40s} {'IS Sh':>6s} {'IS DD':>7s} {'OOS Sh':>7s} {'OOS DD':>7s}")
    print("  " + "-" * 68)

    # Individual
    for period in ["is", "oos"]:
        m_pdt = compute_metrics_v2(results[f"pdt_{period}"])
        m_vega = compute_metrics_v2(results[f"vega_{period}"])
        if period == "is":
            print(f"  {'PDT fixed':<40s} {m_pdt['sharpe']:>6.2f} {m_pdt['max_drawdown_pct']:>7.1f}")
            print(f"  {'Top Vega (vega=50)':<40s} {m_vega['sharpe']:>6.2f} {m_vega['max_drawdown_pct']:>7.1f}")
        else:
            # Complete the row
            pass

    # Combined entries - recompute for table
    for period in ["is", "oos"]:
        df_pdt = results[f"pdt_{period}"]
        df_vega = results[f"vega_{period}"]
        dp = df_pdt.copy()
        dv = df_vega.copy()
        dp.attrs = {}
        dv.attrs = {}
        dp["entry_date"] = pd.to_datetime(dp["entry_date"])
        dp["exit_date"] = pd.to_datetime(dp["exit_date"])
        dv["entry_date"] = pd.to_datetime(dv["entry_date"])
        dv["exit_date"] = pd.to_datetime(dv["exit_date"])
        dp["strategy"] = "PDT"
        dv["strategy"] = "Vega"
        df_all = pd.concat([dp, dv], ignore_index=True).sort_values("exit_date")
        equity = initial_eq
        rets = []
        eqs = []
        for _, row in df_all.iterrows():
            r = row["total_pnl"] / equity * 100
            equity += row["total_pnl"]
            rets.append(r)
            eqs.append(equity)
        rets = np.array(rets)
        n = len(df_all)
        td = (df_all["exit_date"].max() - df_all["entry_date"].min()).days
        tpy = n / (td / 365) if td > 0 else 12
        sh = rets.mean() / rets.std() * np.sqrt(tpy) if rets.std() > 0 else 0
        eq_arr = np.array(eqs)
        dd = ((eq_arr - np.maximum.accumulate(eq_arr)) / np.maximum.accumulate(eq_arr)).min() * 100

        if period == "is":
            print(f"  {'Combined mixed (trade-level DD)':<40s} {sh:>6.2f} {dd:>7.1f}")

    # Print the VRP vs theta mechanism note
    print(f"\n" + "=" * 70)
    print("MECHANISM ANALYSIS — VRP vs Theta Decay")
    print("=" * 70)

    for period in ["is", "oos"]:
        df_pdt = results[f"pdt_{period}"]
        df_vega = results[f"vega_{period}"]
        if df_pdt.empty or df_vega.empty:
            continue

        pdt_dte = df_pdt["entry_dte"].mean()
        vega_dte = df_vega["entry_dte"].mean()
        pdt_vega_raw = df_pdt["entry_vega"].mean() if "entry_vega" in df_pdt.columns else 0
        vega_vega_raw = df_vega["entry_vega"].mean() if "entry_vega" in df_vega.columns else 0
        pdt_holding = (pd.to_datetime(df_pdt["exit_date"]) - pd.to_datetime(df_pdt["entry_date"])).dt.days.mean()
        vega_holding = (pd.to_datetime(df_vega["exit_date"]) - pd.to_datetime(df_vega["entry_date"])).dt.days.mean()

        # Option vs Hedge PnL breakdown
        pdt_opt = df_pdt["option_pnl"].mean()
        pdt_hedge = df_pdt["hedge_pnl_total"].mean()
        vega_opt = df_vega["option_pnl"].mean()
        vega_hedge = df_vega["hedge_pnl_total"].mean()

        print(f"\n  {period.upper()}:")
        print(f"    {'Metric':<25s} {'Premium/DTE':>12s} {'Top Vega':>12s}")
        print(f"    {'-'*50}")
        print(f"    {'Mean DTE at entry':<25s} {pdt_dte:>12.0f} {vega_dte:>12.0f}")
        print(f"    {'Mean holding days':<25s} {pdt_holding:>12.1f} {vega_holding:>12.1f}")
        print(f"    {'Raw vega/contract':<25s} {pdt_vega_raw:>12.1f} {vega_vega_raw:>12.1f}")
        print(f"    {'Option PnL/trade':<25s} ${pdt_opt:>11,.0f} ${vega_opt:>11,.0f}")
        print(f"    {'Hedge PnL/trade':<25s} ${pdt_hedge:>11,.0f} ${vega_hedge:>11,.0f}")
        print(f"    {'Profit source':<25s} {'Theta decay':>12s} {'VRP':>12s}")


if __name__ == "__main__":
    main()
