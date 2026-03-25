"""
Money Management — Vega-based Position Sizing.

Compares:
1. Fixed sizing (1 contract) — current baseline
2. Vega-normalized sizing — each trade has same raw vega exposure
3. Premium % sizing — each trade sells premium = X% of equity

Also verifies mark-to-market drawdown.
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


def print_metrics_line(df_trades, label):
    if df_trades.empty:
        print(f"  {label}: no trades")
        return {}
    m = compute_metrics_v2(df_trades)
    n_info = ""
    if "n_contracts" in df_trades.columns:
        n_info = f"  Ncontr={df_trades['n_contracts'].mean():.2f}±{df_trades['n_contracts'].std():.2f}"
    vega_info = ""
    if "entry_vega" in df_trades.columns:
        raw_vega = df_trades["entry_vega"] * df_trades["n_contracts"]
        vega_info = f"  TotVega={raw_vega.mean():.1f}±{raw_vega.std():.1f}"
    print(f"  {label:<35s} N={m['n_trades']:>3d}  Sh={m['sharpe']:>5.2f}  "
          f"Ret={m['total_return_pct']:>7.1f}%  DD={m['max_drawdown_pct']:>6.1f}%  "
          f"$/tr=${m['mean_pnl']:>7,.0f}{n_info}{vega_info}")
    return m


def bootstrap_sharpe(pnls, tpy):
    rng = np.random.default_rng(42)
    n = len(pnls)
    sharpes = []
    for _ in range(N_BOOTSTRAP):
        sample = rng.choice(pnls, size=n, replace=True)
        if sample.std() > 0:
            sharpes.append(sample.mean() / sample.std() * np.sqrt(tpy))
    return np.array(sharpes)


def combined_portfolio(df_pdt, df_vega, initial_equity, label):
    """Merge two strategies, compute combined metrics."""
    df_p = df_pdt.copy()
    df_v = df_vega.copy()

    # Clear attrs to avoid concat issue
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

    eq_arr = np.array(equities)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak
    max_dd = dd.min() * 100
    wr = (pnls > 0).sum() / n * 100

    print(f"  {label}:")
    print(f"    N={n} (PDT={len(df_p)}, Vega={len(df_v)})  Sh={sharpe:.2f}  "
          f"Ret={(equity/initial_equity-1)*100:.1f}%  DD={max_dd:.1f}%  "
          f"WR={wr:.1f}%  $/tr=${pnls.mean():,.0f}")

    # Bootstrap
    boot = bootstrap_sharpe(pnls, tpy)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    print(f"    Bootstrap 95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]  "
          f"P(>0)={( boot > 0).mean()*100:.0f}%  P(>1)={(boot > 1).mean()*100:.0f}%")

    return sharpe, max_dd


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
    common_base = dict(
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

    score_pdt = lambda df: df["bid_price"] * df["spot"] / df["dte"].clip(lower=1)
    score_vega = lambda df: df["vega"].abs()

    # ================================================================
    # 0. DRAWDOWN — MTM vs Trade-level
    # ================================================================
    print("\n" + "=" * 70)
    print("0. DRAWDOWN — Mark-to-Market vs Trade-Level")
    print("=" * 70)

    for name, fn in [("Premium/DTE", score_pdt), ("Top Vega", score_vega)]:
        for period in ["is", "oos"]:
            df_t = run_strat(fn, df_scores_raw, common_base, config, period)
            if df_t.empty:
                continue
            eq_trade = df_t["equity_after"].values
            dd_trade = ((eq_trade - np.maximum.accumulate(eq_trade)) / np.maximum.accumulate(eq_trade)).min() * 100
            m = compute_metrics_v2(df_t)
            print(f"  {name:>12s} {period.upper():>3s}: Trade-level={dd_trade:>6.1f}%  MTM={m['max_drawdown_pct']:>6.1f}%  Δ={m['max_drawdown_pct']-dd_trade:>+5.1f}pp")

    # ================================================================
    # 1. RAW VEGA ANALYSIS — What are the actual vega values?
    # ================================================================
    print("\n" + "=" * 70)
    print("1. RAW VEGA VALUES — Fixed sizing baseline")
    print("=" * 70)

    for name, fn in [("Premium/DTE", score_pdt), ("Top Vega", score_vega)]:
        df_is = run_strat(fn, df_scores_raw, common_base, config, "is")
        if df_is.empty or "entry_vega" not in df_is.columns:
            continue
        v = df_is["entry_vega"]
        prem = df_is["entry_price"] * df_is["entry_spot"]
        print(f"\n  {name} IS:")
        print(f"    Raw vega:    mean={v.mean():>8.1f}  std={v.std():>8.1f}  min={v.min():>8.1f}  max={v.max():>8.1f}")
        print(f"    Premium USD: mean=${prem.mean():>8,.0f}  std=${prem.std():>8,.0f}")
        print(f"    Vega/Premium: {(v/prem*100).mean():.2f}% per 1pp IV")

    # ================================================================
    # 2. VEGA-NORMALIZED SIZING — Grid search
    # ================================================================
    print("\n" + "=" * 70)
    print("2. VEGA-NORMALIZED SIZING — target = raw vega per trade")
    print("=" * 70)

    # Target raw vega values to test
    vega_targets = [5, 10, 20, 50, 100, 200]

    for name, fn in [("Premium/DTE", score_pdt), ("Top Vega", score_vega)]:
        print(f"\n  {name}:")
        print(f"    {'Target':>8s} {'IS_Sh':>6s} {'IS_Ret':>8s} {'IS_DD':>7s} {'IS_$/tr':>8s} "
              f"{'OOS_Sh':>7s} {'OOS_Ret':>8s} {'OOS_DD':>7s} {'OOS_$/tr':>9s} {'Ncontr':>7s}")
        print("    " + "-" * 86)

        for tv in vega_targets:
            df_is = run_strat(fn, df_scores_raw, common_base, config, "is",
                              sizing="vega", target_vega=tv)
            df_oos = run_strat(fn, df_scores_raw, common_base, config, "oos",
                               sizing="vega", target_vega=tv)
            if df_is.empty:
                continue
            m_is = compute_metrics_v2(df_is)
            m_oos = compute_metrics_v2(df_oos) if not df_oos.empty else {}
            avg_n = df_is["n_contracts"].mean()
            oos_str = (f"{m_oos['sharpe']:>7.2f} {m_oos['total_return_pct']:>8.1f} "
                       f"{m_oos['max_drawdown_pct']:>7.1f} {m_oos['mean_pnl']:>9,.0f}"
                       if m_oos else "     N/A")
            print(f"    {tv:>8d} {m_is['sharpe']:>6.2f} {m_is['total_return_pct']:>8.1f} "
                  f"{m_is['max_drawdown_pct']:>7.1f} {m_is['mean_pnl']:>8,.0f} "
                  f"{oos_str} {avg_n:>7.2f}")

    # ================================================================
    # 3. PREMIUM % SIZING — Grid search
    # ================================================================
    print("\n" + "=" * 70)
    print("3. PREMIUM % SIZING — premium sold = X% of equity")
    print("=" * 70)

    prem_pcts = [1, 2, 3, 5, 7, 10]

    for name, fn in [("Premium/DTE", score_pdt), ("Top Vega", score_vega)]:
        print(f"\n  {name}:")
        print(f"    {'Prem%':>6s} {'IS_Sh':>6s} {'IS_Ret':>8s} {'IS_DD':>7s} {'IS_$/tr':>8s} "
              f"{'OOS_Sh':>7s} {'OOS_Ret':>8s} {'OOS_DD':>7s} {'OOS_$/tr':>9s} {'Ncontr':>7s}")
        print("    " + "-" * 86)

        for pp in prem_pcts:
            df_is = run_strat(fn, df_scores_raw, common_base, config, "is",
                              sizing="premium_pct", target_vega=pp)
            df_oos = run_strat(fn, df_scores_raw, common_base, config, "oos",
                               sizing="premium_pct", target_vega=pp)
            if df_is.empty:
                continue
            m_is = compute_metrics_v2(df_is)
            m_oos = compute_metrics_v2(df_oos) if not df_oos.empty else {}
            avg_n = df_is["n_contracts"].mean()
            oos_str = (f"{m_oos['sharpe']:>7.2f} {m_oos['total_return_pct']:>8.1f} "
                       f"{m_oos['max_drawdown_pct']:>7.1f} {m_oos['mean_pnl']:>9,.0f}"
                       if m_oos else "     N/A")
            print(f"    {pp:>5d}% {m_is['sharpe']:>6.2f} {m_is['total_return_pct']:>8.1f} "
                  f"{m_is['max_drawdown_pct']:>7.1f} {m_is['mean_pnl']:>8,.0f} "
                  f"{oos_str} {avg_n:>7.2f}")

    # ================================================================
    # 4. BEST SIZING — Detailed comparison
    # ================================================================
    print("\n" + "=" * 70)
    print("4. COMPARISON — Fixed vs Best Vega vs Best Premium")
    print("=" * 70)

    # We'll pick reasonable targets after seeing the grids
    # For now, use intermediate values
    configs_to_test = [
        ("Fixed (1 contr)", "fixed", 0),
        ("Vega target=20", "vega", 20),
        ("Vega target=50", "vega", 50),
        ("Premium 2%", "premium_pct", 2),
        ("Premium 3%", "premium_pct", 3),
        ("Premium 5%", "premium_pct", 5),
    ]

    for name, fn in [("Premium/DTE", score_pdt), ("Top Vega", score_vega)]:
        print(f"\n  {name}:")
        for sizing_label, sizing_mode, target in configs_to_test:
            for period in ["is", "oos"]:
                df_t = run_strat(fn, df_scores_raw, common_base, config, period,
                                 sizing=sizing_mode, target_vega=target)
                label = f"{sizing_label} {period.upper()}"
                print_metrics_line(df_t, label)
            print()

    # ================================================================
    # 5. COMBINED PORTFOLIO with best sizing
    # ================================================================
    print("\n" + "=" * 70)
    print("5. COMBINED PORTFOLIO — Fixed vs Sized")
    print("=" * 70)

    initial_eq = bt["initial_equity"]

    for sizing_label, sizing_mode, target in configs_to_test:
        print(f"\n  --- {sizing_label} ---")
        for period in ["is", "oos"]:
            df_pdt = run_strat(score_pdt, df_scores_raw, common_base, config, period,
                               sizing=sizing_mode, target_vega=target)
            df_vega = run_strat(score_vega, df_scores_raw, common_base, config, period,
                                sizing=sizing_mode, target_vega=target)
            if df_pdt.empty or df_vega.empty:
                continue
            label = f"{period.upper()} Combined"
            combined_portfolio(df_pdt, df_vega, initial_eq, label)


if __name__ == "__main__":
    main()
