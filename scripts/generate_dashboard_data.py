"""
Pre-generate all dashboard data — run once, then dashboard loads instantly.

Strategy: Top Vega (VRP Selling) — single-position, capital-based vega sizing.
Three risk configurations: Conservative, Moderate, Aggressive.

Usage:
    python scripts/generate_dashboard_data.py
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_config, load_options, load_funding_rates
from src.backtest_engine_v2 import (
    run_backtest_v2, compute_metrics_v2, load_precomputed_scores,
    _build_option_lookup, _build_quarterly_lookup,
)

SCORES_PATH = "Data/scores_v2_dte3.csv"
FUTURES_PATH = "Data/deribit_futures_daily.csv"
OUT_DIR = Path("dashboard_data")
N_BOOTSTRAP = 10_000

# Strategy parameters (final research)
EXIT_DTE = 5
PROFIT_TARGET = 0.50
BASE_VEGA_TARGET = 50            # Base backtest uses fixed vega=50
MAX_ABS_DELTA = 0.50
OPTION_FEE_PCT = 0.0003
WORST_LOSS_PER_VEGA = 54.93     # From IS worst trade analysis

# Risk configurations (post-hoc rescaling from base vega=50)
RISK_CONFIGS = {
    "cons": {"label": "Conservativa", "risk_per_trade": 0.01, "buffer": 2.0},
    "mod":  {"label": "Moderata",     "risk_per_trade": 0.02, "buffer": 1.5},
    "agg":  {"label": "Aggressiva",   "risk_per_trade": 0.03, "buffer": 1.0},
}
INITIAL_EQUITY = 100_000


# ── Post-hoc rescaling ──────────────────────────────────────

def _rescale_trades(df_trades_base, daily_eq_base, risk, buffer,
                    initial_equity=INITIAL_EQUITY):
    """
    Post-hoc rescale trade PnLs and daily equity for capital-based sizing.

    vega_target = (current_equity * risk) / (WORST_LOSS_PER_VEGA * buffer)
    scale_factor = vega_target / BASE_VEGA_TARGET
    """
    df = df_trades_base.copy()
    deq = daily_eq_base.copy()

    # Rescale trade PnLs
    equity = initial_equity
    scale_factors = []
    rescaled_pnls = []

    for idx in df.index:
        vega_target = (equity * risk) / (WORST_LOSS_PER_VEGA * buffer)
        sf = vega_target / BASE_VEGA_TARGET
        rpnl = df.loc[idx, "total_pnl"] * sf
        scale_factors.append(sf)
        rescaled_pnls.append(rpnl)
        equity += rpnl

    df["_scale_factor"] = scale_factors
    df["total_pnl"] = rescaled_pnls

    # Rescale component PnLs
    for col in ["option_pnl", "hedge_pnl_total", "hedge_txn_cost_total",
                "option_txn_cost_total", "tail_hedge_cost_total"]:
        if col in df.columns:
            df[col] = df_trades_base[col] * df["_scale_factor"]

    # Recompute return_pct and equity_after
    eq = initial_equity
    ret_pcts = []
    eq_afters = []
    for idx in df.index:
        ret_pcts.append(df.loc[idx, "total_pnl"] / eq * 100)
        eq += df.loc[idx, "total_pnl"]
        eq_afters.append(eq)
    df["return_pct"] = ret_pcts
    df["equity_after"] = eq_afters
    df["equity_before"] = [initial_equity] + eq_afters[:-1]

    # Rescale n_contracts for display
    df["n_contracts"] = df_trades_base["n_contracts"] * df["_scale_factor"]

    # Rescale daily equity
    # Build trade exit map: date -> (orig_pnl, rescaled_pnl)
    trade_exits = {}
    for i, idx in enumerate(df.index):
        exit_d = pd.Timestamp(df.loc[idx, "exit_date"]).normalize()
        if exit_d not in trade_exits:
            trade_exits[exit_d] = []
        trade_exits[exit_d].append((
            df_trades_base.loc[idx, "total_pnl"],
            rescaled_pnls[i],
        ))

    # Build day -> scale_factor map for unrealized MTM
    day_sf = {}
    for i, idx in enumerate(df.index):
        entry = pd.Timestamp(df.loc[idx, "entry_date"]).normalize()
        exit_d = pd.Timestamp(df.loc[idx, "exit_date"]).normalize()
        for d in pd.date_range(entry, exit_d):
            day_sf[d.normalize()] = scale_factors[i]

    # Reconstruct daily equity
    orig_cash = initial_equity
    rescaled_cash = initial_equity
    orig_eq_vals = deq["equity"].values
    dates = deq["date"].values

    orig_cash_arr = np.empty(len(deq))
    rescaled_cash_arr = np.empty(len(deq))
    oc = initial_equity
    rc = initial_equity
    for j in range(len(deq)):
        d = pd.Timestamp(dates[j]).normalize()
        if d in trade_exits:
            for op, rp in trade_exits[d]:
                oc += op
                rc += rp
            trade_exits[d] = []  # consume
        orig_cash_arr[j] = oc
        rescaled_cash_arr[j] = rc

    orig_unrealized = orig_eq_vals - orig_cash_arr
    sf_arr = np.array([day_sf.get(pd.Timestamp(d).normalize(),
                                   scale_factors[0] if scale_factors else 1.0)
                       for d in dates])
    rescaled_unrealized = orig_unrealized * sf_arr
    deq_out = deq.copy()
    deq_out["equity"] = rescaled_cash_arr + rescaled_unrealized

    df = df.drop(columns=["_scale_factor"], errors="ignore")
    return df, deq_out


# ── Main ─────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(exist_ok=True)
    config = load_config()

    print("Loading data...", flush=True)
    df_scores_raw = load_precomputed_scores(SCORES_PATH)
    df_options = load_options(config)
    df_funding = load_funding_rates(config)

    df_futures = pd.read_csv(FUTURES_PATH)
    df_futures["date"] = pd.to_datetime(df_futures["date"])
    df_futures["expiry"] = pd.to_datetime(df_futures["expiry"])

    # BTC spot daily
    df_opt_temp = df_options.copy()
    df_opt_temp["date"] = pd.to_datetime(df_opt_temp["date"])
    btc_daily = df_opt_temp.groupby("date")["underlying_price"].first().sort_index()
    btc_daily.to_csv(OUT_DIR / "btc_daily_spot.csv")

    print("Building lookups...", flush=True)
    option_lookup = _build_option_lookup(df_options)
    qtr_price_lookup, qtr_avail_lookup = _build_quarterly_lookup(df_futures)

    bt = config["backtest"]
    common_base = dict(
        df_options=df_options, df_funding=df_funding, df_futures=df_futures,
        exit_dte=EXIT_DTE, profit_target_pct=PROFIT_TARGET,
        initial_equity=INITIAL_EQUITY,
        tail_strike_pct=bt["tail_strike_pct"],
        gap_anomaly_z=bt["gap_anomaly_z_threshold"],
        max_abs_delta=MAX_ABS_DELTA, option_fee_pct=OPTION_FEE_PCT,
        verbose=False, max_parallel_positions=1,
        _option_lookup=option_lookup,
        _quarterly_price_lookup=qtr_price_lookup,
        _quarterly_avail_lookup=qtr_avail_lookup,
    )

    score_vega = lambda df: df["vega"].abs()

    # ─── Step 1: Run BASE backtest (vega=50 fixed) ───────────
    all_metrics = {}
    base_results = {}   # {period: df_trades}
    base_daily = {}     # {period: df_daily_equity}

    for period in ["is", "oos"]:
        split = config["split"]
        start = split["is_start"] if period == "is" else split["oos_start"]
        end = split["is_end"] if period == "is" else split["oos_end"]

        print(f"  Running base vega=50 {period.upper()}...", flush=True)
        df_scores = df_scores_raw.copy()
        df_scores["score"] = score_vega(df_scores)

        df_t = run_backtest_v2(
            df_scores=df_scores, start_date=start, end_date=end,
            sizing="vega", target_vega_usd=BASE_VEGA_TARGET,
            **common_base,
        )
        daily_eq = df_t.attrs.get("daily_equity", pd.DataFrame())
        base_results[period] = df_t
        base_daily[period] = daily_eq

        # Save base trades/equity (used for statistical validation)
        df_save = df_t.copy()
        df_save.attrs = {}
        df_save.to_csv(OUT_DIR / f"trades_vega_{period}.csv", index=False)
        if not daily_eq.empty:
            daily_eq.to_csv(OUT_DIR / f"daily_equity_vega_{period}.csv", index=False)

        # Base metrics
        m = compute_metrics_v2(df_t)
        m["key"] = f"vega_{period}"
        if not df_t.empty:
            eq_arr = df_t["equity_after"].values
            m["trade_level_dd"] = ((eq_arr - np.maximum.accumulate(eq_arr)) /
                                   np.maximum.accumulate(eq_arr)).min() * 100
        all_metrics[f"vega_{period}"] = m

    # ─── Step 2: Post-hoc rescale for 3 risk configs ─────────
    for ckey, cfg in RISK_CONFIGS.items():
        risk = cfg["risk_per_trade"]
        buff = cfg["buffer"]
        vega1 = (INITIAL_EQUITY * risk) / (WORST_LOSS_PER_VEGA * buff)
        print(f"  Rescaling {ckey} (risk={risk*100:.0f}%, buff={buff}, vega1={vega1:.1f})...",
              flush=True)

        for period in ["is", "oos"]:
            key = f"{ckey}_{period}"
            df_base = base_results[period]
            deq_base = base_daily[period]

            if df_base.empty:
                continue

            df_resc, deq_resc = _rescale_trades(
                df_base, deq_base, risk, buff, INITIAL_EQUITY)

            # Save
            df_save = df_resc.copy()
            df_save.attrs = {}
            df_save.to_csv(OUT_DIR / f"trades_{key}.csv", index=False)
            if not deq_resc.empty:
                deq_resc.to_csv(OUT_DIR / f"daily_equity_{key}.csv", index=False)

            # Metrics
            n = len(df_resc)
            pnls = df_resc["total_pnl"].values
            rets = df_resc["return_pct"].values
            td = (pd.to_datetime(df_resc["exit_date"]).max() -
                  pd.to_datetime(df_resc["entry_date"]).min()).days
            tpy = n / (td / 365) if td > 0 else 12
            sharpe = rets.mean() / rets.std() * np.sqrt(tpy) if rets.std() > 0 else 0
            final_eq = df_resc["equity_after"].iloc[-1]

            # MTM drawdown from daily equity
            if not deq_resc.empty:
                eq_vals = deq_resc["equity"].values
                peak = np.maximum.accumulate(eq_vals)
                dd = ((eq_vals - peak) / peak).min() * 100
            else:
                dd = 0

            all_metrics[key] = {
                "n_trades": n,
                "sharpe": float(sharpe),
                "total_return_pct": float((final_eq / INITIAL_EQUITY - 1) * 100),
                "max_drawdown_pct": float(dd),
                "win_rate": float((pnls > 0).sum() / n * 100),
                "mean_pnl": float(pnls.mean()),
                "trades_per_year": float(tpy),
                "initial_equity": INITIAL_EQUITY,
                "final_equity": float(final_eq),
                "key": key,
                "vega_initial": float(vega1),
                "risk_per_trade": risk,
                "buffer_multiplier": buff,
                "option_pnl_total": float(df_resc["option_pnl"].sum()) if "option_pnl" in df_resc else 0,
                "hedge_pnl_total": float(df_resc["hedge_pnl_total"].sum()) if "hedge_pnl_total" in df_resc else 0,
                "hedge_txn_cost_total": float(df_resc["hedge_txn_cost_total"].sum()) if "hedge_txn_cost_total" in df_resc else 0,
                "option_txn_cost_total": float(df_resc["option_txn_cost_total"].sum()) if "option_txn_cost_total" in df_resc else 0,
                "tail_hedge_cost_total": float(df_resc["tail_hedge_cost_total"].sum()) if "tail_hedge_cost_total" in df_resc else 0,
            }

    # ─── Step 3: Build full period (IS + OOS stitched) ───────
    print("  Building full period...", flush=True)
    for prefix in ["vega", "cons", "mod", "agg"]:
        is_key = f"{prefix}_is"
        oos_key = f"{prefix}_oos"
        is_path = OUT_DIR / f"trades_{is_key}.csv"
        oos_path = OUT_DIR / f"trades_{oos_key}.csv"
        if not is_path.exists() or not oos_path.exists():
            continue

        df_is = pd.read_csv(is_path)
        df_oos = pd.read_csv(oos_path)
        df_full = pd.concat([df_is, df_oos], ignore_index=True)
        df_full.to_csv(OUT_DIR / f"trades_{prefix}_full.csv", index=False)

        deq_is_path = OUT_DIR / f"daily_equity_{is_key}.csv"
        deq_oos_path = OUT_DIR / f"daily_equity_{oos_key}.csv"
        if deq_is_path.exists() and deq_oos_path.exists():
            d_is = pd.read_csv(deq_is_path, parse_dates=["date"])
            d_oos = pd.read_csv(deq_oos_path, parse_dates=["date"])
            final_is = d_is["equity"].iloc[-1]
            oos_start = d_oos["equity"].iloc[0]
            d_oos["equity"] += (final_is - oos_start)
            d_full = pd.concat([d_is, d_oos], ignore_index=True)
            d_full.to_csv(OUT_DIR / f"daily_equity_{prefix}_full.csv", index=False)

    # ─── Step 4: Statistical validation (on BASE vega=50) ────
    # These stats don't change with sizing — computed once on base.

    # Bootstrap Sharpe CI
    print("  Computing bootstrap CIs...", flush=True)
    bootstrap_results = {}
    rng = np.random.default_rng(42)

    for key in ["vega_is", "vega_oos"]:
        df_t = base_results[key.split("_")[1]]
        if df_t.empty or len(df_t) < 5:
            continue
        m = compute_metrics_v2(df_t)
        pnls = df_t["total_pnl"].values
        tpy = m["trades_per_year"]
        n = len(pnls)

        boot_sharpes = []
        for _ in range(N_BOOTSTRAP):
            sample = rng.choice(pnls, size=n, replace=True)
            if sample.std() > 0:
                boot_sharpes.append(sample.mean() / sample.std() * np.sqrt(tpy))
        boot_sharpes = np.array(boot_sharpes)

        ci_lo, ci_hi = np.percentile(boot_sharpes, [2.5, 97.5])
        bootstrap_results[key] = {
            "sharpe": m["sharpe"],
            "ci_95_lo": ci_lo, "ci_95_hi": ci_hi,
            "p_gt_0": float((boot_sharpes > 0).mean()),
            "p_gt_1": float((boot_sharpes > 1).mean()),
            "p_gt_2": float((boot_sharpes > 2).mean()),
        }
        np.save(OUT_DIR / f"bootstrap_{key}.npy", boot_sharpes)

    # Regime analysis (volatility)
    print("  Computing regime analysis...", flush=True)
    log_rets = np.log(btc_daily / btc_daily.shift(1)).dropna()
    rv30 = log_rets.rolling(30).std() * np.sqrt(365) * 100
    rv_dict = rv30.to_dict()

    regime_results = {}
    for key in ["vega_is", "vega_oos"]:
        df_t = base_results[key.split("_")[1]].copy()
        if df_t.empty:
            continue
        df_t["entry_date"] = pd.to_datetime(df_t["entry_date"])
        df_t["rv30"] = df_t["entry_date"].map(lambda d: rv_dict.get(d, np.nan))
        df_t = df_t.dropna(subset=["rv30"])
        if len(df_t) < 5:
            continue

        q33 = df_t["rv30"].quantile(0.33)
        q67 = df_t["rv30"].quantile(0.67)
        regimes = {
            "Low": df_t[df_t["rv30"] < q33],
            "Mid": df_t[(df_t["rv30"] >= q33) & (df_t["rv30"] < q67)],
            "High": df_t[df_t["rv30"] >= q67],
        }
        regime_data = {"thresholds": [float(q33), float(q67)]}
        for rname, df_r in regimes.items():
            n = len(df_r)
            if n < 2:
                regime_data[rname] = {"n": n}
                continue
            wr = (df_r["total_pnl"] > 0).sum() / n * 100
            regime_data[rname] = {
                "n": n, "win_rate": round(wr, 1),
                "mean_pnl": round(float(df_r["total_pnl"].mean()), 0),
                "total_pnl": round(float(df_r["total_pnl"].sum()), 0),
            }
        regime_results[key] = regime_data

    # Outlier dependence
    print("  Computing outlier dependence...", flush=True)
    outlier_results = {}
    for key in ["vega_is", "vega_oos"]:
        df_t = base_results[key.split("_")[1]]
        if df_t.empty:
            continue
        pnls = df_t["total_pnl"].values
        rets = df_t["return_pct"].values
        n = len(pnls)
        td = (pd.to_datetime(df_t["exit_date"]).max() -
              pd.to_datetime(df_t["entry_date"]).min()).days
        tpy = n / (td / 365) if td > 0 else 12

        rows = []
        for remove in [0, 1, 2, 3, 5]:
            if remove >= n:
                break
            idx = np.argsort(pnls)[::-1][remove:]
            sub_p = pnls[idx]
            sub_r = rets[idx]
            sh = sub_r.mean() / sub_r.std() * np.sqrt(tpy) if sub_r.std() > 0 else 0
            rows.append({
                "removed": remove, "n": len(sub_p),
                "mean_pnl": round(float(sub_p.mean()), 0),
                "sharpe": round(sh, 2),
                "win_rate": round((sub_p > 0).sum() / len(sub_p) * 100, 1),
            })
        outlier_results[key] = rows

    # Cost stress test
    print("  Computing cost stress test...", flush=True)
    cost_results = {}
    for key in ["vega_is", "vega_oos"]:
        df_t = base_results[key.split("_")[1]]
        if df_t.empty:
            continue
        pnls = df_t["total_pnl"].values
        costs = df_t["hedge_txn_cost_total"].values
        rets = df_t["return_pct"].values
        n = len(pnls)
        td = (pd.to_datetime(df_t["exit_date"]).max() -
              pd.to_datetime(df_t["entry_date"]).min()).days
        tpy = n / (td / 365) if td > 0 else 12

        rows = []
        for mult in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
            extra = costs * (mult - 1)
            adj_p = pnls - extra
            adj_r = rets - (extra / df_t["equity_before"].values * 100)
            sh = adj_r.mean() / adj_r.std() * np.sqrt(tpy) if adj_r.std() > 0 else 0
            rows.append({
                "multiplier": mult,
                "mean_pnl": round(float(adj_p.mean()), 0),
                "sharpe": round(sh, 2),
            })
        cost_results[key] = rows

    # T-tests
    print("  Computing t-tests...", flush=True)
    ttest_results = {}
    for key in ["vega_is", "vega_oos"]:
        df_t = base_results[key.split("_")[1]]
        if df_t.empty or len(df_t) < 3:
            continue
        pnls = df_t["total_pnl"].values
        t_stat, p_val = stats.ttest_1samp(pnls, 0)
        p_one = p_val / 2 if t_stat > 0 else 1.0
        ttest_results[key] = {
            "n": len(pnls),
            "mean_pnl": round(float(np.mean(pnls)), 0),
            "std_pnl": round(float(np.std(pnls, ddof=1)), 0),
            "t_stat": round(float(t_stat), 4),
            "p_two_sided": round(float(p_val), 6),
            "p_one_sided": round(float(p_one), 6),
            "significant_10pct": p_one < 0.10,
            "significant_5pct": p_one < 0.05,
            "significant_1pct": p_one < 0.01,
        }

    # Trend regime (60d BTC return)
    print("  Computing trend regimes...", flush=True)
    ret60 = btc_daily.pct_change(60)
    ret60_dict = ret60.to_dict()

    trend_results = {}
    for key in ["vega_is", "vega_oos"]:
        df_t = base_results[key.split("_")[1]].copy()
        if df_t.empty:
            continue
        df_t["entry_date"] = pd.to_datetime(df_t["entry_date"])
        df_t["ret60"] = df_t["entry_date"].map(lambda d: ret60_dict.get(d, np.nan))
        df_t = df_t.dropna(subset=["ret60"])
        if len(df_t) < 3:
            continue

        regimes_trend = {
            "Bull (>+10%)": df_t[df_t["ret60"] > 0.10],
            "Sideways": df_t[(df_t["ret60"] >= -0.10) & (df_t["ret60"] <= 0.10)],
            "Bear (<-10%)": df_t[df_t["ret60"] < -0.10],
        }
        trend_data = {}
        for rname, df_r in regimes_trend.items():
            n = len(df_r)
            if n < 2:
                trend_data[rname] = {"n": n}
                continue
            wr = (df_r["total_pnl"] > 0).sum() / n * 100
            trend_data[rname] = {
                "n": n, "win_rate": round(wr, 1),
                "mean_pnl": round(float(df_r["total_pnl"].mean()), 0),
                "total_pnl": round(float(df_r["total_pnl"].sum()), 0),
            }
        trend_results[key] = trend_data

    # Call/Put breakdown
    print("  Computing call/put breakdown...", flush=True)
    callput_results = {}
    for key in ["vega_is", "vega_oos"]:
        df_t = base_results[key.split("_")[1]]
        if df_t.empty or "option_type" not in df_t.columns:
            continue
        cp_data = {}
        for otype in ["call", "put"]:
            sub = df_t[df_t["option_type"] == otype]
            n = len(sub)
            if n < 1:
                cp_data[otype] = {"n": 0}
                continue
            wr = (sub["total_pnl"] > 0).sum() / n * 100
            cp_data[otype] = {
                "n": n, "win_rate": round(wr, 1),
                "mean_pnl": round(float(sub["total_pnl"].mean()), 0),
                "total_pnl": round(float(sub["total_pnl"].sum()), 0),
                "mean_entry_dte": round(float(sub["entry_dte"].mean()), 1),
            }
        callput_results[key] = cp_data

    # Sensitivity (from existing file)
    sens_path = Path("Data/sensitivity_v2_IS.csv")
    if sens_path.exists():
        import shutil
        shutil.copy(sens_path, OUT_DIR / "sensitivity.csv")

    # ─── Save all metrics to JSON ────────────────────────────
    print("  Saving metrics...", flush=True)

    for k, m in all_metrics.items():
        for mk, mv in m.items():
            if isinstance(mv, (np.integer, np.int64)):
                m[mk] = int(mv)
            elif isinstance(mv, (np.floating, np.float64)):
                m[mk] = float(mv)
            elif isinstance(mv, dict):
                m[mk] = {str(kk): int(vv) if isinstance(vv, (np.integer,)) else vv
                          for kk, vv in mv.items()}

    boot_for_json = {k: v for k, v in bootstrap_results.items()}

    dashboard_json = {
        "metrics": all_metrics,
        "bootstrap": boot_for_json,
        "ttest": ttest_results,
        "regimes": regime_results,
        "trend_regimes": trend_results,
        "callput": callput_results,
        "outliers": outlier_results,
        "cost_stress": cost_results,
        "risk_configs": {
            k: {
                "label": v["label"],
                "risk_per_trade": v["risk_per_trade"],
                "buffer": v["buffer"],
                "vega_initial": round((INITIAL_EQUITY * v["risk_per_trade"]) /
                                      (WORST_LOSS_PER_VEGA * v["buffer"]), 1),
            }
            for k, v in RISK_CONFIGS.items()
        },
        "config": {
            "exit_dte": EXIT_DTE,
            "profit_target": PROFIT_TARGET,
            "base_vega_target": BASE_VEGA_TARGET,
            "worst_loss_per_vega": WORST_LOSS_PER_VEGA,
            "initial_equity": INITIAL_EQUITY,
            "is_start": config["split"]["is_start"],
            "is_end": config["split"]["is_end"],
            "oos_start": config["split"]["oos_start"],
            "oos_end": config["split"]["oos_end"],
        },
    }

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(dashboard_json, f, indent=2, default=str)

    print(f"\nDone! Dashboard data saved to {OUT_DIR}/")
    print(f"Files: {len(list(OUT_DIR.iterdir()))}")
    for p in sorted(OUT_DIR.iterdir()):
        size = p.stat().st_size
        if size > 1_000_000:
            print(f"  {p.name}: {size/1_000_000:.1f} MB")
        elif size > 1_000:
            print(f"  {p.name}: {size/1_000:.0f} KB")
        else:
            print(f"  {p.name}: {size} B")


if __name__ == "__main__":
    main()
