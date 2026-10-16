#!/usr/bin/env python3
"""
Figures for the short outreach paper (LinkedIn/prop-firm version).

Outputs (300 dpi) to /outputs/paper_figures/:
  fig1_vrp.png         - DVOL vs Parkinson RV30, VRP as filled area
  fig2_equity.png      - Full-period equity for 3 risk configs + BTC B&H,
                         IS/OOS split marked
  fig3_stoploss.png    - Per-trade worst unrealized P&L vs final P&L scatter
                         (the "why we don't use stop losses" chart)
  fig4_crossasset.png  - BTC (IS) equity vs SPY equity, same rules

Re-run any time with:
    /opt/anaconda3/bin/python3 scripts/generate_outreach_paper_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths --
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "dashboard_data"
RAW  = BASE / "Data"
SPY  = BASE / "SPY Options"
OUT  = BASE / "outputs" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ style --
C_CONS   = "#0EA5E9"  # sky       — conservative
C_MOD    = "#1D4ED8"  # blue      — moderate (primary)
C_AGG    = "#0B3B8C"  # deep blue — aggressive
C_BTC    = "#9CA3AF"  # gray
C_SPY    = "#7C3AED"  # purple
C_RED    = "#DC2626"
C_GREEN  = "#16A34A"
C_IV     = "#1D4ED8"
C_RV     = "#F59E0B"

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
    "axes.grid":         True,
    "grid.color":        "#E5E7EB",
    "grid.linewidth":    0.55,
    "grid.alpha":        1.0,
    "axes.axisbelow":    True,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  True,
    "axes.spines.bottom":True,
    "axes.edgecolor":    "#D1D5DB",
    "axes.linewidth":    0.8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.labelsize":    9,
    "axes.titlesize":    10.5,
    "axes.titleweight":  "semibold",
    "legend.fontsize":   8,
    "legend.frameon":    True,
    "legend.framealpha": 0.93,
    "legend.edgecolor":  "#E5E7EB",
})

DPI = 300
W_FULL = 5.25   # ~1575 px, fits LaTeX textwidth at 11pt / 1" margins
H_WIDE = 2.70
H_TALL = 3.30

IS_START = pd.Timestamp("2021-06-01")
IS_END   = pd.Timestamp("2023-09-30")
OOS_END  = pd.Timestamp("2024-09-30")

metrics = json.loads((DATA / "metrics.json").read_text())


# =========================================================================
# FIG 1 — VRP existence: DVOL vs Parkinson RV30
# =========================================================================
def fig1_vrp():
    print("Generating fig1_vrp.png ...")

    dvol = pd.read_csv(RAW / "dvol_btc_daily.csv", parse_dates=["date"])
    ohlc = pd.read_csv(RAW / "btc_daily_ohlc.csv", parse_dates=["timestamp"])
    ohlc = ohlc.sort_values("timestamp").reset_index(drop=True)
    ohlc = ohlc.rename(columns={"timestamp": "date"})

    # Parkinson 30d annualized (%)
    ohlc["lnhl2"] = (np.log(ohlc["high"] / ohlc["low"])) ** 2
    park_var = ohlc["lnhl2"].rolling(30).mean() / (4 * np.log(2))
    ohlc["park_rv30"] = np.sqrt(park_var * 252) * 100.0

    df = ohlc[["date", "park_rv30"]].merge(
        dvol[["date", "dvol_close"]], on="date", how="inner"
    ).dropna()
    df = df[(df["date"] >= IS_START) & (df["date"] <= OOS_END)]

    mean_gap = (df["dvol_close"] - df["park_rv30"]).mean()
    pct_pos  = float((df["dvol_close"] > df["park_rv30"]).mean()) * 100

    fig, ax = plt.subplots(figsize=(W_FULL, H_WIDE + 0.15))

    # shaded premium area
    ax.fill_between(
        df["date"], df["park_rv30"], df["dvol_close"],
        where=df["dvol_close"] >= df["park_rv30"],
        color=C_IV, alpha=0.12, linewidth=0, zorder=1,
        label="VRP (IV − RV)",
    )
    ax.fill_between(
        df["date"], df["park_rv30"], df["dvol_close"],
        where=df["dvol_close"] < df["park_rv30"],
        color=C_RED, alpha=0.14, linewidth=0, zorder=1,
    )

    ax.plot(df["date"], df["dvol_close"],
            color=C_IV, linewidth=1.35,
            label="Implied Vol (Deribit DVOL)", zorder=3)
    ax.plot(df["date"], df["park_rv30"],
            color=C_RV, linewidth=1.35,
            label="Realized Vol (Parkinson 30d)", zorder=3)

    ax.set_title(
        f"Bitcoin Volatility Risk Premium  \u2013  "
        f"IV exceeds RV on {pct_pos:.0f}% of days, avg.\u00a0gap +{mean_gap:.1f} pp"
    )
    ax.set_ylabel("Annualized Volatility (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.set_xlim(df["date"].min(), df["date"].max())
    # headroom for legend
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi + (y_hi - y_lo) * 0.12)

    ax.legend(loc="upper right", ncol=3, fontsize=7.5,
              handlelength=1.6, columnspacing=1.2)

    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_vrp.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved")


# =========================================================================
# FIG 2 — Full-period equity, 3 configs + BTC benchmark
# =========================================================================
def fig2_equity():
    print("Generating fig2_equity.png ...")

    cfgs = [
        ("cons", "Conservative  (1% risk, 2.0× buffer)",  C_CONS, 1.35),
        ("mod",  "Moderate  (2% risk, 1.5× buffer)",       C_MOD,  1.80),
        ("agg",  "Aggressive  (3% risk, 1.0× buffer)",     C_AGG,  1.55),
    ]
    btc = pd.read_csv(DATA / "btc_daily_spot.csv", parse_dates=["date"])
    btc = btc[(btc["date"] >= IS_START) & (btc["date"] <= OOS_END)].copy()
    btc["ret"] = (btc["underlying_price"] / btc["underlying_price"].iloc[0] - 1) * 100

    fig, ax = plt.subplots(figsize=(W_FULL, H_TALL))

    ax.plot(btc["date"], btc["ret"],
            color=C_BTC, linewidth=1.25, label="BTC Buy & Hold",
            zorder=2)

    for key, label, color, lw in cfgs:
        eq = pd.read_csv(DATA / f"daily_equity_{key}_full.csv", parse_dates=["date"])
        eq = eq[(eq["date"] >= IS_START) & (eq["date"] <= OOS_END)].copy()
        eq["ret"] = (eq["equity"] / eq["equity"].iloc[0] - 1) * 100
        ax.plot(eq["date"], eq["ret"],
                color=color, linewidth=lw, label=label, zorder=4)

    ax.axhline(0, color="#6B7280", linewidth=0.75, linestyle="--", alpha=0.8, zorder=1)
    ax.axvline(IS_END, color="#374151", linewidth=0.9, linestyle=":", zorder=3)
    # Expand ylim slightly so the "OOS start" label has breathing room
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi + (y_hi - y_lo) * 0.10)
    ax.text(
        IS_END + pd.Timedelta(days=8),
        ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03,
        "  IS  \u2502  OOS",
        fontsize=7.2, color="#374151", style="italic", va="bottom", ha="left",
    )

    ax.set_title("Strategy Equity vs BTC Buy & Hold (Full Period, Jun 2021 – Sep 2024)")
    ax.set_ylabel("Cumulative Return (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.set_xlim(IS_START, OOS_END)
    ax.legend(loc="upper left", ncol=1)

    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_equity.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved")


# =========================================================================
# FIG 3 — Stop-loss insight: worst unrealized vs final P&L
# =========================================================================
def fig3_stoploss():
    print("Generating fig3_stoploss.png ...")

    sl = pd.read_csv(RAW / "stoploss_exploratory_is.csv")
    winners = sl[sl["winner"] == "W"]
    losers  = sl[sl["winner"] == "L"]

    fig, ax = plt.subplots(figsize=(W_FULL, H_WIDE + 0.25))

    # Shaded "stop-loss kill zones"
    ax.axvspan(-16, -5, facecolor=C_RED, alpha=0.08, zorder=0)
    ax.axvspan(-5,  -2, facecolor=C_RED, alpha=0.04, zorder=0)
    ax.axvline(-5, color=C_RED, linewidth=0.9, linestyle="--", alpha=0.7, zorder=1)
    ax.axvline(-2, color=C_RED, linewidth=0.9, linestyle=":",  alpha=0.7, zorder=1)

    ax.axhline(0, color="#374151", linewidth=0.7, zorder=2)

    ax.scatter(winners["worst_unreal_%"], winners["final_pnl_%"],
               s=42, color=C_GREEN, edgecolor="white", linewidth=0.9,
               zorder=4, label=f"Winners  (N={len(winners)})")
    ax.scatter(losers["worst_unreal_%"], losers["final_pnl_%"],
               s=42, color=C_RED,   edgecolor="white", linewidth=0.9,
               zorder=4, label=f"Losers    (N={len(losers)})")

    # Annotate the two most extreme winners-that-recovered
    extreme = winners.nsmallest(2, "worst_unreal_%")
    for _, r in extreme.iterrows():
        ax.annotate(
            f"+{r['final_pnl_%']:.1f}%",
            xy=(r["worst_unreal_%"], r["final_pnl_%"]),
            xytext=(r["worst_unreal_%"] + 1.8, r["final_pnl_%"] + 0.9),
            fontsize=7, color="#111827",
            arrowprops=dict(arrowstyle="-", color="#6B7280", lw=0.6, alpha=0.9),
        )

    killed_n = int((winners["worst_unreal_%"] < -2).sum())
    killed_pnl = float(winners.loc[winners["worst_unreal_%"] < -2, "final_pnl_$"].sum())
    ax.text(
        0.015, 0.95,
        f"A −2% stop would have killed {killed_n} winners worth ${killed_pnl:,.0f}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=7.8, color="#111827", fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                  edgecolor="#E5E7EB", alpha=0.95, linewidth=0.7),
    )
    ax.text(
        -5.1, ax.get_ylim()[0] + 0.3, " −5% stop",
        fontsize=6.8, color=C_RED, ha="right", va="bottom",
    )
    ax.text(
        -2.05, ax.get_ylim()[0] + 0.3, " −2% stop",
        fontsize=6.8, color=C_RED, ha="right", va="bottom",
    )

    ax.set_title("Why No Stop Loss: Intraday Drawdown vs Final Trade P&L (IS, 35 trades)")
    ax.set_xlabel("Worst Intraday Unrealized P&L  (% of equity)")
    ax.set_ylabel("Final Trade P&L  (% of equity)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT / "fig3_stoploss.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved")


# =========================================================================
# FIG 4 — Cross-asset: BTC (IS) vs SPY (full), same rules
# =========================================================================
def fig4_crossasset():
    print("Generating fig4_crossasset.png ...")

    eq_btc = pd.read_csv(DATA / "daily_equity_vega_is.csv", parse_dates=["date"])
    eq_spy = pd.read_csv(SPY  / "spy_daily_equity.csv",     parse_dates=["date"])

    eq_btc = eq_btc[(eq_btc["date"] >= IS_START) & (eq_btc["date"] <= IS_END)].copy()
    eq_btc["ret"] = (eq_btc["equity"] / eq_btc["equity"].iloc[0] - 1) * 100
    eq_spy["ret"] = (eq_spy["equity"] / eq_spy["equity"].iloc[0] - 1) * 100

    btc_sharpe = metrics["metrics"]["vega_is"]["sharpe"]
    btc_n      = metrics["metrics"]["vega_is"]["n_trades"]
    spy_sharpe = 0.79
    spy_n      = 147

    fig, (ax_b, ax_s) = plt.subplots(
        1, 2, figsize=(W_FULL, H_WIDE + 0.25),
        gridspec_kw={"wspace": 0.34},
    )

    def panel(ax, dates, rets, color, title, sharpe, n, period, year_interval):
        ax.plot(dates, rets, color=color, linewidth=1.5, zorder=3)
        ax.axhline(0, color="#6B7280", linewidth=0.6, linestyle="--", alpha=0.8)
        ax.fill_between(dates, rets, 0, where=(rets >= 0),
                        color=color, alpha=0.08, zorder=1)
        ax.fill_between(dates, rets, 0, where=(rets < 0),
                        color=C_RED, alpha=0.08, zorder=1)
        ax.set_title(title, pad=6)
        ax.set_ylabel("Cumulative Return (%)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("'%y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(year_interval))
        # headroom for stat-box
        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(y_lo - (y_hi - y_lo) * 0.18, y_hi)
        info = f"Sharpe = {sharpe:.2f}   |   N = {n} trades\n{period}"
        ax.text(0.035, 0.035, info,
                transform=ax.transAxes, va="bottom", ha="left",
                fontsize=7.2, color="#374151", linespacing=1.45,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#E5E7EB", alpha=0.92, linewidth=0.7))

    panel(ax_b,
          eq_btc["date"].values, eq_btc["ret"].values,
          C_MOD, "BTC / Deribit",
          btc_sharpe, btc_n, "Jun 2021 – Sep 2023  |  IS window",
          year_interval=1)
    panel(ax_s,
          eq_spy["date"].values, eq_spy["ret"].values,
          C_SPY, "SPY / OptionsDX",
          spy_sharpe, spy_n, "Jan 2010 – Dec 2023  |  14-year history",
          year_interval=3)

    ax_s.set_ylabel("")

    fig.suptitle("Cross-Asset Validation: Identical Parameters, Different Market",
                 fontsize=10.5, fontweight="semibold", y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig4_crossasset.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved")


# =========================================================================
if __name__ == "__main__":
    fig1_vrp()
    fig2_equity()
    fig3_stoploss()
    fig4_crossasset()
    print(f"\nAll figures saved to: {OUT}")
