#!/usr/bin/env python3
"""
Generate publication-quality figures for the VRP paper.

Outputs (300 dpi, ~1400 px wide each):
  fig1_equity.png     — IS equity curve vs BTC benchmark (moderate config)
  fig2_trades.png     — Per-trade P&L barchart, 35 trades
  fig3_bootstrap.png  — Bootstrap Sharpe distribution
  fig4_crossasset.png — BTC vs SPY cross-asset validation (side-by-side)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "dashboard_data"
SPY  = BASE / "SPY Options"
OUT  = BASE / "outputs" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette & global rcParams ───────────────────────────────────────────────
C_BLUE   = "#1D4ED8"   # strategy / BTC
C_GRAY   = "#9CA3AF"   # BTC benchmark (light gray)
C_RED    = "#DC2626"   # losses / threshold
C_GREEN  = "#16A34A"   # profits
C_PURPLE = "#7C3AED"   # SPY

plt.rcParams.update({
    "font.family":            "sans-serif",
    "font.sans-serif":        ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.facecolor":         "white",
    "figure.facecolor":       "white",
    "axes.grid":              True,
    "grid.color":             "#E5E7EB",
    "grid.linewidth":         0.55,
    "grid.alpha":             1.0,
    "axes.axisbelow":         True,
    "axes.spines.top":        False,
    "axes.spines.right":      False,
    "axes.spines.left":       True,
    "axes.spines.bottom":     True,
    "axes.edgecolor":         "#D1D5DB",
    "axes.linewidth":         0.8,
    "xtick.labelsize":        8,
    "ytick.labelsize":        8,
    "axes.labelsize":         9,
    "axes.titlesize":         10.5,
    "axes.titleweight":       "semibold",
    "legend.fontsize":        8,
    "legend.frameon":         True,
    "legend.framealpha":      0.9,
    "legend.edgecolor":       "#E5E7EB",
})

DPI    = 300
W1     = 1400 / DPI   # ≈ 4.67 in  — single-panel width
H1     = W1 * 0.68    # ≈ 3.17 in  — single-panel height

IS_START = pd.Timestamp("2021-06-01")
IS_END   = pd.Timestamp("2023-09-30")

# ── Load shared data ────────────────────────────────────────────────────────
with open(DATA / "metrics.json") as f:
    metrics = json.load(f)

# =============================================================================
# FIG 1 — IS Equity Curve vs BTC Buy & Hold
# =============================================================================
print("Generating fig1_equity.png …")

eq_mod = pd.read_csv(DATA / "daily_equity_mod_is.csv", parse_dates=["date"])
btc    = pd.read_csv(DATA / "btc_daily_spot.csv",      parse_dates=["date"])

eq_mod = eq_mod[(eq_mod["date"] >= IS_START) & (eq_mod["date"] <= IS_END)].copy()
btc_is = btc[   (btc["date"]    >= IS_START) & (btc["date"]    <= IS_END)].copy()

eq_mod["ret"] = (eq_mod["equity"]             / eq_mod["equity"].iloc[0]             - 1) * 100
btc_is["ret"] = (btc_is["underlying_price"] / btc_is["underlying_price"].iloc[0] - 1) * 100

fig1, ax1 = plt.subplots(figsize=(W1 * 1.45, H1))

ax1.plot(btc_is["date"], btc_is["ret"],
         color=C_GRAY,  linewidth=1.3, label="BTC Buy & Hold",           zorder=2)
ax1.plot(eq_mod["date"], eq_mod["ret"],
         color=C_BLUE,  linewidth=1.7, label="VRP Strategy (Moderate)",  zorder=3)
ax1.axhline(0, color="#6B7280", linewidth=0.75, linestyle="--", zorder=1, alpha=0.8)

ax1.set_title("Strategy vs BTC Buy & Hold (IS Period)")
ax1.set_ylabel("Cumulative Return (%)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
ax1.legend(loc="upper left")
fig1.autofmt_xdate(rotation=0, ha="center")
fig1.tight_layout()
fig1.savefig(OUT / "fig1_equity.png", dpi=DPI, bbox_inches="tight")
plt.close(fig1)
print("  ✓ saved")


# =============================================================================
# FIG 2 — Per-Trade P&L Barchart (35 trades, vega IS)
# =============================================================================
print("Generating fig2_trades.png …")

trades = pd.read_csv(DATA / "trades_vega_is.csv", parse_dates=["entry_date", "exit_date"])
trades = trades.sort_values("entry_date").reset_index(drop=True)
trades["trade_num"] = range(1, len(trades) + 1)

colors = [C_GREEN if r >= 0 else C_RED for r in trades["return_pct"]]

# Identify 2022 bear market trades (entry_date within 2022)
bear_mask = (trades["entry_date"] >= "2022-01-01") & (trades["entry_date"] <= "2022-12-31")
bear_nums = trades.loc[bear_mask, "trade_num"].values

fig2, ax2 = plt.subplots(figsize=(W1 * 1.9, H1))

ax2.bar(trades["trade_num"], trades["return_pct"],
        color=colors, width=0.72, zorder=3, linewidth=0)
ax2.axhline(0, color="#374151", linewidth=0.7, zorder=4)

# 2022 bear market shaded region + label
if len(bear_nums) > 0:
    x0 = float(bear_nums[0])  - 0.55
    x1 = float(bear_nums[-1]) + 0.55

    ax2.axvspan(x0, x1, alpha=0.10, color=C_RED, zorder=1)

    # Expand y-axis headroom so the bracket sits inside the plot area
    y_lo, y_hi = ax2.get_ylim()
    y_data_max = trades["return_pct"].max()
    headroom   = (y_hi - y_lo) * 0.18          # 18 % of current range
    new_y_hi   = max(y_hi, y_data_max + headroom)
    ax2.set_ylim(y_lo, new_y_hi)

    bracket_y = y_data_max + headroom * 0.35    # sits comfortably below top
    text_y    = bracket_y + (new_y_hi - y_lo) * 0.03

    # Horizontal bracket line (stays inside axes)
    ax2.annotate("",
                 xy=(x1, bracket_y), xytext=(x0, bracket_y),
                 arrowprops=dict(arrowstyle="-", color=C_RED, lw=1.1),
                 annotation_clip=False)
    ax2.text((x0 + x1) / 2, text_y, "2022 Bear Market",
             ha="center", va="bottom", clip_on=False,
             color=C_RED, fontsize=7.5, fontweight="semibold")

ax2.set_title("Per-Trade P&L — 35 Trades")
ax2.set_xlabel("Trade #")
ax2.set_ylabel("Return on Equity (%)")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
ax2.set_xticks(trades["trade_num"])
ax2.tick_params(axis="x", labelsize=6.5)
ax2.set_xlim(0.3, len(trades) + 0.7)

fig2.tight_layout()
fig2.savefig(OUT / "fig2_trades.png", dpi=DPI, bbox_inches="tight")
plt.close(fig2)
print("  ✓ saved")


# =============================================================================
# FIG 3 — Bootstrap Sharpe Distribution
# =============================================================================
print("Generating fig3_bootstrap.png …")

boot      = np.load(DATA / "bootstrap_vega_is.npy")
is_sharpe = metrics["metrics"]["vega_is"]["sharpe"]   # 1.7397 (drawn at this value)
p_gt_1    = metrics["bootstrap"]["vega_is"]["p_gt_1"] # 0.9272 → "93%"

fig3, ax3 = plt.subplots(figsize=(W1 * 1.35, H1))

n_bins = 75
ax3.hist(boot, bins=n_bins, color=C_BLUE, alpha=0.60, edgecolor="white",
         linewidth=0.3, zorder=3)

# Shade area where Sharpe > 1 (highlight the winning region)
counts, edges = np.histogram(boot, bins=n_bins)
for i, (e0, e1) in enumerate(zip(edges[:-1], edges[1:])):
    if e0 >= 1.0:
        ax3.bar(e0, counts[i], width=(e1 - e0), align="edge",
                color=C_BLUE, alpha=0.80, edgecolor="white", linewidth=0.3, zorder=4)

# Vertical lines
ax3.axvline(1.0,       color=C_RED,  linewidth=1.6, linestyle="--", zorder=5,
            label='Threshold  (Sharpe = 1.0)')
ax3.axvline(is_sharpe, color=C_BLUE, linewidth=2.0, linestyle="-",  zorder=6,
            label='IS Sharpe = 1.75')

# P-value annotation box
ax3.text(0.97, 0.94,
         f"P(Sharpe > 1) = {p_gt_1 * 100:.0f}%",
         transform=ax3.transAxes, ha="right", va="top",
         fontsize=9, color="#111827", fontweight="semibold",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                   edgecolor="#D1D5DB", linewidth=0.8, alpha=0.95))

ax3.set_title("Bootstrap Distribution of Sharpe Ratio (10,000 resamples)")
ax3.set_xlabel("Sharpe Ratio")
ax3.set_ylabel("Count")
ax3.legend(loc="upper left")
ax3.set_xlim(boot.min() - 0.1, boot.max() + 0.1)

fig3.tight_layout()
fig3.savefig(OUT / "fig3_bootstrap.png", dpi=DPI, bbox_inches="tight")
plt.close(fig3)
print("  ✓ saved")


# =============================================================================
# FIG 4 — Cross-Asset Validation (BTC | SPY, side-by-side)
# =============================================================================
print("Generating fig4_crossasset.png …")

eq_btc = pd.read_csv(DATA / "daily_equity_vega_is.csv", parse_dates=["date"])
eq_spy = pd.read_csv(SPY  / "spy_daily_equity.csv",     parse_dates=["date"])

eq_btc = eq_btc[(eq_btc["date"] >= IS_START) & (eq_btc["date"] <= IS_END)].copy()
eq_btc["ret"] = (eq_btc["equity"] / eq_btc["equity"].iloc[0] - 1) * 100

eq_spy["ret"] = (eq_spy["equity"] / eq_spy["equity"].iloc[0] - 1) * 100

btc_sharpe = metrics["metrics"]["vega_is"]["sharpe"]
btc_n      = metrics["metrics"]["vega_is"]["n_trades"]
btc_period = "Jun 2021 – Sep 2023"

spy_sharpe = 0.79
spy_n      = 147
spy_start  = eq_spy["date"].iloc[0].strftime("%b %Y")
spy_end    = eq_spy["date"].iloc[-1].strftime("%b %Y")
spy_period = f"{spy_start} – {spy_end}"

fig4, (ax_b, ax_s) = plt.subplots(
    1, 2,
    figsize=(W1 * 2.1, H1),
    gridspec_kw={"wspace": 0.32}
)

def _equity_panel(ax, dates, rets, color, title, sharpe, n_trades, period):
    ax.plot(dates, rets, color=color, linewidth=1.5, zorder=3)
    ax.axhline(0, color="#6B7280", linewidth=0.7, linestyle="--", zorder=2, alpha=0.8)
    ax.fill_between(dates, rets, 0,
                    where=(rets >= 0), color=color,  alpha=0.07, zorder=1)
    ax.fill_between(dates, rets, 0,
                    where=(rets < 0),  color=C_RED,  alpha=0.07, zorder=1)
    ax.set_title(title, pad=7)
    ax.set_ylabel("Cumulative Return (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    info = (f"Sharpe = {sharpe:.2f}   |   N = {n_trades} trades\n"
            f"{period}")
    ax.text(0.04, 0.05, info,
            transform=ax.transAxes, va="bottom", ha="left",
            fontsize=7.5, color="#374151", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#E5E7EB", alpha=0.9, linewidth=0.7))

_equity_panel(ax_b,
              eq_btc["date"].values, eq_btc["ret"].values,
              C_BLUE, "BTC / Deribit",
              btc_sharpe, btc_n, btc_period)

_equity_panel(ax_s,
              eq_spy["date"].values, eq_spy["ret"].values,
              C_PURPLE, "SPY / OptionsDX",
              spy_sharpe, spy_n, spy_period)

# Remove ylabel on right panel (shared y-scale intent)
ax_s.set_ylabel("")

fig4.autofmt_xdate(rotation=0, ha="center")
fig4.suptitle("Structural Validation: Same Parameters, Different Asset",
              fontsize=11.5, fontweight="semibold", y=1.01)
fig4.tight_layout()
fig4.savefig(OUT / "fig4_crossasset.png", dpi=DPI, bbox_inches="tight")
plt.close(fig4)
print("  ✓ saved")

print(f"\nAll 4 figures saved to: {OUT}")
