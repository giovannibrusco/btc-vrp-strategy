"""
Narrative Dashboard — Visual story of the BTC VRP research journey.

Designed for non-technical audiences (interviewers, LinkedIn).
Scrollable single page, dark theme, minimal text, big numbers.

Usage:
    streamlit run dashboard/story.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Style ──────────────────────────────────────────────────
BLUE = "#2962FF"
GREEN = "#26A69A"
RED = "#EF5350"
ORANGE = "#FF9800"
PURPLE = "#AB47BC"
CYAN = "#00BCD4"
GREY = "#787B86"
GOLD = "#FFD54F"
WHITE = "#E0E0E0"
BG = "#0E1117"
CARD_BG = "#1A1D29"

LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=WHITE),
    margin=dict(l=40, r=40, t=40, b=40),
)

DATA_DIR = Path("dashboard_data")
# Small research extracts (DVOL, spot, exploratory trades, SPY results) shipped
# with the repo so the story runs without the raw ~5GB dataset in Data/
RAW_DIR = DATA_DIR / "research"
SPY_DIR = DATA_DIR / "research"

st.set_page_config(
    page_title="BTC VRP Strategy — Research Story",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown("""
<style>
    .block-container { max-width: 1100px; padding-top: 2rem; }
    h1 { font-size: 2.4rem !important; font-weight: 700 !important; letter-spacing: -0.5px; }
    h2 { font-size: 1.8rem !important; font-weight: 600 !important; margin-top: 3rem !important;
         border-bottom: 2px solid #2962FF; padding-bottom: 0.5rem; }
    h3 { font-size: 1.2rem !important; font-weight: 600 !important; }
    .big-number { font-size: 2.5rem; font-weight: 700; line-height: 1; }
    .caption { font-size: 1.05rem; color: #9E9E9E; line-height: 1.6; margin-top: 1rem; }
    .card { background: #1A1D29; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;
            border: 1px solid #2A2D3A; }
    .rejected-label { color: #EF5350; font-weight: 700; font-size: 0.85rem;
                      text-transform: uppercase; letter-spacing: 1px; }
    .stMetric > div { background: transparent !important; }
    div[data-testid="stMetricValue"] { font-size: 2rem !important; }
    .kpi-box { text-align: center; padding: 0.8rem 0; }
    .kpi-box .kpi-label { color: #787B86; font-size: 0.8rem; text-transform: uppercase;
                          letter-spacing: 1px; margin-bottom: 0.2rem; }
    .kpi-box .kpi-value { color: #E0E0E0; font-size: 2rem; font-weight: 600; }
    .caption { text-align: center !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 1 — The VRP on BTC
# ══════════════════════════════════════════════════════════════

st.title("BTC Volatility Risk Premium")
st.markdown("##### Systematic short-volatility strategy on Deribit options with delta hedge")

st.markdown("---")
st.header("1 — The VRP on BTC")

@st.cache_data
def load_vrp_data():
    dvol = pd.read_csv(RAW_DIR / "dvol_btc_daily.csv", parse_dates=["date"])
    spot = pd.read_csv(RAW_DIR / "btc_spot_daily.csv", parse_dates=["date"])

    # Parkinson Realized Vol (30-day rolling)
    spot = spot.sort_values("date")
    log_hl = np.log(spot["high"] / spot["low"])
    park_var = log_hl ** 2 / (4 * np.log(2))
    spot["park_rv30"] = park_var.rolling(30).mean().apply(lambda x: np.sqrt(x * 365) * 100)

    merged = dvol.merge(spot[["date", "park_rv30"]], on="date", how="inner").dropna()
    return merged

vrp = load_vrp_data()

fig_vrp = go.Figure()

# Fill area between IV and RV
fig_vrp.add_trace(go.Scatter(
    x=vrp["date"], y=vrp["dvol_close"],
    name="Implied Vol (DVOL)",
    line=dict(color=BLUE, width=2),
))
fig_vrp.add_trace(go.Scatter(
    x=vrp["date"], y=vrp["park_rv30"],
    name="Realized Vol (Parkinson 30d)",
    line=dict(color=ORANGE, width=2),
    fill="tonexty",
    fillcolor="rgba(41,98,255,0.12)",
))

fig_vrp.update_layout(
    **LAYOUT,
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    yaxis_title="Annualized Volatility (%)",
    xaxis_title="",
)
st.plotly_chart(fig_vrp, use_container_width=True)

# VRP stats
mean_iv = vrp["dvol_close"].mean()
mean_rv = vrp["park_rv30"].mean()
mean_spread = mean_iv - mean_rv

cols = st.columns(3)
for col, label, val in [
    (cols[0], "Mean Implied Vol", f"{mean_iv:.0f}%"),
    (cols[1], "Mean Realized Vol", f"{mean_rv:.0f}%"),
    (cols[2], "Mean VRP Spread", f"+{mean_spread:.0f}pp"),
]:
    col.markdown(
        f'<div class="kpi-box"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{val}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<p class="caption">Implied volatility on BTC is structurally above realized volatility. '
    'This spread — the <b>Volatility Risk Premium</b> — is the edge this strategy harvests. '
    'Selling options and delta-hedging captures this premium systematically.</p>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════
# SECTION 2 — What We Tried and Rejected
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.header("2 — What We Tried and Rejected")

st.markdown(
    '<p class="caption">Every added complexity was tested. Five approaches were rejected '
    'before arriving at the final strategy — the simplest one.</p>',
    unsafe_allow_html=True,
)

# --- Card 1: RND-KDE Gap ---
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown('<span class="rejected-label">✗ Rejected</span>', unsafe_allow_html=True)
        st.markdown("### RND-KDE Gap Scoring")

        @st.cache_data
        def load_rnd_data():
            t = pd.read_csv(RAW_DIR / "trades_v2_IS.csv")
            return t[["gap_prob", "total_pnl"]].dropna()

        rnd = load_rnd_data()
        fig_rnd = go.Figure()
        fig_rnd.add_trace(go.Scatter(
            x=rnd["gap_prob"], y=rnd["total_pnl"],
            mode="markers",
            marker=dict(color=GREY, size=8, opacity=0.7),
        ))
        # Regression line
        if len(rnd) > 2:
            z = np.polyfit(rnd["gap_prob"], rnd["total_pnl"], 1)
            x_line = np.linspace(rnd["gap_prob"].min(), rnd["gap_prob"].max(), 50)
            fig_rnd.add_trace(go.Scatter(
                x=x_line, y=np.polyval(z, x_line),
                mode="lines", line=dict(color=RED, dash="dash", width=2),
                showlegend=False,
            ))
        fig_rnd.update_layout(**LAYOUT, height=220, showlegend=False,
                              xaxis_title="Gap probability", yaxis_title="Trade PnL ($)")
        st.plotly_chart(fig_rnd, use_container_width=True)
        st.markdown("No predictive power. β < 0, p = 0.37. Random selection performs equally.")

# --- Card 2: Premium/DTE ---
with col2:
    with st.container(border=True):
        st.markdown('<span class="rejected-label">✗ Rejected</span>', unsafe_allow_html=True)
        st.markdown("### Premium/DTE Selection")

        @st.cache_data
        def load_pdt_data():
            t = pd.read_csv(RAW_DIR / "trades_v2_IS.csv")
            return t

        pdt = load_pdt_data()
        colors = [RED if p == 0 else GREEN if pnl > 0 else GREY
                  for p, pnl in zip(pdt["exit_price"], pdt["total_pnl"])]

        fig_pdt = go.Figure()
        fig_pdt.add_trace(go.Bar(
            x=list(range(len(pdt))),
            y=pdt["total_pnl"],
            marker_color=colors,
        ))
        fig_pdt.update_layout(**LAYOUT, height=220, showlegend=False,
                              xaxis_title="Trade #", yaxis_title="PnL ($)")
        st.plotly_chart(fig_pdt, use_container_width=True)

        n_zero = (pdt["exit_price"] == 0).sum()
        pnl_zero = pdt[pdt["exit_price"] == 0]["total_pnl"].sum()
        pnl_total = pdt["total_pnl"].sum()
        pct = pnl_zero / pnl_total * 100 if pnl_total != 0 else 0
        st.markdown(
            f'{n_zero} trades with exit\\_price = 0 (red bars) account for '
            f'**{pct:.0f}% of total profits**. A data artifact from illiquid options.'
        )

# --- Card 3: Multi-Position & Card 4: Regime Filters ---
col3, col4 = st.columns(2)

with col3:
    with st.container(border=True):
        st.markdown('<span class="rejected-label">✗ Rejected</span>', unsafe_allow_html=True)
        st.markdown("### Multi-Position Mode")

        fig_multi = go.Figure()
        fig_multi.add_trace(go.Bar(
            x=["Single position", "Multi position"],
            y=[-18.4, -49.7],
            marker_color=[GREEN, RED],
            text=["-18.4%", "-49.7%"],
            textposition="outside",
            textfont=dict(size=14, color=WHITE),
        ))
        fig_multi.update_layout(**LAYOUT, height=220, showlegend=False,
                                yaxis_title="Max Drawdown (%)",
                                yaxis_range=[-60, 0])
        st.plotly_chart(fig_multi, use_container_width=True)
        st.markdown("Allowing multiple simultaneous trades pushed drawdown from **-18% to -50%** during the 2022 crash.")

with col4:
    with st.container(border=True):
        st.markdown('<span class="rejected-label">✗ Rejected</span>', unsafe_allow_html=True)
        st.markdown("### Regime Filters")

        @st.cache_data
        def load_regime_corr():
            vm = pd.read_csv(RAW_DIR / "trades_is_with_vol_metrics.csv")
            indicators = {
                "DVOL level": vm["dvol"].corr(vm["total_pnl"]),
                "Parkinson RV30": vm["park_rv30"].corr(vm["total_pnl"]),
                "VRP spread": vm["vrp_spread"].corr(vm["total_pnl"]),
            }
            return indicators

        corrs = load_regime_corr()

        fig_regime = go.Figure()
        names = list(corrs.keys())
        vals = list(corrs.values())
        fig_regime.add_trace(go.Bar(
            x=vals, y=names, orientation="h",
            marker_color=[GREY if abs(v) < 0.25 else BLUE for v in vals],
            text=[f"r = {v:.2f}" for v in vals],
            textposition="outside",
        ))
        fig_regime.add_vline(x=-0.25, line=dict(color=RED, dash="dot", width=1))
        fig_regime.add_vline(x=0.25, line=dict(color=RED, dash="dot", width=1))
        fig_regime.update_layout(**LAYOUT, height=220, showlegend=False,
                                xaxis_title="Correlation with trade PnL",
                                xaxis_range=[-0.5, 0.5])
        st.plotly_chart(fig_regime, use_container_width=True)
        st.markdown("All correlations below ±0.25 (red lines). No indicator predicts strategy returns.")

# --- Card 5: Stop Loss ---
with st.container(border=True):
    st.markdown('<span class="rejected-label">✗ Rejected</span>', unsafe_allow_html=True)
    st.markdown("### Stop Loss")

    col5a, col5b = st.columns([2, 1])

    with col5a:
        @st.cache_data
        def load_stoploss():
            sl = pd.read_csv(RAW_DIR / "stoploss_exploratory_is.csv")
            return sl

        sl_data = load_stoploss()

        # Compute stop loss impact at different thresholds
        thresholds = [0.5, 1, 1.5, 2, 3, 5, 8]
        results = []
        for pct in thresholds:
            # Winners that would have been stopped
            w_stopped = sl_data[(sl_data["winner"] == "W") &
                               (sl_data["worst_unreal_%"].abs() > pct)]
            winner_pnl_lost = w_stopped["final_pnl_$"].sum() if len(w_stopped) > 0 else 0

            # Losers that would have been stopped (loss avoided)
            l_stopped = sl_data[(sl_data["winner"] == "L") &
                               (sl_data["worst_unreal_%"].abs() > pct)]
            loser_pnl_saved = l_stopped["final_pnl_$"].abs().sum() if len(l_stopped) > 0 else 0

            label = f"-{pct:.1f}%" if pct != int(pct) else f"-{int(pct)}%"
            results.append({
                "threshold": label,
                "winner_pnl_lost": winner_pnl_lost,
                "loser_pnl_saved": loser_pnl_saved,
            })

        fig_sl = go.Figure()
        fig_sl.add_trace(go.Bar(
            x=[r["threshold"] for r in results],
            y=[r["winner_pnl_lost"] for r in results],
            name="Winner PnL destroyed",
            marker_color=RED,
        ))
        fig_sl.add_trace(go.Bar(
            x=[r["threshold"] for r in results],
            y=[r["loser_pnl_saved"] for r in results],
            name="Loser PnL saved",
            marker_color=GREEN,
        ))
        fig_sl.update_layout(**LAYOUT, height=250, barmode="group",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                            xaxis_title="Stop loss level",
                            yaxis_title="PnL ($)")
        st.plotly_chart(fig_sl, use_container_width=True)

    with col5b:
        st.markdown("<br><br>", unsafe_allow_html=True)
        # Key stat
        n_recovered = sl_data[sl_data["recovered_from_deep_loss"] == True].shape[0]
        n_winners = sl_data[sl_data["winner"] == "W"].shape[0]
        st.metric("Winners recovered from deep loss", f"{n_recovered} / {n_winners}")
        st.markdown(
            "At every threshold, more winner PnL is destroyed than loser PnL saved. "
            "Winners need room to recover."
        )


# ══════════════════════════════════════════════════════════════
# SECTION 3 — The Final Strategy
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.header("3 — The Final Strategy: Top Vega")

st.markdown(
    '<p class="caption">Select the highest-vega OTM option each day. Delta-hedge with quarterly futures. '
    'Exit at 50% profit target or DTE ≤ 5. The simplest approach — validated by the failure of complexity.</p>',
    unsafe_allow_html=True,
)

# Equity curves — 3 configs
@st.cache_data
def load_strategy_data():
    metrics = json.loads((DATA_DIR / "metrics.json").read_text())
    configs = {}
    for key, info in [("cons", "Conservative"), ("mod", "Moderate"), ("agg", "Aggressive")]:
        eq = pd.read_csv(DATA_DIR / f"daily_equity_{key}_is.csv", parse_dates=["date"])
        configs[key] = {"equity": eq, "label": info}
    bootstrap = np.load(DATA_DIR / "bootstrap_vega_is.npy")
    return metrics, configs, bootstrap

metrics, configs, bootstrap = load_strategy_data()

# Equity overlay
fig_eq = go.Figure()
colors_cfg = {"cons": GREEN, "mod": BLUE, "agg": ORANGE}
for key, cfg in configs.items():
    eq = cfg["equity"]
    fig_eq.add_trace(go.Scatter(
        x=eq["date"], y=eq["equity"],
        name=cfg["label"],
        line=dict(color=colors_cfg[key], width=2.5 if key == "mod" else 1.5),
    ))

fig_eq.update_layout(
    **LAYOUT,
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    yaxis_title="Equity ($)",
    yaxis_tickformat="$,.0f",
)
st.plotly_chart(fig_eq, use_container_width=True)

# Key metrics table
m = metrics["metrics"]
rc = metrics["risk_configs"]

col_a, col_b, col_c = st.columns(3)
for col, key, color in [(col_a, "cons", GREEN), (col_b, "mod", BLUE), (col_c, "agg", ORANGE)]:
    d = m[f"{key}_is"]
    with col:
        st.markdown(
            f'<div style="text-align:center; margin-bottom:0.8rem;">'
            f'<span style="color:{color}; font-weight:700; font-size:1.1rem;">{rc[key]["label"]}</span></div>',
            unsafe_allow_html=True,
        )
        for label, val in [
            ("SHARPE", f'{d["sharpe"]:.2f}'),
            ("RETURN", f'+{d["total_return_pct"]:.1f}%'),
            ("MAX DRAWDOWN", f'{d["max_drawdown_pct"]:.1f}%'),
            ("WIN RATE", f'{d["win_rate"]:.0f}%'),
        ]:
            st.markdown(
                f'<div class="kpi-box"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{val}</div></div>',
                unsafe_allow_html=True,
            )

# Bootstrap distribution
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### Sharpe Ratio Bootstrap Distribution (10,000 samples)")

fig_boot = go.Figure()
fig_boot.add_trace(go.Histogram(
    x=bootstrap,
    nbinsx=80,
    marker_color=BLUE,
    opacity=0.7,
))
# CI lines
ci_lo = np.percentile(bootstrap, 2.5)
ci_hi = np.percentile(bootstrap, 97.5)
fig_boot.add_vline(x=0, line=dict(color=RED, width=2))
fig_boot.add_vline(x=ci_lo, line=dict(color=GOLD, dash="dash", width=1.5))
fig_boot.add_vline(x=ci_hi, line=dict(color=GOLD, dash="dash", width=1.5))
fig_boot.add_vline(x=np.mean(bootstrap), line=dict(color=WHITE, width=2))

fig_boot.add_annotation(x=0, y=0.95, yref="paper", text="Zero",
                        font=dict(color=RED), showarrow=False)
fig_boot.add_annotation(x=ci_lo, y=0.85, yref="paper", text=f"2.5%: {ci_lo:.2f}",
                        font=dict(color=GOLD), showarrow=False)
fig_boot.add_annotation(x=ci_hi, y=0.85, yref="paper", text=f"97.5%: {ci_hi:.2f}",
                        font=dict(color=GOLD), showarrow=False)

fig_boot.update_layout(**LAYOUT, height=300, showlegend=False,
                       xaxis_title="Sharpe Ratio", yaxis_title="Frequency")
st.plotly_chart(fig_boot, use_container_width=True)

p_gt_0 = (bootstrap > 0).mean() * 100
st.markdown(
    f'<p class="caption">95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]. '
    f'P(Sharpe > 0) = {p_gt_0:.1f}%. '
    f'Sharpe is <b>invariant to sizing</b> (1.75 across all three configurations). '
    f'The choice is purely return vs drawdown.</p>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════
# SECTION 4 — Cross-Asset Validation
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.header("4 — Cross-Asset Validation: SPY")

st.markdown(
    '<p class="caption">Same strategy applied to SPY options (2010–2023). '
    'Zero re-optimization — identical parameters, scoring, and exit rules.</p>',
    unsafe_allow_html=True,
)

@st.cache_data
def load_cross_asset():
    btc_eq = pd.read_csv(DATA_DIR / "daily_equity_vega_is.csv", parse_dates=["date"])
    spy_eq = pd.read_csv(SPY_DIR / "spy_daily_equity.csv", parse_dates=["date"])
    btc_trades = pd.read_csv(DATA_DIR / "trades_vega_is.csv")
    spy_trades = pd.read_csv(SPY_DIR / "spy_trades.csv")
    return btc_eq, spy_eq, btc_trades, spy_trades

btc_eq, spy_eq, btc_tr, spy_tr = load_cross_asset()

# Side-by-side equity curves
col_btc, col_spy = st.columns(2)

with col_btc:
    st.markdown(f'<h3 style="color:{BLUE}">BTC (35 trades, 2021–2023)</h3>', unsafe_allow_html=True)
    fig_btc = go.Figure()
    btc_norm = btc_eq["equity"] / btc_eq["equity"].iloc[0] * 100
    fig_btc.add_trace(go.Scatter(
        x=btc_eq["date"], y=btc_norm,
        line=dict(color=BLUE, width=2.5),
    ))
    fig_btc.update_layout(**LAYOUT, height=280, showlegend=False,
                          yaxis_title="Equity (indexed to 100)")
    st.plotly_chart(fig_btc, use_container_width=True)

with col_spy:
    st.markdown(f'<h3 style="color:{ORANGE}">SPY (147 trades, 2010–2023)</h3>', unsafe_allow_html=True)
    fig_spy = go.Figure()
    spy_norm = spy_eq["equity"] / spy_eq["equity"].iloc[0] * 100
    fig_spy.add_trace(go.Scatter(
        x=spy_eq["date"], y=spy_norm,
        line=dict(color=ORANGE, width=2.5),
    ))
    fig_spy.update_layout(**LAYOUT, height=280, showlegend=False,
                          yaxis_title="Equity (indexed to 100)")
    st.plotly_chart(fig_spy, use_container_width=True)

# Comparison — versus layout
st.markdown("#### Head-to-Head")

btc_wr = (btc_tr["total_pnl"] > 0).mean() * 100
spy_wr = (spy_tr["total_pnl"] > 0).mean() * 100
btc_opt_wr = (btc_tr["option_pnl"] > 0).mean() * 100
spy_opt_wr = (spy_tr["option_pnl"] > 0).mean() * 100

# Header row
st.markdown(
    f'<div style="display:flex; justify-content:space-between; padding:0 2rem; margin-bottom:0.5rem;">'
    f'<span style="color:{BLUE}; font-weight:700; font-size:1.3rem;">BTC</span>'
    f'<span style="color:{ORANGE}; font-weight:700; font-size:1.3rem;">SPY</span></div>',
    unsafe_allow_html=True,
)

versus_rows = [
    ("Win Rate", f"{btc_wr:.0f}%", f"{spy_wr:.0f}%"),
    ("Option-Only Win Rate", f"{btc_opt_wr:.0f}%", f"{spy_opt_wr:.0f}%"),
    ("Sharpe Ratio", f"{m['vega_is']['sharpe']:.2f}", "0.79"),
    ("Avg Option PnL", f"${btc_tr['option_pnl'].mean():,.0f}", f"${spy_tr['option_pnl'].mean():,.0f}"),
    ("Avg Hedge PnL", f"${btc_tr['hedge_pnl_total'].mean():+,.0f}", f"${spy_tr['hedge_pnl_total'].mean():+,.0f}"),
    ("Avg Total PnL", f"${btc_tr['total_pnl'].mean():+,.0f}", f"${spy_tr['total_pnl'].mean():+,.0f}"),
]

for metric_label, btc_val, spy_val in versus_rows:
    st.markdown(
        f'<div style="display:flex; align-items:center; justify-content:space-between; '
        f'padding:0.6rem 2rem; border-bottom:1px solid #1A1D29;">'
        f'<span style="color:{WHITE}; font-size:1.2rem; font-weight:600; width:30%; text-align:left;">{btc_val}</span>'
        f'<span style="color:#616161; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; '
        f'width:40%; text-align:center;">{metric_label}</span>'
        f'<span style="color:{WHITE}; font-size:1.2rem; font-weight:600; width:30%; text-align:right;">{spy_val}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# PnL decomposition bar chart
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### PnL Decomposition: Option Premium vs Hedge")

fig_decomp = go.Figure()
fig_decomp.add_trace(go.Bar(
    x=["BTC", "SPY"],
    y=[btc_tr["option_pnl"].mean(), spy_tr["option_pnl"].mean()],
    name="Option PnL",
    marker_color=GREEN,
))
fig_decomp.add_trace(go.Bar(
    x=["BTC", "SPY"],
    y=[btc_tr["hedge_pnl_total"].mean(), spy_tr["hedge_pnl_total"].mean()],
    name="Hedge PnL",
    marker_color=CYAN,
))
fig_decomp.update_layout(
    **LAYOUT, height=320, barmode="group",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    yaxis_title="Avg PnL per trade ($)",
)
st.plotly_chart(fig_decomp, use_container_width=True)

st.markdown(
    '<p class="caption">Same strategy, zero re-optimization. Option-only win rate is '
    '<b>82% on both assets</b> — the VRP exists universally. But the hedge economics diverge: '
    'BTC\'s high realized volatility makes gamma rebalancing profitable (+$298/trade), '
    'while SPY\'s low vol means hedging costs eat the premium (-$129/trade). '
    'BTC\'s structurally wide IV spread makes it the better venue.</p>',
    unsafe_allow_html=True,
)

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#616161; font-size:0.9rem;">'
    'IS: Jun 2021 – Sep 2023 &nbsp;|&nbsp; OOS: Oct 2023 – Sep 2024 &nbsp;|&nbsp; '
    'SPY: Jan 2010 – Dec 2023<br>'
    'All results net of transaction costs (Deribit 0.03% taker + hedge spread). '
    'Past performance does not guarantee future results.</p>',
    unsafe_allow_html=True,
)
