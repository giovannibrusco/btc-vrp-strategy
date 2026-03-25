"""
Streamlit Dashboard V2 — BTC Options VRP Trading System.

Strategy: Top Vega (VRP Selling) — single-position, capital-based vega sizing.
Three risk configurations: Conservativa, Moderata, Aggressiva.

Usage:
    streamlit run dashboard/dashboard.py

Pre-generate data first:
    python scripts/generate_dashboard_data.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Colour palette (TradingView-inspired) ──────────────────
BLUE = "#2962FF"
GREEN = "#26A69A"
RED = "#EF5350"
ORANGE = "#FF9800"
PURPLE = "#AB47BC"
CYAN = "#00BCD4"
GREY = "#787B86"
GOLD = "#FFD54F"
BG_CARD = "#131722"
BG_PAGE = "#0E1117"
BG_PANEL = "#1E222D"

PLOTLY_TEMPLATE = "plotly_dark"
DATA_DIR = Path("dashboard_data")

# Risk config display info
RISK_CONFIG_INFO = {
    "cons": {"label": "Conservativa", "color": GREEN,  "emoji": ""},
    "mod":  {"label": "Moderata",     "color": BLUE,   "emoji": ""},
    "agg":  {"label": "Aggressiva",   "color": ORANGE, "emoji": ""},
}


# ════════════════════════════════════════════════════════════
# Data Loading (cached)
# ════════════════════════════════════════════════════════════

@st.cache_data
def load_all_data():
    """Load all pre-generated dashboard data."""
    data = {}

    # Metrics JSON
    with open(DATA_DIR / "metrics.json") as f:
        data["meta"] = json.load(f)

    # All possible keys: base (vega) + risk configs (cons, mod, agg) × (is, oos, full)
    all_prefixes = ["vega", "cons", "mod", "agg"]
    all_periods = ["is", "oos", "full"]

    # Trades
    for prefix in all_prefixes:
        for period in all_periods:
            key = f"{prefix}_{period}"
            p = DATA_DIR / f"trades_{key}.csv"
            if p.exists():
                data[f"trades_{key}"] = pd.read_csv(p, parse_dates=["entry_date", "exit_date"])

    # Daily equity
    for prefix in all_prefixes:
        for period in all_periods:
            key = f"{prefix}_{period}"
            p = DATA_DIR / f"daily_equity_{key}.csv"
            if p.exists():
                data[f"daily_eq_{key}"] = pd.read_csv(p, parse_dates=["date"])

    # Bootstrap distributions (base strategy only)
    for key in ["vega_is", "vega_oos"]:
        p = DATA_DIR / f"bootstrap_{key}.npy"
        if p.exists():
            data[f"boot_{key}"] = np.load(p)

    # Sensitivity
    p = DATA_DIR / "sensitivity.csv"
    if p.exists():
        data["sensitivity"] = pd.read_csv(p)

    # BTC daily
    p = DATA_DIR / "btc_daily_spot.csv"
    if p.exists():
        data["btc_daily"] = pd.read_csv(p, parse_dates=["date"], index_col="date")

    # ── Build "full" period data if not on disk ──────────
    initial_eq = data["meta"]["config"]["initial_equity"]
    m = data["meta"]["metrics"]

    for strat in all_prefixes:
        # Full trades
        is_key = f"trades_{strat}_is"
        oos_key = f"trades_{strat}_oos"
        full_key = f"trades_{strat}_full"
        if is_key in data and oos_key in data and full_key not in data:
            data[full_key] = pd.concat([data[is_key], data[oos_key]], ignore_index=True)

        # Full daily equity
        is_deq = f"daily_eq_{strat}_is"
        oos_deq = f"daily_eq_{strat}_oos"
        full_deq = f"daily_eq_{strat}_full"
        if is_deq in data and oos_deq in data and full_deq not in data:
            d_is = data[is_deq]
            d_oos = data[oos_deq].copy()
            final_is_eq = d_is["equity"].iloc[-1]
            oos_start_eq = d_oos["equity"].iloc[0]
            d_oos["equity"] = d_oos["equity"] + (final_is_eq - oos_start_eq)
            data[full_deq] = pd.concat([d_is, d_oos], ignore_index=True)

        # Full metrics
        full_mkey = f"{strat}_full"
        tk = f"trades_{strat}_full"
        if full_mkey not in m and tk in data:
            df_t = data[tk]
            if not df_t.empty:
                pnls = df_t["total_pnl"].values
                n = len(df_t)
                equity = initial_eq
                rets = []
                for _, row in df_t.sort_values("exit_date").iterrows():
                    r = row["total_pnl"] / equity * 100
                    equity += row["total_pnl"]
                    rets.append(r)
                rets = np.array(rets)
                td = (df_t["exit_date"].max() - df_t["entry_date"].min()).days
                tpy = n / (td / 365) if td > 0 else 12
                sharpe = rets.mean() / rets.std() * np.sqrt(tpy) if rets.std() > 0 else 0
                deq_key = f"daily_eq_{strat}_full"
                dd = 0
                if deq_key in data:
                    eq_mtm = data[deq_key]["equity"].values
                    peak_mtm = np.maximum.accumulate(eq_mtm)
                    dd = ((eq_mtm - peak_mtm) / peak_mtm).min() * 100
                m[full_mkey] = {
                    "n_trades": n, "sharpe": float(sharpe),
                    "total_return_pct": float((equity / initial_eq - 1) * 100),
                    "max_drawdown_pct": float(dd),
                    "win_rate": float((pnls > 0).sum() / n * 100),
                    "mean_pnl": float(pnls.mean()),
                    "trades_per_year": float(tpy),
                    "initial_equity": initial_eq,
                    "final_equity": float(equity),
                }

    return data


# ── Helpers ────────────────────────────────────────────────

def _dark_fig(fig, height=400):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor=BG_PAGE,
        plot_bgcolor=BG_PAGE,
        font=dict(color="white", size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    return fig


def _kpi_card(label, value, color=BLUE):
    st.markdown(
        f"""
        <div style='background:{BG_PANEL}; border-left:4px solid {color};
                    padding:14px 18px; border-radius:8px; margin-bottom:8px;'>
            <div style='color:{GREY}; font-size:0.7rem; text-transform:uppercase;
                        letter-spacing:1px;'>{label}</div>
            <div style='color:white; font-size:1.5rem; font-weight:700;
                        margin-top:4px;'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _drawdown_series(equity_values):
    peak = np.maximum.accumulate(equity_values)
    return (equity_values - peak) / peak * 100


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="BTC VRP Trading — Capital-Based Sizing",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        f"""
        <style>
        .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
        .stTabs [data-baseweb="tab"] {{
            padding: 8px 16px;
            font-weight: 600;
            font-size: 0.85rem;
        }}
        .stMetric {{ background: {BG_PANEL}; padding: 12px; border-radius: 8px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not DATA_DIR.exists() or not (DATA_DIR / "metrics.json").exists():
        st.error("Dashboard data not found. Run `python generate_dashboard_data.py` first.")
        return

    data = load_all_data()
    meta = data["meta"]
    metrics = meta["metrics"]
    config = meta["config"]
    risk_configs = meta.get("risk_configs", {})

    # ── Sidebar ────────────────────────────────────────────
    st.sidebar.markdown("## Top Vega (VRP Selling)")
    st.sidebar.markdown("Capital-Based Vega Sizing")
    st.sidebar.markdown("---")

    period = st.sidebar.radio("Period", ["IS", "OOS", "Full"], horizontal=True)
    period_key = period.lower()

    # Risk configuration selector
    risk_labels = {
        "Conservativa (1%/2.0x)": "cons",
        "Moderata (2%/1.5x)": "mod",
        "Aggressiva (3%/1.0x)": "agg",
    }
    selected_risk = st.sidebar.radio(
        "Risk Configuration",
        list(risk_labels.keys()),
        index=1,  # default to Moderata
        horizontal=False,
    )
    variant_key = risk_labels[selected_risk]

    st.sidebar.markdown("---")

    # Risk config details
    rc = risk_configs.get(variant_key, {})
    st.sidebar.markdown(
        f"""
        <div style='background:{BG_PANEL}; padding:14px; border-radius:8px;
                    border:1px solid #333; font-size:0.82rem;'>
            <b style='color:{PURPLE};'>Strategy:</b> Top Vega (VRP Selling)<br>
            <b style='color:{BLUE};'>Exit DTE:</b> {config['exit_dte']} |
            <b style='color:{BLUE};'>Profit Target:</b> {config['profit_target']*100:.0f}%<br>
            <b style='color:{ORANGE};'>Risk/Trade:</b> {rc.get('risk_per_trade', 0)*100:.0f}% |
            <b style='color:{ORANGE};'>Buffer:</b> {rc.get('buffer', 0):.1f}x<br>
            <b style='color:{CYAN};'>Initial Vega:</b> {rc.get('vega_initial', 0):.1f}<br>
            <b style='color:{GREEN};'>IS:</b> {config['is_start']} to {config['is_end']}<br>
            <b style='color:{PURPLE};'>OOS:</b> {config['oos_start']} to {config['oos_end']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style='background:{BG_PANEL}; padding:10px; border-radius:8px;
                    border:1px solid #333; font-size:0.75rem; color:{GREY};'>
            <b>Money Management:</b> Capital-based vega sizing<br>
            <code>vega = (equity × risk%) / (worst_loss/vega × buffer)</code><br>
            Worst loss/vega: ${config.get('worst_loss_per_vega', 54.93):.2f}<br>
            Base vega: {config.get('base_vega_target', 50)}<br>
            Post-hoc rescaling preserves Sharpe ratio.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tabs ───────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "Equity & Drawdown",
        "Trade Analysis",
        "IS vs OOS",
        "Statistical Validation",
        "Robustness",
    ])

    # Current data key
    data_key = f"{variant_key}_{period_key}"
    m_cur = metrics.get(data_key, {})
    trade_key = f"trades_{data_key}"
    eq_key = f"daily_eq_{data_key}"

    # ================================================================
    # TAB 1 — Overview
    # ================================================================
    with tab1:
        st.markdown("### Top Vega — Command Center")

        cols = st.columns(6)
        with cols[0]:
            _kpi_card("Sharpe", f"{m_cur.get('sharpe', 0):.2f}",
                      color=GREEN if m_cur.get('sharpe', 0) > 1 else ORANGE)
        with cols[1]:
            _kpi_card("Total Return", f"{m_cur.get('total_return_pct', 0):.0f}%",
                      color=GREEN if m_cur.get('total_return_pct', 0) > 0 else RED)
        with cols[2]:
            _kpi_card("MTM Max DD", f"{m_cur.get('max_drawdown_pct', 0):.1f}%", color=RED)
        with cols[3]:
            _kpi_card("Win Rate", f"{m_cur.get('win_rate', 0):.0f}%",
                      color=GREEN if m_cur.get('win_rate', 0) > 50 else RED)
        with cols[4]:
            _kpi_card("Trades", f"{m_cur.get('n_trades', 0)}", color=BLUE)
        with cols[5]:
            _kpi_card("Final Equity", f"${m_cur.get('final_equity', 0):,.0f}", color=GREEN)

        # Mini equity curve
        if eq_key in data:
            daily_eq = data[eq_key]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_eq["date"], y=daily_eq["equity"],
                mode="lines", line=dict(color=PURPLE, width=2),
                name="Top Vega",
            ))
            fig.update_layout(yaxis_title="Equity ($)", showlegend=False,
                              yaxis_tickformat="$,.0f")
            _dark_fig(fig, height=280)
            st.plotly_chart(fig, use_container_width=True)

        # Risk config comparison table
        st.markdown("#### Risk Configuration Comparison")
        comp_rows = []
        for ckey, info in RISK_CONFIG_INFO.items():
            mk = f"{ckey}_{period_key}"
            mv = metrics.get(mk, {})
            rc_info = risk_configs.get(ckey, {})
            if mv:
                comp_rows.append({
                    "Config": info["label"],
                    "Risk/Trade": f"{rc_info.get('risk_per_trade', 0)*100:.0f}%",
                    "Buffer": f"{rc_info.get('buffer', 0):.1f}x",
                    "Vega Init": f"{rc_info.get('vega_initial', 0):.1f}",
                    "Trades": mv.get("n_trades", 0),
                    "Sharpe": f"{mv.get('sharpe', 0):.2f}",
                    "Return": f"{mv.get('total_return_pct', 0):.1f}%",
                    "Max DD": f"{mv.get('max_drawdown_pct', 0):.1f}%",
                    "Win Rate": f"{mv.get('win_rate', 0):.0f}%",
                    "Final Equity": f"${mv.get('final_equity', 0):,.0f}",
                })
        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        # Overlay equity comparison
        st.markdown("#### Equity Curves — All Configurations")
        fig_comp = go.Figure()
        for ckey, info in RISK_CONFIG_INFO.items():
            ceq_key = f"daily_eq_{ckey}_{period_key}"
            if ceq_key in data:
                ceq = data[ceq_key]
                fig_comp.add_trace(go.Scatter(
                    x=ceq["date"], y=ceq["equity"],
                    mode="lines", line=dict(color=info["color"], width=2),
                    name=info["label"],
                ))
        fig_comp.update_layout(yaxis_title="Equity ($)", yaxis_tickformat="$,.0f")
        _dark_fig(fig_comp, height=350)
        st.plotly_chart(fig_comp, use_container_width=True)

    # ================================================================
    # TAB 2 — Equity & Drawdown
    # ================================================================
    with tab2:
        st.markdown("### Equity Curves & Drawdown")

        show_btc = st.checkbox("Overlay BTC buy-and-hold", value=True)

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.04,
            subplot_titles=("Mark-to-Market Equity", "Drawdown (%)"),
        )

        if eq_key in data:
            daily_eq = data[eq_key]
            eq_vals = daily_eq["equity"].values
            dates = daily_eq["date"]

            fig.add_trace(go.Scatter(
                x=dates, y=eq_vals,
                mode="lines", name="Top Vega",
                line=dict(color=PURPLE, width=2),
            ), row=1, col=1)

            dd = _drawdown_series(eq_vals)
            fig.add_trace(go.Scatter(
                x=dates, y=dd,
                mode="lines", name="Drawdown",
                line=dict(color=PURPLE, width=1),
                fill="tozeroy",
                fillcolor="rgba(171,71,188,0.15)",
                showlegend=False,
            ), row=2, col=1)

        # BTC overlay
        if show_btc and "btc_daily" in data and eq_key in data:
            btc = data["btc_daily"]
            daily_eq = data[eq_key]
            dates = daily_eq["date"]
            btc_sub = btc.loc[dates.min():dates.max()]
            if len(btc_sub) > 1:
                btc_norm = btc_sub["underlying_price"] / btc_sub["underlying_price"].iloc[0] * config["initial_equity"]
                fig.add_trace(go.Scatter(
                    x=btc_sub.index, y=btc_norm,
                    mode="lines", name="BTC B&H",
                    line=dict(color=ORANGE, width=1, dash="dash"),
                    opacity=0.6,
                ), row=1, col=1)

        # IS/OOS boundary
        if period_key == "full":
            oos_start = config["oos_start"]
            fig.add_shape(type="line", x0=oos_start, x1=oos_start,
                          y0=0, y1=1, yref="paper",
                          line=dict(dash="dash", color=GREY, width=1))
            fig.add_annotation(x=oos_start, y=1, yref="paper",
                               text="IS | OOS", showarrow=False,
                               font=dict(color=GREY, size=11), yshift=10)

        fig.update_yaxes(title_text="Equity ($)", tickformat="$,.0f", row=1, col=1)
        fig.update_yaxes(title_text="DD (%)", row=2, col=1)
        _dark_fig(fig, height=600)
        st.plotly_chart(fig, use_container_width=True)

    # ================================================================
    # TAB 3 — Trade Analysis
    # ================================================================
    with tab3:
        st.markdown("### Trade Analysis")

        if trade_key not in data:
            st.warning(f"No trade data for {data_key}")
        else:
            df_t = data[trade_key]

            # PnL distribution
            st.markdown("#### P&L Distribution")
            fig = go.Figure()
            pnl_vals = df_t["total_pnl"].values
            pnl_min, pnl_max = pnl_vals.min(), pnl_vals.max()
            bin_width = (pnl_max - pnl_min) / 30
            neg_edges = np.arange(0, pnl_min - bin_width, -bin_width)[::-1]
            pos_edges = np.arange(0, pnl_max + bin_width, bin_width)
            bin_edges_custom = np.unique(np.concatenate([neg_edges, pos_edges]))
            counts, bin_edges = np.histogram(pnl_vals, bins=bin_edges_custom)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bar_colors = [GREEN if c >= 0 else RED for c in bin_centers]
            bar_widths = np.diff(bin_edges)

            fig.add_trace(go.Bar(
                x=bin_centers, y=counts,
                marker_color=bar_colors,
                width=bar_widths * 0.9,
            ))
            fig.add_vline(x=float(np.mean(pnl_vals)), line_dash="dash",
                          line_color="white",
                          annotation_text=f"Mean ${np.mean(pnl_vals):,.0f}")
            fig.update_layout(xaxis_title="P&L ($)", yaxis_title="Count",
                              showlegend=False, bargap=0)
            _dark_fig(fig, height=350)
            st.plotly_chart(fig, use_container_width=True)

            # P&L attribution
            st.markdown("#### P&L Attribution")
            col1, col2 = st.columns(2)
            with col1:
                fig_attr = go.Figure()
                components = [
                    ("option_pnl", "Option P&L", CYAN),
                    ("hedge_pnl_total", "Hedge P&L", PURPLE),
                ]
                for comp, lbl, clr in components:
                    if comp in df_t.columns:
                        fig_attr.add_trace(go.Bar(
                            x=[lbl], y=[df_t[comp].sum()], marker_color=clr, name=lbl,
                        ))
                cost_items = [
                    ("hedge_txn_cost_total", "Hedge Costs", RED),
                    ("option_txn_cost_total", "Option Costs", GOLD),
                    ("tail_hedge_cost_total", "Tail Cost", ORANGE),
                ]
                for comp, lbl, clr in cost_items:
                    if comp in df_t.columns:
                        fig_attr.add_trace(go.Bar(
                            x=[lbl], y=[-df_t[comp].sum()], marker_color=clr, name=lbl,
                        ))
                fig_attr.update_layout(yaxis_title="Total ($)", yaxis_tickformat="$,.0f",
                                        showlegend=False)
                _dark_fig(fig_attr, height=300)
                st.plotly_chart(fig_attr, use_container_width=True)

            with col2:
                if "exit_reason" in df_t.columns:
                    exit_counts = df_t["exit_reason"].value_counts()
                    fig_exit = go.Figure(go.Pie(
                        labels=exit_counts.index, values=exit_counts.values,
                        marker=dict(colors=[GREEN, CYAN, ORANGE, RED, GREY]),
                        hole=0.45,
                    ))
                    fig_exit.update_layout(title="Exit Reasons")
                    _dark_fig(fig_exit, height=300)
                    st.plotly_chart(fig_exit, use_container_width=True)

            # Trade scatter
            st.markdown("#### Trade Timeline")
            colors = [GREEN if p > 0 else RED for p in df_t["total_pnl"]]
            fig_sc = go.Figure(go.Scatter(
                x=df_t["exit_date"], y=df_t["total_pnl"],
                mode="markers",
                marker=dict(color=colors, size=8, opacity=0.7),
                hovertemplate=(
                    "Date: %{x}<br>PnL: $%{y:,.0f}<br>"
                    "Strike: %{customdata[0]:,.0f}<br>"
                    "DTE: %{customdata[1]:.0f}<extra></extra>"
                ),
                customdata=df_t[["strike", "entry_dte"]].values,
            ))
            fig_sc.add_hline(y=0, line_dash="dot", line_color=GREY)
            fig_sc.update_layout(xaxis_title="Exit Date", yaxis_title="P&L ($)",
                                  yaxis_tickformat="$,.0f")
            _dark_fig(fig_sc, height=350)
            st.plotly_chart(fig_sc, use_container_width=True)

            # Trade characteristics
            st.markdown("#### Trade Characteristics")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                fig_dte = go.Figure(go.Histogram(
                    x=df_t["entry_dte"], nbinsx=20, marker_color=BLUE))
                fig_dte.update_layout(xaxis_title="Entry DTE", yaxis_title="Count")
                _dark_fig(fig_dte, height=250)
                st.plotly_chart(fig_dte, use_container_width=True)
            with col_b:
                if "entry_iv" in df_t.columns:
                    fig_iv = go.Figure(go.Histogram(
                        x=df_t["entry_iv"], nbinsx=20, marker_color=ORANGE))
                    fig_iv.update_layout(xaxis_title="Entry IV (%)", yaxis_title="Count")
                    _dark_fig(fig_iv, height=250)
                    st.plotly_chart(fig_iv, use_container_width=True)
            with col_c:
                holding = (df_t["exit_date"] - df_t["entry_date"]).dt.days
                fig_hold = go.Figure(go.Histogram(
                    x=holding, nbinsx=20, marker_color=GREEN))
                fig_hold.update_layout(xaxis_title="Holding Days", yaxis_title="Count")
                _dark_fig(fig_hold, height=250)
                st.plotly_chart(fig_hold, use_container_width=True)

            # Trade table
            with st.expander("Full Trade Table"):
                display_cols = ["entry_date", "exit_date", "strike", "option_type",
                                "entry_price", "exit_price", "entry_dte", "entry_iv",
                                "n_contracts", "option_pnl", "hedge_pnl_total",
                                "hedge_txn_cost_total", "option_txn_cost_total",
                                "total_pnl", "return_pct", "exit_reason"]
                available = [c for c in display_cols if c in df_t.columns]
                st.dataframe(df_t[available].sort_values("exit_date", ascending=False),
                             use_container_width=True, hide_index=True)

    # ================================================================
    # TAB 4 — IS vs OOS
    # ================================================================
    with tab4:
        st.markdown("### In-Sample vs Out-of-Sample")

        comp_keys = [
            ("n_trades", "Trades", lambda x: f"{x:.0f}"),
            ("sharpe", "Sharpe", lambda x: f"{x:.2f}"),
            ("total_return_pct", "Return (%)", lambda x: f"{x:.1f}"),
            ("max_drawdown_pct", "MTM Max DD (%)", lambda x: f"{x:.1f}"),
            ("win_rate", "Win Rate (%)", lambda x: f"{x:.1f}"),
            ("mean_pnl", "Mean PnL ($)", lambda x: f"${x:,.0f}"),
        ]

        rows = []
        for mk, label, fmt in comp_keys:
            row = {"Metric": label}
            for per in ["is", "oos"]:
                for ckey, info in RISK_CONFIG_INFO.items():
                    key = f"{ckey}_{per}"
                    m = metrics.get(key, {})
                    row[f"{info['label'][:4]} {per.upper()}"] = fmt(m.get(mk, 0))
            rows.append(row)

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Dual equity charts
        st.markdown("#### Equity Comparison")
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("In-Sample", "Out-of-Sample"))
        for i, per in enumerate(["is", "oos"], 1):
            for ckey, info in RISK_CONFIG_INFO.items():
                deq = data.get(f"daily_eq_{ckey}_{per}")
                if deq is not None:
                    fig.add_trace(go.Scatter(
                        x=deq["date"], y=deq["equity"],
                        mode="lines", line=dict(color=info["color"], width=2),
                        name=info["label"] if i == 1 else None,
                        showlegend=(i == 1),
                    ), row=1, col=i)
        fig.update_yaxes(tickformat="$,.0f")
        _dark_fig(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)

        # Sharpe Degradation
        st.markdown("#### Sharpe Degradation OOS/IS")
        cols = st.columns(3)
        for i, (ckey, info) in enumerate(RISK_CONFIG_INFO.items()):
            m_is = metrics.get(f"{ckey}_is", {})
            m_oos = metrics.get(f"{ckey}_oos", {})
            sh_is = m_is.get("sharpe", 0)
            sh_oos = m_oos.get("sharpe", 0)
            ratio = sh_oos / sh_is if sh_is != 0 else 0
            with cols[i]:
                _kpi_card(f"{info['label']} OOS/IS", f"{ratio:.2f}",
                          color=GREEN if ratio > 0.7 else RED)

    # ================================================================
    # TAB 5 — Statistical Validation
    # ════════════════════════════════════════════════════════════
    # Stats are computed on base vega=50 strategy (sizing doesn't change
    # the statistical properties — Sharpe, significance, regimes are identical)
    # ================================================================
    with tab5:
        st.markdown("### Statistical Validation")
        st.caption("Based on base strategy (vega=50). Post-hoc sizing preserves all statistical properties.")

        stat_keys = ["vega_is", "vega_oos"] if period_key == "full" else [f"vega_{period_key}"]
        strat_names = {
            "vega_is": "Base Strategy IS",
            "vega_oos": "Base Strategy OOS",
        }

        # ── T-test ──
        st.markdown("#### T-test (H0: mean PnL = 0)")
        ttest_data = meta.get("ttest", {})
        ttest_rows = []
        for key in stat_keys:
            tt = ttest_data.get(key, {})
            if tt:
                sig = ""
                if tt.get("significant_1pct"): sig = "***"
                elif tt.get("significant_5pct"): sig = "**"
                elif tt.get("significant_10pct"): sig = "*"
                else: sig = "n.s."
                ttest_rows.append({
                    "Period": strat_names.get(key, key),
                    "N": tt["n"],
                    "Mean PnL": f"${tt['mean_pnl']:,.0f}",
                    "Std PnL": f"${tt['std_pnl']:,.0f}",
                    "t-stat": f"{tt['t_stat']:.3f}",
                    "p (one-sided)": f"{tt['p_one_sided']:.4f}",
                    "Sig.": sig,
                })
        if ttest_rows:
            st.dataframe(pd.DataFrame(ttest_rows), use_container_width=True, hide_index=True)

        # ── Bootstrap Sharpe CI ──
        st.markdown("#### Bootstrap Sharpe CI (10,000 resamples)")
        boot_meta = meta.get("bootstrap", {})

        boot_keys = [k for k in stat_keys if f"boot_{k}" in data]
        if boot_keys:
            cols_boot = st.columns(len(boot_keys))
            for col, key in zip(cols_boot, boot_keys):
                bm = boot_meta.get(key, {})
                with col:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=data[f"boot_{key}"], nbinsx=50,
                        marker_color=PURPLE, opacity=0.7,
                    ))
                    if bm:
                        fig.add_vline(x=bm["sharpe"], line_dash="solid",
                                      line_color="white",
                                      annotation_text=f"Sharpe={bm['sharpe']:.2f}")
                        fig.add_vline(x=bm["ci_95_lo"], line_dash="dash", line_color=RED)
                        fig.add_vline(x=bm["ci_95_hi"], line_dash="dash", line_color=RED)
                        fig.update_layout(
                            title=f"{strat_names.get(key, key)} [{bm['ci_95_lo']:.2f}, {bm['ci_95_hi']:.2f}]",
                        )
                    fig.update_layout(xaxis_title="Sharpe", showlegend=False)
                    _dark_fig(fig, height=280)
                    st.plotly_chart(fig, use_container_width=True)

            boot_rows = []
            for key in boot_keys:
                bm = boot_meta.get(key, {})
                if bm:
                    boot_rows.append({
                        "Period": strat_names.get(key, key),
                        "Sharpe": f"{bm['sharpe']:.2f}",
                        "95% CI": f"[{bm['ci_95_lo']:.2f}, {bm['ci_95_hi']:.2f}]",
                        "P(S>0)": f"{bm['p_gt_0']*100:.0f}%",
                        "P(S>1)": f"{bm['p_gt_1']*100:.0f}%",
                        "P(S>2)": f"{bm.get('p_gt_2',0)*100:.0f}%",
                    })
            if boot_rows:
                st.dataframe(pd.DataFrame(boot_rows), use_container_width=True, hide_index=True)

        # ── Regime analysis (volatility) ──
        st.markdown("#### Volatility Regime (30d Realized Vol)")
        regimes = meta.get("regimes", {})
        for key in stat_keys:
            rd = regimes.get(key, {})
            if not rd:
                continue
            thresholds = rd.get("thresholds", [0, 0])
            st.markdown(f"**{strat_names.get(key, key)}** "
                        f"(Low < {thresholds[0]:.0f}% | Mid | High > {thresholds[1]:.0f}%)")
            regime_rows = []
            for rname in ["Low", "Mid", "High"]:
                r = rd.get(rname, {})
                if r.get("n", 0) < 2:
                    regime_rows.append({"Regime": rname, "N": r.get("n", 0),
                                        "Win Rate": "-", "Mean PnL": "-", "Total PnL": "-"})
                else:
                    regime_rows.append({
                        "Regime": rname, "N": r["n"],
                        "Win Rate": f"{r['win_rate']:.0f}%",
                        "Mean PnL": f"${r['mean_pnl']:,.0f}",
                        "Total PnL": f"${r['total_pnl']:,.0f}",
                    })
            st.dataframe(pd.DataFrame(regime_rows), use_container_width=True, hide_index=True)

        # ── Trend regime ──
        st.markdown("#### Trend Regime (60d BTC Return)")
        trends = meta.get("trend_regimes", {})
        for key in stat_keys:
            td = trends.get(key, {})
            if not td:
                continue
            st.markdown(f"**{strat_names.get(key, key)}**")
            trend_rows = []
            for rname in ["Bull (>+10%)", "Sideways", "Bear (<-10%)"]:
                r = td.get(rname, {})
                if r.get("n", 0) < 2:
                    trend_rows.append({"Regime": rname, "N": r.get("n", 0),
                                       "Win Rate": "-", "Mean PnL": "-", "Total PnL": "-"})
                else:
                    trend_rows.append({
                        "Regime": rname, "N": r["n"],
                        "Win Rate": f"{r['win_rate']:.0f}%",
                        "Mean PnL": f"${r['mean_pnl']:,.0f}",
                        "Total PnL": f"${r['total_pnl']:,.0f}",
                    })
            st.dataframe(pd.DataFrame(trend_rows), use_container_width=True, hide_index=True)

        # ── Call/Put ──
        st.markdown("#### Call vs Put Breakdown")
        callput = meta.get("callput", {})
        for key in stat_keys:
            cp = callput.get(key, {})
            if not cp:
                continue
            st.markdown(f"**{strat_names.get(key, key)}**")
            cp_rows = []
            for otype in ["call", "put"]:
                r = cp.get(otype, {})
                n = r.get("n", 0)
                if n < 1:
                    cp_rows.append({"Type": otype.upper(), "N": 0,
                                    "Win Rate": "-", "Mean PnL": "-", "Total PnL": "-"})
                else:
                    cp_rows.append({
                        "Type": otype.upper(), "N": n,
                        "Win Rate": f"{r['win_rate']:.0f}%",
                        "Mean PnL": f"${r['mean_pnl']:,.0f}",
                        "Total PnL": f"${r['total_pnl']:,.0f}",
                        "Avg DTE": f"{r.get('mean_entry_dte', 0):.0f}",
                    })
            st.dataframe(pd.DataFrame(cp_rows), use_container_width=True, hide_index=True)

    # ================================================================
    # TAB 6 — Robustness
    # ================================================================
    with tab6:
        st.markdown("### Robustness Tests")
        st.caption("Based on base strategy (vega=50). Post-hoc sizing preserves all statistical properties.")

        stat_keys_rob = ["vega_is", "vega_oos"] if period_key == "full" else [f"vega_{period_key}"]
        strat_names_rob = {
            "vega_is": "Base Strategy IS",
            "vega_oos": "Base Strategy OOS",
        }

        # ── Sensitivity heatmap ──
        if "sensitivity" in data:
            df_sens = data["sensitivity"]
            if "profit_target" in df_sens.columns and "exit_dte" in df_sens.columns:
                st.markdown("#### Parameter Sensitivity: Exit DTE vs Profit Target")
                col_hm1, col_hm2 = st.columns(2)
                with col_hm1:
                    pivot = df_sens.pivot_table(values="sharpe", index="exit_dte",
                                                 columns="profit_target")
                    fig_hm = go.Figure(go.Heatmap(
                        z=pivot.values,
                        x=[f"{c:.0%}" for c in pivot.columns],
                        y=[str(int(r)) for r in pivot.index],
                        colorscale="RdYlGn",
                        text=np.round(pivot.values, 2), texttemplate="%{text}",
                        colorbar=dict(title="Sharpe"),
                    ))
                    fig_hm.update_layout(title="Sharpe Ratio",
                                          xaxis_title="Profit Target", yaxis_title="Exit DTE")
                    _dark_fig(fig_hm, height=380)
                    st.plotly_chart(fig_hm, use_container_width=True)

                with col_hm2:
                    pivot_wr = df_sens.pivot_table(values="win_rate", index="exit_dte",
                                                    columns="profit_target")
                    fig_hm2 = go.Figure(go.Heatmap(
                        z=pivot_wr.values,
                        x=[f"{c:.0%}" for c in pivot_wr.columns],
                        y=[str(int(r)) for r in pivot_wr.index],
                        colorscale="RdYlGn",
                        text=np.round(pivot_wr.values, 1), texttemplate="%{text}%",
                        colorbar=dict(title="Win Rate %"),
                    ))
                    fig_hm2.update_layout(title="Win Rate (%)",
                                           xaxis_title="Profit Target", yaxis_title="Exit DTE")
                    _dark_fig(fig_hm2, height=380)
                    st.plotly_chart(fig_hm2, use_container_width=True)

        # ── Outlier dependence ──
        st.markdown("#### Outlier Dependence")
        outliers = meta.get("outliers", {})
        ol_keys = [k for k in stat_keys_rob if k in outliers]
        if ol_keys:
            cols_ol = st.columns(len(ol_keys))
            for col, key in zip(cols_ol, ol_keys):
                ol = outliers[key]
                df_ol = pd.DataFrame(ol)
                with col:
                    fig_ol = go.Figure()
                    fig_ol.add_trace(go.Scatter(
                        x=df_ol["removed"], y=df_ol["sharpe"],
                        mode="lines+markers", line=dict(color=PURPLE, width=2),
                        marker=dict(size=8),
                    ))
                    fig_ol.add_hline(y=0, line_dash="dot", line_color=RED)
                    fig_ol.add_hline(y=1, line_dash="dot", line_color=GREY,
                                     annotation_text="Sharpe=1")
                    fig_ol.update_layout(
                        title=strat_names_rob.get(key, key),
                        xaxis_title="Top N trades removed", yaxis_title="Sharpe",
                    )
                    _dark_fig(fig_ol, height=300)
                    st.plotly_chart(fig_ol, use_container_width=True)
                    st.dataframe(df_ol[["removed", "n", "sharpe", "win_rate", "mean_pnl"]].rename(
                        columns={"removed": "Removed", "n": "N", "sharpe": "Sharpe",
                                 "win_rate": "WR%", "mean_pnl": "Mean$"}
                    ), use_container_width=True, hide_index=True)

        # ── Cost stress test ──
        st.markdown("#### Cost Stress Test")
        costs = meta.get("cost_stress", {})
        fig_cost = go.Figure()
        has_cost_data = False
        dash_map = {"is": "solid", "oos": "dash"}
        for key in stat_keys_rob:
            cl = costs.get(key, [])
            if not cl:
                continue
            has_cost_data = True
            df_c = pd.DataFrame(cl)
            per = key.split("_")[-1]
            fig_cost.add_trace(go.Scatter(
                x=df_c["multiplier"], y=df_c["sharpe"],
                mode="lines+markers",
                name=strat_names_rob.get(key, key),
                line=dict(color=PURPLE, width=2, dash=dash_map.get(per, "solid")),
            ))
        if has_cost_data:
            fig_cost.add_hline(y=0, line_dash="dot", line_color=RED,
                               annotation_text="Break-even")
            fig_cost.add_hline(y=1, line_dash="dot", line_color=GREY,
                               annotation_text="Sharpe=1")
            fig_cost.update_layout(xaxis_title="Cost Multiplier", yaxis_title="Sharpe")
            _dark_fig(fig_cost, height=350)
            st.plotly_chart(fig_cost, use_container_width=True)

        # ── Robustness summary ──
        st.markdown("#### Robustness Summary")
        summary_rows = []
        for key in stat_keys_rob:
            ol = outliers.get(key, [])
            cl = costs.get(key, [])
            if not ol or not cl:
                continue
            ol_5 = [x for x in ol if x["removed"] == 5]
            sharpe_no5 = ol_5[0]["sharpe"] if ol_5 else "-"
            break_mult = "-"
            for c in cl:
                if c["sharpe"] < 0:
                    break_mult = f"{c['multiplier']:.1f}x"
                    break
            if break_mult == "-":
                break_mult = ">10x"
            m_key = metrics.get(key, {})
            summary_rows.append({
                "Period": strat_names_rob.get(key, key),
                "Base Sharpe": f"{m_key.get('sharpe', 0):.2f}",
                "Sharpe w/o top 5": f"{sharpe_no5:.2f}" if isinstance(sharpe_no5, float) else sharpe_no5,
                "Cost break-even": break_mult,
                "Robust?": "Yes" if (isinstance(sharpe_no5, float) and sharpe_no5 > 0 and break_mult != "1.0x") else "Check",
            })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
