"""
BTC VRP TERMINAL — Live Paper Trading Monitor

Terminal-style dashboard. Real-time data from Deribit API. Auto-refreshes 60s.
Usage: streamlit run dashboard/live.py
"""

import sys, re, json, math, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Constants ────────────────────────────────────────────
STATE_FILE = Path("Data/live_state.json")
SIGNALS_FILE = Path("logs/live_signals.csv")
BT_EQUITY = Path("dashboard_data/daily_equity_mod_full.csv")
LOG_DIR = Path("logs")

RISK_PCT, BUFFER, WORST_LOSS = 0.02, 1.5, 54.93
INITIAL_CAPITAL = 100_000
PROFIT_TARGET_PCT = 0.50

# ── Colors ───────────────────────────────────────────────
G="#00ff41"; GD="#00aa2a"; GK="#004d00"; R="#ff0040"; CY="#00ffff"
YL="#ffff00"; AM="#ffaa00"; WH="#b0b0b0"; GR="#555555"
BG="#0a0a0a"; BP="#0d1117"; BD="#1a3a1a"; PU="#a855f7"

# ── Page config + autorefresh ────────────────────────────
st.set_page_config(page_title="BTC VRP TERMINAL", layout="wide",
                   initial_sidebar_state="collapsed")
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, limit=None, key="tr")
except ImportError:
    pass

# ── CSS ──────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500;600;700&display=swap');
.stApp{{background-color:{BG}!important}}
.block-container{{max-width:1400px;padding-top:.5rem;padding-bottom:.5rem}}
#MainMenu,footer,header{{visibility:hidden}}.stDeployButton{{display:none}}
iframe[title="streamlit_autorefresh.st_autorefresh"]{{display:none!important;height:0!important;position:absolute!important}}
p,span,div,td,th,li,label,h1,h2,h3{{font-family:'Fira Code','Courier New',monospace!important}}
h1,h2,h3{{color:{G}!important;text-transform:uppercase;letter-spacing:2px}}
.tp{{background:{BP};border:1px solid {BD};padding:1rem;margin-bottom:.5rem}}
.pt{{color:{G};font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:3px;
    border-bottom:1px solid {BD};padding-bottom:.4rem;margin-bottom:.8rem}}
.dr{{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #111}}
.dk{{color:{GR};font-size:.85rem}}.dv{{color:{WH};font-size:.85rem;font-weight:500}}
.hero{{font-size:2.8rem;font-weight:700;text-shadow:0 0 30px rgba(0,255,65,.3);line-height:1}}
.kl{{color:{GR};font-size:.7rem;text-transform:uppercase;letter-spacing:2px}}
.kv{{font-size:1.3rem;font-weight:700}}.kvs{{font-size:1rem;font-weight:600}}
.badge-open{{color:{BG};background:{G};padding:2px 8px;font-size:.7rem;font-weight:700;letter-spacing:1px}}
.badge-flat{{color:{BG};background:{YL};padding:2px 8px;font-size:.7rem;font-weight:700;letter-spacing:1px}}
.badge-legacy{{color:{BG};background:{AM};padding:1px 6px;font-size:.6rem;font-weight:700;letter-spacing:1px;margin-left:6px}}
.pbar{{background:#111;height:6px;width:100%;margin-top:4px}}.pfill{{height:6px}}
.tt{{width:100%;border-collapse:collapse;font-size:.8rem}}
.tt th{{color:{GD};text-align:left;padding:5px 6px;border-bottom:1px solid {BD};
        text-transform:uppercase;font-size:.7rem;letter-spacing:1px}}
.tt td{{padding:4px 6px;border-bottom:1px solid #111;color:{WH}}}
.tt tr:hover{{background:#111!important}}
.scanlines{{pointer-events:none;position:fixed;top:0;left:0;width:100%;height:100%;z-index:9999;
            background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,255,65,.012) 2px,rgba(0,255,65,.012) 4px)}}
.sep{{border-top:1px solid {BD};margin:.3rem 0}}
/* box-sizing for panels */
.tp{{box-sizing:border-box}}
/* equal-height columns: stretch .tp to fill Streamlit column */
[data-testid="stHorizontalBlock"]{{align-items:stretch}}
[data-testid="stHorizontalBlock"] [data-testid="stColumn"] > div{{height:100%}}
[data-testid="stHorizontalBlock"] [data-testid="stColumn"] > div > div{{height:100%}}
[data-testid="stHorizontalBlock"] [data-testid="stColumn"] > div > div > div{{height:100%}}
[data-testid="stHorizontalBlock"] [data-testid="stColumn"] .tp{{height:100%}}
</style>
<div class="scanlines"></div>
""", unsafe_allow_html=True)

PLOT = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,17,23,0.8)",
    font=dict(family="Fira Code, Courier New, monospace", color=GD, size=10),
    margin=dict(l=40, r=10, t=25, b=30),
    xaxis=dict(gridcolor="#0a1a0a", zerolinecolor="#1a3a1a"),
    yaxis=dict(gridcolor="#0a1a0a", zerolinecolor="#1a3a1a"),
)


# ── Data helpers ─────────────────────────────────────────

@st.cache_resource
def get_client():
    try:
        from src.deribit_client import DeribitClient
        return DeribitClient()
    except Exception:
        return None

@st.cache_data(ttl=30)
def fetch_spot():
    cl = get_client()
    try: return cl.get_index_price() if cl else None
    except Exception: return None

@st.cache_data(ttl=30)
def fetch_ticker(instrument: str):
    cl = get_client()
    try: return cl.get_ticker(instrument) if cl else None
    except Exception: return None

@st.cache_data(ttl=30)
def fetch_positions():
    cl = get_client()
    if not cl: return {}
    try:
        positions = cl.get_positions("BTC")
        sp = cl.get_index_price()
        out = {}
        for p in positions:
            pnl_btc = p.get("total_profit_loss", 0)
            out[p["instrument_name"]] = {"pnl_usd": pnl_btc * sp}
        out["_spot"] = sp
        return out
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def fetch_iv_rv():
    """Fetch DVOL + compute Parkinson RV30 from daily candles."""
    cl = get_client()
    if not cl: return None, None
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - 120 * 86400_000  # 120d so RV30 has 90 valid points

    dvol = rv = None
    try:
        d = cl._request('public/get_volatility_index_data', {
            'currency': 'BTC', 'start_timestamp': str(start_ms),
            'end_timestamp': str(now_ms), 'resolution': '86400',
        }, authenticated=False)
        if isinstance(d, dict) and 'data' in d:
            df = pd.DataFrame(d['data'], columns=['ts','o','h','l','close'])
            df['date'] = pd.to_datetime(df['ts'], unit='ms')
            dvol = df[['date','close']].rename(columns={'close':'dvol'})
    except Exception:
        pass
    try:
        ca = cl._request('public/get_tradingview_chart_data', {
            'instrument_name': 'BTC-PERPETUAL', 'start_timestamp': str(start_ms),
            'end_timestamp': str(now_ms), 'resolution': '1D',
        }, authenticated=False)
        if isinstance(ca, dict) and 'ticks' in ca:
            df = pd.DataFrame({'date': pd.to_datetime(ca['ticks'], unit='ms'),
                               'high': ca['high'], 'low': ca['low']})
            df['hl2'] = np.log(df['high']/df['low'])**2
            w = 30
            f = 1.0 / (4 * w * math.log(2))
            df['rv30'] = df['hl2'].rolling(w).apply(
                lambda x: math.sqrt(f * x.sum()) * math.sqrt(365) * 100, raw=True)
            rv = df[['date','rv30']].dropna()
    except Exception:
        pass
    return dvol, rv


def reconstruct_unrealized_pnl(pos_info):
    """Reconstruct unrealized PnL (option + hedge) from trader logs."""
    if not pos_info:
        return pd.DataFrame()
    opened_at = pos_info.get("opened_at", "")
    entry_price = pos_info["entry_price"]
    n_contracts = pos_info["n_contracts"]

    try:
        open_dt = datetime.fromisoformat(opened_at)
    except Exception:
        return pd.DataFrame()

    # Skip bids from the opening cycle (first bid may belong to previous position)
    min_ts = open_dt + timedelta(hours=2)

    # Parse all log files (trader_*.log + cron.log)
    log_files = sorted(glob.glob(str(LOG_DIR / "trader_*.log")))
    opt_points = []
    hedge_points = []
    spot_val = 0.0

    re_spot = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .* BTC Spot: \$([\d,]+)')
    re_bid = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .* \[top_vega\] bid=([\d.]+)')
    re_hedge = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .* \[top_vega\] Hedge PnL this cycle: .+ \(accumulated: \$([-+\d,.]+)\)')

    for lf in log_files:
        try:
            with open(lf) as f:
                for line in f:
                    m_spot = re_spot.search(line)
                    if m_spot:
                        spot_val = float(m_spot.group(2).replace(',', ''))
                        continue
                    m_hedge = re_hedge.search(line)
                    if m_hedge:
                        ts = datetime.strptime(m_hedge.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                        acc = float(m_hedge.group(2).replace(',', ''))
                        hedge_points.append({"ts": ts, "hedge_acc": acc})
                        continue
                    m_bid = re_bid.search(line)
                    if m_bid and spot_val > 0:
                        ts = datetime.strptime(m_bid.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                        if ts < min_ts:
                            continue
                        bid_val = float(m_bid.group(2))
                        opt_pnl = (entry_price - bid_val) * spot_val * n_contracts
                        opt_points.append({"ts": ts, "opt_pnl": round(opt_pnl, 2)})
        except Exception:
            continue

    if not opt_points:
        return pd.DataFrame()

    df = pd.DataFrame(opt_points).sort_values("ts").reset_index(drop=True)

    # Merge hedge accumulated PnL
    if hedge_points:
        hdf = pd.DataFrame(hedge_points).sort_values("ts").drop_duplicates("ts")
        first_hedge_ts = hdf["ts"].iloc[0]
        first_hedge_val = hdf["hedge_acc"].iloc[0]
        # Interpolate linearly from 0 at opening to first known value
        # for the gap before hedge PnL logging was enabled
        df = pd.merge_asof(df.sort_values("ts"), hdf, on="ts", direction="backward")
        df["hedge_acc"] = df["hedge_acc"].fillna(0.0)
        if first_hedge_val != 0:
            gap = df["ts"] < first_hedge_ts
            if gap.any():
                t0_s = df["ts"].iloc[0].timestamp()
                t1_s = first_hedge_ts.timestamp()
                span = t1_s - t0_s
                if span > 0:
                    df.loc[gap, "hedge_acc"] = df.loc[gap, "ts"].apply(
                        lambda t: first_hedge_val * (t.timestamp() - t0_s) / span)
    else:
        df["hedge_acc"] = 0.0

    df["combined_pnl"] = df["opt_pnl"] + df["hedge_acc"]
    return df


def cc(v):
    return G if v > 0 else R if v < 0 else GR


# ── Load ─────────────────────────────────────────────────
state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else None
signals = pd.read_csv(SIGNALS_FILE) if SIGNALS_FILE.exists() else pd.DataFrame()
if not state:
    st.markdown(f'<p style="color:{R};">ERROR: live_state.json not found</p>', unsafe_allow_html=True)
    st.stop()

all_trades = state.get("trades_history", [])
trades_tv = [t for t in all_trades if t.get("strategy") == "top_vega"]
tv_pnl = sum(t["total_pnl"] for t in trades_tv)
equity = INITIAL_CAPITAL + tv_pnl
target_vega = (equity * RISK_PCT) / (WORST_LOSS * BUFFER)
pos = state["positions"].get("top_vega")

spot = fetch_spot()
deribit_pos = fetch_positions()
if spot is None:
    spot = deribit_pos.get("_spot") or (pos["entry_spot"] if pos else 0)
now = datetime.now(timezone.utc)

# Live PnL
opt_pnl = hdg_pnl = combo_pnl = 0.0
opt_ticker = hdg_ticker = None
delta = gamma = vega = theta = iv_now = mark = bid = ask = 0.0
dte = days_held = 0
hedge_inst = None
is_legacy = False

if pos:
    opt_ticker = fetch_ticker(pos["instrument"])
    hedge_inst = pos.get("hedge_instrument")
    hdg_ticker = fetch_ticker(hedge_inst) if hedge_inst else None
    gk = opt_ticker.get("greeks", {}) if opt_ticker else {}
    mark = opt_ticker.get("mark_price", 0) if opt_ticker else 0
    bid = opt_ticker.get("best_bid_price", 0) if opt_ticker else 0
    ask = opt_ticker.get("best_ask_price", 0) if opt_ticker else 0
    iv_now = opt_ticker.get("mark_iv", 0) if opt_ticker else 0
    delta, gamma = gk.get("delta", 0), gk.get("gamma", 0)
    vega, theta = gk.get("vega", 0), gk.get("theta", 0)
    try:
        exp_dt = datetime.strptime(pos["instrument"].split("-")[1], "%d%b%y").replace(tzinfo=timezone.utc, hour=8)
        dte = (exp_dt - now).total_seconds() / 86400
    except Exception: pass
    try: days_held = (now - datetime.fromisoformat(pos["opened_at"])).days
    except Exception: pass
    od = deribit_pos.get(pos["instrument"])
    if od: opt_pnl = od["pnl_usd"]
    if hedge_inst:
        hd = deribit_pos.get(hedge_inst)
        if hd: hdg_pnl = hd["pnl_usd"]
    combo_pnl = opt_pnl + hdg_pnl
    # Legacy: position vega exposure >> current target
    pos_vega = pos["n_contracts"] * abs(pos.get("entry_vega", 0))
    if pos_vega > target_vega * 1.3:
        is_legacy = True

# Equity including unrealized
equity_display = equity + combo_pnl


# ══════════════════════════════════════════════════════════
#  STATUS BAR
# ══════════════════════════════════════════════════════════
ago_str = "—"
lu = state.get("last_update", "")
if lu:
    try:
        d = now - datetime.fromisoformat(lu)
        ago_str = f"{int(d.total_seconds()//3600)}h {int((d.total_seconds()%3600)//60)}m ago"
    except Exception: pass
badge = '<span class="badge-open">POSITION OPEN</span>' if pos else '<span class="badge-flat">FLAT — SCANNING</span>'
st.markdown(f'''<div style="display:flex;justify-content:space-between;align-items:center;padding:.3rem 0;margin-bottom:.2rem;">
<span style="color:{GD};font-size:.7rem;letter-spacing:2px;">BTC VRP TERMINAL // DERIBIT TESTNET</span>
<span style="color:{GR};font-size:.7rem;">BTC ${spot:,.0f} | {now.strftime("%H:%M UTC")} | Cycle {ago_str} | {badge}</span>
</div>''', unsafe_allow_html=True)
st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  HERO METRICS
# ══════════════════════════════════════════════════════════
m1, m2, m3, m4, m5 = st.columns([2.5, 1.5, 1.5, 1.5, 1.5])
with m1:
    st.markdown(f'''<div class="tp" style="padding:1rem 1.2rem;">
    <span class="kl">EQUITY</span><br>
    <span class="hero" style="color:{G};">${equity_display:,.0f}</span><br>
    <span style="color:{GR};font-size:.75rem;">from ${INITIAL_CAPITAL:,} initial</span>
    </div>''', unsafe_allow_html=True)
with m2:
    st.markdown(f'''<div class="tp" style="text-align:center;padding:1rem .5rem;">
    <span class="kl">OPEN P&L</span><br>
    <span class="kv" style="color:{cc(combo_pnl)};">${combo_pnl:+,.0f}</span><br>
    <span style="color:{GR};font-size:.65rem;">OPT ${opt_pnl:+,.0f} | HDG ${hdg_pnl:+,.0f}</span>
    </div>''', unsafe_allow_html=True)
with m3:
    st.markdown(f'''<div class="tp" style="text-align:center;padding:1rem .5rem;">
    <span class="kl">CLOSED P&L</span><br>
    <span class="kv" style="color:{cc(tv_pnl)};">${tv_pnl:+,.0f}</span><br>
    <span style="color:{GR};font-size:.65rem;">{len(trades_tv)} trades</span>
    </div>''', unsafe_allow_html=True)
with m4:
    total_pnl = tv_pnl + combo_pnl
    ret = total_pnl / INITIAL_CAPITAL * 100
    st.markdown(f'''<div class="tp" style="text-align:center;padding:1rem .5rem;">
    <span class="kl">TOTAL RETURN</span><br>
    <span class="kv" style="color:{cc(total_pnl)};">{ret:+.2f}%</span><br>
    <span style="color:{GR};font-size:.65rem;">${total_pnl:+,.0f}</span>
    </div>''', unsafe_allow_html=True)
with m5:
    wr = sum(1 for t in trades_tv if t["total_pnl"] > 0) / len(trades_tv) * 100 if trades_tv else 0
    st.markdown(f'''<div class="tp" style="text-align:center;padding:1rem .5rem;">
    <span class="kl">WIN RATE</span><br>
    <span class="kv" style="color:{G};">{wr:.0f}%</span><br>
    <span style="color:{GR};font-size:.65rem;">vega tgt {target_vega:.1f}</span>
    </div>''', unsafe_allow_html=True)
st.markdown('<div class="sep"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  POSITION + GREEKS + HEDGE
# ══════════════════════════════════════════════════════════
if pos:
    ep = pos["entry_price"]
    cap_pct = ((ep - ask) / ep * 100) if ep > 0 and ask > 0 else 0
    progress = min(max(cap_pct / (PROFIT_TARGET_PCT * 100) * 100, 0), 100)
    prog_col = G if progress >= 80 else AM if progress >= 50 else GR
    h_btc = pos.get("current_hedge_btc", 0)
    h_dir = "LONG" if h_btc > 0 else "SHORT" if h_btc < 0 else "FLAT"
    h_price = 0
    if hdg_ticker: h_price = hdg_ticker.get("last_price") or hdg_ticker.get("mark_price", 0)
    net_d = (-pos["n_contracts"] * abs(delta)) + h_btc
    d_ok = abs(net_d) < 0.02
    leg = '<span class="badge-legacy">LEGACY SIZING</span>' if is_legacy else ''

    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        st.markdown(f'''<div class="tp">
        <div class="pt">POSITION</div>
        <div style="display:flex;justify-content:space-between;margin-bottom:.5rem;">
            <span style="color:{G};font-size:1rem;font-weight:700;">{pos["instrument"]}</span>
            <span style="color:{R};font-size:.8rem;font-weight:600;">SHORT {pos["option_type"].upper()}</span>
        </div>
        <div class="dr"><span class="dk">SIZE</span>
            <span class="dv">{pos["n_contracts"]:.1f} contracts{leg}</span></div>
        <div class="dr"><span class="dk">ENTRY</span>
            <span class="dv">{ep:.4f} BTC (${ep*spot:,.0f})</span></div>
        <div class="dr"><span class="dk">MARK</span>
            <span class="dv">{mark:.4f} BTC (${mark*spot:,.0f})</span></div>
        <div class="dr"><span class="dk">BID / ASK</span>
            <span class="dv">{bid:.4f} / {ask:.4f}</span></div>
        <div class="dr"><span class="dk">HELD / DTE</span>
            <span class="dv">{days_held}d / <span style="color:{'#ff0' if dte<15 else WH};">{dte:.0f}d</span></span></div>
        <div class="dr"><span class="dk">IV ENTRY / NOW</span>
            <span class="dv">{pos.get("entry_iv",0):.1f}% / {iv_now:.1f}%</span></div>
        <div style="margin-top:.5rem;padding-top:.3rem;border-top:1px solid {BD};">
            <span class="dk">TARGET {PROFIT_TARGET_PCT*100:.0f}% PREMIUM</span>
            <div class="pbar"><div class="pfill" style="width:{progress:.0f}%;background:{prog_col};"></div></div>
            <span style="color:{GD};font-size:.65rem;">{cap_pct:.1f}% captured — {progress:.0f}% to target</span>
        </div>
        </div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''<div class="tp">
        <div class="pt">GREEKS</div>
        <div class="dr"><span class="dk">DELTA</span><span class="dv">{delta:.4f}</span></div>
        <div class="dr"><span class="dk">GAMMA</span><span class="dv">{gamma:.6f}</span></div>
        <div class="dr"><span class="dk">VEGA</span><span class="dv">{vega:.2f}</span></div>
        <div class="dr"><span class="dk">THETA</span>
            <span class="dv" style="color:{G};">{theta:.2f}</span></div>
        <div style="margin-top:.8rem;padding-top:.4rem;border-top:1px solid {BD};">
            <span class="dk">NET DELTA (OPT+HDG)</span><br>
            <span style="color:{G if d_ok else AM};font-size:1rem;font-weight:700;">{net_d:+.4f} BTC</span><br>
            <span style="color:{GR};font-size:.55rem;">-{pos["n_contracts"]:.1f}×|{abs(delta):.3f}| + {h_btc:.4f} = {net_d:+.4f}</span><br>
            <span style="color:{GR};font-size:.55rem;">{"HEDGED" if d_ok else "DRIFT"} — rebal every 4h (last Δ={pos.get("last_delta",0):.3f})</span>
        </div>
        </div>''', unsafe_allow_html=True)
    with c3:
        st.markdown(f'''<div class="tp">
        <div class="pt">HEDGE</div>
        <div style="margin-bottom:.4rem;">
            <span style="color:{CY};font-size:.85rem;font-weight:600;">{hedge_inst or "N/A"}</span>
        </div>
        <div class="dr"><span class="dk">DIRECTION</span>
            <span class="dv" style="color:{CY};">{h_dir}</span></div>
        <div class="dr"><span class="dk">SIZE</span>
            <span class="dv">{abs(h_btc):.4f} BTC</span></div>
        <div class="dr"><span class="dk">QTR PRICE</span>
            <span class="dv">${h_price:,.0f}</span></div>
        <div style="margin-top:.8rem;padding-top:.4rem;border-top:1px solid {BD};">
            <span class="dk">SIZING CONFIG</span>
            <div class="dr"><span class="dk">RISK/TRADE</span><span class="dv">{RISK_PCT*100:.0f}%</span></div>
            <div class="dr"><span class="dk">BUFFER</span><span class="dv">{BUFFER:.1f}x</span></div>
            <div class="dr"><span class="dk">VEGA TARGET</span>
                <span class="dv" style="color:{CY};">{target_vega:.1f}</span></div>
        </div>
        </div>''', unsafe_allow_html=True)
else:
    st.markdown(f'''<div class="tp" style="text-align:center;padding:1.5rem;">
    <span style="color:{YL};font-size:1rem;font-weight:700;letter-spacing:3px;">
    NO ACTIVE POSITION — SCANNING</span>
    </div>''', unsafe_allow_html=True)

# ── LATEST SIGNAL ────────────────────────────────────────
if len(signals) > 0:
    la = signals.iloc[-1]
    si = la.get("vega_top_instrument", "N/A")
    sv = la.get("vega_top_score", 0)
    ss = la.get("spot", 0)
    st_ts = str(la.get("timestamp", ""))[:19]
    st.markdown(f'''<div class="tp" style="display:flex;justify-content:space-between;align-items:center;padding:.5rem 1rem;">
    <div><span style="color:{GR};font-size:.65rem;letter-spacing:2px;">LATEST TOP VEGA SIGNAL</span><br>
    <span style="color:{YL};font-size:1rem;font-weight:700;">{si}</span></div>
    <div style="text-align:right;">
        <span style="color:{GR};font-size:.7rem;">VEGA</span><span style="color:{G};font-weight:600;">{sv:.1f}</span>
        <span style="color:{GR};font-size:.7rem;"> | SPOT </span><span style="color:{WH};">${ss:,.0f}</span>
        <span style="color:{GR};font-size:.7rem;"> | {st_ts}</span>
    </div></div>''', unsafe_allow_html=True)
st.markdown('<div class="sep"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  CHARTS ROW 1: Equity Curves (backtest + live) + Unrealized PnL
# ══════════════════════════════════════════════════════════

# Compute Y-axis range for backtest chart only
_eq_ymin, _eq_ymax = INITIAL_CAPITAL, INITIAL_CAPITAL
if BT_EQUITY.exists():
    _bt = pd.read_csv(BT_EQUITY, parse_dates=["date"])
    _eq_ymin = min(_eq_ymin, _bt["equity"].min())
    _eq_ymax = max(_eq_ymax, _bt["equity"].max())
_eq_pad = (_eq_ymax - _eq_ymin) * 0.05
_eq_yrange = [_eq_ymin - _eq_pad, _eq_ymax + _eq_pad]

ch1a, ch1b, ch2 = st.columns([1, 1, 2])

with ch1a:
    st.markdown(f'<div class="pt">BACKTEST (MOD) — 2021-2024</div>', unsafe_allow_html=True)
    fig_bt = go.Figure()
    if BT_EQUITY.exists():
        bt = pd.read_csv(BT_EQUITY, parse_dates=["date"])
        fig_bt.add_trace(go.Scatter(
            x=bt["date"], y=bt["equity"],
            line=dict(color=GR, width=1.5), name='Backtest',
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
        ))
    fig_bt.add_hline(y=INITIAL_CAPITAL, line_dash="dash", line_color="#1a3a1a", line_width=1)
    fig_bt.update_layout(**PLOT, height=220, showlegend=False,
                         yaxis_title="USD", yaxis_tickformat="$,.0f",
                         yaxis_range=_eq_yrange,
                         xaxis_dtick="M12", xaxis_tickformat="%Y")
    st.plotly_chart(fig_bt, use_container_width=True)

with ch1b:
    st.markdown(f'<div class="pt">LIVE — 2026</div>', unsafe_allow_html=True)
    fig_lv = go.Figure()
    if trades_tv:
        live_pts = []
        eq_run = INITIAL_CAPITAL
        try:
            t0 = datetime.fromisoformat(trades_tv[0]["opened_at"])
        except Exception:
            t0 = now
        live_pts.append((t0, eq_run))
        for t in trades_tv:
            try:
                td = datetime.fromisoformat(t.get("exit_date", t.get("opened_at", "")))
            except Exception:
                td = t0
            eq_run += t["total_pnl"]
            live_pts.append((td, eq_run))
        lx, ly = zip(*live_pts)
        fig_lv.add_trace(go.Scatter(
            x=list(lx), y=list(ly),
            line=dict(color=G, width=2.5), mode='lines+markers',
            marker=dict(size=5, color=G), name='Live',
            hovertemplate="%{x|%b %d}<br>$%{y:,.0f}<extra></extra>",
        ))
    fig_lv.add_hline(y=INITIAL_CAPITAL, line_dash="dash", line_color="#1a3a1a", line_width=1)
    fig_lv.update_layout(**PLOT, height=220, showlegend=False,
                         yaxis_tickformat="$,.0f")
    st.plotly_chart(fig_lv, use_container_width=True)

with ch2:
    st.markdown(f'<div class="pt">UNREALIZED P&L — CURRENT POSITION</div>', unsafe_allow_html=True)
    if pos:
        pnl_df = reconstruct_unrealized_pnl(pos)
        # Append live point from Deribit API so chart ends at current value
        if not pnl_df.empty and combo_pnl != 0:
            live_row = pd.DataFrame([{
                "ts": now, "opt_pnl": opt_pnl, "hedge_acc": hdg_pnl,
                "combined_pnl": combo_pnl,
            }])
            pnl_df = pd.concat([pnl_df, live_row], ignore_index=True)
        if not pnl_df.empty and len(pnl_df) > 1:
            fig_u = go.Figure()
            # Combined P&L (option + hedge) — main line
            fig_u.add_trace(go.Scatter(
                x=pnl_df['ts'], y=pnl_df['combined_pnl'],
                line=dict(color=G, width=2.5), name='Combined P&L',
                fill='tozeroy', fillcolor='rgba(0,255,65,0.05)',
                hovertemplate="%{x|%b %d %H:%M}<br>Combined: $%{y:+,.0f}<extra></extra>",
            ))
            # Option-only — thin dashed reference
            fig_u.add_trace(go.Scatter(
                x=pnl_df['ts'], y=pnl_df['opt_pnl'],
                line=dict(color=CY, width=1, dash='dot'), name='Option only',
                hovertemplate="%{x|%b %d %H:%M}<br>Option: $%{y:+,.0f}<extra></extra>",
            ))
            fig_u.add_hline(y=0, line_color=GR, line_width=1)
            fig_u.update_layout(**PLOT, height=220, showlegend=True,
                                legend=dict(x=.02, y=.98, font=dict(size=9)),
                                yaxis_tickformat="$,.0f")
            st.plotly_chart(fig_u, use_container_width=True)
        elif not pnl_df.empty:
            st.markdown(f'''<div class="tp" style="text-align:center;padding:2rem;">
            <span style="color:{GR};font-size:.7rem;">1 data point — chart populates each cycle</span><br>
            <span style="color:{cc(combo_pnl)};font-size:1.5rem;font-weight:700;">${combo_pnl:+,.0f}</span>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown(f'''<div class="tp" style="text-align:center;padding:2rem;">
            <span style="color:{GR};font-size:.7rem;">No log data found</span><br>
            <span style="color:{cc(combo_pnl)};font-size:1.5rem;font-weight:700;">${combo_pnl:+,.0f}</span>
            </div>''', unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color:{GR};text-align:center;padding:2rem;">No open position</p>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  CHARTS ROW 2: IV vs RV + Trade PnL
# ══════════════════════════════════════════════════════════
ch3, ch4 = st.columns(2)

with ch3:
    st.markdown(f'<div class="pt">DVOL (IV) VS PARKINSON RV30</div>', unsafe_allow_html=True)
    dvol_df, rv_df = fetch_iv_rv()
    if dvol_df is not None or rv_df is not None:
        # Trim to common date range (RV30 needs 30 days to produce first value)
        common_start = None
        if dvol_df is not None and len(dvol_df) and rv_df is not None and len(rv_df):
            common_start = max(dvol_df['date'].min(), rv_df['date'].min())
            dvol_df = dvol_df[dvol_df['date'] >= common_start]
            rv_df = rv_df[rv_df['date'] >= common_start]
        fig_v = go.Figure()
        if dvol_df is not None and len(dvol_df):
            fig_v.add_trace(go.Scatter(x=dvol_df['date'], y=dvol_df['dvol'],
                line=dict(color=CY, width=2), name='DVOL (implied)',
                hovertemplate="%{x|%b %d}<br>IV: %{y:.1f}%<extra></extra>"))
        if rv_df is not None and len(rv_df):
            fig_v.add_trace(go.Scatter(x=rv_df['date'], y=rv_df['rv30'],
                line=dict(color=AM, width=2), name='RV30 (realized)',
                hovertemplate="%{x|%b %d}<br>RV: %{y:.1f}%<extra></extra>"))
        fig_v.update_layout(**PLOT, height=220, showlegend=True,
            legend=dict(x=.02, y=.98, font=dict(size=9)),
            yaxis_title="Vol (%)", yaxis_ticksuffix="%")
        st.plotly_chart(fig_v, use_container_width=True)
    else:
        st.markdown(f'<p style="color:{GR};text-align:center;padding:2rem;">API unavailable</p>',
                    unsafe_allow_html=True)

with ch4:
    st.markdown(f'<div class="pt">TRADE P&L</div>', unsafe_allow_html=True)
    if trades_tv:
        colors = ["rgba(0,255,65,0.7)" if t["total_pnl"] > 0 else "rgba(255,0,64,0.7)" for t in trades_tv]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=list(range(1, len(trades_tv)+1)),
            y=[t["total_pnl"] for t in trades_tv], marker_color=colors, width=0.5,
            hovertext=[t["instrument"] for t in trades_tv],
            hovertemplate="#%{x} %{hovertext}<br>$%{y:+,.0f}<extra></extra>"))
        fig2.add_hline(y=0, line_color=GR, line_width=1)
        fig2.update_layout(**PLOT, height=220, showlegend=False,
            xaxis_title="Trade #", yaxis_title="PnL ($)", yaxis_tickformat="$,.0f", xaxis_dtick=1)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.markdown(f'<p style="color:{GR};text-align:center;padding:2rem;">Awaiting first trade...</p>',
                    unsafe_allow_html=True)
st.markdown('<div class="sep"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  TRADE HISTORY + STATS
# ══════════════════════════════════════════════════════════
t1, t2 = st.columns([4, 1])
with t1:
    st.markdown(f'<div class="pt">TRADE HISTORY // TOP VEGA</div>', unsafe_allow_html=True)
    if trades_tv:
        rows = ""
        for i, t in enumerate(trades_tv):
            tot, opt, hdg = t["total_pnl"], t.get("option_pnl", 0), t.get("hedge_pnl", 0)
            rows += f'''<tr><td style="color:{GR};">{i+1}</td>
            <td>{t["instrument"]}</td><td style="color:{CY};">{t.get("hedge_instrument","—")}</td>
            <td style="color:{GR};">{str(t.get("opened_at",""))[:10]}</td>
            <td style="color:{GR};">{str(t.get("exit_date",""))[:10]}</td>
            <td style="text-align:right;color:{cc(opt)};">${opt:+,.0f}</td>
            <td style="text-align:right;color:{cc(hdg)};">${hdg:+,.0f}</td>
            <td style="text-align:right;color:{cc(tot)};font-weight:600;">${tot:+,.0f}</td>
            <td style="color:{GR};">{t.get("exit_reason","—")}</td></tr>'''
        st.markdown(f'''<table class="tt"><tr><th>#</th><th>Instrument</th><th>Hedge</th><th>Entry</th><th>Exit</th>
        <th style="text-align:right;">Opt</th><th style="text-align:right;">Hdg</th>
        <th style="text-align:right;">Total</th><th>Reason</th></tr>{rows}</table>''', unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color:{GR};">No completed trades yet.</p>', unsafe_allow_html=True)
with t2:
    st.markdown(f'<div class="pt">STATS</div>', unsafe_allow_html=True)
    if trades_tv:
        n = len(trades_tv)
        w = sum(1 for t in trades_tv if t["total_pnl"] > 0)
        avg = tv_pnl / n
        st.markdown(f'''<div class="tp">
        <div class="dr"><span class="dk">TRADES</span><span class="dv">{n}</span></div>
        <div class="dr"><span class="dk">WIN RATE</span><span class="dv" style="color:{G};">{w/n*100:.0f}%</span></div>
        <div class="dr"><span class="dk">AVG PNL</span><span class="dv" style="color:{cc(avg)};">${avg:+,.0f}</span></div>
        <div class="dr"><span class="dk">BEST</span><span class="dv" style="color:{G};">${max(t["total_pnl"] for t in trades_tv):+,.0f}</span></div>
        <div class="dr"><span class="dk">WORST</span><span class="dv" style="color:{R};">${min(t["total_pnl"] for t in trades_tv):+,.0f}</span></div>
        <div class="dr"><span class="dk">TOTAL</span><span class="dv" style="color:{cc(tv_pnl)};font-weight:700;">${tv_pnl:+,.0f}</span></div>
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown(f'''<div class="tp"><div class="dr"><span class="dk">TRADES</span><span class="dv">0</span></div></div>''',
                    unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────
st.markdown(f'''<div style="text-align:center;padding:.6rem 0;border-top:1px solid {BD};margin-top:.3rem;">
<span style="color:{GK};font-size:.6rem;letter-spacing:3px;">
MODERATE (2% RISK / 1.5X BUFFER) // TOP VEGA + QTR FUTURES HEDGE // AUTO-REFRESH 60S
</span></div>''', unsafe_allow_html=True)
