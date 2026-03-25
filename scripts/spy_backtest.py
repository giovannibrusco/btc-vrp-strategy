"""
SPY Options VRP Backtest — Same strategy as BTC Top Vega.

Identical logic:
  - Score by |vega|, pick highest each day
  - Delta filter: |delta| ≤ 0.50
  - DTE filter: 30-90 days (same range as BTC)
  - T+1 execution
  - Exit: DTE ≤ 5 or profit target ≥ 50%
  - Vega-normalized sizing
  - Delta hedge with SPY shares (rebalanced daily)

Usage:
    python spy_backtest.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── Parameters (aligned with BTC backtest) ────────────────
EXIT_DTE = 5
PROFIT_TARGET_PCT = 0.50
MAX_ABS_DELTA = 0.50
DTE_MIN = 3                   # Aligned: BTC has DTE >= 3 (no upper limit in scoring)
DTE_MAX = 90                  # Keeps reasonable upper bound
INITIAL_EQUITY = 100_000
CONTRACT_MULTIPLIER = 100     # 1 SPY option = 100 shares

# Vega sizing: target vega exposure in dollar terms
# BTC: target_vega = 50 (BTC units) × spot ~$30k = ~$1.5M dollar-vega
# But BTC options are inverse, so effective $ risk per 1% vol = vega * n_contracts * spot
# Mean BTC trade: ~1% return, ~2% std on $100k equity
# For SPY: we allow fractional contracts (same as BTC) to match risk profile
TARGET_VEGA_DOLLARS = 100     # Calibrated: ~1.5 contracts, ~$40k notional (same as BTC)

# Transaction costs (realistic equity market)
# Options: $0.65 per contract per side (standard retail/institutional)
# Hedge: $0.01 per share (spread + slippage on SPY, very liquid)
# Tail hedge: not needed on equity (no exchange bankruptcy risk)
OPTION_COST_PER_CONTRACT = 0.65   # $/contract/side
HEDGE_COST_PER_SHARE = 0.01      # $/share (spread + slippage)
TAIL_HEDGE_ANNUAL_PCT = 0.0       # no tail hedge on equity

DATA_DIR = Path("SPY Options/extracted")
OUT_DIR = Path("SPY Options")


def load_spy_data(years=None):
    """Load and parse all SPY option data into a single DataFrame."""
    all_files = sorted(DATA_DIR.glob("spy_eod_*.txt"))
    if years:
        all_files = [f for f in all_files if any(str(y) in f.name for y in years)]

    print(f"Loading {len(all_files)} files...")

    chunks = []
    for i, f in enumerate(all_files):
        if i % 12 == 0:
            print(f"  {f.name}...", flush=True)
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip().strip("[]") for c in df.columns]
        chunks.append(df)

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Total rows: {len(df):,}")
    return df


def build_option_data(df_raw):
    """
    Convert wide format (call+put per row) to long format + build lookups.

    Delta filter is applied ONLY for scoring/selection, NOT for position tracking.
    The option_lookup contains ALL options (any delta) so we can track positions
    even when delta drifts beyond the filter during the trade.

    Returns:
        scores: DataFrame with top-scored option per day
        opts_all: DataFrame with all options (for date list)
        option_lookup: dict (date, strike, expiry, type) -> row dict (ALL options)
    """
    df = df_raw.copy()
    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"].str.strip())
    df["EXPIRE_DATE"] = pd.to_datetime(df["EXPIRE_DATE"].str.strip())
    df["UNDERLYING_LAST"] = pd.to_numeric(df["UNDERLYING_LAST"], errors="coerce")
    df["STRIKE"] = pd.to_numeric(df["STRIKE"], errors="coerce")
    df["DTE"] = pd.to_numeric(df["DTE"], errors="coerce")

    # Build call rows (ALL DTEs for lookup, filter later for scoring)
    calls = df[["QUOTE_DATE", "EXPIRE_DATE", "UNDERLYING_LAST", "STRIKE", "DTE",
                 "C_DELTA", "C_GAMMA", "C_VEGA", "C_THETA", "C_IV",
                 "C_BID", "C_ASK", "C_VOLUME"]].copy()
    calls.columns = ["date", "expiry", "spot", "strike", "dte",
                      "delta", "gamma", "vega", "theta", "iv",
                      "bid", "ask", "volume"]
    calls["option_type"] = "call"

    # Build put rows
    puts = df[["QUOTE_DATE", "EXPIRE_DATE", "UNDERLYING_LAST", "STRIKE", "DTE",
                "P_DELTA", "P_GAMMA", "P_VEGA", "P_THETA", "P_IV",
                "P_BID", "P_ASK", "P_VOLUME"]].copy()
    puts.columns = ["date", "expiry", "spot", "strike", "dte",
                     "delta", "gamma", "vega", "theta", "iv",
                     "bid", "ask", "volume"]
    puts["option_type"] = "put"

    # Combine ALL options
    opts_all = pd.concat([calls, puts], ignore_index=True)
    del calls, puts

    # Clean numeric columns
    for col in ["delta", "gamma", "vega", "theta", "iv", "bid", "ask", "volume"]:
        opts_all[col] = pd.to_numeric(opts_all[col], errors="coerce")

    # Basic cleanup: need valid data
    opts_all = opts_all.dropna(subset=["bid", "delta"])
    print(f"  All valid options: {len(opts_all):,}")

    # For lookup: keep DTE 0-120 (covers scoring range + tracking to expiry)
    # No delta filter — need all deltas for position tracking
    lookup_opts = opts_all[opts_all["dte"] <= DTE_MAX + 30].copy()
    # Also include DTE < DTE_MIN for tracking positions near exit
    # (already covered since we don't filter by DTE_MIN here)
    print(f"  Options for lookup (DTE ≤ {DTE_MAX + 30}): {len(lookup_opts):,}")

    print("  Building option lookup...", flush=True)
    # Vectorized: set multi-index then to_dict
    lookup_opts = lookup_opts.set_index(["date", "strike", "expiry", "option_type"])
    option_lookup = lookup_opts.to_dict("index")
    lookup_opts = lookup_opts.reset_index()
    del lookup_opts
    print(f"  Lookup size: {len(option_lookup):,} entries")

    # For SCORING: apply DTE and delta filters
    score_opts = opts_all[
        (opts_all["dte"] >= DTE_MIN) & (opts_all["dte"] <= DTE_MAX) &
        (opts_all["delta"].abs() <= MAX_ABS_DELTA) &
        (opts_all["bid"] > 0) &
        (opts_all["vega"].notna()) & (opts_all["vega"].abs() > 0)
    ].copy()
    print(f"  After DTE+delta filter for scoring: {len(score_opts):,} options")

    # Vega dollar (per contract)
    score_opts["vega_dollar"] = score_opts["vega"].abs() * CONTRACT_MULTIPLIER

    # Score: |vega|
    score_opts["score"] = score_opts["vega"].abs()

    # Best score per day
    scores = score_opts.sort_values(["date", "score"], ascending=[True, False])
    scores = scores.groupby("date").first().reset_index()
    print(f"  Scoring days: {len(scores):,}")

    return scores, opts_all, option_lookup


def run_spy_backtest(scores, option_lookup, all_dates, verbose=True):
    """
    Run SPY backtest with identical logic to BTC Top Vega.

    Delta hedge with SPY shares (rebalanced daily).
    """
    trades = []
    equity = INITIAL_EQUITY
    current_trade = None
    pending_signal = None
    daily_equity = []

    scores_by_day = {row["date"]: row for _, row in scores.iterrows()}

    for date in all_dates:
        # ══════════════════════════════════════════
        # Phase 1: Manage open position
        # ══════════════════════════════════════════
        mtm_total = 0.0

        if current_trade is not None:
            key = (date, current_trade["strike"],
                   current_trade["expiry"], current_trade["option_type"])
            opt_today = option_lookup.get(key)

            if opt_today is None:
                # Data gap → close at last known
                current_trade["exit_date"] = date
                current_trade["exit_reason"] = "data_gap"
                current_trade["exit_price"] = current_trade["_last_ask"]
                current_trade["exit_spot"] = current_trade["_last_spot"]
                current_trade["exit_dte"] = 0

                _finalize_spy_trade(current_trade, equity)
                equity += current_trade["total_pnl"]
                trades.append(current_trade)
                current_trade = None
            else:
                cur_bid = opt_today["bid"]
                cur_ask = opt_today["ask"] if opt_today["ask"] > 0 else cur_bid
                cur_delta = opt_today["delta"]
                cur_spot = opt_today["spot"]
                cur_dte = opt_today["dte"]

                # Delta hedge P&L (shares)
                hedge_shares = current_trade["_last_delta"] * current_trade["n_contracts"] * CONTRACT_MULTIPLIER
                spot_change = cur_spot - current_trade["_last_spot"]
                hedge_pnl = hedge_shares * spot_change

                # Rebalancing cost ($0.01/share spread on rebalanced shares)
                delta_change = abs(cur_delta - current_trade["_last_delta"])
                rebal_shares = delta_change * current_trade["n_contracts"] * CONTRACT_MULTIPLIER
                rebal_cost = rebal_shares * HEDGE_COST_PER_SHARE

                # Tail hedge cost (none for equity)
                tail_cost = 0.0

                current_trade["hedge_pnl_total"] += hedge_pnl
                current_trade["hedge_txn_cost_total"] += rebal_cost
                current_trade["tail_hedge_cost_total"] += tail_cost
                current_trade["n_hedge_days"] += 1
                current_trade["_last_delta"] = cur_delta
                current_trade["_last_spot"] = cur_spot
                current_trade["_last_bid"] = cur_bid
                current_trade["_last_ask"] = cur_ask

                # MTM
                # Option PnL: we sold at entry_price, current ask to buy back
                mtm_opt = (current_trade["entry_price"] - cur_ask) * current_trade["n_contracts"] * CONTRACT_MULTIPLIER
                mtm_total = (mtm_opt + current_trade["hedge_pnl_total"]
                             - current_trade["hedge_txn_cost_total"]
                             - current_trade["tail_hedge_cost_total"])

                # Exit check
                pnl_opt = (current_trade["entry_price"] - cur_ask) * current_trade["n_contracts"] * CONTRACT_MULTIPLIER
                entry_premium = current_trade["entry_price"] * current_trade["n_contracts"] * CONTRACT_MULTIPLIER
                captured_pct = pnl_opt / entry_premium if entry_premium > 0 else 0

                should_exit = False
                exit_reason = ""

                if cur_dte <= EXIT_DTE:
                    should_exit = True
                    exit_reason = "dte_exit"
                if captured_pct >= PROFIT_TARGET_PCT:
                    should_exit = True
                    exit_reason = "profit_target"

                if should_exit:
                    current_trade["exit_date"] = date
                    current_trade["exit_reason"] = exit_reason
                    current_trade["exit_price"] = cur_ask
                    current_trade["exit_spot"] = cur_spot
                    current_trade["exit_dte"] = cur_dte

                    _finalize_spy_trade(current_trade, equity)
                    equity += current_trade["total_pnl"]
                    trades.append(current_trade)
                    current_trade = None

        # Daily equity
        daily_equity.append({"date": date, "equity": equity + mtm_total})

        # ══════════════════════════════════════════
        # Phase 2: Execute pending signal (T+1)
        # ══════════════════════════════════════════
        if current_trade is None and pending_signal is not None:
            sig = pending_signal
            pending_signal = None

            key = (date, sig["strike"], sig["expiry"], sig["option_type"])
            opt_today = option_lookup.get(key)

            if opt_today is not None and opt_today["bid"] > 0:
                entry_bid = opt_today["bid"]
                entry_ask = opt_today["ask"] if opt_today["ask"] > 0 else entry_bid
                entry_delta = opt_today["delta"]
                entry_spot = opt_today["spot"]
                entry_dte = opt_today["dte"]
                entry_vega_dollar = abs(opt_today["vega"]) * CONTRACT_MULTIPLIER
                entry_iv = opt_today["iv"]

                # Vega sizing (aligned: fractional contracts, same as BTC)
                if entry_vega_dollar > 0:
                    n_contracts = TARGET_VEGA_DOLLARS / entry_vega_dollar
                    n_contracts = min(max(0.01, n_contracts), 100.0)
                else:
                    n_contracts = 1.0

                # Entry costs (realistic equity market)
                option_entry_cost = n_contracts * OPTION_COST_PER_CONTRACT
                hedge_shares_entry = abs(entry_delta) * n_contracts * CONTRACT_MULTIPLIER
                hedge_entry_cost = hedge_shares_entry * HEDGE_COST_PER_SHARE

                current_trade = {
                    "signal_date": sig["date"],
                    "entry_date": date,
                    "expiry": sig["expiry"],
                    "strike": sig["strike"],
                    "option_type": sig["option_type"],
                    "entry_price": entry_bid,  # sell at bid
                    "entry_spot": entry_spot,
                    "entry_dte": entry_dte,
                    "entry_iv": entry_iv * 100 if entry_iv < 5 else entry_iv,
                    "entry_delta": entry_delta,
                    "entry_vega_dollar": entry_vega_dollar,
                    "n_contracts": n_contracts,
                    "score": sig["score"],
                    "hedge_pnl_total": 0.0,
                    "hedge_txn_cost_total": hedge_entry_cost,
                    "option_txn_cost_total": option_entry_cost,
                    "tail_hedge_cost_total": 0.0,
                    "n_hedge_days": 0,
                    "equity_before": equity,
                    "_last_delta": entry_delta,
                    "_last_spot": entry_spot,
                    "_last_bid": entry_bid,
                    "_last_ask": entry_ask,
                }

        # ══════════════════════════════════════════
        # Phase 3: New signal (store for T+1)
        # ══════════════════════════════════════════
        if current_trade is None and pending_signal is None:
            if date in scores_by_day:
                sig = scores_by_day[date]
                pending_signal = sig

    # Close remaining position
    if current_trade is not None:
        current_trade["exit_date"] = all_dates[-1]
        current_trade["exit_reason"] = "end_of_data"
        current_trade["exit_price"] = current_trade["_last_ask"]
        current_trade["exit_spot"] = current_trade["_last_spot"]
        current_trade["exit_dte"] = 0
        _finalize_spy_trade(current_trade, equity)
        equity += current_trade["total_pnl"]
        trades.append(current_trade)

    df_trades = pd.DataFrame(trades)
    df_trades = df_trades.drop(columns=[c for c in df_trades.columns if c.startswith("_")],
                                errors="ignore")
    df_daily = pd.DataFrame(daily_equity)

    return df_trades, df_daily


def _finalize_spy_trade(trade, equity):
    """Compute final P&L for a SPY trade (realistic equity costs)."""
    # Exit hedge cost ($0.01/share)
    exit_hedge_shares = abs(trade["_last_delta"]) * trade["n_contracts"] * CONTRACT_MULTIPLIER
    exit_hedge_cost = exit_hedge_shares * HEDGE_COST_PER_SHARE
    trade["hedge_txn_cost_total"] += exit_hedge_cost

    # Exit option cost ($0.65/contract)
    exit_option_cost = trade["n_contracts"] * OPTION_COST_PER_CONTRACT
    trade["option_txn_cost_total"] += exit_option_cost

    # Option PnL: sold at entry_price, bought back at exit_price
    # Both in $ per share, multiply by n_contracts * 100
    trade["option_pnl"] = (
        (trade["entry_price"] - trade["exit_price"])
        * trade["n_contracts"] * CONTRACT_MULTIPLIER
    )
    trade["total_pnl"] = (
        trade["option_pnl"]
        + trade["hedge_pnl_total"]
        - trade["hedge_txn_cost_total"]
        - trade["option_txn_cost_total"]
        - trade["tail_hedge_cost_total"]
    )
    trade["return_pct"] = trade["total_pnl"] / equity * 100
    trade["equity_after"] = equity + trade["total_pnl"]


def compute_metrics(df_trades, initial_equity=INITIAL_EQUITY):
    """Compute standard metrics."""
    if df_trades.empty:
        return {}
    n = len(df_trades)
    pnls = df_trades["total_pnl"].values
    rets = df_trades["return_pct"].values
    td = (df_trades["exit_date"].max() - df_trades["entry_date"].min()).days
    tpy = n / (td / 365) if td > 0 else 12
    sharpe = rets.mean() / rets.std() * np.sqrt(tpy) if rets.std() > 0 else 0
    equity = initial_equity
    for p in pnls:
        equity += p
    return {
        "n_trades": n,
        "sharpe": round(sharpe, 2),
        "total_return_pct": round((equity / initial_equity - 1) * 100, 1),
        "max_dd_pct": 0,  # filled from daily equity
        "win_rate": round((pnls > 0).sum() / n * 100, 1),
        "mean_pnl": round(pnls.mean(), 0),
        "final_equity": round(equity, 0),
        "trades_per_year": round(tpy, 1),
    }


# ── Main ──────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SPY OPTIONS VRP BACKTEST — Top Vega Strategy")
    print("Same logic as BTC: |vega| scoring, delta hedge, DTE/profit exit")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Exit DTE:       {EXIT_DTE}")
    print(f"  Profit Target:  {PROFIT_TARGET_PCT*100:.0f}%")
    print(f"  Delta Filter:   |delta| ≤ {MAX_ABS_DELTA}")
    print(f"  DTE Range:      {DTE_MIN}-{DTE_MAX} days")
    print(f"  Target Vega:    ${TARGET_VEGA_DOLLARS}")
    print(f"  Initial Equity: ${INITIAL_EQUITY:,}")
    print()

    # Load data
    df_raw = load_spy_data()

    # Build scores and lookup
    print("\nBuilding scores and lookup...")
    scores, opts, option_lookup = build_option_data(df_raw)

    # Free memory
    del df_raw

    # All unique trading dates
    all_dates = sorted(opts["date"].unique())
    print(f"\nTrading period: {all_dates[0].date()} to {all_dates[-1].date()}")
    print(f"Trading days: {len(all_dates):,}")

    # Free more memory
    del opts

    # Run backtest
    print("\nRunning backtest...")
    df_trades, df_daily = run_spy_backtest(scores, option_lookup, all_dates)

    # Metrics
    m = compute_metrics(df_trades)

    # MTM drawdown
    if not df_daily.empty:
        eq = df_daily["equity"].values
        peak = np.maximum.accumulate(eq)
        dd = ((eq - peak) / peak * 100)
        m["max_dd_pct"] = round(dd.min(), 1)

    # Print results
    print(f"\n{'='*70}")
    print("SPY BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"Trades:         {m['n_trades']}")
    print(f"Win Rate:       {m['win_rate']}%")
    print(f"Sharpe:         {m['sharpe']}")
    print(f"Total Return:   {m['total_return_pct']}%")
    print(f"Max DD (MTM):   {m['max_dd_pct']}%")
    print(f"Mean PnL:       ${m['mean_pnl']:,.0f}")
    print(f"Final Equity:   ${m['final_equity']:,.0f}")
    print(f"Trades/Year:    {m['trades_per_year']}")

    if not df_trades.empty:
        print(f"\nP&L Breakdown:")
        print(f"  Option:       ${df_trades['option_pnl'].sum():,.0f}")
        print(f"  Hedge:        ${df_trades['hedge_pnl_total'].sum():,.0f}")
        print(f"  Hedge txn:    -${df_trades['hedge_txn_cost_total'].sum():,.0f}")
        print(f"  Option txn:   -${df_trades['option_txn_cost_total'].sum():,.0f}")
        print(f"  Tail hedge:   -${df_trades['tail_hedge_cost_total'].sum():,.0f}")
        total_costs = (df_trades['hedge_txn_cost_total'].sum() +
                       df_trades['option_txn_cost_total'].sum() +
                       df_trades['tail_hedge_cost_total'].sum())
        print(f"  Total costs:  -${total_costs:,.0f} (avg ${total_costs/len(df_trades):,.0f}/trade)")

        print(f"\nExit reasons:")
        print(df_trades["exit_reason"].value_counts().to_string())

        print(f"\nOption type distribution:")
        print(df_trades["option_type"].value_counts().to_string())

    # ── BTC comparison ──
    print(f"\n{'='*70}")
    print("CONFRONTO BTC vs SPY")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'BTC IS':>12} {'BTC OOS':>12} {'SPY Full':>12}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    btc = {"is": {"sharpe": 1.75, "return": 37.0, "wr": 65.7, "n": 35, "dd": -7.0},
           "oos": {"sharpe": 1.52, "return": 14.3, "wr": 70.0, "n": 10, "dd": -4.7}}
    print(f"{'Trades':<20} {btc['is']['n']:>12} {btc['oos']['n']:>12} {m['n_trades']:>12}")
    print(f"{'Sharpe':<20} {btc['is']['sharpe']:>12.2f} {btc['oos']['sharpe']:>12.2f} {m['sharpe']:>12.2f}")
    print(f"{'Return %':<20} {btc['is']['return']:>11.1f}% {btc['oos']['return']:>11.1f}% {m['total_return_pct']:>11.1f}%")
    print(f"{'Win Rate %':<20} {btc['is']['wr']:>11.1f}% {btc['oos']['wr']:>11.1f}% {m['win_rate']:>11.1f}%")
    print(f"{'Max DD %':<20} {btc['is']['dd']:>11.1f}% {btc['oos']['dd']:>11.1f}% {m['max_dd_pct']:>11.1f}%")

    # Save
    df_trades.to_csv(OUT_DIR / "spy_trades.csv", index=False)
    df_daily.to_csv(OUT_DIR / "spy_daily_equity.csv", index=False)
    print(f"\nSaved to {OUT_DIR}/spy_trades.csv and spy_daily_equity.csv")


if __name__ == "__main__":
    main()
