"""
Backtest Engine V2 — Phase 2: Trading Simulation.

Reads precomputed scores (from Phase 1) and simulates:
- Sell the best-scored option each day (if flat)
- Delta hedge daily with Deribit quarterly futures
- Account for basis cost (no funding — quarterly futures)
- Exit at DTE threshold or profit target
- One position at a time

Timing: signal generated at day T close → trade executed at day T+1.
  - Option entry: T+1 bid_price (EOD snapshot — only data available)
  - Quarterly hedge entry: T+1 open price (OHLC data)
  This eliminates any forward-looking bias in signal → execution.

Hedge instrument: quarterly futures (fixed basis cost, no funding).
Option P&L: computed on Deribit option prices.
Hedge P&L: computed on Deribit quarterly future prices.
"""

import numpy as np
import pandas as pd

from src.scoring import check_anomalous_gap


def load_precomputed_scores(path: str) -> pd.DataFrame:
    """Load precomputed scores CSV from Phase 1."""
    df = pd.read_csv(path, parse_dates=["date", "expiry"])
    return df


def _build_option_lookup(df_options: pd.DataFrame) -> dict:
    """
    Build a fast lookup: (date, strike, expiry, option_type) → row dict.

    Used to track a sold option on subsequent days.
    """
    df_options = df_options.copy()
    df_options["date"] = pd.to_datetime(df_options["date"])

    expiry_col = "expiry_date" if "expiry_date" in df_options.columns else "expiry"
    df_options["_expiry"] = pd.to_datetime(df_options[expiry_col])

    lookup = {}
    for _, row in df_options.iterrows():
        key = (
            row["date"].normalize(),
            row["strike"],
            row["_expiry"].normalize(),
            row["option_type"],
        )
        lookup[key] = row

    return lookup


def _build_quarterly_lookup(df_futures: pd.DataFrame) -> tuple[dict, dict]:
    """
    Build lookups for quarterly futures.

    Returns:
        price_lookup: (date, instrument) → close price
        available_lookup: date → list of {instrument, price, dte, expiry}
    """
    df = df_futures.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["dte"] = (df["expiry"] - df["date"]).dt.days

    # Filter out perpetual
    df = df[df["instrument"] != "BTC-PERPETUAL"]

    price_lookup = {}
    available_lookup = {}

    for _, row in df.iterrows():
        d = row["date"].normalize()
        inst = row["instrument"]
        price_lookup[(d, inst)] = row["close"]

        if d not in available_lookup:
            available_lookup[d] = []
        available_lookup[d].append({
            "instrument": inst,
            "price": row["close"],
            "open": row["open"],
            "dte": row["dte"],
            "expiry": row["expiry"],
        })

    return price_lookup, available_lookup


def _select_quarterly(available_lookup: dict, date, min_dte_needed: int) -> dict | None:
    """
    Select the front-month quarterly that expires at least min_dte_needed days from date.

    Returns dict with instrument info, or None if not available.
    """
    candidates = available_lookup.get(date, [])
    valid = [c for c in candidates if c["dte"] >= min_dte_needed]
    if not valid:
        return None
    # Pick shortest DTE among valid (front-month that doesn't expire too early)
    return min(valid, key=lambda x: x["dte"])


def _build_funding_lookup(df_funding: pd.DataFrame) -> dict:
    """Build a fast lookup: date → daily funding rate."""
    lookup = {}
    for _, row in df_funding.iterrows():
        d = pd.Timestamp(row["date"]).normalize()
        lookup[d] = row["funding_rate_daily"]
    return lookup


def _build_spot_lookup(df_spot: pd.DataFrame) -> dict:
    """Build a fast lookup: date → Binance BTCUSDT close price."""
    lookup = {}
    for _, row in df_spot.iterrows():
        d = pd.Timestamp(row["date"]).normalize()
        lookup[d] = row["close"]
    return lookup


# Hedge transaction costs for quarterly futures on Deribit
# Fee: 0.05% taker entry + 0.05% taker exit = 0.10%
# Half-spread: ~0.03% each side = 0.06%
# Slippage: ~0.02% each side = 0.04%
# Total roundtrip: ~0.20%
DEFAULT_HEDGE_FEE_PCT = 0.0005       # 0.05% per side
DEFAULT_HEDGE_SPREAD_PCT = 0.0003    # 0.03% half-spread
DEFAULT_HEDGE_SLIPPAGE_PCT = 0.0002  # 0.02% slippage

# Option transaction costs on Deribit
# Taker fee: 0.03% of underlying per side, capped at 12.5% of option price
DEFAULT_OPTION_FEE_PCT = 0.0003      # 0.03% of underlying per side


def _option_txn_cost_one_side(n_contracts: float, spot: float,
                               option_price: float, fee_pct: float) -> float:
    """
    Transaction cost for one side of an option trade (Deribit).

    Fee = min(fee_pct × spot, 12.5% × option_price × spot) per contract.
    """
    fee_per_contract = min(fee_pct * spot, 0.125 * option_price * spot)
    return fee_per_contract * n_contracts


def _hedge_txn_cost_one_side(volume_btc: float, price: float) -> float:
    """Transaction cost for one side (entry or exit) of a quarterly hedge."""
    cost_pct = DEFAULT_HEDGE_FEE_PCT + DEFAULT_HEDGE_SPREAD_PCT + DEFAULT_HEDGE_SLIPPAGE_PCT
    return volume_btc * price * cost_pct


def _get_expiry_bucket(dte: float) -> str:
    """Classify option into expiry bucket for diversification."""
    if dte <= 21:
        return "short"
    elif dte <= 45:
        return "medium"
    else:
        return "long"


def run_backtest_v2(df_scores: pd.DataFrame,
                    df_options: pd.DataFrame,
                    df_funding: pd.DataFrame,
                    df_spot: pd.DataFrame | None = None,
                    df_futures: pd.DataFrame | None = None,
                    exit_dte: int = 2,
                    profit_target_pct: float = 0.75,
                    initial_equity: float = 100_000,
                    max_vega_loss_pct: float = 0.02,
                    iv_shock_pct: float = 0.10,
                    tail_strike_pct: float = 0.60,
                    gap_anomaly_z: float = 2.0,
                    sizing: str = "fixed",
                    target_vega_usd: float = 500.0,
                    max_abs_delta: float | None = None,
                    option_fee_pct: float = DEFAULT_OPTION_FEE_PCT,
                    max_parallel_positions: int = 1,
                    start_date: str | None = None,
                    end_date: str | None = None,
                    verbose: bool = True,
                    _option_lookup: dict | None = None,
                    _quarterly_price_lookup: dict | None = None,
                    _quarterly_avail_lookup: dict | None = None,
                    **kwargs) -> pd.DataFrame:
    """
    PHASE 2: Run backtest from precomputed scores.

    Hedging with quarterly futures (no funding, fixed basis cost).

    Parameters:
        df_scores: precomputed scores from Phase 1
        df_options: full options dataset (for tracking positions)
        df_funding: daily funding rates (kept for compatibility, not used for quarterly hedge)
        df_spot: Binance spot prices (not used for quarterly hedge)
        df_futures: quarterly futures daily data
        exit_dte: close position when DTE reaches this
        profit_target_pct: close if this fraction of premium captured
        initial_equity: starting equity in USD
        tail_strike_pct: tail hedge put strike as fraction of spot
        gap_anomaly_z: z-score threshold for anomalous gap filter
        sizing: "fixed" (1 contract) or "vega" (target vega per trade)
        target_vega_usd: target vega exposure in USD (only used if sizing="vega")
        verbose: print summary

    Returns:
        DataFrame with one row per completed trade.
        Also stores daily_equity list for mark-to-market drawdown.
    """
    # Build option lookup
    if _option_lookup is not None:
        option_lookup = _option_lookup
    else:
        if verbose:
            print("Building option lookup...")
        option_lookup = _build_option_lookup(df_options)

    # Build quarterly futures lookups
    if _quarterly_price_lookup is not None and _quarterly_avail_lookup is not None:
        qtr_price_lookup = _quarterly_price_lookup
        qtr_avail_lookup = _quarterly_avail_lookup
    elif df_futures is not None:
        if verbose:
            print("Building quarterly futures lookup...")
        qtr_price_lookup, qtr_avail_lookup = _build_quarterly_lookup(df_futures)
    else:
        qtr_price_lookup, qtr_avail_lookup = {}, {}

    # Best score per day (the option we'd sell)
    df_scores["date"] = pd.to_datetime(df_scores["date"])
    df_scores["expiry"] = pd.to_datetime(df_scores["expiry"])

    # Delta filter: exclude deep ITM options (Issue 1)
    if max_abs_delta is not None and "delta" in df_scores.columns:
        before = len(df_scores)
        df_scores = df_scores[df_scores["delta"].abs() <= max_abs_delta]
        if verbose:
            print(f"Delta filter (|delta| <= {max_abs_delta}): {before} → {len(df_scores)} scores")

    # Build scores-by-day lookup for signal selection
    _scores_sorted = df_scores.sort_values(["date", "score"], ascending=[True, False])
    scores_by_day = {date: group for date, group in _scores_sorted.groupby("date")}

    # All trading dates from options
    all_option_dates = sorted(df_options["date"].unique())
    all_option_dates = [pd.Timestamp(d).normalize() for d in all_option_dates]

    # Filter by date range
    if start_date:
        sd = pd.Timestamp(start_date).normalize()
        all_option_dates = [d for d in all_option_dates if d >= sd]
    if end_date:
        ed = pd.Timestamp(end_date).normalize()
        all_option_dates = [d for d in all_option_dates if d <= ed]

    trades = []
    equity = initial_equity
    open_positions = []       # list of trade dicts (replaces single current_trade)
    historical_gaps = []
    pending_signal = None     # Signal from day T, executed at T+1
    daily_equity = []         # Mark-to-market daily equity for accurate drawdown

    for date_ts in all_option_dates:
        date_norm = date_ts.normalize()

        # ============================================
        # Phase 1: Manage ALL open positions
        # ============================================
        to_close = []
        mtm_total = 0.0

        for idx, trade in enumerate(open_positions):
            key = (
                date_norm,
                trade["strike"],
                pd.Timestamp(trade["expiry"]).normalize(),
                trade["option_type"],
            )
            opt_today = option_lookup.get(key)

            if opt_today is None:
                # Data gap → mark for close at last known values
                trade["exit_date"] = date_norm
                trade["exit_reason"] = "data_gap"
                trade["exit_price"] = trade["_last_ask"]
                trade["exit_spot"] = trade["_last_spot"]
                trade["exit_dte"] = 0
                # MTM at last known values
                mtm_opt = (trade["entry_price"] - trade["_last_ask"]) * trade["n_contracts"] * trade["_last_spot"]
                mtm_total += mtm_opt + trade["hedge_pnl_total"] - trade["hedge_txn_cost_total"] - trade["tail_hedge_cost_total"]
                to_close.append(idx)
                continue

            current_bid = opt_today.get("bid_price", 0)
            current_ask = opt_today.get("ask_price", 0) or current_bid
            current_delta = opt_today.get("delta", trade["_last_delta"])
            deribit_spot = opt_today["underlying_price"]
            current_dte = opt_today.get("dte", 0)

            # Track current DTE for bucket diversification
            trade["_current_dte"] = current_dte

            # --- Quarterly future price today ---
            qtr_instrument = trade["_hedge_instrument"]
            qtr_price_today = qtr_price_lookup.get((date_norm, qtr_instrument))
            if qtr_price_today is None:
                qtr_price_today = trade["_last_hedge_price"]

            # --- Delta hedge P&L (on quarterly future) ---
            hedge_pos = trade["_last_delta"] * trade["n_contracts"]
            hedge_price_change = qtr_price_today - trade["_last_hedge_price"]
            hedge_pnl = hedge_pos * hedge_price_change

            # --- Rebalancing txn cost ---
            delta_change = abs(current_delta - trade["_last_delta"])
            rebal_volume = delta_change * trade["n_contracts"]
            rebal_txn = _hedge_txn_cost_one_side(rebal_volume, qtr_price_today)

            # --- Tail hedge cost (amortized) ---
            tail_cost = _estimate_tail_cost(
                deribit_spot, tail_strike_pct, trade["n_contracts"], current_dte
            )

            # Accumulate
            trade["hedge_pnl_total"] += hedge_pnl
            trade["hedge_txn_cost_total"] += rebal_txn
            trade["tail_hedge_cost_total"] += tail_cost
            trade["n_hedge_days"] += 1
            trade["_last_delta"] = current_delta
            trade["_last_spot"] = deribit_spot
            trade["_last_hedge_price"] = qtr_price_today
            trade["_last_bid"] = current_bid
            trade["_last_ask"] = current_ask

            # --- MTM for this position ---
            mtm_opt = (trade["entry_price"] - current_ask) * trade["n_contracts"] * deribit_spot
            mtm_total += mtm_opt + trade["hedge_pnl_total"] - trade["hedge_txn_cost_total"] - trade["tail_hedge_cost_total"]

            # --- Check exit ---
            pnl_btc = (trade["entry_price"] - current_ask) * trade["n_contracts"]
            entry_premium_btc = trade["entry_price"] * trade["n_contracts"]
            captured_pct = pnl_btc / entry_premium_btc if entry_premium_btc > 0 else 0

            should_exit = False
            exit_reason = ""

            if current_dte <= exit_dte:
                should_exit = True
                exit_reason = "dte_exit"
            if captured_pct >= profit_target_pct:
                should_exit = True
                exit_reason = "profit_target"

            if should_exit:
                trade["exit_date"] = date_norm
                trade["exit_reason"] = exit_reason
                trade["exit_price"] = current_ask
                trade["exit_spot"] = deribit_spot
                trade["exit_dte"] = current_dte
                to_close.append(idx)

        # Daily equity: cash + unrealized MTM of all positions (pre-exit)
        daily_equity.append({"date": date_norm, "equity": equity + mtm_total})

        # Finalize and remove closed positions
        for idx in reversed(to_close):
            trade = open_positions[idx]
            if "exit_date" not in trade:
                # safety — shouldn't happen
                continue
            _finalize_trade(trade, equity)
            equity += trade["total_pnl"]
            historical_gaps.append(trade["gap_prob"])
            trades.append(trade)
            open_positions.pop(idx)

        # ============================================
        # Phase 2: Execute pending signal from previous day
        # ============================================
        if len(open_positions) < max_parallel_positions and pending_signal is not None:
            sig = pending_signal
            pending_signal = None  # consume

            # Check bucket availability
            occupied_buckets = {
                _get_expiry_bucket(t.get("_current_dte", t["entry_dte"]))
                for t in open_positions
            }
            sig_bucket = _get_expiry_bucket(sig["dte"])

            if max_parallel_positions <= 1 or sig_bucket not in occupied_buckets:
                # Look up the SAME option on TODAY (T+1)
                key = (
                    date_norm,
                    sig["strike"],
                    pd.Timestamp(sig["expiry"]).normalize(),
                    sig["option_type"],
                )
                opt_today = option_lookup.get(key)

                if opt_today is not None:
                    entry_dte = opt_today.get("dte", sig["dte"])

                    # Re-check bucket with actual T+1 DTE
                    actual_bucket = _get_expiry_bucket(entry_dte)
                    if max_parallel_positions <= 1 or actual_bucket not in occupied_buckets:
                        expected_holding = max(int(entry_dte) - exit_dte, 1)
                        min_dte_needed = expected_holding + 7
                        qtr_selected = _select_quarterly(qtr_avail_lookup, date_norm, min_dte_needed)

                        if qtr_selected is not None:
                            entry_bid = opt_today.get("bid_price", 0)
                            entry_delta = opt_today.get("delta", sig["delta"])
                            entry_spot = opt_today["underlying_price"]
                            entry_iv = opt_today.get("mark_iv", sig["mark_iv"])
                            entry_vega = abs(opt_today.get("vega", sig.get("vega", 1.0)))

                            # Position sizing
                            if sizing == "vega" and entry_vega > 0:
                                n_contracts = target_vega_usd / entry_vega if entry_vega > 0 else 1.0
                                n_contracts = min(max(0.01, n_contracts), 10.0)
                            elif sizing == "capital_vega" and entry_vega > 0:
                                # Capital-based vega sizing:
                                #   vega_target = (equity * risk_per_trade) / (worst_loss_per_vega * buffer_mult)
                                _risk = kwargs.get("risk_per_trade", 0.02)
                                _wlpv = kwargs.get("worst_loss_per_vega", 54.93)
                                _buff = kwargs.get("buffer_multiplier", 1.5)
                                _vega_target = (equity * _risk) / (_wlpv * _buff)
                                n_contracts = _vega_target / entry_vega
                                n_contracts = min(max(0.01, n_contracts), 20.0)
                            elif sizing == "premium_pct":
                                premium_usd = entry_bid * entry_spot
                                target_premium = (target_vega_usd / 100.0) * equity
                                n_contracts = target_premium / premium_usd if premium_usd > 0 else 1.0
                                n_contracts = min(max(0.01, n_contracts), 10.0)
                            else:
                                n_contracts = 1.0

                            qtr_entry_price = qtr_selected["open"]
                            init_hedge_volume = abs(entry_delta) * n_contracts
                            entry_txn = _hedge_txn_cost_one_side(init_hedge_volume, qtr_entry_price)

                            ask_price = opt_today.get("ask_price", 0) or entry_bid
                            option_entry_txn = _option_txn_cost_one_side(
                                n_contracts, entry_spot, entry_bid, option_fee_pct)

                            new_trade = {
                                "signal_date": sig["signal_date"],
                                "entry_date": date_norm,
                                "expiry": sig["expiry"],
                                "strike": sig["strike"],
                                "option_type": sig["option_type"],
                                "entry_price": entry_bid,
                                "entry_spot": entry_spot,
                                "entry_dte": entry_dte,
                                "entry_iv": entry_iv,
                                "entry_delta": entry_delta,
                                "score": sig["score"],
                                "gap_prob": sig["gap_prob"],
                                "n_contracts": n_contracts,
                                "hedge_instrument": qtr_selected["instrument"],
                                "hedge_entry_price": qtr_entry_price,
                                "hedge_entry_dte": qtr_selected["dte"],
                                "hedge_pnl_total": 0.0,
                                "hedge_txn_cost_total": entry_txn,
                                "tail_hedge_cost_total": 0.0,
                                "n_hedge_days": 0,
                                "_hedge_instrument": qtr_selected["instrument"],
                                "_last_delta": entry_delta,
                                "_last_spot": entry_spot,
                                "_last_hedge_price": qtr_entry_price,
                                "_last_bid": entry_bid,
                                "_last_ask": ask_price,
                                "_current_dte": entry_dte,
                                "option_txn_cost_total": option_entry_txn,
                                "_option_fee_pct": option_fee_pct,
                                "rnd_fair_value": sig.get("rnd_fair_value", np.nan),
                                "rnd_std": sig.get("rnd_std", np.nan),
                                "kde_bandwidth": sig.get("kde_bandwidth", np.nan),
                                "equity_before": equity,
                                "entry_vega": entry_vega,
                                "vega_usd": entry_vega * entry_spot,
                            }
                            open_positions.append(new_trade)
                        elif verbose:
                            print(f"  {date_norm.strftime('%Y-%m-%d')}: no quarterly for pending signal, skipping")
                elif verbose:
                    print(f"  {date_norm.strftime('%Y-%m-%d')}: pending option not found on T+1, skipping")

        # ============================================
        # Phase 3: Look for new signal (store as pending for T+1)
        # ============================================
        if len(open_positions) < max_parallel_positions and pending_signal is None:
            occupied_buckets = {
                _get_expiry_bucket(t.get("_current_dte", t["entry_dte"]))
                for t in open_positions
            }

            if date_norm in scores_by_day:
                candidates = scores_by_day[date_norm]
                for _, row in candidates.iterrows():
                    # Bucket diversification check
                    if max_parallel_positions > 1:
                        candidate_bucket = _get_expiry_bucket(row["dte"])
                        if candidate_bucket in occupied_buckets:
                            continue

                    # Anomalous gap filter
                    if check_anomalous_gap(row["gap_prob"], historical_gaps, gap_anomaly_z):
                        continue

                    pending_signal = {
                        "signal_date": date_norm,
                        "expiry": row["expiry"],
                        "strike": row["strike"],
                        "option_type": row["option_type"],
                        "delta": row["delta"],
                        "vega": row.get("vega", 0),
                        "dte": row["dte"],
                        "mark_iv": row["mark_iv"],
                        "bid_price": row["bid_price"],
                        "spot": row["spot"],
                        "score": row["score"],
                        "gap_prob": row["gap_prob"],
                        "rnd_fair_value": row.get("rnd_fair_value", np.nan),
                        "rnd_std": row.get("rnd_std", np.nan),
                        "kde_bandwidth": row.get("kde_bandwidth", np.nan),
                    }
                    break  # one pending signal per day

    # Close any remaining positions
    for trade in open_positions:
        trade["exit_date"] = all_option_dates[-1]
        trade["exit_reason"] = "end_of_data"
        trade["exit_price"] = trade["_last_ask"]
        trade["exit_spot"] = trade["_last_spot"]
        trade["exit_dte"] = 0
        _finalize_trade(trade, equity)
        equity += trade["total_pnl"]
        trades.append(trade)

    if not trades:
        if verbose:
            print("No trades generated.")
        return pd.DataFrame()

    df_trades = pd.DataFrame(trades)

    # Drop internal tracking columns
    df_trades = df_trades.drop(columns=[c for c in df_trades.columns if c.startswith("_")],
                               errors="ignore")

    # Attach daily equity as metadata for mark-to-market drawdown
    df_daily_eq = pd.DataFrame(daily_equity)
    if not df_daily_eq.empty:
        df_trades.attrs["daily_equity"] = df_daily_eq

    if verbose:
        _print_summary(df_trades)

    return df_trades


def _finalize_trade(trade: dict, equity: float):
    """Compute final P&L for a trade."""
    # Exit hedge txn cost
    exit_hedge_volume = abs(trade["_last_delta"]) * trade["n_contracts"]
    exit_txn = _hedge_txn_cost_one_side(exit_hedge_volume, trade["_last_hedge_price"])
    trade["hedge_txn_cost_total"] += exit_txn

    # Exit option txn cost
    fee_pct = trade.get("_option_fee_pct", DEFAULT_OPTION_FEE_PCT)
    exit_option_txn = _option_txn_cost_one_side(
        trade["n_contracts"], trade["exit_spot"],
        trade["exit_price"], fee_pct)
    trade["option_txn_cost_total"] = trade.get("option_txn_cost_total", 0.0) + exit_option_txn

    # Option P&L: compute in BTC, convert to USD at exit spot
    pnl_btc = (trade["entry_price"] - trade["exit_price"]) * trade["n_contracts"]
    trade["option_pnl"] = pnl_btc * trade["exit_spot"]
    trade["total_pnl"] = (
        trade["option_pnl"]
        + trade["hedge_pnl_total"]
        - trade["hedge_txn_cost_total"]
        - trade["tail_hedge_cost_total"]
        - trade["option_txn_cost_total"]
    )
    trade["return_pct"] = trade["total_pnl"] / equity * 100
    trade["equity_after"] = equity + trade["total_pnl"]


def _estimate_tail_cost(spot: float, tail_strike_pct: float,
                        n_contracts: float, dte: float) -> float:
    """
    Estimate daily tail hedge cost.

    Approximation: deep OTM put costs ~0.5-1% of notional annually.
    We use 0.75% annualized as conservative estimate.
    """
    notional = spot * n_contracts
    annual_cost_pct = 0.0075  # 0.75% annualized
    daily_cost = notional * annual_cost_pct / 365
    return daily_cost


def _print_summary(df_trades: pd.DataFrame):
    """Print backtest summary."""
    n = len(df_trades)
    wins = (df_trades["total_pnl"] > 0).sum()
    print(f"\n{'='*50}")
    print(f"BACKTEST V2 RESULTS")
    print(f"{'='*50}")
    print(f"Trades:       {n}")
    print(f"Win rate:     {wins/n*100:.1f}%")
    print(f"Total P&L:    ${df_trades['total_pnl'].sum():,.0f}")
    print(f"Mean P&L:     ${df_trades['total_pnl'].mean():,.0f}")
    print(f"Final equity: ${df_trades['equity_after'].iloc[-1]:,.0f}")
    print(f"\nP&L Breakdown:")
    print(f"  Option:     ${df_trades['option_pnl'].sum():,.0f}")
    print(f"  Hedge:      ${df_trades['hedge_pnl_total'].sum():,.0f}")
    print(f"  Hedge txn:  -${df_trades['hedge_txn_cost_total'].sum():,.0f}")
    print(f"  Option txn: -${df_trades['option_txn_cost_total'].sum():,.0f}")
    print(f"  Tail cost:  -${df_trades['tail_hedge_cost_total'].sum():,.0f}")
    print(f"\nExit reasons:")
    for reason, count in df_trades["exit_reason"].value_counts().items():
        print(f"  {reason}: {count}")


def compute_metrics_v2(df_trades: pd.DataFrame) -> dict:
    """Compute performance metrics for V2 backtest."""
    if df_trades.empty:
        return {}

    pnls = df_trades["total_pnl"].values
    returns_pct = df_trades["return_pct"].values
    n_trades = len(df_trades)
    winners = (pnls > 0).sum()

    # Max drawdown — use mark-to-market daily equity if available
    daily_eq = df_trades.attrs.get("daily_equity", None)
    if daily_eq is not None and not daily_eq.empty:
        eq_mtm = daily_eq["equity"].values
        peak_mtm = np.maximum.accumulate(eq_mtm)
        dd_mtm = (eq_mtm - peak_mtm) / peak_mtm
        max_dd = dd_mtm.min()
    else:
        # Fallback to trade-level equity
        equity_curve = df_trades["equity_after"].values
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_dd = drawdown.min()

    # Sharpe
    df_trades = df_trades.copy()
    df_trades["entry_date"] = pd.to_datetime(df_trades["entry_date"])
    df_trades["exit_date"] = pd.to_datetime(df_trades["exit_date"])
    total_days = (df_trades["exit_date"].max() - df_trades["entry_date"].min()).days
    trades_per_year = n_trades / (total_days / 365) if total_days > 0 else 0
    sharpe = (returns_pct.mean() / returns_pct.std() * np.sqrt(trades_per_year)
              if returns_pct.std() > 0 else 0)

    holding_days = (df_trades["exit_date"] - df_trades["entry_date"]).dt.days
    initial_equity = df_trades["equity_before"].iloc[0]
    final_equity = df_trades["equity_after"].iloc[-1]

    return {
        "total_return_pct": (final_equity / initial_equity - 1) * 100,
        "total_pnl": pnls.sum(),
        "n_trades": n_trades,
        "win_rate": winners / n_trades * 100,
        "mean_pnl": pnls.mean(),
        "median_pnl": np.median(pnls),
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "avg_holding_days": holding_days.mean(),
        "option_pnl_total": df_trades["option_pnl"].sum(),
        "hedge_pnl_total": df_trades["hedge_pnl_total"].sum(),
        "hedge_txn_cost_total": df_trades["hedge_txn_cost_total"].sum(),
        "option_txn_cost_total": df_trades["option_txn_cost_total"].sum() if "option_txn_cost_total" in df_trades.columns else 0.0,
        "tail_hedge_cost_total": df_trades["tail_hedge_cost_total"].sum(),
        "exit_reasons": df_trades["exit_reason"].value_counts().to_dict(),
        "trades_per_year": trades_per_year,
        "initial_equity": initial_equity,
        "final_equity": final_equity,
    }
