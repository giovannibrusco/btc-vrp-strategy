"""
Money Management module.

Features:
- Position sizing strategies (fixed fraction, Kelly, volatility-adjusted)
- Equity curve simulation with MM applied
- Risk metrics (Sharpe, Sortino, Max DD, Calmar)
- Comparative analysis vs BTC Buy & Hold
"""

import numpy as np
import pandas as pd


def fixed_fraction_sizing(equity, risk_pct=0.02):
    """
    Fixed fractional position sizing.

    Parameters:
        equity: current equity
        risk_pct: fraction of equity to risk per trade

    Returns:
        position_size (in USD)
    """
    return equity * risk_pct


def kelly_sizing(win_rate, avg_win, avg_loss, max_fraction=0.25):
    """
    Kelly Criterion position sizing.

    f* = (p * b - q) / b
    where p = win_rate, q = 1-p, b = avg_win / avg_loss

    Parameters:
        win_rate: probability of winning
        avg_win: average winning trade return
        avg_loss: average losing trade return (positive number)
        max_fraction: cap on Kelly fraction

    Returns:
        kelly_fraction (0 to max_fraction)
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0

    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p

    kelly = (p * b - q) / b
    kelly = max(0, min(kelly, max_fraction))
    return kelly


def volatility_adjusted_sizing(equity, realized_vol, target_vol=0.15, max_leverage=3.0):
    """
    Volatility-targeting position sizing.

    leverage = target_vol / realized_vol (capped at max_leverage)

    Parameters:
        equity: current equity
        realized_vol: annualized realized volatility (decimal)
        target_vol: target annualized volatility
        max_leverage: maximum leverage allowed

    Returns:
        position_size (in USD)
    """
    if realized_vol <= 0:
        return equity
    leverage = min(target_vol / realized_vol, max_leverage)
    return equity * leverage


def simulate_equity_curve(df_trades, initial_equity=10_000_000, sizing="fixed",
                          risk_pct=0.02, rolling_vol_window=30,
                          prior_returns=None):
    """
    Simulate equity curve with position sizing.

    Parameters:
        df_trades: DataFrame with Return_Pct, Entry_Date columns
        initial_equity: starting capital
        sizing: "fixed", "kelly_full", "kelly_half", or "vol_target"
        risk_pct: risk per trade (for fixed sizing)
        rolling_vol_window: window for vol estimation
        prior_returns: array of prior trade returns (decimal) for Kelly warmup
                       (e.g. IS returns when simulating OOS)

    Returns:
        DataFrame with Date, Equity, Position_Size, Return
    """
    df = df_trades.sort_values("Entry_Date").reset_index(drop=True)
    equity = initial_equity
    equity_curve = []

    returns = df["Return_Pct"].values / 100
    has_sl = "SL_Pct" in df.columns
    has_spot = "Spot_Entry" in df.columns

    # Prepend prior returns for Kelly walk-forward warmup
    if prior_returns is not None and len(prior_returns) > 0:
        _prior = np.asarray(prior_returns, dtype=float)
    else:
        _prior = np.array([], dtype=float)
    # Combined history: prior + current (used for Kelly lookback)
    all_returns = np.concatenate([_prior, returns])
    n_prior = len(_prior)

    for i, row in df.iterrows():
        trade_return = row["Return_Pct"] / 100
        # Index into all_returns for this trade
        hist_idx = n_prior + i

        if sizing in ("kelly_full", "kelly_half") and hist_idx >= 10:
            # Walk-forward Kelly (matches notebook 04 logic)
            past_returns = all_returns[:hist_idx]
            wins = past_returns[past_returns > 0]
            losses = past_returns[past_returns < 0]
            wr = len(wins) / len(past_returns) if len(past_returns) > 0 else 0
            avg_w = wins.mean() if len(wins) > 0 else 0
            avg_l = -losses.mean() if len(losses) > 0 else 0
            wlr = avg_w / avg_l if avg_l > 0 else 1.0
            kelly_f = wr - (1 - wr) / wlr
            kelly_f = max(0, kelly_f)
            if sizing == "kelly_half":
                kelly_f *= 0.5

            sl_pct = row["SL_Pct"] if has_sl else 0.05
            btc_price = row["Spot_Entry"] if has_spot else 1.0
            dollar_risk = equity * kelly_f
            position_usd = dollar_risk / sl_pct if sl_pct > 0 else 0
            contracts = max(0, np.floor(position_usd / (10 * btc_price)))
            position_size = contracts * 10 * btc_price
            position_frac = kelly_f
        elif sizing in ("kelly_full", "kelly_half") and hist_idx < 10:
            # Warmup: no position (matches notebook fallback)
            position_size = 0
            position_frac = 0
        elif sizing == "vol_target" and i >= rolling_vol_window:
            past_vol = np.std(returns[max(0, i - rolling_vol_window):i]) * np.sqrt(252)
            position_frac = min(0.15 / past_vol, 3.0) if past_vol > 0 else 1.0
            position_size = equity * position_frac
        else:
            position_frac = risk_pct if sizing == "fixed" else 0.02
            position_size = equity * position_frac

        pnl = position_size * trade_return
        equity += pnl

        equity_curve.append({
            "Date": row["Entry_Date"],
            "Equity": equity,
            "Position_Size": position_size,
            "Position_Fraction": position_frac,
            "Return_Pct": trade_return * 100,
            "PnL": pnl,
        })

    return pd.DataFrame(equity_curve)


def compute_risk_metrics(equity_curve, initial_equity=10_000_000):
    """
    Compute comprehensive risk metrics from an equity curve.

    Returns:
        dict with Sharpe, Sortino, Calmar, MaxDD, CAGR, TotalReturn, etc.
    """
    df = equity_curve.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    if len(df) < 2:
        return {}

    returns = df["Return_Pct"].values / 100
    n_days = (df["Date"].max() - df["Date"].min()).days + 1
    n_years = max(n_days / 365, 0.01)
    trades_per_year = len(df) / n_days * 365

    total_return = (df["Equity"].iloc[-1] / initial_equity - 1) * 100
    cagr = ((df["Equity"].iloc[-1] / initial_equity) ** (1 / n_years) - 1) * 100

    avg_ret = returns.mean()
    std_ret = returns.std()
    sharpe = avg_ret / std_ret * np.sqrt(trades_per_year) if std_ret > 0 else 0

    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-8
    sortino = avg_ret / downside_std * np.sqrt(trades_per_year) if downside_std > 0 else 0

    equity_series = df["Equity"].values
    peak = np.maximum.accumulate(equity_series)
    drawdown = (equity_series - peak) / peak
    max_dd = drawdown.min() * 100

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        "Total_Return_Pct": total_return,
        "CAGR_Pct": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max_Drawdown_Pct": max_dd,
        "N_Trades": len(df),
        "Win_Rate_Pct": (returns > 0).mean() * 100,
        "Avg_Return_Pct": avg_ret * 100,
    }


def compare_with_btc(equity_curve, df_btc_daily, initial_equity=10_000_000):
    """
    Compare strategy equity curve with BTC buy-and-hold.

    Returns:
        dict with strategy_metrics, btc_metrics, differential
    """
    strat_metrics = compute_risk_metrics(equity_curve, initial_equity)

    btc = df_btc_daily.copy()
    btc["date"] = pd.to_datetime(btc["date"])
    btc = btc.sort_values("date")

    # Filter to same date range
    start = pd.to_datetime(equity_curve["Date"].min())
    end = pd.to_datetime(equity_curve["Date"].max())
    btc = btc[(btc["date"] >= start) & (btc["date"] <= end)]

    if btc.empty:
        return {"strategy": strat_metrics, "btc": {}, "alpha_vs_btc": None}

    btc_returns = btc["close"].pct_change().dropna().values
    n_years = (btc["date"].max() - btc["date"].min()).days / 365
    btc_total = (btc["close"].iloc[-1] / btc["close"].iloc[0] - 1) * 100
    btc_cagr = ((btc["close"].iloc[-1] / btc["close"].iloc[0]) ** (1 / max(n_years, 0.01)) - 1) * 100
    btc_sharpe = btc_returns.mean() / btc_returns.std() * np.sqrt(365) if btc_returns.std() > 0 else 0

    btc_equity = btc["close"] / btc["close"].iloc[0] * initial_equity
    btc_peak = btc_equity.cummax()
    btc_dd = ((btc_equity - btc_peak) / btc_peak).min() * 100

    btc_metrics = {
        "Total_Return_Pct": btc_total,
        "CAGR_Pct": btc_cagr,
        "Sharpe": btc_sharpe,
        "Max_Drawdown_Pct": btc_dd,
    }

    return {
        "strategy": strat_metrics,
        "btc": btc_metrics,
        "alpha_vs_btc": strat_metrics.get("CAGR_Pct", 0) - btc_cagr,
    }
