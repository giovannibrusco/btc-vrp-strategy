"""
Statistical Validation module.

Features:
- T-test on alpha (H0: mean return = 0)
- Bootstrap confidence intervals for Sharpe ratio
- Hansen SPA test for multiple testing correction
- Factor benchmark regression (momentum, IV change, carry)
- Regime analysis by realized volatility
- Exit type breakdown and P&L contribution
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ============================================================
# Alpha significance tests
# ============================================================

def ttest_alpha(returns):
    """
    T-test on mean return (H0: alpha = 0).

    Parameters:
        returns: array-like of trade returns (in decimal, not %)

    Returns:
        dict with t_stat, p_value, mean, se, significant_5pct
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    if len(returns) < 2:
        return {"t_stat": np.nan, "p_value": np.nan, "mean": np.nan,
                "se": np.nan, "significant_5pct": False}

    t_stat, p_value = sp_stats.ttest_1samp(returns, 0)
    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "mean": returns.mean(),
        "se": returns.std(ddof=1) / np.sqrt(len(returns)),
        "significant_5pct": p_value < 0.05,
        "n_obs": len(returns),
    }


def bootstrap_sharpe_ci(returns, n_bootstrap=10000, confidence=0.95, seed=42):
    """
    Bootstrap confidence interval for Sharpe ratio (Ledoit-Wolf 2008 approach).

    Parameters:
        returns: array of trade returns (decimal)
        n_bootstrap: number of bootstrap samples
        confidence: confidence level (e.g. 0.95)
        seed: random seed

    Returns:
        dict with sharpe, ci_lower, ci_upper, se
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    n = len(returns)

    if n < 5:
        return {"sharpe": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "se": np.nan}

    rng = np.random.RandomState(seed)
    sharpe_orig = returns.mean() / returns.std(ddof=1) if returns.std(ddof=1) > 0 else 0

    # Bootstrap
    sharpe_boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = returns[rng.randint(0, n, size=n)]
        std_s = sample.std(ddof=1)
        sharpe_boot[b] = sample.mean() / std_s if std_s > 0 else 0

    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(sharpe_boot, alpha * 100)
    ci_upper = np.percentile(sharpe_boot, (1 - alpha) * 100)

    return {
        "sharpe": sharpe_orig,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": sharpe_boot.std(),
    }


def hansen_spa_test(base_returns, model_returns_list, n_bootstrap=10000, seed=42):
    """
    Simplified Hansen's Superior Predictive Ability (SPA) test.

    Tests H0: no model beats the base (buy-and-hold or zero benchmark).

    Parameters:
        base_returns: array of benchmark returns (e.g., zeros or buy-and-hold)
        model_returns_list: list of arrays, each a strategy's returns
        n_bootstrap: number of bootstrap iterations
        seed: random seed

    Returns:
        dict with p_value, max_t_stat, n_models
    """
    base_returns = np.asarray(base_returns, dtype=float)
    n = len(base_returns)
    n_models = len(model_returns_list)

    if n_models == 0 or n < 5:
        return {"p_value": 1.0, "max_t_stat": 0, "n_models": 0}

    # Compute excess returns for each model
    d_list = []
    for model_ret in model_returns_list:
        model_ret = np.asarray(model_ret, dtype=float)
        min_len = min(len(model_ret), n)
        d = model_ret[:min_len] - base_returns[:min_len]
        d_list.append(d)

    # Truncate all to the same minimum length
    min_obs = min(len(d) for d in d_list)
    d_matrix = np.array([d[:min_obs] for d in d_list])  # shape: (n_models, min_obs)
    d_bar = d_matrix.mean(axis=1)

    # Standard errors
    se = np.array([d_matrix[i].std(ddof=1) / np.sqrt(min_obs) for i in range(n_models)])
    se[se == 0] = 1e-8

    # Test statistic
    t_stats = d_bar / se
    max_t_stat = t_stats.max()

    # Bootstrap p-value
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_bootstrap):
        idx = rng.randint(0, min_obs, size=min_obs)
        d_boot = d_matrix[:, idx]
        d_bar_boot = d_boot.mean(axis=1) - d_bar  # center
        se_boot = np.array([d_boot[i].std(ddof=1) / np.sqrt(min_obs) for i in range(n_models)])
        se_boot[se_boot == 0] = 1e-8
        t_boot = d_bar_boot / se_boot
        if t_boot.max() >= max_t_stat:
            count += 1

    p_value = count / n_bootstrap

    return {
        "p_value": p_value,
        "max_t_stat": max_t_stat,
        "n_models": n_models,
    }


# ============================================================
# Factor benchmark regression
# ============================================================

def compute_factor_returns(df_trades, df_btc_daily, df_signals=None):
    """
    Compute factor returns aligned to trade dates.

    Factors:
    1. BTC Momentum (5-day return)
    2. IV ATM change (if available in signals)
    3. BTC daily return (buy-and-hold proxy)

    Parameters:
        df_trades: DataFrame of trades with Entry_Date, Return_Pct
        df_btc_daily: daily BTC OHLC
        df_signals: signals with RND features (optional)

    Returns:
        DataFrame with trade returns and factor columns
    """
    df = df_trades.copy()
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"]).dt.normalize()

    btc = df_btc_daily.copy()
    btc["date"] = pd.to_datetime(btc["date"]).dt.normalize()
    btc = btc.sort_values("date")

    # BTC daily return
    btc["btc_daily_ret"] = btc["close"].pct_change()

    # BTC momentum (5-day)
    btc["btc_momentum_5d"] = btc["close"].pct_change(5)

    # Realized volatility (30-day rolling)
    btc["realized_vol_30d"] = btc["btc_daily_ret"].rolling(30).std() * np.sqrt(365) * 100

    # Merge
    df = df.merge(btc[["date", "btc_daily_ret", "btc_momentum_5d", "realized_vol_30d"]],
                  left_on="Entry_Date", right_on="date", how="left")

    # IV change factor (if signals have RND_Std)
    if df_signals is not None and "RND_Std" in df_signals.columns:
        sig = df_signals[["Date", "RND_Std"]].copy()
        sig["Date"] = pd.to_datetime(sig["Date"]).dt.normalize()
        sig["iv_change"] = sig["RND_Std"].pct_change()
        df = df.merge(sig[["Date", "iv_change"]], left_on="Entry_Date",
                      right_on="Date", how="left", suffixes=("", "_sig"))

    return df


def factor_regression(df_trades_with_factors):
    """
    OLS regression of trade returns on factors.
    The intercept is the alpha after controlling for factors.

    Returns:
        dict with alpha, alpha_tstat, alpha_pvalue, r_squared, factor_betas
    """
    df = df_trades_with_factors.dropna(subset=["Return_Pct"]).copy()
    y = df["Return_Pct"].values / 100  # Convert to decimal

    # Build factor matrix
    factor_cols = []
    for col in ["btc_daily_ret", "btc_momentum_5d", "iv_change"]:
        if col in df.columns:
            factor_cols.append(col)

    if not factor_cols:
        # No factors: just return t-test
        result = ttest_alpha(y)
        result["r_squared"] = 0
        result["factor_betas"] = {}
        result["alpha"] = result["mean"]
        result["alpha_tstat"] = result["t_stat"]
        result["alpha_pvalue"] = result["p_value"]
        return result

    X = df[factor_cols].values
    X = np.column_stack([np.ones(len(X)), X])  # Add intercept

    # Drop NaN rows
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    if len(y) < len(factor_cols) + 2:
        return {"alpha": np.nan, "alpha_tstat": np.nan, "alpha_pvalue": np.nan,
                "r_squared": np.nan, "factor_betas": {}}

    # OLS
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return {"alpha": np.nan, "alpha_tstat": np.nan, "alpha_pvalue": np.nan,
                "r_squared": np.nan, "factor_betas": {}}

    y_hat = X @ beta
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Standard errors
    n, k = X.shape
    sigma2 = ss_res / (n - k)
    try:
        cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        cov_beta = np.full((k, k), np.nan)
    se_beta = np.sqrt(np.diag(cov_beta))

    alpha = beta[0]
    alpha_se = se_beta[0]
    alpha_tstat = alpha / alpha_se if alpha_se > 0 else 0
    alpha_pvalue = 2 * (1 - sp_stats.t.cdf(abs(alpha_tstat), df=n - k))

    factor_betas = {col: {"beta": beta[i + 1], "se": se_beta[i + 1]}
                    for i, col in enumerate(factor_cols)}

    return {
        "alpha": alpha,
        "alpha_tstat": alpha_tstat,
        "alpha_pvalue": alpha_pvalue,
        "r_squared": r_squared,
        "factor_betas": factor_betas,
        "n_obs": n,
    }


# ============================================================
# Regime analysis
# ============================================================

def regime_analysis(df_trades_with_factors, config):
    """
    Segment performance by realized volatility regime.

    Regimes:
    - Low: RV < low_vol threshold
    - Medium: low_vol <= RV < high_vol
    - High: RV >= high_vol

    Returns:
        DataFrame with regime, n_trades, sharpe, win_rate, tp_rate, avg_return
    """
    thresholds = config["statistics"]["regime_thresholds"]
    low = thresholds["low_vol"]
    high = thresholds["high_vol"]

    df = df_trades_with_factors.dropna(subset=["realized_vol_30d"]).copy()

    if df.empty:
        return pd.DataFrame()

    df["regime"] = pd.cut(
        df["realized_vol_30d"],
        bins=[-np.inf, low, high, np.inf],
        labels=["Low", "Medium", "High"]
    )

    results = []
    for regime, group in df.groupby("regime", observed=True):
        returns = group["Return_Pct"] / 100
        n = len(group)
        if n < 2:
            continue

        avg_ret = returns.mean()
        std_ret = returns.std()
        sharpe = avg_ret / std_ret if std_ret > 0 else 0
        win_rate = (returns > 0).mean() * 100
        tp_rate = (group["Exit_Type"] == "TP").mean() * 100 if "Exit_Type" in group.columns else 0

        results.append({
            "regime": regime,
            "n_trades": n,
            "avg_return_pct": avg_ret * 100,
            "std_return_pct": std_ret * 100,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "tp_rate": tp_rate,
        })

    return pd.DataFrame(results)


# ============================================================
# Exit type breakdown
# ============================================================

def exit_type_breakdown(df_trades):
    """
    Breakdown P&L by exit type (TP, SL, EOD, Trailing_SL).

    Returns:
        DataFrame with exit_type, count, pct, avg_return, total_pnl_contribution
    """
    if df_trades.empty or "Exit_Type" not in df_trades.columns:
        return pd.DataFrame()

    df = df_trades.copy()
    total_pnl = df["Return_Pct"].sum()

    results = []
    for exit_type, group in df.groupby("Exit_Type"):
        n = len(group)
        avg_ret = group["Return_Pct"].mean()
        contribution = group["Return_Pct"].sum()
        pct_contribution = contribution / total_pnl * 100 if total_pnl != 0 else 0

        results.append({
            "exit_type": exit_type,
            "count": n,
            "pct_of_trades": n / len(df) * 100,
            "avg_return_pct": avg_ret,
            "total_pnl_pct": contribution,
            "pnl_contribution_pct": pct_contribution,
        })

    return pd.DataFrame(results)


def direction_breakdown(df_trades):
    """Breakdown P&L by direction (LONG/SHORT)."""
    if df_trades.empty or "Direction" not in df_trades.columns:
        return pd.DataFrame()

    df = df_trades.copy()
    results = []
    for direction, group in df.groupby("Direction"):
        returns = group["Return_Pct"] / 100
        n = len(group)
        avg_ret = returns.mean()
        std_ret = returns.std()
        sharpe = avg_ret / std_ret if std_ret > 0 else 0
        win_rate = (returns > 0).mean() * 100

        results.append({
            "direction": direction,
            "n_trades": n,
            "avg_return_pct": avg_ret * 100,
            "win_rate": win_rate,
            "sharpe": sharpe,
        })

    return pd.DataFrame(results)
