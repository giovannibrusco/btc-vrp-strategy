"""
Historical Distribution Engine — P-measure density via KDE.

Computes the real-world (physical) probability distribution of BTC returns
using Kernel Density Estimation on rolling historical log-returns.

This is compared with the RND (Q-measure) to find mispriced options.

Key concepts:
- KDE places a Gaussian kernel on each observed return, sums them → smooth density
- Silverman bandwidth: h = 1.06 * σ * N^(-1/5)
- Returns are computed for T-day horizons (matching option DTE)
- Output density is in strike space for direct comparison with RND
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def compute_t_day_returns(prices: pd.Series, t_days: int,
                          overlapping: bool = True) -> np.ndarray:
    """
    Compute log-returns over a T-day horizon.

    Parameters:
        prices: daily close prices (sorted by date ascending)
        t_days: return horizon in days
        overlapping: if True, use overlapping windows (more data, autocorrelated)
                     if False, non-overlapping (independent, fewer points)

    Returns:
        array of log-returns: ln(P_{i+T} / P_i)
    """
    prices = prices.values if isinstance(prices, pd.Series) else np.asarray(prices)

    if len(prices) <= t_days:
        return np.array([])

    if overlapping:
        returns = np.log(prices[t_days:] / prices[:-t_days])
    else:
        # Non-overlapping: step by t_days
        idx = np.arange(0, len(prices) - t_days, t_days)
        returns = np.log(prices[idx + t_days] / prices[idx])

    # Remove any NaN or Inf
    returns = returns[np.isfinite(returns)]
    return returns


def fit_kde(returns: np.ndarray, bw_method: str = "silverman") -> gaussian_kde | None:
    """
    Fit a Gaussian KDE to observed returns.

    Parameters:
        returns: array of log-returns
        bw_method: bandwidth selection ('silverman', 'scott', or float)

    Returns:
        fitted gaussian_kde object, or None if insufficient data
    """
    if len(returns) < 10:
        return None

    try:
        kde = gaussian_kde(returns, bw_method=bw_method)
        return kde
    except Exception:
        return None


def kde_to_strike_density(kde: gaussian_kde, strikes: np.ndarray,
                          spot: float) -> np.ndarray:
    """
    Evaluate KDE (in return space) at given strikes, converting to strike-space density.

    The KDE is fitted on log-returns r = ln(K/S).
    To convert to strike-space density: f_K(K) = f_r(ln(K/S)) / K

    This ensures the P-density is in the same units as the RND for direct comparison.

    Parameters:
        kde: fitted gaussian_kde on log-returns
        strikes: array of strike prices
        spot: current spot price

    Returns:
        array of density values in strike space (same shape as strikes)
    """
    log_returns = np.log(strikes / spot)
    density_return_space = kde.evaluate(log_returns)

    # Change of variable: f_K(K) = f_r(ln(K/S)) / K
    density_strike_space = density_return_space / strikes

    return density_strike_space


def build_p_distribution(daily_prices: pd.DataFrame, current_date,
                         t_days: int, spot: float, strikes: np.ndarray,
                         lookback_days: int = 365,
                         bw_method: str = "silverman") -> dict | None:
    """
    Build the P-measure (historical) density for a given date and horizon.

    Full pipeline:
    1. Take rolling window of daily prices up to current_date
    2. Compute T-day log-returns
    3. Fit KDE
    4. Convert to strike-space density
    5. Return density aligned with strikes (for comparison with RND)

    Parameters:
        daily_prices: DataFrame with 'date' and 'close' columns
        current_date: the date we're computing for (only use data before this)
        t_days: return horizon in days (matching option DTE)
        spot: current spot price
        strikes: array of strike prices (from RND grid)
        lookback_days: rolling window size for historical returns
        bw_method: KDE bandwidth method

    Returns:
        dict with: kde, density_strike, density_return, returns, bandwidth, n_returns
        or None if insufficient data
    """
    # Ensure date types are compatible
    current_date = pd.Timestamp(current_date)

    dates = pd.to_datetime(daily_prices["date"])
    window_start = current_date - pd.Timedelta(days=lookback_days + t_days)

    # Only use data strictly before current_date (no look-ahead)
    mask = (dates >= window_start) & (dates < current_date)
    df_window = daily_prices[mask].sort_values("date")

    if len(df_window) < t_days + 10:
        return None

    prices = df_window["close"].values

    # Compute T-day returns
    returns = compute_t_day_returns(prices, t_days, overlapping=True)

    if len(returns) < 10:
        return None

    # Fit KDE
    kde = fit_kde(returns, bw_method=bw_method)
    if kde is None:
        return None

    # Evaluate in strike space
    density_strike = kde_to_strike_density(kde, strikes, spot)

    # Also keep return-space density for diagnostics
    log_returns_grid = np.log(strikes / spot)
    density_return = kde.evaluate(log_returns_grid)

    return {
        "kde": kde,
        "density_strike": density_strike,
        "density_return": density_return,
        "returns": returns,
        "bandwidth": kde.factor,
        "n_returns": len(returns),
        "lookback_days": lookback_days,
        "t_days": t_days,
    }


def compute_gap_probability(rnd_density: np.ndarray, p_density: np.ndarray) -> np.ndarray:
    """
    Compute the gap between Q-measure (RND) and P-measure (historical) densities.

    gap_probability(K) = RND(K) - P(K)

    Positive gap → market overprices this region → good candidate for selling.

    Parameters:
        rnd_density: RND density array (Q-measure, from rnd_engine)
        p_density: historical density array (P-measure, from KDE)

    Returns:
        array of gap values (same shape as inputs)
    """
    return rnd_density - p_density
