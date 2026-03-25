"""
RND Engine — Risk-Neutral Density extraction pipeline.

Improvements over notebook 01:
- UnivariateSpline instead of interp1d (C2 continuity)
- Analytic 2nd derivative via spline.derivative(n=2)
- Put-call parity OTM stitching (calls + puts)
- Liquidity filter (OI, bid/ask, spread)
- Proper normalization and moment calculation
"""

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm, lognorm

# Compatibility: _trapz (NumPy 2.0+) vs np.trapz (NumPy 1.x)
_trapz = getattr(np, "trapezoid", np.trapz)


# ============================================================
# Pricing
# ============================================================

def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call price (no dividends)."""
    if sigma <= 0 or T <= 0:
        return np.nan
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes put price (no dividends)."""
    if sigma <= 0 or T <= 0:
        return np.nan
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def put_to_call_parity(put_price, S, K, T, r):
    """Convert put price to call price via put-call parity: C = P + S - K*exp(-rT)."""
    return put_price + S - K * np.exp(-r * T)


# ============================================================
# Liquidity filter
# ============================================================

def apply_liquidity_filter(df_exp, config):
    """
    Filter options by liquidity criteria.

    Removes options with:
    - open_interest < min_oi
    - bid_price <= 0
    - ask_price <= 0
    - bid-ask spread > max_spread_pct
    """
    liq = config["liquidity"]
    min_oi = liq["min_open_interest"]
    max_spread = liq["max_spread_pct"]

    mask = (
        (df_exp["open_interest"] >= min_oi) &
        (df_exp["bid_price"] > 0) &
        (df_exp["ask_price"] > 0)
    )

    # Spread filter: (ask - bid) / ask < max_spread
    spread_ratio = (df_exp["ask_price"] - df_exp["bid_price"]) / df_exp["ask_price"]
    mask = mask & (spread_ratio < max_spread)

    return df_exp[mask].copy()


# ============================================================
# OTM stitching (put-call parity)
# ============================================================

def costruisci_smile_otm(df_exp, S, T, r):
    """
    Build OTM-stitched IV smile using:
    - Put OTM for K < S (forward)
    - Call OTM for K > S
    - Weighted average for K ≈ S (within 2% of spot)

    Parameters:
        df_exp: DataFrame with columns [strike, mark_iv, option_type, ...]
        S: spot/underlying price
        T: time to maturity in years
        r: risk-free rate

    Returns:
        DataFrame with columns [strike, iv_otm] sorted by strike
    """
    df_calls = df_exp[df_exp["option_type"] == "call"].copy()
    df_puts = df_exp[df_exp["option_type"] == "put"].copy()

    if df_calls.empty and df_puts.empty:
        return pd.DataFrame(columns=["strike", "iv_otm"])

    # Aggregate: for each strike, get the IV from the OTM side
    strikes = sorted(df_exp["strike"].unique())
    iv_otm_list = []

    blend_band = 0.02  # 2% around ATM for blending

    for K in strikes:
        call_row = df_calls[df_calls["strike"] == K]
        put_row = df_puts[df_puts["strike"] == K]

        iv_call = call_row["mark_iv"].values[0] if len(call_row) > 0 else np.nan
        iv_put = put_row["mark_iv"].values[0] if len(put_row) > 0 else np.nan

        moneyness = K / S

        if abs(moneyness - 1.0) < blend_band:
            # Near ATM: blend if both available
            if not np.isnan(iv_call) and not np.isnan(iv_put):
                # Weight by distance from ATM
                w = (moneyness - (1.0 - blend_band)) / (2 * blend_band)
                w = np.clip(w, 0, 1)
                iv = w * iv_call + (1 - w) * iv_put
            elif not np.isnan(iv_call):
                iv = iv_call
            elif not np.isnan(iv_put):
                iv = iv_put
            else:
                continue
        elif K > S:
            # OTM call
            iv = iv_call if not np.isnan(iv_call) else iv_put
        else:
            # OTM put (K < S)
            iv = iv_put if not np.isnan(iv_put) else iv_call

        if np.isnan(iv) or iv <= 0:
            continue

        iv_otm_list.append({"strike": K, "iv_otm": iv})

    return pd.DataFrame(iv_otm_list)


# ============================================================
# RND extraction for a single expiry
# ============================================================

def extract_rnd_for_expiry(df_day, expiry, config, use_otm_stitching=True):
    """
    Extract Risk-Neutral Density for a single expiry on a single day.

    Improvements:
    1. Liquidity filter
    2. OTM stitching (put + call)
    3. UnivariateSpline interpolation (C2)
    4. Analytic 2nd derivative

    Parameters:
        df_day: DataFrame of all options for one day
        expiry: the expiry date to process
        config: configuration dict
        use_otm_stitching: if True, use OTM puts + calls

    Returns:
        dict with keys: expiry, K_valid, density, fair_value, std, skewness,
        kurtosis, coverage, oi_total, dte, T, spot
        or None if insufficient data
    """
    rnd_cfg = config["rnd"]
    r = rnd_cfg["risk_free_rate"]
    smoothing = rnd_cfg["spline_smoothing"]
    m_range = tuple(rnd_cfg["moneyness_range"])
    n_grid = rnd_cfg["n_grid_points"]

    # Filter for this expiry
    if "expiry_date" in df_day.columns:
        df_exp = df_day[df_day["expiry_date"] == expiry].copy()
    else:
        df_exp = df_day[df_day["expiry"] == expiry].copy()

    if df_exp.empty:
        return None

    S = df_exp["underlying_price"].iloc[0]
    T = df_exp["T"].iloc[0]
    dte = df_exp["dte"].iloc[0] if "dte" in df_exp.columns else T * 365

    if T <= 0:
        return None

    # Moneyness filter
    min_K, max_K = m_range[0] * S, m_range[1] * S
    df_exp = df_exp[(df_exp["strike"] >= min_K) & (df_exp["strike"] <= max_K)]

    # Liquidity filter
    required_cols = {"open_interest", "bid_price", "ask_price"}
    if required_cols.issubset(df_exp.columns):
        df_exp = apply_liquidity_filter(df_exp, config)

    if len(df_exp) < 5:
        return None

    # Total OI for this expiry
    oi_total = df_exp["open_interest"].sum() if "open_interest" in df_exp.columns else 0

    # Build IV smile
    if use_otm_stitching:
        df_smile = costruisci_smile_otm(df_exp, S, T, r)
        if len(df_smile) < 5:
            # Fallback to calls only
            df_calls = df_exp[df_exp["option_type"] == "call"].dropna(subset=["mark_iv"])
            df_smile = df_calls[["strike", "mark_iv"]].rename(columns={"mark_iv": "iv_otm"})
    else:
        df_calls = df_exp[df_exp["option_type"] == "call"].dropna(subset=["mark_iv"])
        df_smile = df_calls[["strike", "mark_iv"]].rename(columns={"mark_iv": "iv_otm"})

    df_smile = df_smile.sort_values("strike").drop_duplicates(subset=["strike"])

    if len(df_smile) < 5:
        return None

    strikes = df_smile["strike"].values
    ivs = df_smile["iv_otm"].values / 100.0  # Convert from % to decimal

    # Coverage: fraction of moneyness range covered
    coverage = (strikes.max() - strikes.min()) / (max_K - min_K)

    # Fit UnivariateSpline to IV smile
    try:
        iv_spline = UnivariateSpline(strikes, ivs, s=smoothing, k=3)
    except Exception:
        return None

    # Dense strike grid
    K_dense = np.linspace(strikes.min(), strikes.max(), n_grid)
    iv_dense = iv_spline(K_dense)

    # Ensure IV is positive
    iv_dense = np.clip(iv_dense, 1e-4, None)

    # Price calls on the dense grid
    prices = np.array([black_scholes_call(S, K, T, r, sigma)
                       for K, sigma in zip(K_dense, iv_dense)])

    # Fit spline to call prices for analytic 2nd derivative
    try:
        price_spline = UnivariateSpline(K_dense, prices, s=0, k=4)
    except Exception:
        return None

    # Analytic 2nd derivative: d²C/dK²
    d2C_dK2 = price_spline.derivative(n=2)(K_dense)

    # RND = exp(rT) * d²C/dK²
    density = np.exp(r * T) * d2C_dK2

    # Keep only positive density
    mask = (density > 0) & np.isfinite(density)
    K_valid = K_dense[mask]
    rho_valid = density[mask]

    if len(rho_valid) < 5:
        return None

    # Normalize to integrate to 1
    area = _trapz(rho_valid, K_valid)
    if area <= 0:
        return None
    rho_valid = rho_valid / area

    # Moments
    mean = _trapz(K_valid * rho_valid, K_valid)
    var = _trapz((K_valid - mean)**2 * rho_valid, K_valid)
    std = np.sqrt(max(var, 0))

    if std > 0:
        skewness = _trapz(((K_valid - mean) / std)**3 * rho_valid, K_valid)
        kurt = _trapz(((K_valid - mean) / std)**4 * rho_valid, K_valid)
    else:
        skewness = 0.0
        kurt = 3.0

    return {
        "expiry": expiry,
        "K_valid": K_valid,
        "density": rho_valid,
        "fair_value": mean,
        "std": std,
        "skewness": skewness,
        "kurtosis": kurt,
        "coverage": coverage,
        "oi_total": oi_total,
        "dte": dte,
        "T": T,
        "spot": S,
    }


# ============================================================
# Composite scoring for expiry selection
# ============================================================

def score_coverage(df_day, expiries, config):
    """Score each expiry by number of strikes in moneyness range."""
    m_range = tuple(config["rnd"]["moneyness_range"])
    scores = {}

    for expiry in expiries:
        if "expiry_date" in df_day.columns:
            df_exp = df_day[df_day["expiry_date"] == expiry]
        else:
            df_exp = df_day[df_day["expiry"] == expiry]

        if df_exp.empty:
            continue

        S = df_exp["underlying_price"].iloc[0]
        in_range = df_exp[
            (df_exp["strike"] >= m_range[0] * S) &
            (df_exp["strike"] <= m_range[1] * S)
        ]
        scores[expiry] = in_range["strike"].nunique()

    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    mu, sigma = vals.mean(), vals.std()
    if sigma == 0:
        sigma = 1.0
    return {k: (v - mu) / sigma for k, v in scores.items()}


def score_steepness(df_day, expiries, config):
    """Score each expiry by IV smile steepness (mean |dIV/dK|)."""
    m_range = tuple(config["rnd"]["moneyness_range"])
    smoothing = config["rnd"]["spline_smoothing"]
    scores = {}

    for expiry in expiries:
        if "expiry_date" in df_day.columns:
            df_exp = df_day[df_day["expiry_date"] == expiry]
        else:
            df_exp = df_day[df_day["expiry"] == expiry]

        df_calls = df_exp[df_exp["option_type"] == "call"].dropna(subset=["mark_iv"])
        if len(df_calls) < 5:
            continue

        S = df_calls["underlying_price"].iloc[0]
        df_calls = df_calls[
            (df_calls["strike"] >= m_range[0] * S) &
            (df_calls["strike"] <= m_range[1] * S)
        ].sort_values("strike")

        if len(df_calls) < 5:
            continue

        try:
            spline = UnivariateSpline(df_calls["strike"].values,
                                      df_calls["mark_iv"].values, s=smoothing)
            deriv = spline.derivative()(df_calls["strike"].values)
            scores[expiry] = np.mean(np.abs(deriv))
        except Exception:
            continue

    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    mu, sigma = vals.mean(), vals.std()
    if sigma == 0:
        sigma = 1.0
    return {k: (v - mu) / sigma for k, v in scores.items()}


def score_stability(df_day, expiries, config):
    """Score each expiry by IV smile smoothness (inverse residual variance)."""
    m_range = tuple(config["rnd"]["moneyness_range"])
    smoothing = config["rnd"]["spline_smoothing"]
    scores = {}

    for expiry in expiries:
        if "expiry_date" in df_day.columns:
            df_exp = df_day[df_day["expiry_date"] == expiry]
        else:
            df_exp = df_day[df_day["expiry"] == expiry]

        df_calls = df_exp[df_exp["option_type"] == "call"].dropna(subset=["mark_iv"])
        if len(df_calls) < 5:
            continue

        S = df_calls["underlying_price"].iloc[0]
        df_calls = df_calls[
            (df_calls["strike"] >= m_range[0] * S) &
            (df_calls["strike"] <= m_range[1] * S)
        ].sort_values("strike")

        if len(df_calls) < 5:
            continue

        try:
            spline = UnivariateSpline(df_calls["strike"].values,
                                      df_calls["mark_iv"].values, s=smoothing)
            iv_pred = spline(df_calls["strike"].values)
            resid_var = np.var(df_calls["mark_iv"].values - iv_pred)
            scores[expiry] = resid_var
        except Exception:
            continue

    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    mu, sigma = vals.mean(), vals.std()
    if sigma == 0:
        sigma = 1.0
    # Inverse: lower variance = higher score
    return {k: -(v - mu) / sigma for k, v in scores.items()}


def score_informativeness(rnd_results):
    """
    Score each expiry by how much the RND deviates from lognormal.
    Higher deviation = more informative signal.
    """
    scores = {}
    for res in rnd_results:
        if res is None:
            continue
        expiry = res["expiry"]
        K = res["K_valid"]
        rho = res["density"]
        mean = res["fair_value"]
        std = res["std"]

        if std <= 0 or mean <= 0:
            continue

        # Fit lognormal
        var = std**2
        mu_ln = np.log(mean**2 / np.sqrt(var + mean**2))
        sigma_ln = np.sqrt(np.log(1 + var / mean**2))
        lognorm_fit = lognorm.pdf(K, s=sigma_ln, scale=np.exp(mu_ln))

        # L1 deviation
        deviation = np.sum(np.abs(rho - lognorm_fit))
        scores[expiry] = deviation

    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    mu, sigma = vals.mean(), vals.std()
    if sigma == 0:
        sigma = 1.0
    return {k: (v - mu) / sigma for k, v in scores.items()}


def select_best_expiry(df_day, config, rnd_results):
    """
    Select the best expiry using composite score.

    Steps:
    1. Filter expiries by DTE range [min_dte, max_dte]
    2. Compute 4 sub-scores
    3. Weighted average → composite score
    4. Return best expiry

    Returns:
        best_expiry or None
    """
    exp_cfg = config["expiry_selection"]
    min_dte = exp_cfg["min_dte"]
    max_dte = exp_cfg["max_dte"]
    weights = exp_cfg["score_weights"]

    # Get valid RND results within DTE range
    valid_results = [r for r in rnd_results
                     if r is not None and min_dte <= r["dte"] <= max_dte]

    if not valid_results:
        return None

    valid_expiries = [r["expiry"] for r in valid_results]

    # Compute sub-scores
    s_cov = score_coverage(df_day, valid_expiries, config)
    s_steep = score_steepness(df_day, valid_expiries, config)
    s_stab = score_stability(df_day, valid_expiries, config)
    s_info = score_informativeness(valid_results)

    # Composite score
    best_expiry = None
    best_score = -np.inf

    for expiry in valid_expiries:
        sc = (
            weights["coverage"] * s_cov.get(expiry, 0) +
            weights["steepness"] * s_steep.get(expiry, 0) +
            weights["stability"] * s_stab.get(expiry, 0) +
            weights["informativeness"] * s_info.get(expiry, 0)
        )
        if sc > best_score:
            best_score = sc
            best_expiry = expiry

    return best_expiry


def select_expiry_max_gap(rnd_results, config):
    """
    Fallback: select expiry with maximum |gap| (original method).
    Still constrained by DTE range.
    """
    exp_cfg = config["expiry_selection"]
    min_dte = exp_cfg["min_dte"]
    max_dte = exp_cfg["max_dte"]

    best = None
    best_gap = 0

    for r in rnd_results:
        if r is None:
            continue
        if not (min_dte <= r["dte"] <= max_dte):
            continue
        gap = abs((r["fair_value"] - r["spot"]) / r["spot"])
        if gap > best_gap:
            best_gap = gap
            best = r["expiry"]

    return best
