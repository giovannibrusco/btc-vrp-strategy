"""
Live Option Scorer — Score options from Deribit real-time data.

Implements two scoring strategies:
  1. Premium/DTE (Theta Harvesting): score = bid_price × spot / DTE
  2. Top Vega (VRP Selling):         score = |vega|

Both select OTM options only (calls K > spot, puts K < spot).
Filters: positive bid, positive vega, positive OI.
"""

import logging
from datetime import datetime, timezone

import numpy as np

from src.deribit_client import DeribitClient

logger = logging.getLogger(__name__)

# Strategy parameter defaults (from backtest research)
PDT_MIN_DTE = 3
PDT_MAX_DTE = 30
VEGA_MIN_DTE = 30
VEGA_MAX_DTE = 120
MIN_BID = 0.0001          # Minimum bid price in BTC
MIN_VEGA = 0.1            # Minimum vega for Top Vega strategy
MIN_OPEN_INTEREST = 0     # Testnet may have 0 OI
MAX_SPREAD_PCT = 0.50     # (ask-bid)/ask < 50%
MAX_ABS_DELTA = 0.50      # Exclude deep ITM options


def fetch_and_score(client: DeribitClient) -> dict:
    """
    Fetch all BTC options from Deribit and score them.

    Returns dict with:
        spot: current BTC price
        timestamp: when data was fetched
        pdt_candidates: list of scored options for Premium/DTE (sorted desc)
        vega_candidates: list of scored options for Top Vega (sorted desc)
        n_options_total: total options fetched
        n_options_filtered: options after filtering
    """
    spot = client.get_index_price()
    logger.info(f"BTC spot: ${spot:,.0f}")

    # Fetch all active options
    instruments = client.get_instruments(currency="BTC", kind="option")
    logger.info(f"Fetched {len(instruments)} option instruments")

    # Get book summary for all options (batch — much faster than individual tickers)
    book_summaries = client.get_book_summary(currency="BTC", kind="option")
    summary_by_name = {s["instrument_name"]: s for s in book_summaries}

    now = datetime.now(timezone.utc)
    options = []

    for inst in instruments:
        name = inst["instrument_name"]
        summary = summary_by_name.get(name)
        if summary is None:
            continue

        # Parse instrument name: BTC-26DEC25-100000-C
        parts = name.split("-")
        if len(parts) != 4:
            continue

        strike = inst["strike"]
        option_type = "call" if parts[3] == "C" else "put"
        expiry_ts = inst["expiration_timestamp"] / 1000
        expiry_dt = datetime.fromtimestamp(expiry_ts, tz=timezone.utc)
        dte = (expiry_dt - now).total_seconds() / 86400

        if dte < 1:
            continue

        # OTM filter
        if option_type == "call" and strike <= spot:
            continue
        if option_type == "put" and strike >= spot:
            continue

        bid = summary.get("bid_price") or 0
        ask = summary.get("ask_price") or 0
        oi = summary.get("open_interest") or 0
        mark_iv = summary.get("mark_iv") or 0
        volume = summary.get("volume") or 0

        # Get greeks from ticker (book_summary doesn't have them)
        # We'll fetch ticker only for filtered candidates later
        moneyness = strike / spot

        options.append({
            "instrument": name,
            "strike": strike,
            "option_type": option_type,
            "expiry": expiry_dt.strftime("%Y-%m-%d"),
            "dte": round(dte, 1),
            "moneyness": round(moneyness, 4),
            "bid_price": bid,
            "ask_price": ask,
            "open_interest": oi,
            "mark_iv": mark_iv,
            "volume": volume,
            "spot": spot,
        })

    n_total = len(options)
    logger.info(f"Parsed {n_total} OTM options")

    # Basic filters
    filtered = []
    for opt in options:
        if opt["bid_price"] < MIN_BID:
            continue
        if opt["ask_price"] > 0 and opt["bid_price"] > 0:
            spread = (opt["ask_price"] - opt["bid_price"]) / opt["ask_price"]
            if spread > MAX_SPREAD_PCT:
                continue
        filtered.append(opt)

    logger.info(f"After basic filters: {len(filtered)} options")

    # Enrich with greeks (fetch tickers for filtered options)
    # Rate limit: batch in groups
    enriched = _enrich_with_greeks(client, filtered)

    # Score for Premium/DTE
    pdt_candidates = _score_premium_dte(enriched, spot)

    # Score for Top Vega
    vega_candidates = _score_top_vega(enriched, spot)

    return {
        "spot": spot,
        "timestamp": now.isoformat(),
        "pdt_candidates": pdt_candidates,
        "vega_candidates": vega_candidates,
        "n_options_total": n_total,
        "n_options_filtered": len(enriched),
    }


def _enrich_with_greeks(client: DeribitClient, options: list) -> list:
    """Fetch ticker data to get greeks (delta, vega, gamma, theta)."""
    enriched = []
    for opt in options:
        try:
            ticker = client.get_ticker(opt["instrument"])
            greeks = ticker.get("greeks", {})
            opt["delta"] = greeks.get("delta", 0)
            opt["vega"] = greeks.get("vega", 0)
            opt["gamma"] = greeks.get("gamma", 0)
            opt["theta"] = greeks.get("theta", 0)

            # Update bid/ask from ticker (more recent)
            opt["bid_price"] = ticker.get("best_bid_price") or opt["bid_price"]
            opt["ask_price"] = ticker.get("best_ask_price") or opt["ask_price"]
            opt["mark_price"] = ticker.get("mark_price", 0)

            enriched.append(opt)
        except Exception as e:
            logger.warning(f"Failed to get ticker for {opt['instrument']}: {e}")
            continue

    logger.info(f"Enriched {len(enriched)} options with greeks")
    return enriched


def _score_premium_dte(options: list, spot: float) -> list:
    """
    Score options for Premium/DTE strategy.

    Score = bid_price × spot / DTE
    Selects: short-DTE OTM options (theta decay).
    """
    candidates = []
    for opt in options:
        dte = opt["dte"]
        if dte < PDT_MIN_DTE or dte > PDT_MAX_DTE:
            continue
        if opt["bid_price"] <= 0:
            continue
        # Delta filter: exclude deep ITM options
        if abs(opt.get("delta", 0)) > MAX_ABS_DELTA:
            continue

        score = opt["bid_price"] * spot / dte
        candidates.append({**opt, "score": score, "strategy": "premium_dte"})

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def _score_top_vega(options: list, spot: float) -> list:
    """
    Score options for Top Vega strategy.

    Score = |vega|
    Selects: long-DTE near-ATM options (VRP selling).
    """
    candidates = []
    for opt in options:
        dte = opt["dte"]
        if dte < VEGA_MIN_DTE or dte > VEGA_MAX_DTE:
            continue

        vega = abs(opt.get("vega", 0))
        if vega < MIN_VEGA:
            continue
        if opt["bid_price"] <= 0:
            continue
        # Delta filter: exclude deep ITM options
        if abs(opt.get("delta", 0)) > MAX_ABS_DELTA:
            continue

        score = vega
        candidates.append({**opt, "score": score, "strategy": "top_vega"})

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def format_signal_report(scoring_result: dict, top_n: int = 5) -> str:
    """Format scoring results for logging/display."""
    lines = []
    lines.append(f"=== Live Scoring Report ===")
    lines.append(f"Time: {scoring_result['timestamp']}")
    lines.append(f"BTC Spot: ${scoring_result['spot']:,.0f}")
    lines.append(f"Options scanned: {scoring_result['n_options_total']} "
                 f"(filtered: {scoring_result['n_options_filtered']})")

    lines.append(f"\n--- Premium/DTE Top {top_n} ---")
    for i, c in enumerate(scoring_result["pdt_candidates"][:top_n]):
        lines.append(
            f"  {i+1}. {c['instrument']:30s}  DTE={c['dte']:5.1f}  "
            f"bid={c['bid_price']:.4f}  IV={c['mark_iv']:.1f}%  "
            f"delta={c['delta']:.3f}  vega={c.get('vega',0):.1f}  "
            f"score={c['score']:.1f}"
        )

    lines.append(f"\n--- Top Vega Top {top_n} ---")
    for i, c in enumerate(scoring_result["vega_candidates"][:top_n]):
        lines.append(
            f"  {i+1}. {c['instrument']:30s}  DTE={c['dte']:5.1f}  "
            f"bid={c['bid_price']:.4f}  IV={c['mark_iv']:.1f}%  "
            f"delta={c['delta']:.3f}  vega={c.get('vega',0):.1f}  "
            f"score={c['score']:.1f}"
        )

    return "\n".join(lines)
