"""
Live Trader — Fully automated paper trading daemon.

Orchestrates the complete trading loop:
  1. Score options (Premium/DTE + Top Vega)
  2. Manage open positions (exit checks, hedge rebalancing)
  3. Open new positions when flat
  4. Log everything

Runs on a configurable schedule (default: every 4 hours).
Falls back to dry-run mode if Deribit authentication fails.
"""

import time
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.deribit_client import DeribitClient
from src.live_scorer import fetch_and_score, format_signal_report
from src.position_manager import PositionManager
from src.order_executor import OrderExecutor

# ── Config ─────────────────────────────────────────────────

# Strategy parameters (from backtest research)
EXIT_DTE = 5                      # Close at ≤5 DTE
PROFIT_TARGET_PCT = 0.50          # Close at ≥50% premium captured

# Capital-based vega sizing — Moderate config
RISK_PCT = 0.02                   # 2% of equity at risk per trade
BUFFER = 1.5                      # Safety multiplier
WORST_LOSS_PER_VEGA = 54.93       # From IS data (worst hist loss per unit vega)
VEGA_MAX_CONTRACTS = 10.0         # Cap

# Schedule
RUN_INTERVAL_HOURS = 4            # Run every N hours
MAX_RETRIES = 3                   # Max retries per cycle

# Paths
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
STATE_DIR = Path(__file__).resolve().parent.parent / "data"
TRADE_LOG = LOG_DIR / "live_trades.csv"
SIGNAL_LOG = LOG_DIR / "live_signals.csv"


# ── Logging Setup ──────────────────────────────────────────

def setup_logging():
    """Configure logging to both file and console."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOG_DIR / f"trader_{datetime.now().strftime('%Y%m%d')}.log"

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))

    root.addHandler(fh)
    root.addHandler(ch)

    return logging.getLogger(__name__)


# ── Core Trading Logic ─────────────────────────────────────

def run_cycle(client: DeribitClient, executor: OrderExecutor,
              pm: PositionManager, logger: logging.Logger):
    """
    Run one complete trading cycle:
      1. Check open positions for exits
      2. Rebalance hedges
      3. Score new candidates
      4. Open new positions if flat
    """
    logger.info("=" * 60)
    logger.info("TRADING CYCLE START")
    logger.info("=" * 60)

    spot = client.get_index_price()
    logger.info(f"BTC Spot: ${spot:,.0f}")
    logger.info(pm.summary())

    # ── Step 1: Manage open positions ──────────────────────
    # Check both slots for exit (legacy PDT may still be open)
    for strategy in ["premium_dte", "top_vega"]:
        pos = pm.get_position(strategy)
        if pos is None:
            continue

        logger.info(f"[{strategy}] Checking position: {pos['instrument']}")

        try:
            ticker = client.get_ticker(pos["instrument"])
        except Exception as e:
            logger.error(f"[{strategy}] Failed to get ticker: {e}")
            continue

        current_bid = ticker.get("best_bid_price") or ticker.get("mark_price", 0)
        current_ask = ticker.get("best_ask_price") or current_bid
        current_delta = ticker.get("greeks", {}).get("delta", pos.get("entry_delta", 0))

        # Compute current DTE
        parts = pos["instrument"].split("-")
        try:
            # Parse expiry from instrument name: BTC-26DEC25-100000-C
            expiry_str = parts[1]
            # Deribit format: DDMMMYY
            expiry_dt = datetime.strptime(expiry_str, "%d%b%y").replace(
                tzinfo=timezone.utc, hour=8)  # Deribit expires at 08:00 UTC
            current_dte = (expiry_dt - datetime.now(timezone.utc)).total_seconds() / 86400
        except Exception:
            current_dte = pos.get("entry_dte", 30) - 1  # rough fallback

        logger.info(f"[{strategy}] bid={current_bid:.4f} delta={current_delta:.3f} "
                     f"DTE={current_dte:.1f}")

        # Check exit conditions (use ask for realistic buy-back cost)
        exit_signal = pm.check_exit(strategy, current_bid, current_dte, spot,
                                     current_ask=current_ask)

        if exit_signal:
            logger.info(f"[{strategy}] EXIT SIGNAL: {exit_signal['exit_reason']}")

            # Close option position
            try:
                executor.close_option_position(pos["instrument"])
            except Exception as e:
                logger.error(f"[{strategy}] Failed to close option: {e}")

            # Close hedge
            hedge_inst = pos.get("hedge_instrument")
            if hedge_inst:
                try:
                    executor.hedge_with_quarterly(
                        hedge_inst,
                        target_delta_btc=0.0,
                        current_hedge_btc=pos.get("current_hedge_btc", 0.0)
                    )
                except Exception as e:
                    logger.error(f"[{strategy}] Failed to close hedge: {e}")

            # Record closure
            pm.close_position(
                strategy=strategy,
                exit_price=exit_signal["exit_price"],
                exit_spot=exit_signal["exit_spot"],
                exit_reason=exit_signal["exit_reason"],
                hedge_pnl=pos.get("hedge_pnl_accumulated", 0.0),
                hedge_txn_cost=pos.get("hedge_txn_cost_accumulated", 0.0),
            )
            _log_trade(pm.state["trades_history"][-1])
            continue

        # ── Rebalance hedge ────────────────────────────────
        target_hedge = pm.compute_hedge_delta(strategy, current_delta)
        current_hedge = pos.get("current_hedge_btc", 0.0)
        hedge_inst = pos.get("hedge_instrument")

        if hedge_inst and abs(target_hedge - current_hedge) > 0.01:
            logger.info(f"[{strategy}] Rebalance hedge: "
                        f"{current_hedge:.4f} → {target_hedge:.4f} BTC")
            try:
                executor.hedge_with_quarterly(
                    hedge_inst,
                    target_delta_btc=target_hedge,
                    current_hedge_btc=current_hedge,
                )
                # Update position state
                pos["current_hedge_btc"] = target_hedge
                pos["last_delta"] = current_delta
                pm.save_state()
            except Exception as e:
                logger.error(f"[{strategy}] Hedge rebalance failed: {e}")

    # ── Step 2: Score new candidates ───────────────────────
    needs_vega = pm.is_flat("top_vega")

    if not needs_vega:
        logger.info("Top Vega has an open position, no new signal needed")
        logger.info(f"CYCLE COMPLETE — equity: ${pm.equity:,.0f}")
        return

    logger.info("Scoring options...")
    try:
        scoring = fetch_and_score(client)
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        return

    logger.info(format_signal_report(scoring, top_n=3))
    _log_signal(scoring)

    # ── Step 3: Open new position (Top Vega) ───────────────
    if scoring["vega_candidates"]:
        best_vega = scoring["vega_candidates"][0]
        # Capital-based vega sizing (moderate config)
        target_vega = (pm.equity * RISK_PCT) / (WORST_LOSS_PER_VEGA * BUFFER)
        option_vega = abs(best_vega.get("vega", 1.0))
        if option_vega > 0:
            n_contracts = target_vega / option_vega
            n_contracts = min(max(0.1, n_contracts), VEGA_MAX_CONTRACTS)
        else:
            n_contracts = 0.1

        logger.info(f"[top_vega] Capital-based sizing: equity=${pm.equity:,.0f} "
                     f"→ target_vega={target_vega:.1f} / option_vega={option_vega:.1f} "
                     f"= {n_contracts:.2f} contracts")

        _open_new_position(
            strategy="top_vega",
            candidate=best_vega,
            n_contracts=n_contracts,
            executor=executor,
            pm=pm,
            client=client,
            logger=logger,
        )

    logger.info(f"CYCLE COMPLETE — equity: ${pm.equity:,.0f}")
    logger.info(pm.summary())


def _open_new_position(strategy: str, candidate: dict, n_contracts: float,
                       executor: OrderExecutor, pm: PositionManager,
                       client: DeribitClient, logger: logging.Logger):
    """Open a new short option position with hedge."""
    instrument = candidate["instrument"]
    logger.info(f"[{strategy}] Opening: {instrument} × {n_contracts:.2f}")

    # Round n_contracts to Deribit minimum (0.1 BTC for options)
    n_contracts = round(n_contracts, 1)
    if n_contracts < 0.1:
        n_contracts = 0.1

    # Sell the option
    try:
        executor.sell_option(instrument, n_contracts, label=strategy)
    except Exception as e:
        logger.error(f"[{strategy}] Failed to sell option: {e}")
        return

    # Find and set up quarterly hedge
    entry_dte = candidate["dte"]
    expected_holding = max(int(entry_dte) - EXIT_DTE, 1)
    min_dte_needed = expected_holding + 7
    quarterly = executor.find_hedge_quarterly(min_dte=min_dte_needed)

    hedge_instrument = None
    initial_hedge_btc = 0.0

    if quarterly:
        hedge_instrument = quarterly["instrument"]
        initial_delta = abs(candidate.get("delta", 0.5))
        initial_hedge_btc = initial_delta * n_contracts

        logger.info(f"[{strategy}] Hedge: {hedge_instrument} "
                     f"delta={initial_delta:.3f} × {n_contracts:.2f} = "
                     f"{initial_hedge_btc:.4f} BTC")
        try:
            executor.hedge_with_quarterly(
                hedge_instrument,
                target_delta_btc=initial_hedge_btc,
                current_hedge_btc=0.0,
            )
        except Exception as e:
            logger.error(f"[{strategy}] Failed to place hedge: {e}")
    else:
        logger.warning(f"[{strategy}] No suitable quarterly found for hedge!")

    # Record position
    position = {
        "instrument": instrument,
        "strike": candidate["strike"],
        "option_type": candidate["option_type"],
        "expiry": candidate["expiry"],
        "entry_price": candidate["bid_price"],
        "entry_spot": candidate["spot"],
        "entry_dte": candidate["dte"],
        "entry_delta": candidate.get("delta", 0),
        "entry_vega": candidate.get("vega", 0),
        "entry_iv": candidate.get("mark_iv", 0),
        "n_contracts": n_contracts,
        "score": candidate["score"],
        "hedge_instrument": hedge_instrument,
        "current_hedge_btc": initial_hedge_btc,
        "hedge_pnl_accumulated": 0.0,
        "hedge_txn_cost_accumulated": 0.0,
    }

    pm.open_position(strategy, position)


# ── Logging Helpers ────────────────────────────────────────

def _log_trade(trade: dict):
    """Append trade to CSV log."""
    import csv
    TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)
    file_exists = TRADE_LOG.exists()

    fields = ["strategy", "instrument", "entry_price", "exit_price",
              "entry_spot", "exit_spot", "n_contracts", "option_pnl",
              "hedge_pnl", "total_pnl", "return_pct", "exit_reason",
              "opened_at", "exit_date"]

    with open(TRADE_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(trade)


def _log_signal(scoring: dict):
    """Append scoring summary to CSV log."""
    import csv
    SIGNAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    file_exists = SIGNAL_LOG.exists()

    row = {
        "timestamp": scoring["timestamp"],
        "spot": scoring["spot"],
        "n_options": scoring["n_options_total"],
        "n_filtered": scoring["n_options_filtered"],
        "pdt_top_instrument": (scoring["pdt_candidates"][0]["instrument"]
                               if scoring["pdt_candidates"] else ""),
        "pdt_top_score": (scoring["pdt_candidates"][0]["score"]
                          if scoring["pdt_candidates"] else 0),
        "vega_top_instrument": (scoring["vega_candidates"][0]["instrument"]
                                if scoring["vega_candidates"] else ""),
        "vega_top_score": (scoring["vega_candidates"][0]["score"]
                           if scoring["vega_candidates"] else 0),
    }

    fields = list(row.keys())
    with open(SIGNAL_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ── Main Entry Point ───────────────────────────────────────

def main():
    """Run the live trader daemon."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("LIVE TRADER STARTING")
    logger.info("=" * 60)

    # Initialize components
    client = DeribitClient()
    executor = OrderExecutor(client)
    pm = PositionManager(
        state_file=STATE_DIR / "live_state.json",
        exit_dte=EXIT_DTE,
        profit_target_pct=PROFIT_TARGET_PCT,
    )

    mode = "DRY-RUN" if executor.dry_run else "LIVE"
    logger.info(f"Mode: {mode}")
    logger.info(f"Env: {client.env} ({client.base_url})")
    logger.info(f"Interval: every {RUN_INTERVAL_HOURS}h")
    logger.info(pm.summary())

    # Run first cycle immediately
    try:
        run_cycle(client, executor, pm, logger)
    except Exception as e:
        logger.error(f"Cycle failed: {e}", exc_info=True)

    # Then loop on schedule
    while True:
        sleep_seconds = RUN_INTERVAL_HOURS * 3600
        next_run = datetime.now(timezone.utc).strftime("%H:%M UTC")
        logger.info(f"Next cycle in {RUN_INTERVAL_HOURS}h (sleeping until ~{next_run})")

        time.sleep(sleep_seconds)

        try:
            run_cycle(client, executor, pm, logger)
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
            break
        except Exception as e:
            logger.error(f"Cycle failed: {e}", exc_info=True)
            logger.info("Will retry next cycle")


if __name__ == "__main__":
    main()
