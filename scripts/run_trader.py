#!/usr/bin/env python3
"""
Entry point for the live paper trading system.

Usage:
    python scripts/run_trader.py              # Run daemon (loops every 4h)
    python scripts/run_trader.py --once       # Run one cycle and exit
    python scripts/run_trader.py --status     # Show current positions
    python scripts/run_trader.py --dry-run    # Force dry-run mode
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="BTC Options VRP Paper Trader")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle and exit")
    parser.add_argument("--status", action="store_true",
                        help="Show current positions and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Force dry-run mode (no orders)")
    parser.add_argument("--interval", type=int, default=4,
                        help="Hours between cycles (default: 4)")
    args = parser.parse_args()

    from src.live_trader import setup_logging, run_cycle, STATE_DIR
    from src.deribit_client import DeribitClient
    from src.order_executor import OrderExecutor
    from src.position_manager import PositionManager

    logger = setup_logging()

    # Status mode
    if args.status:
        pm = PositionManager(state_file=STATE_DIR / "live_state.json")
        print(pm.summary())
        if pm.state["trades_history"]:
            print(f"\nLast 5 trades:")
            for t in pm.state["trades_history"][-5:]:
                print(f"  {t.get('exit_date','?')}: {t['instrument']} "
                      f"PnL=${t['total_pnl']:,.0f} ({t['exit_reason']})")
        return

    # Initialize
    client = DeribitClient()
    executor = OrderExecutor(client, dry_run=args.dry_run)
    pm = PositionManager(
        state_file=STATE_DIR / "live_state.json",
        exit_dte=5.0,
        profit_target_pct=0.50,
    )

    mode = "DRY-RUN" if executor.dry_run else "LIVE"
    logger.info(f"Mode: {mode} | Env: {client.env}")

    if args.once:
        # Single cycle
        run_cycle(client, executor, pm, logger)
        logger.info("Single cycle complete.")
        return

    # Daemon mode
    import time
    from datetime import datetime, timezone
    import src.live_trader as lt

    lt.RUN_INTERVAL_HOURS = args.interval
    logger.info(f"Daemon starting — interval: {args.interval}h")

    try:
        run_cycle(client, executor, pm, logger)
    except Exception as e:
        logger.error(f"First cycle failed: {e}", exc_info=True)

    while True:
        logger.info(f"Sleeping {args.interval}h...")
        try:
            time.sleep(args.interval * 3600)
            run_cycle(client, executor, pm, logger)
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
            break
        except Exception as e:
            logger.error(f"Cycle failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
