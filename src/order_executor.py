"""
Order Executor — Execute trades on Deribit with dry-run fallback.

Handles:
- Selling options (open short)
- Buying back options (close short)
- Buying/selling quarterly futures for delta hedge
- Automatic dry-run mode when authentication fails
"""

import logging
from datetime import datetime, timezone

from src.deribit_client import DeribitClient

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Execute orders on Deribit, with transparent dry-run mode."""

    def __init__(self, client: DeribitClient, dry_run: bool = False):
        self.client = client
        self.dry_run = dry_run
        self._check_auth()

    def _check_auth(self):
        """Test authentication, fall back to dry-run if it fails."""
        if self.dry_run:
            logger.info("OrderExecutor in DRY-RUN mode (explicit)")
            return

        try:
            self.client._ensure_auth()
            logger.info("OrderExecutor authenticated — LIVE mode")
        except Exception as e:
            logger.warning(f"Auth failed: {e} — switching to DRY-RUN mode")
            self.dry_run = True

    def sell_option(self, instrument: str, amount: float,
                    label: str = "") -> dict:
        """
        Sell (short) an option.

        Parameters:
            instrument: e.g. "BTC-26DEC25-100000-C"
            amount: in BTC (min 0.1 on mainnet)
            label: optional order label for tracking

        Returns:
            Order result dict (or simulated result in dry-run).
        """
        if self.dry_run:
            result = _dry_run_result("sell", instrument, amount, label)
            logger.info(f"[DRY-RUN] SELL {amount:.2f} {instrument} {label}")
            return result

        logger.info(f"SELL {amount:.2f} {instrument} {label}")
        return self.client.sell(
            instrument=instrument,
            amount=amount,
            order_type="market",
            label=label or None,
        )

    def buy_option(self, instrument: str, amount: float,
                   label: str = "") -> dict:
        """
        Buy back (close short) an option.

        Parameters:
            instrument: option instrument name
            amount: in BTC

        Returns:
            Order result dict.
        """
        if self.dry_run:
            result = _dry_run_result("buy", instrument, amount, label)
            logger.info(f"[DRY-RUN] BUY {amount:.2f} {instrument} {label}")
            return result

        logger.info(f"BUY {amount:.2f} {instrument} {label}")
        return self.client.buy(
            instrument=instrument,
            amount=amount,
            order_type="market",
            label=label or None,
        )

    def close_option_position(self, instrument: str) -> dict:
        """Close an entire option position."""
        if self.dry_run:
            result = _dry_run_result("close", instrument, 0, "close_position")
            logger.info(f"[DRY-RUN] CLOSE POSITION {instrument}")
            return result

        logger.info(f"CLOSE POSITION {instrument}")
        return self.client.close_position(instrument=instrument)

    def hedge_with_quarterly(self, quarterly_instrument: str,
                             target_delta_btc: float,
                             current_hedge_btc: float = 0.0) -> dict:
        """
        Adjust hedge on quarterly futures.

        Parameters:
            quarterly_instrument: e.g. "BTC-26SEP25"
            target_delta_btc: target hedge position (positive = long)
            current_hedge_btc: current hedge position

        Returns:
            Order result or None if no rebalance needed.
        """
        delta_change = target_delta_btc - current_hedge_btc
        # Min trade size check (Deribit: futures in USD, $10 increments)
        min_size_usd = 10
        # We'll approximate: 1 BTC ≈ spot price in USD
        if abs(delta_change) < 0.001:  # Less than ~$70 at current prices
            logger.debug("Hedge delta change too small, skipping")
            return None

        # Futures on Deribit are in USD notional
        # For simplicity, we use BTC amount and let the API convert
        if delta_change > 0:
            return self._buy_future(quarterly_instrument, abs(delta_change))
        else:
            return self._sell_future(quarterly_instrument, abs(delta_change))

    def _buy_future(self, instrument: str, amount_btc: float) -> dict:
        """Buy futures for hedge (long delta)."""
        # Deribit futures amount is in USD
        # Convert BTC to USD notional (approximate)
        spot = self.client.get_index_price()
        amount_usd = round(amount_btc * spot / 10) * 10  # Round to $10
        amount_usd = max(amount_usd, 10)

        if self.dry_run:
            result = _dry_run_result("buy", instrument, amount_usd, "hedge_long")
            logger.info(f"[DRY-RUN] BUY FUTURE {instrument} ${amount_usd:,.0f} "
                        f"({amount_btc:.4f} BTC)")
            return result

        logger.info(f"BUY FUTURE {instrument} ${amount_usd:,.0f}")
        return self.client.buy(
            instrument=instrument,
            amount=amount_usd,
            order_type="market",
            label="hedge",
        )

    def _sell_future(self, instrument: str, amount_btc: float) -> dict:
        """Sell futures for hedge (short delta)."""
        spot = self.client.get_index_price()
        amount_usd = round(amount_btc * spot / 10) * 10
        amount_usd = max(amount_usd, 10)

        if self.dry_run:
            result = _dry_run_result("sell", instrument, amount_usd, "hedge_short")
            logger.info(f"[DRY-RUN] SELL FUTURE {instrument} ${amount_usd:,.0f} "
                        f"({amount_btc:.4f} BTC)")
            return result

        logger.info(f"SELL FUTURE {instrument} ${amount_usd:,.0f}")
        return self.client.sell(
            instrument=instrument,
            amount=amount_usd,
            order_type="market",
            label="hedge",
        )

    def find_hedge_quarterly(self, min_dte: int = 14) -> dict | None:
        """Find the best quarterly future for hedging."""
        return self.client.find_best_quarterly(min_dte=min_dte)

    def get_account_equity(self) -> float | None:
        """Get current account equity in USD."""
        if self.dry_run:
            return None
        try:
            return self.client.get_equity_usd()
        except Exception as e:
            logger.warning(f"Failed to get equity: {e}")
            return None


def _dry_run_result(side: str, instrument: str, amount: float,
                    label: str) -> dict:
    """Generate a simulated order result for dry-run mode."""
    return {
        "dry_run": True,
        "side": side,
        "instrument": instrument,
        "amount": amount,
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "order_id": f"dry_{side}_{instrument}_{int(datetime.now().timestamp())}",
    }
