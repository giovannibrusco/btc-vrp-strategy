"""
Deribit Authenticated API Client.

Supports both testnet and production.
Handles OAuth2 authentication, token refresh, and all trading endpoints.
"""

import json
import time
import logging
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

URLS = {
    "testnet": "https://test.deribit.com/api/v2",
    "production": "https://www.deribit.com/api/v2",
}


class DeribitClient:
    """Authenticated Deribit REST API client."""

    def __init__(self, env: str | None = None,
                 client_id: str | None = None,
                 client_secret: str | None = None):
        self.env = env or os.getenv("DERIBIT_ENV", "testnet")
        self.client_id = client_id or os.getenv("DERIBIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("DERIBIT_CLIENT_SECRET")
        self.base_url = URLS[self.env]
        self.access_token = None
        self.token_expiry = 0

        if not self.client_id or not self.client_secret:
            raise ValueError("DERIBIT_CLIENT_ID and DERIBIT_CLIENT_SECRET required")

        logger.info(f"DeribitClient initialized ({self.env}): {self.base_url}")

    # ── Auth ──────────────────────────────────────────────────

    def _authenticate(self):
        """Get or refresh access token via client_credentials."""
        params = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        resp = self._request("public/auth", params, authenticated=False)
        self.access_token = resp["access_token"]
        self.token_expiry = time.time() + resp["expires_in"] - 60  # refresh 60s early
        logger.info("Authenticated successfully")

    def _ensure_auth(self):
        """Re-authenticate if token expired."""
        if self.access_token is None or time.time() > self.token_expiry:
            self._authenticate()

    # ── HTTP ──────────────────────────────────────────────────

    def _request(self, method: str, params: dict | None = None,
                 authenticated: bool = True, retries: int = 3) -> dict:
        """Make a Deribit JSON-RPC style REST request."""
        if authenticated:
            self._ensure_auth()

        url = f"{self.base_url}/{method}"
        if params:
            query = urllib.parse.urlencode(params)
            url = f"{url}?{query}"

        req = urllib.request.Request(url)
        if authenticated and self.access_token:
            req.add_header("Authorization", f"Bearer {self.access_token}")

        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode())

                if "result" in data:
                    return data["result"]
                elif "error" in data:
                    error = data["error"]
                    logger.error(f"API error: {error}")
                    raise RuntimeError(f"Deribit API error: {error}")
                return data

            except urllib.error.HTTPError as e:
                body = e.read().decode() if e.fp else ""
                logger.warning(f"HTTP {e.code} on {method} (attempt {attempt+1}): {body}")
                if e.code == 429:  # rate limit
                    time.sleep(2 ** attempt)
                    continue
                if e.code == 401:  # token expired
                    self.access_token = None
                    self._ensure_auth()
                    continue
                raise
            except Exception as e:
                logger.warning(f"Request error on {method} (attempt {attempt+1}): {e}")
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                raise

    # ── Account ───────────────────────────────────────────────

    def get_account_summary(self, currency: str = "BTC") -> dict:
        """Get account balance, equity, margin info."""
        return self._request("private/get_account_summary", {
            "currency": currency,
            "extended": "true",
        })

    def get_equity_usd(self) -> float:
        """Get account equity in USD."""
        acc = self.get_account_summary("BTC")
        equity_btc = acc["equity"]
        # Get current BTC price
        spot = self.get_index_price()
        return equity_btc * spot

    # ── Market Data (Public) ──────────────────────────────────

    def get_index_price(self, index: str = "btc_usd") -> float:
        """Get current BTC index price."""
        resp = self._request("public/get_index_price", {
            "index_name": index,
        }, authenticated=False)
        return resp["index_price"]

    def get_instruments(self, currency: str = "BTC", kind: str = "option",
                        expired: bool = False) -> list:
        """Get all active instruments."""
        return self._request("public/get_instruments", {
            "currency": currency,
            "kind": kind,
            "expired": str(expired).lower(),
        }, authenticated=False)

    def get_ticker(self, instrument: str) -> dict:
        """Get ticker data for an instrument."""
        return self._request("public/ticker", {
            "instrument_name": instrument,
        }, authenticated=False)

    def get_book_summary(self, currency: str = "BTC", kind: str = "option") -> list:
        """Get book summary for all instruments of a kind."""
        return self._request("public/get_book_summary_by_currency", {
            "currency": currency,
            "kind": kind,
        }, authenticated=False)

    def get_order_book(self, instrument: str, depth: int = 5) -> dict:
        """Get order book for an instrument."""
        return self._request("public/get_order_book", {
            "instrument_name": instrument,
            "depth": str(depth),
        }, authenticated=False)

    # ── Positions ─────────────────────────────────────────────

    def get_positions(self, currency: str = "BTC", kind: str | None = None) -> list:
        """Get all open positions."""
        params = {"currency": currency}
        if kind:
            params["kind"] = kind
        return self._request("private/get_positions", params)

    def get_position(self, instrument: str) -> dict:
        """Get position for a specific instrument."""
        return self._request("private/get_position", {
            "instrument_name": instrument,
        })

    # ── Orders ────────────────────────────────────────────────

    def sell(self, instrument: str, amount: float,
             order_type: str = "market", price: float | None = None,
             label: str | None = None) -> dict:
        """
        Sell (short) an instrument.

        For options: amount in BTC (min 0.1 on mainnet, may differ on testnet).
        For futures: amount in USD.
        """
        params = {
            "instrument_name": instrument,
            "amount": str(amount),
            "type": order_type,
        }
        if price is not None and order_type == "limit":
            params["price"] = str(price)
        if label:
            params["label"] = label
        return self._request("private/sell", params)

    def buy(self, instrument: str, amount: float,
            order_type: str = "market", price: float | None = None,
            label: str | None = None) -> dict:
        """
        Buy (long) an instrument.

        For options: amount in BTC.
        For futures: amount in USD.
        """
        params = {
            "instrument_name": instrument,
            "amount": str(amount),
            "type": order_type,
        }
        if price is not None and order_type == "limit":
            params["price"] = str(price)
        if label:
            params["label"] = label
        return self._request("private/buy", params)

    def close_position(self, instrument: str,
                       order_type: str = "market") -> dict:
        """Close an entire position."""
        return self._request("private/close_position", {
            "instrument_name": instrument,
            "type": order_type,
        })

    def get_open_orders(self, currency: str = "BTC") -> list:
        """Get all open orders."""
        return self._request("private/get_open_orders_by_currency", {
            "currency": currency,
        })

    def cancel_all(self, currency: str = "BTC") -> int:
        """Cancel all open orders. Returns count cancelled."""
        return self._request("private/cancel_all_by_currency", {
            "currency": currency,
        })

    # ── Helpers ───────────────────────────────────────────────

    def get_quarterly_futures(self, currency: str = "BTC") -> list:
        """Get all active quarterly futures (not perpetual)."""
        instruments = self.get_instruments(currency, kind="future")
        quarterlies = []
        for inst in instruments:
            name = inst["instrument_name"]
            if "PERPETUAL" in name:
                continue
            expiry_ts = inst["expiration_timestamp"] / 1000
            expiry_dt = datetime.utcfromtimestamp(expiry_ts)
            dte = (expiry_dt - datetime.utcnow()).days
            if dte > 0:
                quarterlies.append({
                    "instrument": name,
                    "expiry": expiry_dt,
                    "dte": dte,
                })
        return sorted(quarterlies, key=lambda x: x["dte"])

    def find_best_quarterly(self, min_dte: int = 14) -> dict | None:
        """Find the front-month quarterly with at least min_dte days."""
        quarterlies = self.get_quarterly_futures()
        valid = [q for q in quarterlies if q["dte"] >= min_dte]
        return valid[0] if valid else None
