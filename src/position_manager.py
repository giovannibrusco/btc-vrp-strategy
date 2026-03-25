"""
Position Manager — Track open positions, check exit conditions, persist state.

State is persisted to a JSON file so the system can resume after restarts.
Supports two independent strategy slots (Premium/DTE + Top Vega).
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "live_state.json"


class PositionManager:
    """Manage two independent strategy positions with persistence."""

    def __init__(self, state_file: Path | str | None = None,
                 exit_dte: float = 5.0,
                 profit_target_pct: float = 0.50):
        self.state_file = Path(state_file) if state_file else STATE_FILE
        self.exit_dte = exit_dte
        self.profit_target_pct = profit_target_pct

        self.state = self._load_state()

    def _default_state(self) -> dict:
        return {
            "positions": {
                "premium_dte": None,
                "top_vega": None,
            },
            "equity": 100_000.0,
            "trades_history": [],
            "last_update": None,
        }

    def _load_state(self) -> dict:
        """Load state from disk, or create default."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                logger.info(f"Loaded state from {self.state_file}")
                return state
            except Exception as e:
                logger.warning(f"Failed to load state: {e}, using default")
        return self._default_state()

    def save_state(self):
        """Persist state to disk."""
        self.state["last_update"] = datetime.now(timezone.utc).isoformat()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
        logger.info(f"State saved to {self.state_file}")

    @property
    def equity(self) -> float:
        return self.state["equity"]

    @equity.setter
    def equity(self, value: float):
        self.state["equity"] = value

    def get_position(self, strategy: str) -> dict | None:
        """Get current position for a strategy ('premium_dte' or 'top_vega')."""
        return self.state["positions"].get(strategy)

    def is_flat(self, strategy: str) -> bool:
        """Check if a strategy slot is flat (no open position)."""
        return self.state["positions"].get(strategy) is None

    def open_position(self, strategy: str, position: dict):
        """Record a new position for a strategy."""
        position = deepcopy(position)
        position["opened_at"] = datetime.now(timezone.utc).isoformat()
        position["strategy"] = strategy
        self.state["positions"][strategy] = position
        self.save_state()
        logger.info(f"[{strategy}] Opened: {position['instrument']} "
                     f"@ {position['entry_price']:.4f} BTC")

    def close_position(self, strategy: str, exit_price: float,
                       exit_spot: float, exit_reason: str,
                       hedge_pnl: float = 0.0,
                       hedge_txn_cost: float = 0.0,
                       tail_cost: float = 0.0) -> dict:
        """
        Close a position, compute PnL, move to history.

        Returns the completed trade dict.
        """
        pos = self.state["positions"].get(strategy)
        if pos is None:
            raise ValueError(f"No open position for {strategy}")

        # Option PnL: sold at entry, buy back at exit (BTC → USD)
        pnl_btc = (pos["entry_price"] - exit_price) * pos["n_contracts"]
        option_pnl = pnl_btc * exit_spot

        total_pnl = option_pnl + hedge_pnl - hedge_txn_cost - tail_cost

        trade = {
            **pos,
            "exit_price": exit_price,
            "exit_spot": exit_spot,
            "exit_reason": exit_reason,
            "exit_date": datetime.now(timezone.utc).isoformat(),
            "option_pnl": round(option_pnl, 2),
            "hedge_pnl": round(hedge_pnl, 2),
            "hedge_txn_cost": round(hedge_txn_cost, 2),
            "tail_cost": round(tail_cost, 2),
            "total_pnl": round(total_pnl, 2),
            "return_pct": round(total_pnl / self.equity * 100, 4),
        }

        self.state["equity"] += total_pnl
        self.state["positions"][strategy] = None
        self.state["trades_history"].append(trade)
        self.save_state()

        logger.info(f"[{strategy}] Closed {pos['instrument']}: "
                     f"PnL=${total_pnl:,.0f} ({trade['return_pct']:+.2f}%) "
                     f"reason={exit_reason}")
        return trade

    def check_exit(self, strategy: str, current_bid: float,
                   current_dte: float, current_spot: float,
                   current_ask: float | None = None) -> dict | None:
        """
        Check if a position should be exited.

        Uses current_ask (buy-back cost) for P&L and exit pricing.
        Falls back to current_bid if ask not available.

        Returns dict with exit_reason if should exit, None otherwise.
        """
        pos = self.state["positions"].get(strategy)
        if pos is None:
            return None

        # Use ask for exit pricing (realistic buy-back cost)
        exit_price = current_ask if current_ask else current_bid

        # DTE exit
        if current_dte <= self.exit_dte:
            return {"exit_reason": "dte_exit", "exit_price": exit_price,
                    "exit_spot": current_spot}

        # Profit target — use ask to check captured premium
        entry_premium_btc = pos["entry_price"] * pos["n_contracts"]
        pnl_btc = (pos["entry_price"] - exit_price) * pos["n_contracts"]
        if entry_premium_btc > 0:
            captured_pct = pnl_btc / entry_premium_btc
            if captured_pct >= self.profit_target_pct:
                return {"exit_reason": "profit_target", "exit_price": exit_price,
                        "exit_spot": current_spot}

        return None

    def compute_hedge_delta(self, strategy: str,
                            current_delta: float) -> float:
        """
        Compute the required hedge position in BTC.

        For a short option, we need to buy delta × n_contracts of the underlying.
        Returns the target hedge size (positive = long futures).
        """
        pos = self.state["positions"].get(strategy)
        if pos is None:
            return 0.0
        return current_delta * pos["n_contracts"]

    def summary(self) -> str:
        """Format current state summary."""
        lines = [f"=== Position Manager ==="]
        lines.append(f"Equity: ${self.equity:,.0f}")
        for strat in ["premium_dte", "top_vega"]:
            pos = self.get_position(strat)
            if pos is None:
                lines.append(f"  {strat}: FLAT")
            else:
                lines.append(
                    f"  {strat}: {pos['instrument']} "
                    f"entry={pos['entry_price']:.4f} "
                    f"n={pos['n_contracts']:.2f} "
                    f"delta={pos.get('entry_delta', '?')}"
                )
        n_hist = len(self.state["trades_history"])
        if n_hist > 0:
            total_pnl = sum(t["total_pnl"] for t in self.state["trades_history"])
            lines.append(f"  History: {n_hist} trades, total PnL=${total_pnl:,.0f}")
        return "\n".join(lines)
