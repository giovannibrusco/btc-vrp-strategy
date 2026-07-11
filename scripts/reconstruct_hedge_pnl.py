"""
Reconstruct hedge PnL for the closed Top Vega trade from log data + historical prices.

Trade: BTC-26JUN26-78000-C (short call)
Hedge: BTC-25SEP26 (long futures)
Open: 2026-03-16 22:21 UTC → Close: 2026-03-23 12:00 UTC
"""

from datetime import datetime, timezone
from src.deribit_client import DeribitClient


def get_futures_prices(client, instrument, start_dt, end_dt):
    """Get 1h candle closes for a futures instrument."""
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    result = client._request('public/get_tradingview_chart_data', {
        'instrument_name': instrument,
        'start_timestamp': str(start_ms),
        'end_timestamp': str(end_ms),
        'resolution': '60',  # 1h candles
    }, authenticated=False)

    prices = {}
    if isinstance(result, dict) and 'ticks' in result:
        for ts, close in zip(result['ticks'], result['close']):
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            prices[dt] = close
    return prices


def find_nearest_price(prices: dict, target_dt: datetime) -> float:
    """Find the closest candle close price to a target datetime."""
    best_dt = None
    best_diff = float('inf')
    for dt in prices:
        diff = abs((dt - target_dt).total_seconds())
        if diff < best_diff:
            best_diff = diff
            best_dt = dt
    return prices[best_dt] if best_dt else 0


def main():
    client = DeribitClient()

    # Get 1h candle data for the hedge instrument
    start = datetime(2026, 3, 16, 20, 0, tzinfo=timezone.utc)
    end = datetime(2026, 3, 23, 14, 0, tzinfo=timezone.utc)
    prices = get_futures_prices(client, "BTC-25SEP26", start, end)
    print(f"Loaded {len(prices)} hourly candles for BTC-25SEP26\n")

    # ── Cycle data from logs ──────────────────────────────────
    # Each entry: (timestamp, hedge_position_btc, event)
    # Position changes only on rebalance or close
    cycles = [
        # Entry
        ("2026-03-16 22:21", 0.1465, "ENTRY (buy $10,880)"),
        # Check cycles (no rebalance)
        ("2026-03-17 00:00", 0.1465, "check delta=0.497"),
        ("2026-03-17 04:00", 0.1465, "check delta=0.506"),
        ("2026-03-17 12:00", 0.1465, "check delta=0.475"),
        ("2026-03-17 16:00", 0.1465, "check delta=0.490"),
        ("2026-03-17 20:00", 0.1465, "check delta=0.495"),
        ("2026-03-18 08:09", 0.1465, "check delta=0.477"),
        # Rebalance: sold 0.0305 BTC
        ("2026-03-19 16:00", 0.1160, "REBALANCE 0.1465→0.1160 (sell $2,120)"),
        # Check cycles (no rebalance)
        ("2026-03-20 00:00", 0.1160, "check delta=0.399"),
        ("2026-03-20 20:06", 0.1160, "check delta=0.390"),
        ("2026-03-21 12:00", 0.1160, "check delta=0.406"),
        ("2026-03-21 16:00", 0.1160, "check delta=0.406"),
        ("2026-03-22 12:07", 0.1160, "check delta=0.361"),
        ("2026-03-22 16:00", 0.1160, "check delta=0.364"),
        ("2026-03-22 20:00", 0.1160, "check delta=0.362"),
        # Exit: sold remaining 0.1160 BTC
        ("2026-03-23 12:00", 0.0000, "EXIT profit_target (sell $7,960)"),
    ]

    # ── Compute hedge PnL cycle by cycle ──────────────────────
    # Same formula as backtest:
    #   hedge_pnl = hedge_pos_btc × (price_now - price_last)
    #
    # hedge_pos_btc is the position BEFORE the current cycle's rebalance
    # price_change is from last cycle to current cycle

    print("=" * 90)
    print("HEDGE PnL RECONSTRUCTION — BTC-26JUN26-78000-C")
    print("=" * 90)
    print()
    print(f"{'Timestamp':20s} {'Pos BTC':>8s} {'Qtr Price':>10s} {'ΔPrice':>8s} "
          f"{'Cycle PnL':>10s} {'Cumul PnL':>10s}  Event")
    print("-" * 90)

    cumulative_pnl = 0.0
    last_price = None

    for i, (ts_str, pos_after, event) in enumerate(cycles):
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        qtr_price = find_nearest_price(prices, dt)

        if i == 0:
            # Entry — no PnL yet
            cycle_pnl = 0.0
            delta_price = 0.0
            last_price = qtr_price
            # Position at entry
            pos_btc = pos_after
            print(f"{ts_str:20s} {pos_btc:8.4f} {qtr_price:10,.0f} {'—':>8s} "
                  f"{'—':>10s} {'$0':>10s}  {event}")
        else:
            # Position BEFORE this cycle's rebalance = previous pos_after
            pos_btc = cycles[i-1][1]
            delta_price = qtr_price - last_price
            cycle_pnl = pos_btc * delta_price
            cumulative_pnl += cycle_pnl
            last_price = qtr_price

            print(f"{ts_str:20s} {pos_btc:8.4f} {qtr_price:10,.0f} {delta_price:+8,.0f} "
                  f"${cycle_pnl:+10,.2f} ${cumulative_pnl:+10,.2f}  {event}")

    print("-" * 90)
    print()
    print(f"  TOTAL HEDGE PnL:  ${cumulative_pnl:+,.2f}")
    print()
    print(f"  Option PnL:       $+1,720.64")
    print(f"  Hedge PnL:        ${cumulative_pnl:+,.2f}")
    print(f"  Combined PnL:     ${1720.64 + cumulative_pnl:+,.2f}")
    print()

    # Also compute transaction costs on rebalances
    # Deribit futures taker fee: 0.05% of notional
    rebalance_volumes = [10880, 2120, 7960]  # entry, rebalance, close
    total_fee = sum(v * 0.0005 for v in rebalance_volumes)
    print(f"  Hedge txn costs:  ${total_fee:,.2f} (est. 0.05% taker on ${sum(rebalance_volumes):,})")
    print(f"  Net PnL:          ${1720.64 + cumulative_pnl - total_fee:+,.2f}")


if __name__ == "__main__":
    main()
