# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BTC Volatility Risk Premium (VRP) systematic short-volatility strategy. Sells the highest-vega OTM BTC option daily on Deribit, delta-hedges with quarterly futures, exits at 50% premium capture or DTE <= 5. Includes backtesting, statistical validation, live paper trading, and Streamlit dashboards.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full backtest (Phase 1: scoring ~slow, Phase 2: simulation ~fast)
python scripts/run_backtest.py

# Run backtest Phase 2 only (uses cached scores from Data/scores_v2.csv)
python scripts/run_backtest.py --phase2

# Sensitivity analysis (grid search on profit_target x exit_dte)
python scripts/run_sensitivity.py

# Generate dashboard data (pre-computes all three risk configs)
python scripts/generate_dashboard_data.py

# Launch dashboards
streamlit run dashboard/dashboard.py       # backtest results
streamlit run dashboard/story.py           # narrative research story
streamlit run dashboard/live.py            # live paper trading monitor

# Live paper trading
python scripts/run_trader.py --once        # single cycle
python scripts/run_trader.py               # daemon (every 4h)
python scripts/run_trader.py --status      # check positions
python scripts/run_trader.py --dry-run     # force dry-run mode

# SPY cross-asset validation
python scripts/spy_backtest.py

# Run tests (these are validation scripts, not pytest)
python tests/test_combined_final.py
python tests/test_statistical_validation.py
python tests/test_money_management.py
```

## Architecture

### Two-Phase Backtest Pipeline

The backtest is split for performance:

- **Phase 1** (`src/scoring.py` → `precompute_scores()`): For every trading day, extracts Risk-Neutral Density via spline interpolation (`src/rnd_engine.py`), computes physical distribution via KDE (`src/historical_distribution.py`), and scores all options. Output: `Data/scores_v2.csv`. This is slow (~minutes) and only needs to run once.
- **Phase 2** (`src/backtest_engine_v2.py` → `run_backtest_v2()`): Reads precomputed scores, simulates trading with T+1 execution, daily delta hedging via quarterly futures, exit rules, and transaction costs. Output: trade-level DataFrame with daily MTM equity attached via `df.attrs["daily_equity"]`.

### Live Trading System

Four modules form the live paper trading loop:
- `src/deribit_client.py` — REST API client for Deribit (testnet/production), handles OAuth2
- `src/live_scorer.py` — Scores live option chain using Top Vega strategy (highest |vega|)
- `src/position_manager.py` — Tracks positions, checks exit conditions, persists state to `data/live_state.json`
- `src/order_executor.py` — Executes trades, falls back to dry-run if auth fails

Orchestrated by `src/live_trader.py`, entry point at `scripts/run_trader.py`.

### Key Design Decisions

- **T+1 execution**: Signal at day T close, trade entered at T+1 bid price. Eliminates look-ahead bias.
- **Quarterly futures hedge** (not perpetual): Fixed basis cost, no funding rate exposure. Lookups built once via `_build_quarterly_lookup()`.
- **Single position at a time** (`max_parallel_positions=1`): Multi-position mode was tested and rejected (IS drawdown reached -49.7%).
- **Sizing modes**: `"fixed"` (1 contract), `"vega"` (target vega USD), `"capital_vega"` (equity-based with risk_per_trade/worst_loss_per_vega/buffer params), `"premium_pct"`.
- **Option P&L in BTC converted at exit spot**: `pnl_btc = (entry_price - exit_price) * n_contracts`, then `* exit_spot`.

### Data Flow

All paths are configured in `config/params.yaml`. Raw data lives in `Data/` (gitignored, ~5GB). Pre-generated dashboard data in `dashboard_data/` is committed.

- Options data: monthly EOD CSVs from Deribit → merged into `Data/options_ready_for_backtest.csv`
- BTC prices: 5-min perpetual, daily spot, daily funding rates, quarterly futures OHLC
- SPY data: optionsDX EOD files in `SPY Options/extracted/`

### Config

All strategy parameters centralized in `config/params.yaml` — loaded via `src/data_loader.load_config()`. Live trading constants are hardcoded in `src/live_trader.py` (top of file).

### Tests

Files in `tests/` are standalone validation scripts (not pytest). They run backtests with specific configurations and print statistical results. Run individually with `python tests/<file>.py`.
