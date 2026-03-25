"""
Download historical daily OHLC for all Deribit BTC quarterly futures.
Also downloads BTC-PERPETUAL for basis calculation.
"""

import requests
import time
import pandas as pd
from datetime import datetime

FUTURES = [
    'BTC-24SEP21', 'BTC-31DEC21',
    'BTC-25MAR22', 'BTC-24JUN22', 'BTC-30SEP22', 'BTC-30DEC22',
    'BTC-31MAR23', 'BTC-30JUN23', 'BTC-29SEP23', 'BTC-29DEC23',
    'BTC-29MAR24', 'BTC-28JUN24', 'BTC-27SEP24', 'BTC-27DEC24',
    'BTC-PERPETUAL',
]

# Map instrument name to expiry date (for quarterly only)
EXPIRIES = {
    'BTC-24SEP21': '2021-09-24', 'BTC-31DEC21': '2021-12-31',
    'BTC-25MAR22': '2022-03-25', 'BTC-24JUN22': '2022-06-24',
    'BTC-30SEP22': '2022-09-30', 'BTC-30DEC22': '2022-12-30',
    'BTC-31MAR23': '2023-03-31', 'BTC-30JUN23': '2023-06-30',
    'BTC-29SEP23': '2023-09-29', 'BTC-29DEC23': '2023-12-29',
    'BTC-29MAR24': '2024-03-29', 'BTC-28JUN24': '2024-06-28',
    'BTC-27SEP24': '2024-09-27', 'BTC-27DEC24': '2024-12-27',
}

START_TS = 1609459200000  # 2021-01-01
END_TS   = 1727740800000  # 2024-10-01

URL = 'https://www.deribit.com/api/v2/public/get_tradingview_chart_data'

all_data = []

for name in FUTURES:
    print(f"Downloading {name}...", end=" ", flush=True)
    params = {
        'instrument_name': name,
        'resolution': '1D',
        'start_timestamp': START_TS,
        'end_timestamp': END_TS,
    }
    resp = requests.get(URL, params=params)
    data = resp.json().get('result', {})

    closes = data.get('close', [])
    ticks = data.get('ticks', [])
    opens = data.get('open', [])
    highs = data.get('high', [])
    lows = data.get('low', [])

    if not closes:
        print("no data")
        continue

    # Only keep data up to expiry for quarterly futures
    expiry = EXPIRIES.get(name)

    for i in range(len(closes)):
        dt = datetime.utcfromtimestamp(ticks[i] / 1000)
        date_str = dt.strftime('%Y-%m-%d')

        # Skip data after expiry for quarterly futures
        if expiry and date_str > expiry:
            continue

        all_data.append({
            'date': date_str,
            'instrument': name,
            'open': opens[i],
            'high': highs[i],
            'low': lows[i],
            'close': closes[i],
            'expiry': expiry if expiry else None,
        })

    n_kept = sum(1 for d in all_data if d['instrument'] == name)
    print(f"{n_kept} bars")
    time.sleep(0.3)

df = pd.DataFrame(all_data)
df.to_csv("Data/deribit_futures_daily.csv", index=False)
print(f"\nSaved {len(df)} rows to Data/deribit_futures_daily.csv")
