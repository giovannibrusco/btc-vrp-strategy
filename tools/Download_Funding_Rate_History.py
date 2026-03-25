"""
Download BTC-PERPETUAL funding rate history from Deribit public API.
Covers: 2021-06-01 to present, hourly granularity.
Output: Data/funding_rate_history.csv
"""

import requests
import pandas as pd
import time
from datetime import datetime

INSTRUMENT = "BTC-PERPETUAL"
URL = "https://www.deribit.com/api/v2/public/get_funding_rate_history"

# Range: 2021-06-01 to today
start_dt = datetime(2021, 6, 1)
end_dt = datetime.utcnow()

start_ms = int(start_dt.timestamp() * 1000)
end_ms = int(end_dt.timestamp() * 1000)

# Step: 15 days in ms (safe margin under 500 entries limit)
STEP_MS = 15 * 24 * 3600 * 1000

all_data = []
current_start = start_ms
call_count = 0

print(f"Downloading funding rates from {start_dt.date()} to {end_dt.date()}")
print(f"Instrument: {INSTRUMENT}")
print("-" * 50)

while current_start < end_ms:
    current_end = min(current_start + STEP_MS, end_ms)

    params = {
        "instrument_name": INSTRUMENT,
        "start_timestamp": current_start,
        "end_timestamp": current_end,
    }

    resp = requests.get(URL, params=params)
    call_count += 1

    if resp.status_code != 200:
        print(f"  ERROR: status {resp.status_code}, retrying in 5s...")
        time.sleep(5)
        continue

    data = resp.json()
    result = data.get("result", [])
    all_data.extend(result)

    dt_from = datetime.utcfromtimestamp(current_start / 1000).strftime("%Y-%m-%d")
    dt_to = datetime.utcfromtimestamp(current_end / 1000).strftime("%Y-%m-%d")
    print(f"  Call {call_count}: {dt_from} -> {dt_to} | {len(result)} entries | total: {len(all_data)}")

    current_start = current_end
    time.sleep(0.3)  # rate limit courtesy

print("-" * 50)
print(f"Total API calls: {call_count}")
print(f"Total entries: {len(all_data)}")

# Build DataFrame
df = pd.DataFrame(all_data)
df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)

# Rename for clarity
df = df.rename(columns={
    "interest_8h": "funding_rate_8h",
    "interest_1h": "funding_rate_1h",
})

# Compute daily funding rate (sum of 8h rates × 3 per day, or 1h × 24)
df["date"] = df["datetime"].dt.date
df_daily = (
    df.groupby("date")
    .agg(
        funding_rate_daily=("funding_rate_8h", lambda x: x.sum()),
        mean_index_price=("index_price", "mean"),
        n_observations=("funding_rate_8h", "count"),
    )
    .reset_index()
)

# Save both
output_path = "Data/funding_rate_history.csv"
output_daily_path = "Data/funding_rate_daily.csv"

df.to_csv(output_path, index=False)
df_daily.to_csv(output_daily_path, index=False)

print(f"\nSaved hourly data: {output_path} ({len(df)} rows)")
print(f"Saved daily data:  {output_daily_path} ({len(df_daily)} rows)")
print(f"\nDate range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Mean daily funding rate: {df_daily['funding_rate_daily'].mean():.6f} ({df_daily['funding_rate_daily'].mean()*100:.4f}%)")
print(f"Mean annualized: {df_daily['funding_rate_daily'].mean()*365*100:.2f}%")
