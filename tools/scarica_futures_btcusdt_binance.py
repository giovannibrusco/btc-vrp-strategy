#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 15:41:42 2025

@author: giovannibrusco
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt


def get_monthly_futures_klines(symbol, interval, start_date, end_date):
    """
    Scarica dati futures Binance (BTCUSDT perpetual) su base mensile
    """
    base_url = "https://fapi.binance.com/fapi/v1/klines"
    all_data = []

    current = start_date
    while current < end_date:
        next_month = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
        print(f"📦 Download {current.strftime('%Y-%m')}...", end=" ")

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(current.timestamp() * 1000),
            "endTime": int(next_month.timestamp() * 1000),
            "limit": 1500
        }

        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            if not isinstance(data, list):
                print("❌ Errore:", data)
                break

            for entry in data:
                open_time = datetime.fromtimestamp(entry[0] / 1000)
                open_price = float(entry[1])
                high = float(entry[2])
                low = float(entry[3])
                close = float(entry[4])
                volume = float(entry[5])

                all_data.append({
                    "date": open_time.date(),
                    "open": open_price,
                    "close": close,
                    "high": high,
                    "low": low,
                    "volume": volume
                })

            print("✅")
            time.sleep(0.3)
        except Exception as e:
            print("❌ Exception:", e)
            break

        current = next_month

    return pd.DataFrame(all_data)

# ⚙️ Parametri
symbol = "BTCUSDT"
interval = "1d"
start_date = datetime(2020, 7, 1)
end_date = datetime.today()

# 📥 Scarica i dati
df_fut = get_monthly_futures_klines(symbol, interval, start_date, end_date)

# 💾 Salva
output_folder = "Data"
os.makedirs(output_folder, exist_ok=True)
df_fut.to_csv(f"{output_folder}/futures_btcusdt_daily.csv", index=False)

print(f"✅ File salvato in {output_folder}/futures_btcusdt_daily.csv con {len(df_fut)} righe")









