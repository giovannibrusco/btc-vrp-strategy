#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 12:49:34 2025

@author: giovannibrusco
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

# === CONFIGURAZIONE ===
start_date = datetime(2021, 6, 1)
end_date = datetime(2024, 9, 30)  
instrument = "BTC-PERPETUAL"
resolution = 5  # minuti
save_path = "/Users/giovannibrusco/Desktop/Progetto BTC/Data/btc_perpetual_5min.csv"

# === INIZIALIZZA LISTA RISULTATI ===
all_data = []

# === LOOP GIORNALIERO ===
current_date = start_date
while current_date <= end_date:
    print(f"📅 Scarico {current_date.date()}", end="... ")

    # timestamp in millisecondi
    start_ts = int(current_date.timestamp() * 1000)
    end_ts = int((current_date + timedelta(days=1)).timestamp() * 1000)

    # parametri della richiesta
    url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    params = {
        "instrument_name": instrument,
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "resolution": resolution
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        if "result" not in data or not data["result"]["ticks"]:
            print("❌ Nessun dato")
        else:
            df = pd.DataFrame({
                "timestamp": data["result"]["ticks"],
                "open": data["result"]["open"],
                "high": data["result"]["high"],
                "low": data["result"]["low"],
                "close": data["result"]["close"],
                "volume": data["result"]["volume"]
            })
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            all_data.append(df)
            print(f"✅ {len(df)} record")
        time.sleep(0.3)  # per rispetto del rate limit
    except Exception as e:
        print(f"❌ Errore: {e}")
        time.sleep(1)

    current_date += timedelta(days=1)

# === SALVA IN CSV ===
if all_data:
    full_df = pd.concat(all_data, ignore_index=True)
    full_df.to_csv(save_path, index=False)
    print(f"\n✅ File salvato: {save_path}")
else:
    print("\n⚠️ Nessun dato da salvare.")