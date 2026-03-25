#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 12:03:40 2025

@author: giovannibrusco
"""
import pandas as pd
import requests
import time
from datetime import datetime

def get_binance_klines(symbol, interval, start_time, end_time):
    url = 'https://api.binance.com/api/v3/klines'
    limit = 1000
    start_timestamp = int(start_time.timestamp() * 1000)
    end_timestamp = int(end_time.timestamp() * 1000)
    data = []

    while start_timestamp < end_timestamp:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_timestamp,
            'endTime': end_timestamp,
            'limit': limit
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"❌ Errore richiesta: {response.status_code} – {response.text}")
            break

        klines = response.json()
        if not klines:
            break

        for k in klines:
            data.append({
                'date': datetime.utcfromtimestamp(k[0] / 1000).strftime('%Y-%m-%d'),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5])
            })

        start_timestamp = klines[-1][0] + 1
        time.sleep(0.5)  # per sicurezza contro i limiti

    return pd.DataFrame(data)

# 📅 Periodo da scaricare
start_date = datetime(2020, 6, 1)
end_date = datetime.today()

# ⛏️ Scarica dati
df = get_binance_klines('BTCUSDT', '1d', start_date, end_date)

# 💾 Salva
df.to_csv("/Users/giovannibrusco/Desktop/Progetto BTC/Data/btc_spot_daily.csv", index=False)
print("✅ File salvato in Data/btc_spot_daily.csv")





