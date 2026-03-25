#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 12:43:54 2025

@author: giovannibrusco
"""

# 📦 Import
import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# 📁 Setup
OUTPUT_DIR = "Data/Options"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🔁 Intervallo temporale (trimestri)
start_date = datetime(2020, 7, 1)
end_date = datetime.today()
delta = timedelta(days=90)

# 🔍 Funzione di parsing da instrument_name
def parse_instrument_name(name):
    try:
        parts = name.split("-")
        expiry = datetime.strptime(parts[1], "%d%b%y").date()
        strike = float(parts[2])
        option_type = parts[3]
        return expiry, strike, option_type
    except:
        return None, None, None

# 🔁 Loop sui trimestri
while start_date < end_date:
    end_period = min(start_date + delta, end_date)
    all_data = []

    date_cursor = start_date
    while date_cursor < end_period:
        date_str = date_cursor.strftime("%Y-%m-%d")
        print(f"📅 Scarico opzioni per il {date_str}...")

        try:
            # API Deribit: /v2/public/get_book_summary_by_currency
            url = f"https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option"
            response = requests.get(url)
            data = response.json()["result"]
            df = pd.DataFrame(data)

            # Aggiungi expiry, strike, type
            df["expiry"], df["strike"], df["option_type"] = zip(*df["instrument_name"].map(parse_instrument_name))

            # Filtra solo colonne rilevanti
            cols_to_keep = [
                "instrument_name", "expiry", "strike", "option_type",
                "mark_price", "mark_iv", "underlying_price",
                "open_interest", "volume_usd",
                "bid_price", "ask_price", "mid_price", "creation_timestamp"
            ]
            df = df[cols_to_keep]
            df["date"] = date_cursor.date()

            all_data.append(df)
        except Exception as e:
            print(f"❌ Errore nel giorno {date_str}: {e}")

        date_cursor += timedelta(days=1)

    # Salva il trimestre
    if all_data:
        df_full = pd.concat(all_data, ignore_index=True)
        filename = f"{OUTPUT_DIR}/options_{start_date.strftime('%Y%m%d')}_{end_period.strftime('%Y%m%d')}.csv"
        df_full.to_csv(filename, index=False)
        print(f"✅ Salvato {filename} con {len(df_full)} righe")

    start_date = end_period
    
    
    
    