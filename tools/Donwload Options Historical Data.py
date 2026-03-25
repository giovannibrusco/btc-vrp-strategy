#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 20:04:24 2025

@author: giovannibrusco
"""

import os
import pandas as pd

# 📁 Cartella che contiene i file .txt
cartella = "Data/Options_Historical_DataSet"  
# 🔎 Trova tutti i file .txt nella cartella
file_txt = [f for f in os.listdir(cartella) if f.endswith(".txt")]

# 📦 Lista in cui salviamo i DataFrame temporanei
df_list = []

# 🔁 Legge ogni file e lo aggiunge alla lista
for nome_file in file_txt:
    percorso_completo = os.path.join(cartella, nome_file)
    print(f"📂 Caricamento: {nome_file}")
    
    try:
        df = pd.read_csv(percorso_completo)  # usa ',' come separatore
        df_list.append(df)
    except Exception as e:
        print(f"❌ Errore con {nome_file}: {e}")

# 🔗 Unisce tutti i DataFrame in uno solo
df_completo = pd.concat(df_list, ignore_index=True)

# 👁️ Visualizza informazioni
print(f"✅ Totale righe caricate: {len(df_completo)}")
print(df_completo.head())

# 🔧 Pulisce i nomi delle colonne
df_completo.columns = df_completo.columns.str.strip().str.replace(r"[\[\]]", "", regex=True)

# 👁️ Visualizza intestazioni e alcune statistiche base
print(df_completo.columns)
print(df_completo[['QUOTE_DATE', 'INSTRUMENT_NAME', 'MARK_IV', 'OPTION_RIGHT', 'STRIKE', 'UNDERLYING_PRICE']].head())

# 📂 Percorso del file
file_path = "Data/futures_btcusdt_daily.csv"

# 📄 Caricamento del CSV in un DataFrame
df_futures = pd.read_csv(file_path)

# 👀 Visualizza le prime righe per verifica
print(df_futures.head())
print(df_futures.columns)

# 📅 Assicura che le date siano nel formato corretto
df_completo['QUOTE_DATE'] = pd.to_datetime(df_completo['QUOTE_DATE']).dt.date
df_futures['date'] = pd.to_datetime(df_futures['date']).dt.date

# 🔁 Merge per aggiungere la colonna 'open' da df_futures a df_completo
df_completo = df_completo.merge(
    df_futures[['date', 'open']].rename(columns={'date': 'QUOTE_DATE', 'open': 'open_futures'}),
    on='QUOTE_DATE',
    how='left'
)

# ✅ Controllo del risultato
print(df_completo[['QUOTE_DATE', 'open_futures']].head())

# 📁 Percorso di salvataggio
output_path = "Data/df_completo_opzioni_btc.csv"

# 💾 Salvataggio
df_completo.to_csv(output_path, index=False)

print(f"✅ File salvato in: {output_path}")

df_completo_renamed = df_completo.rename(columns={
    'QUOTE_DATE': 'date',
    'INSTRUMENT_NAME': 'instrument_name',
    'OPTION_RIGHT': 'option_type',
    'STRIKE': 'strike',
    'MARK_IV': 'mark_iv',
    'UNDERLYING_PRICE': 'underlying_price',
    'OPEN_INTEREST': 'open_interest',
    'VOLUME': 'volume_usd',
    'BID_PRICE': 'bid_price',
    'ASK_PRICE': 'ask_price',
    'open_futures': 'future_open'  # se vuoi usarlo nei confronti
})

df_completo_renamed.to_csv("Data/options_ready_for_backtest.csv", index=False)




