#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import urllib.request
from io import StringIO

url = 'ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt'

try:
    with urllib.request.urlopen(url) as response:
        data = response.read().decode('utf-8')
    df = pd.read_csv(StringIO(data), sep='|')
    df = df[df['Symbol'] != 'File Creation Time']
    symbols_list = df['Symbol'].tolist()

    print("\nNASDAQ Symbols Info:")
    df.info()
    print(df.head())

    print("\nList of NASDAQ Symbols:")
    print(symbols_list)
except Exception as e:
    print(f"Error: {e}")
