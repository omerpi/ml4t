#!/usr/bin/env python
# coding: utf-8

"""
Remote Data Access using pandas & pandas-datareader

This script demonstrates how to:
- Scrape an HTML table (S&P 500 Constituents) from Wikipedia
- Retrieve historical market data from various sources via pandas-datareader
- Plot data with matplotlib, mplfinance, and seaborn
"""

import os
from datetime import datetime
from io import StringIO
import warnings

import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import yfinance as yf
import mplfinance as mpf
import seaborn as sns
from tiingo import TiingoClient
import urllib.request

# ------------------------------------------------
# Suppress Warnings
# ------------------------------------------------
warnings.filterwarnings('ignore')

# ------------------------------------------------
# Global Plot Settings
# ------------------------------------------------
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 5)

# ------------------------------------------------
# 1. Download HTML table with S&P 500 constituents
# ------------------------------------------------
sp_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_constituents = pd.read_html(sp_url, header=0)[0]

print("S&P 500 Constituents Info:")
sp500_constituents.info()
print("\nS&P 500 Constituents (head):")
print(sp500_constituents.head())

# ------------------------------------------------
# 2. pandas-datareader for Market Data
# ------------------------------------------------
# Below are examples of using pandas_datareader with different sources.

# -------------------------------
# 2.1. Yahoo! Finance (via yfinance)
# -------------------------------
start = datetime(2014, 1, 1)
end = datetime(2017, 5, 24)
ticker = "MMM"

try:
    yahoo = yf.download(tickers=ticker, start=start, end=end, auto_adjust=False)
    print(yahoo.columns)
    print(yahoo.info())

    yahoo.columns = yahoo.columns.droplevel(1)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        yahoo[col] = pd.to_numeric(yahoo[col], errors="coerce")
    yahoo.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
    if "Adj Close" in yahoo.columns:
        yahoo.drop(columns="Adj Close", inplace=True)

    mpf.plot(
        yahoo,
        type="candle",
        style="yahoo",
        title="MMM Candlestick"
    )
    plt.show()
except Exception as e:
    print(f"Yahoo! Finance retrieval failed: {e}")

# -------------------------------
# 2.2. IEX (may require IEX Cloud API key)
# -------------------------------
# IEX_API_KEY = os.getenv('IEX_API_KEY')
# if IEX_API_KEY:
#     try:
#         start_iex = datetime(2015, 2, 9)
#         iex = web.DataReader('MMM', 'iex', start_iex, api_key=IEX_API_KEY)
#         print("\nIEX MMM data retrieved:")
#         iex.info()
            
#         iex['close'].plot(title="FB Closing Prices (IEX)")
#         plt.tight_layout()
#         plt.show()

#         # Book data (only works on trading days):
#         book = web.get_iex_book('AAPL', api_key=IEX_API_KEY)
#         print("\nAAPL Book Data (IEX):")
#         print(f"Available Keys: {list(book.keys())}")
#     except Exception as e:
#         print(f"IEX retrieval failed: {e}")
# else:
#     print("\nNo IEX_API_KEY environment variable found. Skipping IEX example.")

# -------------------------------
# 2.3. Quandl (requires API key)
# -------------------------------
# if os.getenv('NASADAQ_DATA_LINK_API_KEY'):
#     try:
#         symbol = 'FB.US'
#         quandl_df = web.DataReader(symbol, 'quandl', '2015-01-01')
#         print("\nQuandl FB data retrieved:")
#         quandl_df.info()
#     except Exception as e:
#         print(f"Quandl retrieval failed: {e}")
# else:
#     print("\nNo QUANDL_API_KEY environment variable found. Skipping Quandl example.")

# -------------------------------
# 2.4. FRED (Federal Reserve Economic Data)
# -------------------------------
try:
    start_fred = datetime(2010, 1, 1)
    end_fred = datetime(2013, 1, 27)
    gdp = web.DataReader('GDP', 'fred', start_fred, end_fred)
    print("\nFRED GDP data retrieved:")
    gdp.info()

    inflation = web.DataReader(['CPIAUCSL', 'CPILFESL'], 'fred', start_fred, end_fred)
    print("\nFRED Inflation data retrieved:")
    inflation.info()
except Exception as e:
    print(f"FRED retrieval failed: {e}")

# -------------------------------
# 2.5. Fama/French
# -------------------------------
try:
    from pandas_datareader.famafrench import get_available_datasets
    ff_datasets = get_available_datasets()
    print("\nFama/French Datasets (sample):")
    print(ff_datasets[:10])

    ds = web.DataReader('5_Industry_Portfolios', 'famafrench')
    print("\nFama/French '5_Industry_Portfolios':")
    print(ds['DESCR'])
except Exception as e:
    print(f"Fama/French retrieval failed: {e}")

# -------------------------------
# 2.6. World Bank
# -------------------------------
try:
    from pandas_datareader import wb
    gdp_vars = wb.search('gdp.*capita.*const')
    print("\nSample matches for 'gdp.*capita.*const':")
    print(gdp_vars.head())

    wb_data = wb.download(
        indicator='NY.GDP.PCAP.KD',
        country=['US', 'CA', 'MX'],
        start=1990,
        end=2019
    )
    print("\nWorld Bank data retrieved:")
    print(wb_data.head())
except Exception as e:
    print(f"World Bank retrieval failed: {e}")

# -------------------------------
# 2.7. OECD
# -------------------------------
# try:
#     dataset_code = 'LRUNTTTTJP156S'
#     oecd_data = web.DataReader(dataset_code, 'oecd', start='2010-01-01', end='2019-12-31')
#     print("\nOECD data (TUD) for Japan & US:")
#     print(oecd_data[['Japan', 'United States']].head())
# except Exception as e:
#     print(f"OECD retrieval failed: {e}")

# -------------------------------
# 2.8. Stooq
# -------------------------------
try:
    sp500_stooq = web.DataReader('^SPX', 'stooq')
    print("\nS&P 500 (^SPX) from Stooq (head):")
    print(sp500_stooq.head())

    sp500_stooq['Close'].plot(title="^SPX Closing Prices (Stooq)")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Stooq retrieval failed: {e}")

# -------------------------------
# 2.9. NASDAQ Symbols
# -------------------------------
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

# -------------------------------
# 2.10. Tiingo (requires API key)
# -------------------------------
TIINGO_API_KEY = os.getenv('TIINGO_API_KEY')
if TIINGO_API_KEY:
    try:
        config = {
            'session': True,
            'api_key': TIINGO_API_KEY
        }
        client = TiingoClient(config)

        # Define ticker and date range
        ticker = 'GOOG'
        start_date = '2015-01-01'
        end_date = '2020-12-31'

        # Fetch data
        historical_prices = client.get_ticker_price(ticker, startDate=start_date, endDate=end_date)

        # Convert to DataFrame
        df_tiingo = pd.DataFrame(historical_prices)
        df_tiingo.set_index('date', inplace=True)

        # Display data info
        print("\nTiingo GOOG data retrieved:")
        print(df_tiingo.info())

        # Plot the closing prices
        plt.figure(figsize=(14, 7))
        plt.plot(df_tiingo['close'], label='Closing Price', color='green')
        plt.title(f'{ticker} Closing Prices ({start_date} to {end_date})')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Tiingo retrieval failed: {e}")
else:
    print("\nNo TIINGO_API_KEY environment variable found. Skipping Tiingo example.")
