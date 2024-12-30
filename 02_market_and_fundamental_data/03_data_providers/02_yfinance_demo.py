#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import yfinance as yf
from pathlib import Path

# Define the data path
data_path = Path(r'C:\Users\pinha\OneDrive\Documents\Trading\data')

# How to work with a Ticker object
symbol = 'AAPL'
ticker = yf.Ticker(symbol)

# Show ticker info
print(pd.Series(ticker.info).head(20))

# Get market data
data = ticker.history(period='5d',
                      interval='1m',
                      start=None,
                      end=None,
                      actions=True,
                      auto_adjust=True,
                      back_adjust=False)
print(data.info())

# View company actions
print(ticker.actions)
print(ticker.dividends)
print(ticker.splits)

# Annual and Quarterly Financial Statement Summary
print(ticker.financials)
print(ticker.quarterly_financials)

# Annual and Quarterly Balance Sheet
print(ticker.balance_sheet)
print(ticker.quarterly_balance_sheet)

# Annual and Quarterly Cashflow Statement
print(ticker.cashflow)
print(ticker.quarterly_cashflow)

# Earnings
print(ticker.financials.loc['Net Income'])  # print(ticker.earnings)
print(ticker.income_stmt.loc['Net Income']) # print(ticker.quarterly_earnings)

# Sustainability: Environmental, Social and Governance (ESG)
print(ticker.sustainability)

# Analyst Recommendations
print(ticker.recommendations.info())
print(ticker.recommendations.tail(10))

# Upcoming Events
print(ticker.calendar)

# Option Expiration Dates
print(ticker.options)

# Options Chain
expiration = ticker.options[0]
options = ticker.option_chain(expiration)
print(options.calls.info())
print(options.calls.head())
print(options.puts.info())

# Data Download with proxy server
PROXY_SERVER = 'PROXY_SERVER'

# Example with a proxy (commented out as it requires a valid proxy server)
# msft = yf.Ticker("MSFT")
# msft.history(proxy=PROXY_SERVER)

# Downloading multiple symbols
tickers = yf.Tickers('msft aapl goog')
print(pd.Series(tickers.tickers['MSFT'].info))
print(tickers.tickers['AAPL'].history(period="1mo"))
print(tickers.history(period='1mo').stack(-1))

data = yf.download("SPY AAPL", start="2020-01-01", end="2020-01-05")
print(data.info())

data = yf.download(
    tickers="SPY AAPL MSFT",
    period="5d",
    interval="1m",
    group_by='ticker',
    auto_adjust=True,
    prepost=True,
    threads=True,
    proxy=None
)
print(data.info())

# # Using pandas_datareader
# from pandas_datareader import data as pdr
# yf.pdr_override()

# # Download data with pandas_datareader
# data = pdr.get_data_yahoo('SPY',
#                           start='2017-01-01',
#                           end='2019-04-30',
#                           auto_adjust=False)
# print(data.tail())
