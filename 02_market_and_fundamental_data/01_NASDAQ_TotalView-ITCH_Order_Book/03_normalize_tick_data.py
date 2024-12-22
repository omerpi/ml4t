#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from math import pi
from bokeh.plotting import figure, show
from scipy.stats import normaltest

# Settings
pd.set_option('display.float_format', lambda x: '%.2f' % x)
sns.set_style('whitegrid')
mpl.rcParams['figure.dpi'] = 100

# Set your date & stock symbol here
date = '10302019'
stock = 'AAPL'

# Adjust the paths below as needed
data_path = Path(r'C:\Users\pinha\OneDrive\Documents\Trading\data')
itch_store = str(data_path / f'itch_{date}.h5')
order_book_store = data_path / f'order_book_{stock}_{date}.h5'
title = f'{stock} | {pd.to_datetime(date, format="%m%d%Y").date()}'

# --- Load system event data to get market open/close times ---
with pd.HDFStore(itch_store) as store:
    sys_events = store['S'].set_index('event_code').drop_duplicates()
    sys_events.timestamp = sys_events.timestamp.add(pd.to_datetime(date, format="%m%d%Y")).dt.time
    market_open = sys_events.loc['Q', 'timestamp']
    market_close = sys_events.loc['M', 'timestamp']

# --- Trade Summary: Combine 'P' and 'Q' messages to see volumes ---
with pd.HDFStore(itch_store) as store:
    stocks = store['R']
    stocks = stocks.loc[:, ['stock_locate', 'stock']]
    trades_raw = pd.concat([store['P'], store['Q'].rename(columns={'cross_price': 'price'})], ignore_index=True, sort=False)
    trades_raw = trades_raw.merge(stocks, how='left')

trades_raw['value'] = trades_raw.shares.mul(trades_raw.price)
trades_raw['value_share'] = trades_raw.value.div(trades_raw.value.sum())
trade_summary = trades_raw.groupby('stock').value_share.sum().sort_values(ascending=False)

# Optional plot of top 50 by traded value
fig, ax = plt.subplots(figsize=(14, 6))
trade_summary.iloc[:50].plot.bar(ax=ax, color='darkblue', title='% of Traded Value')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
plt.tight_layout()
plt.show()

# --- Stock Trade Summary ---
with pd.HDFStore(order_book_store) as store:
    trades = store[f'{stock}/trades']
trades.price = trades.price.mul(1e-4)
trades = trades[trades.cross == 0].between_time(market_open, market_close).drop('cross', axis=1)
print(trades.info())

# --- Tick Bars ---
tick_bars = trades.copy()
tick_bars.index = tick_bars.index.time  # Convert DatetimeIndex to just time for plotting
tick_bars.price.plot(figsize=(10, 5), lw=1, title=f'Tick Bars | {stock} | {pd.to_datetime(date, format="%m%d%Y").date()}')
plt.xlabel('')
plt.tight_layout()
plt.show()

# Normality test of tick returns
print("Tick return normality test:", normaltest(tick_bars.price.pct_change().dropna()))

# --- price_volume function (Cleaned & Updated) ---
def price_volume(df, price_col='vwap', vol_col='vol', suptitle='', fname=None):
    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(15,8))
    ax_price, ax_vol = axes

    ax_price.plot(df.index, df[price_col], color='blue', lw=1.5)
    ax_price.set_ylabel('Price', fontsize=12)
    ax_price.set_title(price_col.capitalize(), fontsize=14)

    ax_vol.bar(df.index, df[vol_col], width=1/(4*len(df.index)), color='red', alpha=0.7)
    ax_vol.set_ylabel('Volume', fontsize=12)
    ax_vol.set_title(vol_col.capitalize(), fontsize=14)

    ax_vol.xaxis.set_major_locator(mpl.dates.HourLocator(interval=3))
    ax_vol.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()

    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    if fname:
        plt.savefig(fname, bbox_inches='tight')
    plt.show()

# --- Time Bars ---
def get_bar_stats(agg_trades):
    vwap = agg_trades.apply(lambda x: np.average(x.price, weights=x.shares)).to_frame('vwap')
    ohlc = agg_trades.price.ohlc()
    vol = agg_trades.shares.sum().to_frame('vol')
    txn = agg_trades.shares.size().to_frame('txn')
    return pd.concat([ohlc, vwap, vol, txn], axis=1)

resampled = trades.groupby(pd.Grouper(freq='1Min'))
time_bars = get_bar_stats(resampled)
print("Time bars normality test:", normaltest(time_bars.vwap.pct_change().dropna()))
price_volume(time_bars, price_col='vwap', vol_col='vol',
             suptitle=f'Time Bars | {stock} | {pd.to_datetime(date, format="%m%d%Y").date()}',
             fname='time_bars.png')

# --- Bokeh Candlestick for a 5-min resample ---
resampled_5m = trades.groupby(pd.Grouper(freq='5Min'))
df_5m = get_bar_stats(resampled_5m)
increase = df_5m.close > df_5m.open
decrease = df_5m.open > df_5m.close
w = 2.5 * 60 * 1000  # 2.5 min in ms
WIDGETS = "pan, wheel_zoom, box_zoom, reset, save"
p = figure(x_axis_type='datetime', tools=WIDGETS, width=1500, title=str(stock) + " Candlestick (5Min)")
p.xaxis.major_label_orientation = pi/4
p.grid.grid_line_alpha=0.4
p.segment(df_5m.index, df_5m.high, df_5m.index, df_5m.low, color="black")
p.vbar(df_5m.index[increase], w, df_5m.open[increase], df_5m.close[increase],
       fill_color="#D5E1DD", line_color="black")
p.vbar(df_5m.index[decrease], w, df_5m.open[decrease], df_5m.close[decrease],
       fill_color="#F2583E", line_color="black")
show(p)  # Uncomment to display if running in an environment that supports Bokeh output

# --- Volume Bars ---
with pd.HDFStore(order_book_store) as store:
    trades = store[f'{stock}/trades']
trades.price = trades.price.mul(1e-4)
trades = trades[trades.cross == 0].between_time(market_open, market_close).drop('cross', axis=1)
trades_per_min = trades.shares.sum() / (60*7.5)  # min per trading day
trades['cumul_vol'] = trades.shares.cumsum()

df_vol = trades.reset_index()
by_vol = df_vol.groupby(df_vol.cumul_vol.div(trades_per_min).round().astype(int))
vol_bars = pd.concat([by_vol.timestamp.last().to_frame('timestamp'), get_bar_stats(by_vol)], axis=1)
vol_bars.set_index('timestamp', inplace=True)
price_volume(vol_bars, price_col='vwap', vol_col='vol',
             suptitle=f'Volume Bars | {stock} | {pd.to_datetime(date, format="%m%d%Y").date()}',
             fname='volume_bars.png')
print("Volume bars normality test:", normaltest(vol_bars.vwap.dropna()))

# --- Dollar Bars ---
with pd.HDFStore(order_book_store) as store:
    trades = store[f'{stock}/trades']
trades.price = trades.price.mul(1e-4)
trades = trades[trades.cross == 0].between_time(market_open, market_close).drop('cross', axis=1)
value_per_min = trades.shares.mul(trades.price).sum() / (60*7.5)
trades['cumul_val'] = trades.shares.mul(trades.price).cumsum()

df_val = trades.reset_index()
by_value = df_val.groupby(df_val.cumul_val.div(value_per_min).round().astype(int))
dollar_bars = pd.concat([by_value.timestamp.last().to_frame('timestamp'), get_bar_stats(by_value)], axis=1)
dollar_bars.set_index('timestamp', inplace=True)
price_volume(dollar_bars, price_col='vwap', vol_col='vol',
             suptitle=f'Dollar Bars | {stock} | {pd.to_datetime(date, format="%m%d%Y").date()}',
             fname='dollar_bars.png')
print("Dollar bars normality test:", normaltest(dollar_bars.vwap.dropna()))
