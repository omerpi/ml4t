#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta, timezone
from time import time
from matplotlib.ticker import FuncFormatter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_style('whitegrid')

def format_time(t):
    """Return a formatted time string 'HH:MM:SS' based on a numeric time() value."""
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}'

# Set your date & stock symbol here
date = '10302019'
stock = 'GE'

# Adjust the paths below as needed
data_path = Path(r'C:\Users\pinha\OneDrive\Documents\Trading\data')
itch_store = str(data_path / f'itch_{date}.h5')
order_book_store = data_path / f'order_book_{stock}_{date}.h5'

order_dict = {-1: 'sell', 1: 'buy'}

if not order_book_store.exists():
    def get_messages(date, stock=stock):
        """
        Collect trading messages for given stock from the ITCH HDFStore.
        Merges order information into the trade/cancel/update messages.
        Returns a DataFrame of all relevant messages.
        """
        with pd.HDFStore(itch_store) as store:
            # Correctly filter for the specific stock
            stock_loc = store.select('R', where=f'stock="{stock}"').stock_locate.iloc[0]
            target = f'stock_locate={stock_loc}'

            data_dict = {}
            # Trading message types
            messages_list = ['A', 'F', 'E', 'C', 'X', 'D', 'U', 'P', 'Q']
            for m in messages_list:
                # Each DataFrame gets a 'type' column to identify message type
                data_dict[m] = (
                    store.select(m, where=target)
                    .drop('stock_locate', axis=1)
                    .assign(type=m)
                )

        # Combine Add Order (A) and Add Order with MPID (F)
        order_cols = ['order_reference_number', 'buy_sell_indicator', 'shares', 'price']
        orders = pd.concat([data_dict['A'], data_dict['F']], sort=False, ignore_index=True)[order_cols]

        # Merge E, C, X, D messages with these orders
        for m in messages_list[2:-3]:  # E, C, X, D
            data_dict[m] = data_dict[m].merge(orders, how='left', on='order_reference_number')

        # For 'U' (Replace) messages, merge by the original reference number
        data_dict['U'] = data_dict['U'].merge(
            orders,
            how='left',
            right_on='order_reference_number',
            left_on='original_order_reference_number',
            suffixes=['', '_replaced']
        )

        # For 'Q' messages, rename cross_price -> price
        data_dict['Q'].rename(columns={'cross_price': 'price'}, inplace=True)

        # For 'X' (Cancel), rename cancelled_shares -> shares
        data_dict['X']['shares'] = data_dict['X']['cancelled_shares']
        data_dict['X'] = data_dict['X'].dropna(subset=['price'])

        # Concatenate all messages
        data = pd.concat([data_dict[m] for m in messages_list], ignore_index=True, sort=False)

        # Convert buy_sell_indicator to numeric
        data['buy_sell_indicator'] = pd.to_numeric(data['buy_sell_indicator'], errors='coerce')

        # Convert date/time
        data['date'] = pd.to_datetime(date, format='%m%d%Y')
        data['timestamp'] = data['date'].add(data['timestamp'])

        # Filter out non-printable if that column exists
        if 'printable' in data.columns:
            data = data[data.printable != 0]

        drop_cols = [
            'tracking_number', 'order_reference_number', 'original_order_reference_number',
            'cross_type', 'new_order_reference_number', 'attribution', 'match_number',
            'printable', 'date', 'cancelled_shares'
        ]
        existing_drop_cols = [col for col in drop_cols if col in data.columns]
        data.drop(existing_drop_cols, axis=1, inplace=True)

        return data.sort_values('timestamp').reset_index(drop=True)

    messages = get_messages(date=date, stock=stock)
    messages.info(show_counts=True)

    with pd.HDFStore(order_book_store) as store:
        key = f'{stock}/messages'
        store.put(key, messages)
        print(store.info())

    def get_trades(m):
        """
        Extract and unify trade messages (E, C, P, Q) into a DataFrame.
        """
        trade_dict = {'executed_shares': 'shares', 'execution_price': 'price'}
        cols = ['timestamp', 'executed_shares']
        trades = pd.concat([
            m.loc[m.type == 'E', cols + ['price']].rename(columns=trade_dict),
            m.loc[m.type == 'C', cols + ['execution_price']].rename(columns=trade_dict),
            m.loc[m.type == 'P', ['timestamp', 'price', 'shares']],
            m.loc[m.type == 'Q', ['timestamp', 'price', 'shares']].assign(cross=1),
        ], sort=False).dropna(subset=['price']).fillna(0)
        return trades.set_index('timestamp').sort_index().astype(int)

    trades = get_trades(messages)
    print(trades.info())

    with pd.HDFStore(order_book_store) as store:
        store.put(f'{stock}/trades', trades)

    def add_orders(orders, buysell, nlevels):
        """
        Add orders up to the desired depth (nlevels).
          - If buysell=1 (buy), sort descending by price.
          - If buysell=-1 (sell), sort ascending.
        Returns (updated_counter, list_of_top_levels).
        """
        new_order = []
        items = sorted(orders.copy().items())
        if buysell == 1:
            items = reversed(items)
        for i, (p, s) in enumerate(items, 1):
            new_order.append((p, s))
            if i == nlevels:
                break
        return orders, new_order

    def save_orders(orders, append=False):
        """
        Save the current limit order book to HDFStore for both 'buy' and 'sell'.
        """
        cols = ['price', 'shares']
        for buysell, book in orders.items():
            df_list = []
            for t, data in book.items():
                df_list.append(pd.DataFrame(data=data, columns=cols).assign(timestamp=t))
            if not df_list:
                continue
            df = pd.concat(df_list, ignore_index=True)
            key = f'{stock}/{order_dict[buysell]}'
            df[['price', 'shares']] = df[['price', 'shares']].astype(int)
            with pd.HDFStore(order_book_store) as store:
                if append:
                    store.append(key, df.set_index('timestamp'), format='t')
                else:
                    store.put(key, df.set_index('timestamp'))

    # Initialize
    order_book = {-1: {}, 1: {}}
    current_orders = {-1: Counter(), 1: Counter()}
    message_counter = Counter()
    nlevels = 100

    start_time = time()
    for message in messages.itertuples():
        i = message[0]
        if i % 1e5 == 0 and i > 0:
            print(f'{i:,.0f}\t\t{format_time(time() - start_time)}')
            print(message)
            # Save partial book
            save_orders(order_book, append=True)
            # Reset
            order_book = {-1: {}, 1: {}}
            start_time = time()

        bsi = message.buy_sell_indicator
        if pd.isna(bsi):
            continue
        bsi = int(bsi)
        message_counter.update([message.type])

        price, shares = None, None

        # Handle add or replace
        if message.type in ['A', 'F', 'U']:
            price = int(message.price)
            shares = int(message.shares)
            current_orders[bsi].update({price: shares})
            current_orders[bsi], new_order = add_orders(current_orders[bsi], bsi, nlevels)
            order_book[bsi][message.timestamp] = new_order

        # Handle executions, cancels, deletes, updates
        if message.type in ['E', 'C', 'X', 'D', 'U']:
            if message.type == 'U':
                # For 'U', shares_replaced & price_replaced are negative updates
                if not pd.isna(getattr(message, 'shares_replaced', np.nan)):
                    price = int(getattr(message, 'price_replaced', 0))
                    shares = -int(getattr(message, 'shares_replaced', 0))
            else:
                # Execution or cancel
                if not pd.isna(message.price):
                    price = int(message.price)
                    shares = -int(message.shares)

            if price is not None:
                current_orders[bsi].update({price: shares})
                if current_orders[bsi][price] <= 0:
                    current_orders[bsi].pop(price)
                current_orders[bsi], new_order = add_orders(current_orders[bsi], bsi, nlevels)
                order_book[bsi][message.timestamp] = new_order

    # *** Make sure to save the final chunk after the loop ***
    save_orders(order_book, append=True)

    message_counter = pd.Series(message_counter)
    print("Message Count:\n", message_counter)

# Now read from order_book_store (which should have /ICE/buy, /ICE/sell)
with pd.HDFStore(order_book_store) as store:
    print(store.info())
    buy = store[f'{stock}/buy'].reset_index().drop_duplicates()
    sell = store[f'{stock}/sell'].reset_index().drop_duplicates()

# Convert price to float/dollar form
buy.price = buy.price.mul(1e-4)
sell.price = sell.price.mul(1e-4)

percentiles = [0.01, 0.02, 0.1, 0.25, 0.75, 0.9, 0.98, 0.99]
desc_df = pd.concat([
    buy.price.describe(percentiles=percentiles).to_frame('buy'),
    sell.price.describe(percentiles=percentiles).to_frame('sell')
], axis=1)
print(desc_df)

# Filter extreme outliers
buy = buy[buy.price > buy.price.quantile(0.01)]
sell = sell[sell.price < sell.price.quantile(0.99)]

market_open = '09:30'
market_close = '16:00'

fig, ax = plt.subplots(figsize=(7, 5))
# Plot "Buy"
sns.histplot(
    buy.set_index('timestamp').between_time(market_open, market_close).price,
    ax=ax,
    label='Buy',
    kde=False,
    linewidth=1,
    alpha=0.8,
    color='royalblue'
)

# Plot "Sell"
sns.histplot(
    sell.set_index('timestamp').between_time(market_open, market_close).price,
    ax=ax,
    label='Sell',
    kde=False,
    linewidth=1,
    alpha=0.8,
    color='red'
)

ax.legend(fontsize=10)
ax.set_title('Limit Order Price Distribution')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y/1000):,}'))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${int(x):,}'))
ax.set_xlabel('Price')
ax.set_ylabel("Shares ('000)")
sns.despine()
fig.tight_layout()

utc_offset = timedelta(hours=4)
depth = 100

print("BUY min timestamp:", buy['timestamp'].min())
print("BUY max timestamp:", buy['timestamp'].max())

buy_per_min = (
    buy
    .groupby([pd.Grouper(key='timestamp', freq='Min'), 'price'])['shares']
    .sum()
    .apply(np.log)  # log of the sum of shares
    .to_frame('shares')
    .reset_index('price')
    .between_time(market_open, market_close)
    .groupby(level='timestamp', as_index=False, group_keys=False)
    .apply(lambda x: x.nlargest(columns='price', n=depth))
    .reset_index()
)
buy_per_min.timestamp = buy_per_min.timestamp.view('int64') // int(1e9)
buy_per_min.timestamp = buy_per_min.timestamp.astype(int)
buy_per_min.info()

sell_per_min = (
    sell
    .groupby([pd.Grouper(key='timestamp', freq='Min'), 'price'])['shares']
    .sum()
    .apply(np.log)
    .to_frame('shares')
    .reset_index('price')
    .between_time(market_open, market_close)
    .groupby(level='timestamp', as_index=False, group_keys=False)
    .apply(lambda x: x.nsmallest(columns='price', n=depth))
    .reset_index()
)
sell_per_min.timestamp = sell_per_min.timestamp.view('int64') // int(1e9)
sell_per_min.timestamp = sell_per_min.timestamp.astype(int)
sell_per_min.info()

with pd.HDFStore(order_book_store) as store:
    trades = store[f'{stock}/trades']

# Convert trade prices
trades.price = trades.price.mul(1e-4)
trades = trades[trades.cross == 0].between_time(market_open, market_close)

trades_per_min = (
    trades
    .resample('Min')
    .agg({'price': 'mean', 'shares': 'sum'})
)
trades_per_min.index = trades_per_min.index.view('int64') // int(1e9)
trades_per_min.index = trades_per_min.index.astype(int)
trades_per_min.info()

# Final scatter plot of limit order levels + trade price line
sns.set_style('white')
fig, ax = plt.subplots(figsize=(14, 6))

buy_per_min.plot.scatter(
    x='timestamp', y='price', c='shares',
    ax=ax, cmap='Blues', colorbar=False, alpha=0.2
)

sell_per_min.plot.scatter(
    x='timestamp', y='price', c='shares',
    ax=ax, cmap='Reds', colorbar=False, alpha=0.1
)

title = f'{stock} | {date} | Buy & Sell Limit Order Book | Depth = {depth}'
trades_per_min.price.plot(
    ax=ax, c='k', lw=2, title=title
)

# Format the x-axis as UTC time
ax.xaxis.set_major_formatter(
    FuncFormatter(lambda ts, _: datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%H:%M'))
)

filtered_buy = buy.set_index('timestamp').between_time(market_open, market_close)
print("Filtered BUY range:", filtered_buy.index.min(), filtered_buy.index.max())
print(filtered_buy.head(), filtered_buy.tail())

ax.set_xlabel('')
ax.set_ylabel('Price', fontsize=12)
print("Sample of buy timestamps (raw):\n", buy['timestamp'].head())

red_patch = mpatches.Patch(color='red', label='Sell')
blue_patch = mpatches.Patch(color='royalblue', label='Buy')
plt.legend(handles=[red_patch, blue_patch])
sns.despine()
fig.tight_layout()
plt.show()
