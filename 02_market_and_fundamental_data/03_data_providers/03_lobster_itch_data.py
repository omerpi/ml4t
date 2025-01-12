import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime
from itertools import chain

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# Define the data path
data_path = Path(r'C:\Users\pinha\OneDrive\Documents\Trading\data\LOBSTER_SampleFile_AMZN_2012-06-21_10')

# Column labels for the order book data
price_cols = list(chain(*[('Ask Price {},Bid Price {}'.format(i, i)).split(',') for i in range(10)]))
size_cols = list(chain(*[('Ask Size {},Bid Size {}'.format(i, i)).split(',') for i in range(10)]))
cols = list(chain(*zip(price_cols, size_cols)))

# Load order book data
order_data = 'AMZN_2012-06-21_34200000_57600000_orderbook_10.csv'
orders = pd.read_csv(data_path / order_data, header=None, names=cols)
print(orders.info())
print(orders.head())

# Message Type Codes
types = {
    1: 'submission',
    2: 'cancellation',
    3: 'deletion',
    4: 'execution_visible',
    5: 'execution_hidden',
    7: 'trading_halt'
}

# Load message data
trading_date = '2012-06-21'
levels = 10
message_data = f'AMZN_{trading_date}_34200000_57600000_message_{levels}.csv'
messages = pd.read_csv(
    data_path / message_data,
    header=None,
    names=['time', 'type', 'order_id', 'size', 'price', 'direction']
)
print(messages.info())
print(messages.head())

# Map message types
print(messages['type'].map(types).value_counts())

# Convert time to datetime
messages['time'] = pd.to_timedelta(messages['time'], unit='s')
messages['trading_date'] = pd.to_datetime(trading_date)
messages['time'] = messages['trading_date'] + messages['time']
messages.drop('trading_date', axis=1, inplace=True)

# Combine messages and order data
data = pd.concat([messages, orders], axis=1)
print(data.info())

# Filter executions (visible and hidden)
executions = data[data['type'].isin([4, 5])]
print(executions.head())

# Plot limit order prices for messages with visible or hidden executions
cmaps = {'Bid': 'Blues', 'Ask': 'Reds'}
fig, ax = plt.subplots(figsize=(14, 8))
time = executions['time'].dt.to_pydatetime()
for i in range(10):
    for t in ['Bid', 'Ask']:
        y = executions[f'{t} Price {i}']
        c = executions[f'{t} Size {i}']
        ax.scatter(x=time, y=y, c=c, cmap=cmaps[t], s=1, vmin=1, vmax=c.quantile(0.95))
ax.set_xlim(datetime(2012, 6, 21, 9, 30), datetime(2012, 6, 21, 16, 0))
sns.despine()
plt.title('Executions Only')
plt.tight_layout()
plt.show()

# Plot prices for all order types
fig, ax = plt.subplots(figsize=(14, 8))
time = data['time'].dt.to_pydatetime()
for i in range(10):
    for t in ['Bid', 'Ask']:
        y = data[f'{t} Price {i}']
        c = data[f'{t} Size {i}']
        ax.scatter(x=time, y=y, c=c, cmap=cmaps[t], s=1, vmin=1, vmax=c.quantile(0.95))
ax.set_xlim(datetime(2012, 6, 21, 9, 30), datetime(2012, 6, 21, 16, 0))
sns.despine()
plt.title('All Order Types')
plt.tight_layout()
plt.show()
