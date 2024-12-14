#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import gzip
import shutil
from struct import unpack
from collections import namedtuple, Counter, defaultdict
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urljoin
from datetime import timedelta
from time import time

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

sns.set_style('whitegrid')

##############################################
# Helper Functions
##############################################

def format_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:0>2.0f}:{m:0>2.0f}:{s:0>5.2f}'

def may_be_download(url, data_path):
    if not data_path.exists():
        data_path.mkdir()

    filename = data_path / url.split('/')[-1]        
    if not filename.exists():
        urlretrieve(url, filename)

    unzipped = data_path / (filename.stem + '.bin')
    if not unzipped.exists():
        with gzip.open(str(filename), 'rb') as f_in:
            with open(unzipped, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    return unzipped

def clean_message_types(df):
    df.columns = [c.lower().strip() for c in df.columns]
    df.value = df.value.str.strip()
    df.name = (df.name
               .str.strip()
               .str.lower()
               .str.replace(' ', '_')
               .str.replace('-', '_')
               .str.replace('/', '_'))
    df.notes = df.notes.str.strip()
    df['message_type'] = df.loc[df.name == 'message_type', 'value']
    return df

def format_alpha(mtype, data, alpha_formats, encoding):
    for col in alpha_formats.get(mtype, {}).keys():
        if mtype != 'R' and col == 'stock':
            data = data.drop(col, axis=1)
            continue
        data.loc[:, col] = data.loc[:, col].str.decode("utf-8").str.strip()
        if encoding.get(col):
            data.loc[:, col] = data.loc[:, col].map(encoding.get(col))
    return data

def store_messages(m, itch_store, alpha_formats, encoding, alpha_length):
    with pd.HDFStore(itch_store) as store:
        for mtype, data in m.items():
            data = pd.DataFrame(data)
            data.timestamp = data.timestamp.apply(int.from_bytes, byteorder='big')
            data.timestamp = pd.to_timedelta(data.timestamp)
            if mtype in alpha_formats.keys():
                data = format_alpha(mtype, data, alpha_formats, encoding)
            s = alpha_length.get(mtype)
            if s:
                s = {c: s.get(c) for c in data.columns}
            dc = ['stock_locate']
            if mtype == 'R':
                dc.append('stock')
            store.append(mtype,
                         data,
                         format='t',
                         min_itemsize=s,
                         data_columns=dc)

##############################################
# Configuration
##############################################

data_path = Path('data')
itch_store = str(data_path / 'itch.h5')
order_book_store = data_path / 'order_book.h5'

HTTPS_URL = 'https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/'
SOURCE_FILE = '10302019.NASDAQ_ITCH50.gz'  # Adjust if needed
file_name = may_be_download(urljoin(HTTPS_URL, SOURCE_FILE), data_path)
date = file_name.name.split('.')[0]

event_codes = {'O': 'Start of Messages',
               'S': 'Start of System Hours',
               'Q': 'Start of Market Hours',
               'M': 'End of Market Hours',
               'E': 'End of System Hours',
               'C': 'End of Messages'}

encoding = {
    'primary_market_maker': {'Y': 1, 'N': 0},
    'printable': {'Y': 1, 'N': 0},
    'buy_sell_indicator': {'B': 1, 'S': -1},
    'cross_type': {'O': 0, 'C': 1, 'H': 2},
    'imbalance_direction': {'B': 0, 'S': 1, 'N': 0, 'O': -1}
}

formats = {
    ('integer', 2): 'H',
    ('integer', 4): 'I',
    ('integer', 6): '6s',
    ('integer', 8): 'Q',
    ('alpha',   1): 's',
    ('alpha',   2): '2s',
    ('alpha',   4): '4s',
    ('alpha',   8): '8s',
    ('price_4', 4): 'I',
    ('price_8', 8): 'Q',
}

##############################################
# Load Message Types
##############################################

message_data = pd.read_excel('message_types.xlsx', sheet_name='messages').sort_values('id').drop('id', axis=1)
message_types = clean_message_types(message_data)
message_labels = (message_types.loc[:, ['message_type', 'notes']]
                  .dropna()
                  .rename(columns={'notes': 'name'}))
message_labels.name = (message_labels.name
                       .str.lower()
                       .str.replace('message', '')
                       .str.replace('.', '')
                       .str.strip()
                       .str.replace(' ', '_'))

message_types.message_type = message_types.message_type.ffill()
message_types = message_types[message_types.name != 'message_type']
message_types.value = (message_types.value
                       .str.lower()
                       .str.replace(' ', '_')
                       .str.replace('(', '')
                       .str.replace(')', ''))

message_types['formats'] = message_types[['value', 'length']].apply(tuple, axis=1).map(formats)

alpha_fields = message_types[message_types.value == 'alpha'].set_index('name')
alpha_msgs = alpha_fields.groupby('message_type')
alpha_formats = {k: v.to_dict() for k, v in alpha_msgs.formats}
alpha_length = {k: v.add(5).to_dict() for k, v in alpha_msgs.length}

message_fields, fstring = {}, {}
for t, message in message_types.groupby('message_type'):
    message_fields[t] = namedtuple(typename=t, field_names=message.name.tolist())
    fstring[t] = '>' + ''.join(message.formats.tolist())

##############################################
# Parse Binary Data
##############################################

messages = defaultdict(list)
message_type_counter = Counter()
message_count = 0

start = time()
with file_name.open('rb') as data:
    while True:
        message_size_bytes = data.read(2)
        if not message_size_bytes:
            break
        message_size = int.from_bytes(message_size_bytes, byteorder='big', signed=False)
        message_type = data.read(1).decode('ascii')
        record = data.read(message_size - 1)
        
        try:
            message = message_fields[message_type]._make(unpack(fstring[message_type], record))
        except:
            continue
        
        message_type_counter.update([message_type])
        messages[message_type].append(message)

        if message_type == 'S':
            event_code = message.event_code.decode('ascii')
            if event_code == 'C':
                store_messages(messages, itch_store, alpha_formats, encoding, alpha_length)
                break

        message_count += 1
        if message_count % 25000000 == 0:
            store_messages(messages, itch_store, alpha_formats, encoding, alpha_length)
            messages.clear()

##############################################
# Summary
##############################################

counter = pd.Series(message_type_counter).to_frame('# Trades')
counter['Message Type'] = counter.index.map(message_labels.set_index('message_type').name.to_dict())
counter = counter[['Message Type', '# Trades']].sort_values('# Trades', ascending=False)

with pd.HDFStore(itch_store) as store:
    store.put('summary', counter)

with pd.HDFStore(itch_store) as store:
    stocks = store['R'].loc[:, ['stock_locate', 'stock']]
    trades = store['P'].append(store['Q'].rename(columns={'cross_price': 'price'}), sort=False).merge(stocks)

trades['value'] = trades.shares.mul(trades.price)
trades['value_share'] = trades.value.div(trades.value.sum())

trade_summary = trades.groupby('stock').value_share.sum().sort_values(ascending=False)
trade_summary.iloc[:50].plot.bar(figsize=(14, 6), color='darkblue', title='Share of Traded Value')

plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
sns.despine()
plt.tight_layout()
plt.show()
