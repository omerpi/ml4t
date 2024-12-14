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

def format_time(t):
    """Return a formatted time string 'HH:MM:SS
    based on a numeric time() value"""
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:0>2.0f}:{m:0>2.0f}:{s:0>5.2f}'

date = '10302019'
data_path = Path(r'C:\Users\pinha\OneDrive\Documents\Trading\data')
itch_store = str(data_path / f'itch_{date}.h5')
order_book_store = data_path / 'order_book.h5'
HTTPS_URL = 'https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/'
SOURCE_FILE = f'{date}.NASDAQ_ITCH50.gz'

def may_be_download(url):
    """Download & unzip ITCH data if not yet available"""
    if not data_path.exists():
        print('Creating directory')
        data_path.mkdir()
    else: 
        print('Directory exists')

    filename = data_path / url.split('/')[-1]        
    if not filename.exists():
        print('Downloading...', url)
        urlretrieve(url, filename)
    else: 
        print('File exists')        

    unzipped = data_path / (filename.stem + '.bin')
    if not unzipped.exists():
        print('Unzipping to', unzipped)
        with gzip.open(str(filename), 'rb') as f_in:
            with open(unzipped, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else: 
        print('File already unpacked')
    return unzipped

file_name = may_be_download(urljoin(HTTPS_URL, SOURCE_FILE))
date = file_name.name.split('.')[0]
event_codes = {'O': 'Start of Messages',
               'S': 'Start of System Hours',
               'Q': 'Start of Market Hours',
               'M': 'End of Market Hours',
               'E': 'End of System Hours',
               'C': 'End of Messages'}

encoding = {'primary_market_maker': {'Y': 1, 'N': 0},
            'printable'           : {'Y': 1, 'N': 0},
            'buy_sell_indicator'  : {'B': 1, 'S': -1},
            'cross_type'          : {'O': 0, 'C': 1, 'H': 2},
            'imbalance_direction' : {'B': 0, 'S': 1, 'N': 0, 'O': -1}}
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
message_data = (pd.read_excel(r'C:\Users\pinha\OneDrive\Documents\Trading\ml4t-code\02_market_and_fundamental_data\01_NASDAQ_TotalView-ITCH_Order_Book\message_types.xlsx',
                              sheet_name='messages')
                .sort_values('id')
                .drop('id', axis=1))

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
message_types = clean_message_types(message_data)
message_labels = (message_types.loc[:, ['message_type', 'notes']]
                  .dropna()
                  .rename(columns={'notes': 'name'}))
message_labels.name = (message_labels.name
                       .str.lower()
                       .str.replace('message', '')
                       .str.replace('.', '')
                       .str.strip().str.replace(' ', '_'))
message_types.message_type = message_types.message_type.ffill()
message_types = message_types[message_types.name != 'message_type']
message_types.value = (message_types.value
                       .str.lower()
                       .str.replace(' ', '_')
                       .str.replace('(', '')
                       .str.replace(')', ''))
message_types.to_csv('message_types.csv', index=False)

message_types = pd.read_csv('message_types.csv')

message_types.loc[:, 'formats'] = (message_types[['value', 'length']]
                            .apply(tuple, axis=1).map(formats))

alpha_fields = message_types[message_types.value == 'alpha'].set_index('name')
alpha_msgs = alpha_fields.groupby('message_type')
alpha_formats = {k: v.to_dict() for k, v in alpha_msgs.formats}
alpha_length = {k: v.add(5).to_dict() for k, v in alpha_msgs.length}
message_fields, fstring = {}, {}
for t, message in message_types.groupby('message_type'):
    message_fields[t] = namedtuple(typename=t, field_names=message.name.tolist())
    fstring[t] = '>' + ''.join(message.formats.tolist())
def format_alpha(mtype, data):
    """Process byte strings of type alpha"""

    for col in alpha_formats.get(mtype).keys():
        if mtype != 'R' and col == 'stock':
            data = data.drop(col, axis=1)
            continue
        data.loc[:, col] = data.loc[:, col].str.decode("utf-8").str.strip()
        if encoding.get(col):
            data.loc[:, col] = data.loc[:, col].map(encoding.get(col))
    return data

def store_messages(m):
    """Handle occasional storing of all messages"""
    with pd.HDFStore(itch_store) as store:
        for mtype, data in m.items():
            data = pd.DataFrame(data)

            if 'timestamp' in data.columns:
                data.timestamp = data.timestamp.apply(int.from_bytes, byteorder='big')
                data.timestamp = pd.to_timedelta(data.timestamp)

            if mtype in alpha_formats.keys():
                data = format_alpha(mtype, data)

            for col in data.select_dtypes(include=['object']).columns:
                data[col] = data[col].fillna('Unknown').astype(str)

            min_itemsize = {col: 10 for col in data.select_dtypes(include=['object']).columns}

            data_columns = [col for col in ['stock_locate', 'stock'] if col in data.columns]

            try:
                store.append(
                    mtype,
                    data,
                    format='t',
                    min_itemsize=min_itemsize,
                    data_columns=data_columns
                )
            except Exception as e:
                print(f"Error storing message type {mtype}: {e}")
                print("Data info:")
                print(data.info())
                print("Data sample:")
                print(data.head())
                return 1
    return 0

if not Path(itch_store).exists():
    messages = defaultdict(list)
    message_count = 0
    message_type_counter = Counter()

    start = time()
    with file_name.open('rb') as data:
        while True:

            message_size = int.from_bytes(data.read(2), byteorder='big', signed=False)
            
            message_type = data.read(1).decode('ascii')        
            message_type_counter.update([message_type])

            try:
                record = data.read(message_size - 1)
                message = message_fields[message_type]._make(unpack(fstring[message_type], record))
                messages[message_type].append(message)
            except Exception as e:
                print(e)
                print(message_type)
                print(record)
                print(fstring[message_type])
            
            if message_type == 'S':
                seconds = int.from_bytes(message.timestamp, byteorder='big') * 1e-9
                print('\n', event_codes.get(message.event_code.decode('ascii'), 'Error'))
                print(f'\t{format_time(seconds)}\t{message_count:12,.0f}')
                if message.event_code.decode('ascii') == 'C':
                    store_messages(messages)
                    break
            message_count += 1

            if message_count % 2.5e7 == 0:
                seconds = int.from_bytes(message.timestamp, byteorder='big') * 1e-9
                d = format_time(time() - start)
                print(f'\t{format_time(seconds)}\t{message_count:12,.0f}\t{d}')
                res = store_messages(messages)
                if res == 1:
                    print(pd.Series(dict(message_type_counter)).sort_values())
                    break
                messages.clear()

    print('Duration:', format_time(time() - start))

    counter = pd.Series(message_type_counter).to_frame('# Trades')
    counter['Message Type'] = counter.index.map(message_labels.set_index('message_type').name.to_dict())
    counter = counter[['Message Type', '# Trades']].sort_values('# Trades', ascending=False)

    with pd.HDFStore(itch_store) as store:
        store.put('summary', counter)
else:
    print('itch_store exists')

with pd.HDFStore(itch_store) as store:
    stocks = store['R'].loc[:, ['stock_locate', 'stock']]
    trades = pd.concat([store['P'], store['Q'].rename(columns={'cross_price': 'price'})], ignore_index=True).merge(stocks)

trades['value'] = trades.shares.mul(trades.price)
trades['value_share'] = trades.value.div(trades.value.sum())

trade_summary = trades.groupby('stock').value_share.sum().sort_values(ascending=False)
trade_summary.iloc[:50].plot.bar(figsize=(14, 6), color='darkblue', title='Share of Traded Value')

plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
sns.despine()
plt.tight_layout()

plt.show()
