import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import string
import time

sns.set_style('whitegrid')

results = {}

# Generate Test Data
def generate_test_data(nrows=100000, numerical_cols=2000, text_cols=0, text_length=10):
    s = "".join([random.choice(string.ascii_letters)
                 for _ in range(text_length)])
    data = pd.concat([pd.DataFrame(np.random.random(size=(nrows, numerical_cols))),
                      pd.DataFrame(np.full(shape=(nrows, text_cols), fill_value=s))],
                     axis=1, ignore_index=True)
    data.columns = [str(i) for i in data.columns]
    return data

data_type = 'Numeric'
df = generate_test_data(numerical_cols=1000, text_cols=1000)
df.info()

# Parquet
parquet_file = Path('test.parquet')
df.to_parquet(parquet_file)
size = parquet_file.stat().st_size

start_time = time.time()
df = pd.read_parquet(parquet_file)
read_time = time.time() - start_time

parquet_file.unlink()

start_time = time.time()
df.to_parquet(parquet_file)
write_time = time.time() - start_time
parquet_file.unlink()

results['Parquet'] = {'read': read_time, 'write': write_time, 'size': size}

# HDF5
test_store = Path('index.h5')

with pd.HDFStore(test_store) as store:
    store.put('file', df)
size = test_store.stat().st_size

start_time = time.time()
with pd.HDFStore(test_store) as store:
    store.get('file')
read_time = time.time() - start_time

test_store.unlink()

start_time = time.time()
with pd.HDFStore(test_store) as store:
    store.put('file', df)
write_time = time.time() - start_time
test_store.unlink()

results['HDF Fixed'] = {'read': read_time, 'write': write_time, 'size': size}

# CSV
test_csv = Path('test.csv')
df.to_csv(test_csv)
size = test_csv.stat().st_size

start_time = time.time()
df = pd.read_csv(test_csv)
read_time = time.time() - start_time

test_csv.unlink()

start_time = time.time()
df.to_csv(test_csv)
write_time = time.time() - start_time
test_csv.unlink()

results['CSV'] = {'read': read_time, 'write': write_time, 'size': size}

# Store Results
pd.DataFrame(results).assign(Data=data_type).to_csv(f'{data_type}.csv')

# Display Results
df = (pd.concat([pd.read_csv('Numeric.csv', index_col=0),
                 pd.read_csv('Mixed.csv', index_col=0)])
      .rename(columns=str.capitalize))
df.index.name = 'Storage'
df = df.set_index('Data', append=True).unstack()
df.Size /= 1e9

fig, axes = plt.subplots(ncols=3, figsize=(16, 4))
for i, op in enumerate(['Read', 'Write', 'Size']):
    flag = op in ['Read', 'Write']
    df.loc[:, op].plot.barh(title=op, ax=axes[i], logx=flag)
    if flag:
        axes[i].set_xlabel('seconds (log scale)')
    else:
        axes[i].set_xlabel('GB')
fig.tight_layout()
fig.savefig('storage', dpi=300)

