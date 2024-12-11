#!/usr/bin/env python
# coding: utf-8

# # Data Prep

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


sns.set_style('whitegrid')
idx = pd.IndexSlice
deciles = np.arange(.1, 1, .1).round(1)


# ## Load Data

# In[4]:


DATA_STORE = Path('..', 'data', 'assets.h5')


# In[5]:


with pd.HDFStore(DATA_STORE) as store:
    data = (store['quandl/wiki/prices']
            .loc[idx['2007':'2016', :],
                 ['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']]
            .dropna()
            .swaplevel()
            .sort_index()
            .rename(columns=lambda x: x.replace('adj_', '')))
    metadata = store['us_equities/stocks'].loc[:, ['marketcap', 'sector']]


# In[6]:


data.info(null_counts=True)


# In[7]:


metadata.sector = pd.factorize(metadata.sector)[0]
metadata.info()


# In[8]:


data = data.join(metadata).dropna(subset=['sector'])


# In[9]:


data.info(null_counts=True)


# In[10]:


print(f"# Tickers: {len(data.index.unique('ticker')):,.0f} | # Dates: {len(data.index.unique('date')):,.0f}")


# ## Select 500 most-traded stocks

# In[11]:


dv = data.close.mul(data.volume)


# In[12]:


top500 = (dv.groupby(level='date')
          .rank(ascending=False)
          .unstack('ticker')
          .dropna(thresh=8*252, axis=1)
          .mean()
          .nsmallest(500))


# ### Visualize the 200 most liquid stocks

# In[13]:


top200 = (data.close
          .mul(data.volume)
          .unstack('ticker')
          .dropna(thresh=8*252, axis=1)
          .mean()
          .div(1e6)
          .nlargest(200))
cutoffs = [0, 50, 100, 150, 200]
fig, axes = plt.subplots(ncols=4, figsize=(20, 10), sharex=True)
axes = axes.flatten()

for i, cutoff in enumerate(cutoffs[1:], 1):
    top200.iloc[cutoffs[i-1]:cutoffs[i]
                ].sort_values().plot.barh(logx=True, ax=axes[i-1])
fig.tight_layout()


# In[14]:


to_drop = data.index.unique('ticker').difference(top500.index)


# In[15]:


len(to_drop)


# In[16]:


data = data.drop(to_drop, level='ticker')


# In[17]:


data.info(null_counts=True)


# In[18]:


print(f"# Tickers: {len(data.index.unique('ticker')):,.0f} | # Dates: {len(data.index.unique('date')):,.0f}")


# ### Remove outlier observations based on daily returns

# In[19]:


before = len(data)
data['ret'] = data.groupby('ticker').close.pct_change()
data = data[data.ret.between(-1, 1)].drop('ret', axis=1)
print(f'Dropped {before-len(data):,.0f}')


# In[20]:


tickers = data.index.unique('ticker')
print(f"# Tickers: {len(tickers):,.0f} | # Dates: {len(data.index.unique('date')):,.0f}")


# ### Sample price data for illustration

# In[21]:


ticker = 'AAPL'
# alternative
# ticker = np.random.choice(tickers)
price_sample = data.loc[idx[ticker, :], :].reset_index('ticker', drop=True)


# In[22]:


price_sample.info()


# In[23]:


price_sample.to_hdf('data.h5', 'data/sample')


# ## Compute returns

# Group data by ticker

# In[24]:


by_ticker = data.groupby(level='ticker')


# ### Historical returns

# In[25]:


T = [1, 2, 3, 4, 5, 10, 21, 42, 63, 126, 252]


# In[26]:


for t in T:
    data[f'ret_{t:02}'] = by_ticker.close.pct_change(t)


# ### Forward returns

# In[27]:


data['ret_fwd'] = by_ticker.ret_01.shift(-1)
data = data.dropna(subset=['ret_fwd'])


# ## Persist results

# In[28]:


data.info(null_counts=True)


# In[29]:


data.to_hdf('data.h5', 'data/top500')

