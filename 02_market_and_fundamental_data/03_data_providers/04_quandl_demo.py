# Quandl had been replaced (acquired) by nasdaqdatalink

import os
import nasdaqdatalink
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set up Seaborn style
sns.set_style('whitegrid')

# Load Quandl API Key
api_key = os.getenv('NASDAQ_DATA_LINK_API_KEY')
if not api_key:
    raise ValueError("Please set the NASDAQ_DATA_LINK_API_KEY environment variable.")
nasdaqdatalink.ApiConfig.api_key = api_key

# Fetch oil price 
data = nasdaqdatalink.get_table(
    'WIKI/PRICES',
    qopts={'columns': ['ticker', 'date', 'close']},
    ticker=['AAPL', 'MSFT'],
    date={'gte': '2016-01-01', 'lte': '2024-12-31'}
)

# Ensure the 'date' column is in datetime format for proper plotting
data['date'] = pd.to_datetime(data['date'])

# Filter data for MSFT
msft_data = data[data['ticker'] == 'MSFT']

# Plot MSFT data
plt.figure(figsize=(12, 4))
plt.plot(msft_data['date'], msft_data['close'], linewidth=2, label='MSFT')
plt.title('MSFT Stock Prices (2016)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
sns.despine()
plt.tight_layout()
plt.show()
