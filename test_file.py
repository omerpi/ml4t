import pandas as pd
import matplotlib.pyplot as plt

# Load the performance DataFrame from the pickle file
perf = pd.read_pickle('dma.pickle')

# Display the first few rows of the DataFrame
print(perf.head())

# Plotting settings
plt.figure(figsize=(12, 12))

# Plot Portfolio Value
ax1 = plt.subplot(211)
perf['portfolio_value'].plot(ax=ax1)
ax1.set_ylabel('Portfolio Value')
ax1.set_title('Portfolio Value Over Time')

# Plot AAPL Stock Price
ax2 = plt.subplot(212, sharex=ax1)
perf['AAPL'].plot(ax=ax2)
ax2.set_ylabel('AAPL Stock Price')
ax2.set_title('AAPL Stock Price Over Time')

# Show the plots
plt.tight_layout()
plt.show()
