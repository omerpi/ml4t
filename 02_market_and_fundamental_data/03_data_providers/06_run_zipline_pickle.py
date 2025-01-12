import pandas as pd
import matplotlib.pyplot as plt

# Load the performance DataFrame from the pickle file
perf = pd.read_pickle('dma.pickle')
print(perf.head())

# Plotting
plt.figure(figsize=(12, 12))

# Plot portfolio value
ax1 = plt.subplot(211)
perf['portfolio_value'].plot(ax=ax1, title='Portfolio Value Over Time')
ax1.set_ylabel('Portfolio Value')

# Plot AAPL stock price
ax2 = plt.subplot(212, sharex=ax1)
perf['AAPL'].plot(ax=ax2, title='AAPL Stock Price Over Time')
ax2.set_ylabel('AAPL Stock Price')

# Display the plots
plt.tight_layout()
plt.show()
