import pandas as pd
import matplotlib.pyplot as plt
from zipline.api import order_target, record, symbol
from zipline.data.bundles.core import ingest
from zipline import run_algorithm
from zipline.data.bundles import register
from zipline.data.bundles.quandl import quandl_bundle
from zipline.utils.run_algo import load_extensions
from datetime import datetime
from dateutil.tz import UTC
import pytz
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# ==================================================
# Configure and Register the Quandl Bundle
# ==================================================
def ingest_data():
    # Retrieve the API key from environment variables
    api_key = os.getenv('QUANDL_API_KEY')
    if not api_key:
        raise ValueError("QUANDL_API_KEY environment variable not set. Please set it before running.")

    # Register the Quandl bundle
    register("quandl", quandl_bundle)

    # Ingest the Quandl bundle
    print("Ingesting Quandl data...")
    ingest("quandl", os.environ, show_progress=True)
    print("Ingestion complete.")

# ==================================================
# First Backtest: Save Stock Data to CSV
# ==================================================
def first_backtest(start, end):
    def initialize(context):
        context.i = 0
        context.assets = [symbol('MMM')]

    def handle_data(context, data):
        df = data.history(context.assets, fields=['price', 'volume'], bar_count=1, frequency="1d")
        df = df.reset_index()

        # Save data to CSV
        if context.i == 0:
            df.columns = ['date', 'asset', 'price', 'volume']
            df.to_csv('stock_data.csv', index=False)
        else:
            df.to_csv('stock_data.csv', index=False, mode='a', header=None)
        context.i += 1

    # Run the algorithm
    run_algorithm(
        start=start,
        end=end,
        initialize=initialize,
        handle_data=handle_data,
        capital_base=1_000_000,  # Set initial capital
        data_frequency="daily",
        bundle="quandl",
        benchmark_returns=None,
    )

    # Visualize saved data
    df = pd.read_csv('stock_data.csv')
    df.date = pd.to_datetime(df.date)
    df.set_index('date').groupby('asset').price.plot(lw=2, legend=True, figsize=(14, 6))
    plt.show()

# ==================================================
# Second Backtest: Dual Moving Average Strategy
# ==================================================
def second_backtest(start, end, output_file):
    def initialize(context):
        context.i = 0
        context.asset = symbol('AAPL')

    def handle_data(context, data):
        context.i += 1
        if context.i < 300:  # Skip first 300 days for full moving averages
            return

        # Compute moving averages
        short_mavg = data.history(context.asset, 'price', bar_count=100, frequency="1d").mean()
        long_mavg = data.history(context.asset, 'price', bar_count=300, frequency="1d").mean()

        # Trading logic
        if short_mavg > long_mavg:
            order_target(context.asset, 100)
        elif short_mavg < long_mavg:
            order_target(context.asset, 0)

        # Save values for later inspection
        record(AAPL=data.current(context.asset, 'price'),
               short_mavg=short_mavg,
               long_mavg=long_mavg)

    def analyze(context, perf):
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 8))

        # Portfolio value over time
        perf.portfolio_value.plot(ax=ax1)
        ax1.set_ylabel('Portfolio Value in $')

        # Stock prices and moving averages
        perf['AAPL'].plot(ax=ax2)
        perf[['short_mavg', 'long_mavg']].plot(ax=ax2)

        # Mark buy and sell signals
        perf_trans = perf.loc[[t != [] for t in perf.transactions]]
        buys = perf_trans.loc[[t[0]['amount'] > 0 for t in perf_trans.transactions]]
        sells = perf_trans.loc[[t[0]['amount'] < 0 for t in perf_trans.transactions]]
        ax2.plot(buys.index, perf.short_mavg.loc[buys.index], '^', markersize=10, color='m')
        ax2.plot(sells.index, perf.short_mavg.loc[sells.index], 'v', markersize=10, color='k')

        ax2.set_ylabel('Price in $')
        plt.legend(loc=0)
        plt.show()

    # Run the algorithm
    perf = run_algorithm(
        start=start,
        end=end,
        initialize=initialize,
        handle_data=handle_data,
        analyze=analyze,
        capital_base=1_000_000,  # Set initial capital
        data_frequency="daily",
        bundle="quandl",
        benchmark_returns=None,
    )
    perf.to_pickle("dma.pickle")


# ==================================================
# Main Execution
# ==================================================
if __name__ == "__main__":
    # Load extensions (if necessary)
    load_extensions(
        default=True,
        extensions=[],
        strict=False,
        environ=os.environ,
    )

    # Ingest data
    ingest_data()

    start = pd.Timestamp(datetime(2014, 1, 1))
    end = pd.Timestamp(datetime(2018, 1, 1))

    # First backtest (save data to CSV)
    first_backtest(
        start=start,
        end=end,
    )

    # Second backtest (dual moving average strategy)
    second_backtest(
        start=start,
        end=end,
        output_file="dma.pickle",
    )

plt.show()