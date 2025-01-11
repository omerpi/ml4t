import yfinance as yf
import pandas as pd

# ------------------------------
# 1) Example: fetch Apple splits
# ------------------------------
aapl = yf.Ticker("AAPL")
splits = aapl.splits  # returns a Pandas Series: index=split date, value=split ratio

for split_date, ratio in splits.items():
    split_date = pd.to_datetime(split_date).tz_localize(None)  # Drop timezone
    print(split_date)
    print(ratio)

print(splits)