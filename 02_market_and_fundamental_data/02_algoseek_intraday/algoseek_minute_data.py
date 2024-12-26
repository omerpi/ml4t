from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Adjust the paths below as needed
data_path = Path(r'C:\Users\pinha\OneDrive\Documents\Trading\data')
nasdaq_path = data_path / 'nasdaq100'   # => C:\...\data\nasdaq100

sns.set_style('whitegrid')

# Define columns to drop and rename
tcols = [
    'openbartime', 'firsttradetime', 'highbidtime', 'highasktime', 'hightradetime',
    'lowbidtime', 'lowasktime', 'lowtradetime', 'closebartime', 'lasttradetime'
]
drop_cols = ['unknowntickvolume', 'cancelsize', 'tradeatcrossorlocked']

columns = {
    'volumeweightprice': 'price',
    'finravolume': 'fvolume',
    'finravolumeweightprice': 'fprice',
    'uptickvolume': 'up',
    'downtickvolume': 'down',
    'repeatuptickvolume': 'rup',
    'repeatdowntickvolume': 'rdown',
    'firsttradeprice': 'first',
    'hightradeprice': 'high',
    'lowtradeprice': 'low',
    'lasttradeprice': 'last',
    'nbboquotecount': 'nbbo',
    'totaltrades': 'ntrades',
    'openbidprice': 'obprice',
    'openbidsize': 'obsize',
    'openaskprice': 'oaprice',
    'openasksize': 'oasize',
    'highbidprice': 'hbprice',
    'highbidsize': 'hbsize',
    'highaskprice': 'haprice',
    'highasksize': 'hasize',
    'lowbidprice': 'lbprice',
    'lowbidsize': 'lbsize',
    'lowaskprice': 'laprice',
    'lowasksize': 'lasize',
    'closebidprice': 'cbprice',
    'closebidsize': 'cbsize',
    'closeaskprice': 'caprice',
    'closeasksize': 'casize',
    'firsttradesize': 'firstsize',
    'hightradesize': 'highsize',
    'lowtradesize': 'lowsize',
    'lasttradesize': 'lastsize',
    'tradetomidvolweight': 'volweight',
    'tradetomidvolweightrelative': 'volweightrel'
}

def extract_and_combine_data():
    """
    Reads all .csv.gz files from the 1min_taq directory, combines them into a single DataFrame,
    and saves to algoseek.h5 in the same directory.
    """
    # Directory containing the CSV GZ files
    taq_path = nasdaq_path / '1min_taq'
    # Where we'll save the final HDF5 file
    output_path = taq_path / 'algoseek.h5'

    # If HDF5 file already exists, skip extraction
    if output_path.exists():
        print(f"{output_path} already exists. Skipping data extraction.")
        return

    # Check if the 1min_taq directory exists
    if not taq_path.exists():
        print(f"{taq_path} does not exist! Please make sure CSV data is placed here.")
        # If you want to create the directory automatically, uncomment below:
        # taq_path.mkdir(parents=True, exist_ok=True)
        return

    # Read and combine all CSV GZ files
    print("Reading CSV files and combining data...")
    data_frames = []
    for f in tqdm(list(taq_path.glob('**/*.csv.gz')), desc="Files processed"):
        df = (
            pd.read_csv(f, parse_dates=[['Date', 'TimeBarStart']])
            .rename(columns=str.lower)
            .drop(tcols + drop_cols, axis=1, errors='ignore')
            .rename(columns=columns)
            .set_index('date_timebarstart')
            .sort_index()
            .between_time('9:30', '16:00')
            .set_index('ticker', append=True)
            .swaplevel()
            # e.g. rename 'tradeatcross' to 'atcross'
            .rename(columns=lambda x: x.replace('tradeat', 'at'))
        )
        data_frames.append(df)

    # Concatenate all DataFrames
    if not data_frames:
        print("No CSV.gz files found. Nothing to combine.")
        return

    data = pd.concat(data_frames)
    data = data.apply(pd.to_numeric, downcast='integer', errors='ignore')
    data.index.rename(['ticker', 'date_time'], inplace=True)

    print("Data info before saving:")
    print(data.info())
    # If you want to see null counts:
    print("Null values per column:")
    print(data.isnull().sum())

    # Save to HDF5
    data.to_hdf(output_path, key='min_taq')
    print(f"Data saved to {output_path}")

def analyze_data():
    """
    Loads the HDF5 file from 1min_taq, prints summary info,
    and plots a heatmap of NASDAQ 100 constituents (2015-2017).
    """
    taq_path = nasdaq_path / '1min_taq'
    input_path = taq_path / 'algoseek.h5'

    # Check that file exists before loading
    if not input_path.exists():
        print(f"{input_path} not found. Run extract_and_combine_data() first.")
        return

    # Load the combined data
    df = pd.read_hdf(input_path, 'min_taq')
    print("Data info after loading:")
    print(df.info())
    print("Null values per column:")
    print(df.isnull().sum())

    # Number of unique tickers
    unique_tickers = df.index.unique('ticker')
    print(f"Number of unique tickers: {len(unique_tickers)}")

    # Generate monthly presence heatmap (which tickers have data each month)
    constituents = (
        df.groupby([df.index.get_level_values('date_time').date, 'ticker'])
        .size()
        .unstack('ticker')
        .notnull()
        .astype(int)
        .replace(0, np.nan)
    )

    # Resample by month, take maximum to see if ticker was present at all in that month
    constituents.index = pd.to_datetime(constituents.index)
    constituents = constituents.resample('M').max()
    constituents.index = constituents.index.date  # convert back to date

    fig, ax = plt.subplots(figsize=(20, 20))
    mask = constituents.T.isnull()
    sns.heatmap(constituents.T, mask=mask, cbar=False, ax=ax, cmap='Blues_r')
    ax.set_ylabel('')
    fig.suptitle('NASDAQ100 Constituents (2015-2017)')
    fig.tight_layout()
    plt.show()

# 1) Extract & combine data into HDF5
extract_and_combine_data()

# 2) Analyze the resulting HDF5 file
analyze_data()
