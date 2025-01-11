import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import json
from io import BytesIO
from zipfile import ZipFile, BadZipFile
from tqdm import tqdm
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf

sns.set_style('whitegrid')


def define_paths():
    data_path = Path(r'C:\Users\pinha\OneDrive\Documents\Trading\data\SEC')
    data_path.mkdir(parents=True, exist_ok=True)
    sec_url = 'https://www.sec.gov/files/dera/data/financial-statement-data-sets/'
    filing_periods = [(d.year, d.quarter) for d in pd.date_range('2014', '2025-01-10', freq='Q')]
    headers = {
        "User-Agent": "BNPS PARGON (bnps273@example.com)",
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
        "Connection": "keep-alive"
    }
    return data_path, sec_url, filing_periods, headers


def download_sec_data(data_path, sec_url, filing_periods, headers):
    for yr, qtr in tqdm(filing_periods, desc='Downloading SEC Data'):
        path = data_path / f'{yr}_{qtr}' / 'source'
        if not path.exists():
            path.mkdir(parents=True)

        filing = f'{yr}q{qtr}.zip'
        zip_file_path = data_path / filing
        url = sec_url + filing

        if zip_file_path.exists():
            print(f"{filing} already downloaded, skipping.")
            continue

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            with open(zip_file_path, 'wb') as f:
                f.write(response.content)
            with ZipFile(BytesIO(response.content)) as zip_file:
                for file in zip_file.namelist():
                    local_file = path / file
                    if local_file.exists():
                        print(f"{local_file} already exists, skipping.")
                        continue
                    with local_file.open('wb') as output:
                        for line in zip_file.open(file).readlines():
                            output.write(line)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
        except BadZipFile:
            print(f"Bad zip file: {yr} {qtr}")


def convert_tsv_to_parquet(data_path):
    for f in tqdm(sorted(data_path.glob('**/*.txt')), desc='Converting to Parquet'):
        parquet_path = f.parent.parent / 'parquet'
        parquet_path.mkdir(parents=True, exist_ok=True)

        parquet_file = parquet_path / (f.stem + '.parquet')
        if parquet_file.exists():
            continue

        try:
            df = pd.read_csv(f, sep='\t', encoding='latin1', low_memory=False, on_bad_lines='skip')
            df.to_parquet(parquet_file)
            print(f"Converted {f} to {parquet_file}")
        except Exception as e:
            print(f"Error converting {f}: {e}")


def load_example_metadata(data_path):
    file = data_path / '2018_3' / 'source' / '2018q3_notes-metadata.json'
    metadata = None
    if file.exists():
        with file.open() as f:
            metadata = json.load(f)
            print("Loaded metadata from 2018q3_notes-metadata.json")
    return metadata


def extract_fundamentals(data_path, company_name):
    sub_file = data_path / '2018_3' / 'parquet' / 'sub.parquet'
    if sub_file.exists():
        sub = pd.read_parquet(sub_file)
    else:
        raise FileNotFoundError(f"Expected parquet file not found: {sub_file}")

    company = sub[sub.name == company_name].T.dropna().squeeze()
    subs = pd.DataFrame()
    for sf in data_path.glob('**/sub.parquet'):
        sub_df = pd.read_parquet(sf)
        filtered = sub_df[(sub_df.cik.astype(int) == company.cik) & (sub_df.form.isin(['10-Q', '10-K']))]
        subs = pd.concat([subs, filtered])

    nums = pd.DataFrame()
    for num_file in data_path.glob('**/num.parquet'):
        num = pd.read_parquet(num_file).drop('dimh', axis=1, errors='ignore')
        filtered_num = num[num.adsh.isin(subs.adsh)]
        nums = pd.concat([nums, filtered_num])

    nums.ddate = pd.to_datetime(nums.ddate, format='%Y%m%d')
    nums.to_parquet(data_path / f'{company_name}_nums.parquet', index=False)
    return company, subs, nums


def compute_eps(nums, ticker):
    eps = nums[(nums.tag == 'EarningsPerShareDiluted') & (nums.qtrs == 1)]
    eps = eps.groupby('adsh').apply(lambda x: x.nlargest(n=1, columns=['ddate']))
    tk = yf.Ticker(ticker)
    splits = tk.splits
    for split_date, ratio in splits.items():
        split_date = pd.to_datetime(split_date).tz_localize(None)
        eps.loc[eps.ddate < split_date, 'value'] /= ratio
    eps = eps[['ddate', 'value']].set_index('ddate').squeeze().sort_index()
    eps = eps.rolling(4, min_periods=4).sum().dropna()
    return eps


def download_stock_and_calculate_pe(eps, ticker):
    start_date = eps.index.min()
    end_date = pd.to_datetime('now', utc=True)
    print(f"Downloading {ticker} stock data from {start_date.date()} to {end_date.date()}...")

    stock = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.droplevel(1)
    stock = stock.resample('D').last().ffill()
    stock = stock.loc["2014": eps.index.max()]

    if "Adj Close" in stock.columns:
        stock.rename(columns={"Adj Close": "AdjClose"}, inplace=True)
        pe = stock[["AdjClose"]].rename(columns={"AdjClose": "price"})
    else:
        pe = stock[["Close"]].rename(columns={"Close": "price"})

    pe = pe.join(eps.to_frame("eps"))
    pe["eps"] = pe["eps"].ffill()
    pe["P/E Ratio"] = pe["price"].div(pe["eps"])
    return pe


def plot_ttm_pe(pe, ticker):
    plt.figure(figsize=(14, 6))
    pe['P/E Ratio'].plot(lw=2, title=f'{ticker} TTM P/E Ratio')
    plt.show()

    axes = pe[['price', 'eps', 'P/E Ratio']].plot(subplots=True, figsize=(16, 8), legend=False, lw=2)
    axes[0].set_title('Price')
    axes[1].set_title('EPS')
    axes[2].set_title('TTM P/E')
    plt.tight_layout()
    plt.show()


def plot_dividends(nums):
    nums.tag.value_counts()
    fields = [
        'EarningsPerShareDiluted',
        'CommonStockDividendsPerShareDeclared',
        'WeightedAverageNumberOfDilutedSharesOutstanding',
        'OperatingIncomeLoss',
        'NetIncomeLoss',
        'GrossProfit'
    ]
    dividends = nums.loc[nums.tag == 'CommonStockDividendsPerShareDeclared', ['ddate', 'value']]\
                    .groupby('ddate').mean()
    shares = nums.loc[nums.tag == 'WeightedAverageNumberOfDilutedSharesOutstanding', ['ddate', 'value']]\
                 .drop_duplicates().groupby('ddate').mean()

    print("Dividends DataFrame:")
    print(dividends.head(), dividends.shape)
    print("Shares DataFrame:")
    print(shares.head(), shares.shape)

    if not dividends.empty and not shares.empty:
        df = dividends.dropna()
        if df.empty:
            print("No data available after dividing dividends by shares.")
        else:
            ax = df.plot.bar(figsize=(14, 5), title='Dividends per Share', legend=False)
            ax.xaxis.set_major_formatter(mticker.FixedFormatter(df.index.strftime('%Y-%m')))
            plt.show()
    else:
        print("One of the DataFrames (dividends or shares) is empty.")


def main():
    data_path, sec_url, filing_periods, headers = define_paths()
    download_sec_data(data_path, sec_url, filing_periods, headers)
    convert_tsv_to_parquet(data_path)
    load_example_metadata(data_path)  
    company_name = 'APPLE INC'  # Change here to generalize
    ticker = 'AAPL'             # Change here to generalize
    company, subs, nums = extract_fundamentals(data_path, company_name)
    eps = compute_eps(nums, ticker)
    pe = download_stock_and_calculate_pe(eps, ticker)
    plot_ttm_pe(pe, ticker)
    plot_dividends(nums)


if __name__ == "__main__":
    main()
