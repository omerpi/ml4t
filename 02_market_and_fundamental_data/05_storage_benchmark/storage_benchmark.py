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

# Define the directory for storing benchmark files
data_path = Path(r'C:\Users\pinha\OneDrive\Documents\Trading\data\Storage Benchmark')
data_path.mkdir(parents=True, exist_ok=True)

# Function to generate test data
def generate_test_data(nrows=100000, numerical_cols=1000, text_cols=1000, text_length=10):
    # Create a constant random text string
    s = "".join(random.choice(string.ascii_letters) for _ in range(text_length))
    # Create numerical and text dataframes and concatenate them side by side
    df_numeric = pd.DataFrame(np.random.random(size=(nrows, numerical_cols)))
    df_text = pd.DataFrame(np.full(shape=(nrows, text_cols), fill_value=s))
    df = pd.concat([df_numeric, df_text], axis=1, ignore_index=True)
    # Rename columns for consistency
    df.columns = [f'col_{i}' for i in range(df.shape[1])]
    return df

# Generate a DataFrame for testing
df = generate_test_data(nrows=100000, numerical_cols=1000, text_cols=1000)
print("DataFrame info:")
print(df.info())

# Create a dictionary to store the benchmark results
results = {}

########################################
# Benchmark Parquet
########################################
parquet_file = data_path / 'test.parquet'

# Write benchmark for Parquet
start_time = time.time()
df.to_parquet(parquet_file)
write_time = time.time() - start_time

# Get file size
parquet_size = parquet_file.stat().st_size

# Read benchmark for Parquet
start_time = time.time()
df_read = pd.read_parquet(parquet_file)
read_time = time.time() - start_time

# Clean up file
parquet_file.unlink()

# Save results
results['Parquet'] = {'Read': read_time, 'Write': write_time, 'Size': parquet_size}

########################################
# Benchmark HDF5 (Fixed Format)
########################################
hdf5_file = data_path / 'test.h5'

# Write benchmark for HDF5
start_time = time.time()
with pd.HDFStore(hdf5_file, mode='w') as store:
    store.put('data', df, format='fixed')
write_time = time.time() - start_time

hdf5_size = hdf5_file.stat().st_size

# Read benchmark for HDF5
start_time = time.time()
with pd.HDFStore(hdf5_file, mode='r') as store:
    df_read = store.get('data')
read_time = time.time() - start_time

# Clean up file
hdf5_file.unlink()

results['HDF5'] = {'Read': read_time, 'Write': write_time, 'Size': hdf5_size}

########################################
# Benchmark CSV
########################################
csv_file = data_path / 'test.csv'

# Write benchmark for CSV
start_time = time.time()
df.to_csv(csv_file, index=False)
write_time = time.time() - start_time

csv_size = csv_file.stat().st_size

# Read benchmark for CSV
start_time = time.time()
df_read = pd.read_csv(csv_file)
read_time = time.time() - start_time

# Clean up file
csv_file.unlink()

results['CSV'] = {'Read': read_time, 'Write': write_time, 'Size': csv_size}

########################################
# Create a Results DataFrame
########################################
results_df = pd.DataFrame(results).T  # Transpose to have file types as rows
# Convert size from bytes to gigabytes for readability
results_df['Size'] = results_df['Size'] / 1e9

print("\nBenchmark Results:")
print(results_df)

########################################
# Plotting the Results
########################################
fig, axes = plt.subplots(ncols=3, figsize=(18, 5))

# Plot Read times (in seconds, log scale)
results_df['Read'].plot.barh(ax=axes[0], logx=True, color='steelblue')
axes[0].set_title('Read Time (seconds)')
axes[0].set_xlabel('Seconds (log scale)')

# Plot Write times (in seconds, log scale)
results_df['Write'].plot.barh(ax=axes[1], logx=True, color='seagreen')
axes[1].set_title('Write Time (seconds)')
axes[1].set_xlabel('Seconds (log scale)')

# Plot Sizes (in gigabytes)
results_df['Size'].plot.barh(ax=axes[2], color='indianred')
axes[2].set_title('File Size (GB)')
axes[2].set_xlabel('Size (GB)')

fig.tight_layout()

# Save the plot
plot_file = data_path / 'storage_benchmark.png'
fig.savefig(plot_file, dpi=300)
plt.show()
