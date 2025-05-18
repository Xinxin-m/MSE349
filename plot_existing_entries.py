#%%
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Your list of filepaths
with open('USD_60_filenames.txt', 'r') as f:
    filepaths = [line.strip() for line in f.readlines()]

cutoff = pd.Timestamp('2020-01-01')
cutoff_unix = int(cutoff.timestamp())
timestep = 3600  # 1 hour

# Step 1: Find global min and max timestamps
min_ts, max_ts = float('inf'), float('-inf')
per_file_ts = []

for path in filepaths:
    df = pd.read_parquet(path, columns=['timestamp'])
    ts = df['timestamp'].astype('int64')
    ts_filtered = ts[ts >= cutoff_unix].values
    if len(ts_filtered) == 0:
        per_file_ts.append(np.array([], dtype='int64'))
        continue
    per_file_ts.append(ts_filtered)
    min_ts = min(min_ts, ts_filtered.min())
    max_ts = max(max_ts, ts_filtered.max())

# Step 2: Generate the full timestamp grid
all_ts = np.arange(min_ts, max_ts + 1, timestep)
ts_to_idx = {ts: i for i, ts in enumerate(all_ts)}

# Step 3: Create presence matrix
num_files = len(filepaths)
num_timestamps = len(all_ts)
presence_matrix = np.zeros((num_files, num_timestamps), dtype=bool)

for i, ts_arr in enumerate(per_file_ts):
    if len(ts_arr) == 0:
        continue
    idxs = np.searchsorted(all_ts, ts_arr)
    presence_matrix[i, idxs] = True

# Step 4: Count how many files contain each timestamp
timestamp_file_counts = presence_matrix.sum(axis=0)

# Plot 1: Dot graph
dates = pd.to_datetime(all_ts, unit='s')
plt.figure(figsize=(14, 5))
plt.scatter(dates, timestamp_file_counts, s=10)
plt.title('Number of Files Containing Each Timestamp (After 2020-01-01)')
plt.xlabel('Timestamp')
plt.ylabel('Number of Files')
plt.grid(True)
plt.tight_layout()
plt.savefig('USD_60_timestamp_plot.png', dpi=300, bbox_inches='tight')
plt.show()




# Plot 2: Histogram of timestamps binned by month
# all_ts_combined = np.concatenate(per_file_ts)
# all_dates = pd.to_datetime(all_ts_combined, unit='s')
# all_dates = all_dates[all_dates >= cutoff]

# plt.figure(figsize=(14, 5))
# plt.hist(all_dates, bins=pd.date_range(start=cutoff, end=all_dates.max() + pd.offsets.MonthEnd(1), freq='M'))
# plt.title('Histogram of Timestamps (Monthly Bins)')
# plt.xlabel('Month')
# plt.ylabel('Count of Timestamps')
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
