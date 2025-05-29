import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
import re
from multiprocessing import Pool, cpu_count

def plot_timestamp_coverage_with_percentage(filenames, folder, timestep=3600, cutoff_date='2020-01-01', output_path=None):
    """
    Plot the coverage of timestamps across multiple parquet files, including both raw counts and percentages.
    
    Args:
        filenames (list): List of filenames to process
        folder (str): Path to the folder containing the files
        timestep (int): Time step in seconds (default: 3600 for 1 hour)
        cutoff_date (str): Date to start analysis from (default: '2020-01-01')
        output_path (str, optional): Path to save the plot. If None, plot is shown but not saved.
    """
    # Convert cutoff date to unix timestamp (00:00:00)
    cutoff = pd.Timestamp(cutoff_date).normalize()
    cutoff_unix = int(cutoff.timestamp())

    # Initialize max timestamp and arrays to store all timestamps and time ranges
    max_ts = cutoff_unix
    all_timestamps = []
    dataset_ranges = []  # Store (start_ts, end_ts) for each dataset

    for filename in filenames:
        filepath = os.path.join(folder, filename)
        try:
            # First try to read with timestamp column
            df = pd.read_csv(filepath, usecols=['timestamp'])
            col = 'timestamp'
        except (KeyError, ValueError):
            # If timestamp column not found, read only the first column
            df = pd.read_csv(filepath, usecols=[0], header=None)
            col = 0  # Use integer index since we're reading without headers
           
        ts = df[col].astype('int64')
        ts_filtered = ts[ts >= cutoff_unix]
        if not ts_filtered.empty:
            all_timestamps.append(ts_filtered.values)
            dataset_ranges.append((ts_filtered.min(), ts_filtered.max()))
            max_ts = max(max_ts, ts_filtered.max())

    # Generate the full timestamp grid
    all_ts = np.arange(cutoff_unix, max_ts + 1, timestep)
    ts_to_idx = {ts: i for i, ts in enumerate(all_ts)}

    # Create presence matrix
    num_files = len(filenames)
    num_timestamps = len(all_ts)
    presence_matrix = np.zeros((num_files, num_timestamps), dtype=bool)

    for i, ts_arr in enumerate(all_timestamps):
        if len(ts_arr) == 0:
            continue
        idxs = np.searchsorted(all_ts, ts_arr)
        presence_matrix[i, idxs] = True

    # Count how many files contain each timestamp
    timestamp_file_counts = presence_matrix.sum(axis=0)

    # Calculate percentage of datasets that contain each timestamp
    # Only consider datasets whose time range includes the current timestamp
    timestamp_percentages = np.zeros(num_timestamps)
    for i, ts in enumerate(all_ts):
        # Count how many datasets have this timestamp in their range
        valid_datasets = sum(1 for start, end in dataset_ranges if start <= ts <= end)
        if valid_datasets > 0:
            timestamp_percentages[i] = (timestamp_file_counts[i] / valid_datasets) * 100

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    dates = pd.to_datetime(all_ts, unit='s')

    # Plot 1: Raw counts
    ax1.scatter(dates, timestamp_file_counts, s=10)
    ax1.set_title(f'Number of Files Containing Each Timestamp (After {cutoff_date}, Timestep: {timestep}s)')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Number of Files')
    ax1.grid(True)

    # Plot 2: Percentages
    ax2.scatter(dates, timestamp_percentages, s=10)
    ax2.set_title(f'Percentage of Valid Datasets Containing Each Timestamp (After {cutoff_date}, Timestep: {timestep}s)')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Percentage (%)')
    ax2.grid(True)
    ax2.set_ylim(0, 100)  # Set y-axis limits for percentage

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def process_single_summary_file(summary_file):
    """
    Process a single summary file and generate its plots.
    
    Args:
        summary_file (str): Path to the summary file to process
    """
    # Extract timestep from filename
    match = re.search(r'_(\d+)\.csv$', summary_file)
    if not match:
        print(f"Skipping {summary_file} - could not extract timestep")
        return
        
    timestep = int(match.group(1))
    
    # Read the summary file
    try:
        df = pd.read_csv(summary_file)
        if 'file_name' not in df.columns:
            print(f"Skipping {summary_file} - no file_name column found")
            return
            
        # Generate output path
        output_path = os.path.join(os.path.dirname(summary_file), f'timestamp_coverage_with_percentage_{timestep}.png')
        
        # Generate plot
        print(f"Processing timestep {timestep}s...")
        plot_timestamp_coverage_with_percentage(
            filenames=df['file_name'].tolist(),
            folder='Kraken_OHLCVT',  # Assuming files are in current directory
            timestep=timestep,
            cutoff_date='2020-01-01',
            output_path=output_path
        )
        
    except Exception as e:
        print(f"Error processing {summary_file}: {str(e)}")

def process_summary_files(summary_folder='Kraken_OHLCVT_summary'):
    """
    Process all summary files and generate plots for each timestep using parallel processing.
    
    Args:
        summary_folder (str): Path to the folder containing summary files
    """
    # Find all summary files
    pattern = os.path.join(summary_folder, 'csv_file_summary.csv_*.csv')
    summary_files = glob.glob(pattern)

    if not summary_files:
        print("No files found matching the pattern. Please check the pattern and directory path.")
        return
    
    # Determine number of processes to use (leave one CPU free for system tasks)
    num_processes = max(1, cpu_count() - 1)
    
    # Create a pool of workers and process files in parallel
    with Pool(processes=num_processes) as pool:
        pool.map(process_single_summary_file, summary_files)

if __name__ == "__main__":
    process_summary_files() 