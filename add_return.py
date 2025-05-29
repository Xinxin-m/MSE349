import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os

def add_log_return(input_path, output_path, timestep=3600, colnames=None):
    """
    Add log return column to dataframe from parquet file and save to output path.
    If previous timestamp doesn't exist, uses open price for current row.
    
    Args:
        input_path (str): Path to input parquet file
        output_path (str): Path to save output parquet file
        timestep (int): Time step in seconds (default: 3600 for hourly data)
        colnames (list): List of column names to keep. If None, keep all columns.
    
    Returns:
        bool: True if successful, False if failed
    """
    df = pd.read_parquet(input_path)
    if colnames is not None:
        df = df[colnames]

    df['log_return'] = np.nan
    
    # First row: use open price for return
    df.loc[0, 'log_return'] = np.log(df.loc[0, 'close'] / df.loc[0, 'open'])
    
    # Vectorized computation for all rows
    mask = (df['timestamp'].diff() == timestep) & (~df['close'].isna()) & (~df['close'].shift(1).isna())
    df.loc[mask, 'log_return'] = np.log(df.loc[mask, 'close'] / df.loc[mask, 'close'].shift(1))
    
    # For rows where previous timestamp doesn't exist, use open price
    mask = df['log_return'].isna() & (~df['open'].isna()) & (~df['close'].isna())
    df.loc[mask, 'log_return'] = np.log(df.loc[mask, 'close'] / df.loc[mask, 'open'])
    
    df.to_parquet(output_path)
    return True

def process_files_parallel(folder, filenames, output_folder, timestep=3600, colnames=None):
    """
    Process multiple parquet files in parallel to add log returns.
    
    Args:
        folder (str): Input folder containing the parquet files
        filenames (list): List of filenames to process
        output_folder (str): Folder to save processed files
        timestep (int): Time step in seconds (default: 3600 for hourly data)
        colnames (list): List of column names to keep. If None, keep all columns.
    
    Returns:
        tuple: (number of successful processes, number of failed processes)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Prepare arguments for parallel processing
    args_list = []
    for filename in filenames:
        input_path = os.path.join(folder, filename)
        output_path = os.path.join(output_folder, filename)
        args_list.append((input_path, output_path, timestep, colnames))
    
    # Get number of CPU cores
    num_cores = multiprocessing.cpu_count()
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(lambda x: add_log_return(*x), args_list))
    
    successful = sum(results)
    failed = len(results) - successful
    
    print(f"Successfully processed {successful} files")
    print(f"Failed to process {failed} files")
    
    return successful, failed

if __name__ == "__main__":
    folder = "USD_60"  # folder containing input files
    # Get file names 
    with open('filenames_parquet.txt', 'r') as f:
        filenames = f.read().splitlines()

    output_folder = "USD_60_ohlcvr"
    colnames = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'log_price_range', 'log_return']
    successful, failed = process_files_parallel(folder, filenames, output_folder, colnames=colnames)
