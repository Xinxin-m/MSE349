import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import multiprocessing

def date2unix(time_input):
    """
    Convert time input to unix timestamp.
    time_input can be either a unix timestamp (int/float) or a datetime string.
    If datetime string doesn't include time, assume 00:00:00.
    """
    if isinstance(time_input, (int, float)):
        return int(time_input)
    
    # Convert string to datetime
    if len(time_input) <= 10:  # Only date provided (YYYY-MM-DD)
        dt = pd.to_datetime(time_input).replace(hour=0, minute=0, second=0)
    else:  # Full datetime provided
        dt = pd.to_datetime(time_input)
    
    return int(dt.timestamp())

def dropna_truncate(filepath, output_path, start_time, end_time=None, timestep=3600):
    """
    Process a parquet file by dropping NA values and truncating to the specified time range.
    If end_time is None, only truncate at the beginning.
    
    Args:
        filepath (str): Path to the input parquet file
        output_path (str): Path to save the processed parquet file
        start_time: Start time (unix timestamp or datetime string)
        end_time: End time (unix timestamp or datetime string). If None, keep all data after start_time
        timestep (int): Time step in seconds (default: 3600 for hourly data)
    
    Returns:
        bool: True if successful, False if failed
    """
    try:
        start_unix = date2unix(start_time)
        df = pd.read_parquet(filepath).dropna()
        
        # Apply start time filter
        df = df[df['timestamp'] >= start_unix-timestep].reset_index(drop=True)
        
        # Apply end time filter if provided
        if end_time is not None:
            end_unix = date2unix(end_time)
            df = df[df['timestamp'] <= end_unix].reset_index(drop=True)
        
        # Add index column
        df['index'] = ((df['timestamp'] - start_unix) // timestep).astype(int)
        df.set_index('index', inplace=True)
        
        # Save to parquet
        df.to_parquet(output_path)
        return True
        
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return False

def process_files_parallel(folder, filenames, output_folder, start_time, end_time=None, timestep=3600):
    """
    Process multiple parquet files in parallel.
    
    Args:
        folder (str): Input folder containing the parquet files
        filenames (list): List of filenames to process
        start_time: Start time (unix timestamp or datetime string)
        end_time: End time (unix timestamp or datetime string). If None, keep all data after start_time
        timestep (int): Time step in seconds (default: 3600 for hourly data)
        output_folder (str): Folder to save processed files
    
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
        args_list.append((input_path, output_path, start_time, end_time, timestep))
    
    # Get number of CPU cores
    num_cores = multiprocessing.cpu_count()
    
    # Process files in parallel
    successful = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(dropna_truncate, *zip(*args_list)))
        
        # Count successes and failures
        successful = sum(results)
        failed = len(results) - successful
    
    return successful, failed

if __name__ == "__main__":
    folder = "USD_60_indicators" # folder containing input files
    # Get file names 
    with open('filenames_parquet.txt', 'r') as f:
        filenames = f.read().splitlines()

    start_time = "2022-01-01"
    end_time = None
    outfolder = "USD_60_2022"
    successful, failed = process_files_parallel(folder, filenames, outfolder, start_time, end_time)
    print(f"Successfully processed {successful} files")
    print(f"Failed to process {failed} files") 