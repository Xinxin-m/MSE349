import os, json
import pandas as pd
from datetime import datetime
from typing import Union, List, Optional, Tuple, Dict
import glob
from multiprocessing import Pool, cpu_count

def check_timestamp_coverage(
    folder: str,
    filenames: Optional[List[str]] = None,
    start_time: Union[str, int] = None,
    window: int = None,
    threshold: Union[int, float] = None,
    interval: int = 720  # interval in minutes
) -> List[str]:
    """
    Check timestamp coverage in parquet files and return files meeting the threshold.
    
    Args:
        folder: Base folder path containing parquet files
        filenames: Optional list of specific filenames to process. If None, process all .parquet files in folder
        start_time: Start time as string in format "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS" or unix timestamp
        window: Number of timestamps to check (including start time if valid)
        threshold: Minimum number of timestamps required (integer) or percentage (float <= 1)
        interval: Time interval in minutes between consecutive timestamps (default: 720)
    
    Returns:
        List of filenames that have >= threshold timestamps in the specified range
    """

    # Convert start_time to unix timestamp
    if isinstance(start_time, str):
        if len(start_time.split()) == 1:  # Only date provided
            start_time = pd.to_datetime(start_time + " 00:00:00")
        else:
            start_time = pd.to_datetime(start_time)
        start_ts = int(start_time.timestamp())
    else:
        start_ts = start_time
    
    end_ts = start_ts + window * interval * 60
    
    # Get list of files to process
    if filenames is not None:
        files = [os.path.join(folder, f) for f in filenames]
    else:
        files = glob.glob(os.path.join(folder, "*.parquet"))

    # Convert threshold to absolute number if it's a percentage
    if isinstance(threshold, float) and threshold <= 1:
        threshold = int(window * threshold)
    
    valid_files = []
    for file in files:
        df = pd.read_parquet(file)
        df = df.reset_index()
        ts_col = df['timestamp'] 
        ts_in_range = ((ts_col >= start_ts) & (ts_col < end_ts)).sum()
        if ts_in_range >= threshold:
            valid_files.append(os.path.basename(file))
            
    return valid_files 


def get_monthly_coverage():
    start_date = '2022-01-01'
    end_date = '2025-03-01'
    
    # Generate list of first day of each month
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    all_files = []
    for date in date_range:
        start_ts = int(date.timestamp())
        files = check_timestamp_coverage(
            folder='USD_720',
            start_ts=start_ts,
            window=60,
            threshold=0.8,
            interval=720
        )
        all_files.append(files)
    
    with open('USD_720_info/datasets_2022-1_2025-3_720_0.8.json', 'w') as f:
        json.dump(all_files, f, indent=2)
    
    # Remove duplicates while preserving order
    return all_files




def process_single_file(
    file_path: str,
    start_ts: int,
    end_ts: int,
    columns: Optional[List[str]] = None
) -> Optional[Tuple[pd.DataFrame, int]]:
    """
    Process a single parquet file and return data within the specified timeframe.
    
    Args:
        file_path: Path to the parquet file
        start_ts: Start timestamp (unix)
        end_ts: End timestamp (unix)
        columns: Optional list of column names to select
    
    Returns:
        Tuple of (DataFrame with data from the file within timeframe, number of entries) or None if error
    """

    # Read parquet file with selected columns
    if columns is not None:
        # Always include timestamp column
        read_cols = ['timestamp'] + [col for col in columns if col != 'timestamp']
        df = pd.read_parquet(file_path, columns=read_cols)
    else:
        df = pd.read_parquet(file_path)
    
    # Filter by timestamp range
    df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] < end_ts)]

    df['filename'] = os.path.basename(file_path)
    return df, len(df)



def get_data_in_timeframe(
    folder: str,
    filenames: List[str],
    start_time: Union[str, int],
    window: int,
    interval: int = 720,  # interval in minutes
    columns: Optional[List[str]] = None,
    n_workers: Optional[int] = None,
    output_path: Optional[str] = None,  # New parameter for saving output
    save_output: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Concatenate data from multiple parquet files within a specified timeframe using parallel processing.
    
    Args:
        folder: Base folder path containing parquet files
        filenames: List of filenames to process
        start_time: Start time as string ("YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS") or unix timestamp
        window: Number of timestamps to check
        interval: Time interval in minutes between consecutive timestamps (default: 720)
        columns: Optional list of column names to select from the files
        n_workers: Number of parallel workers (default: number of CPU cores)
        output_path: Optional path to save the output DataFrame as parquet
    
    Returns:
        Tuple of (DataFrame containing concatenated data from all files within the timeframe,
        sorted by timestamp and filename, Dictionary mapping filenames to their entry counts)
    """
    # Convert start_time to unix timestamp if it's a string
    start_ts = int(pd.to_datetime(start_time).timestamp())
    
    # Calculate end timestamp (exclusive)
    end_ts = start_ts + (window * interval * 60)  # Convert minutes to seconds

    # Process files in parallel
    file_paths = [os.path.join(folder, f) for f in filenames]
    args = [(file_path, start_ts, end_ts, columns) for file_path in file_paths]
    n_workers = n_workers or cpu_count()
    with Pool(n_workers) as pool:
        results = pool.starmap(process_single_file, args)
    
    # Filter out None results and collect DataFrames and counts
    dfs = []
    entry_counts = {}
    for result in results:
        df, count = result
        dfs.append(df)
        entry_counts[df['filename'].iloc[0]] = count
    
    result_df = pd.concat(dfs, ignore_index=True)
    result_df = result_df.sort_values(['timestamp', 'filename'])
    
    # Save output if path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_parquet(output_path)
    
    return result_df, entry_counts


def process_month_data(
    month_input: Union[int, str],
    input_json: str = 'USD_720_info/datasets_2022-1_2025-3_720_0.8.json',
    input_folder: str = 'USD_720',
    window: int = 60,
    interval: int = 720,
    columns: Optional[List[str]] = None,
    n_workers: Optional[int] = None,
    output_folder: str = 'USD_720_PCR',
    save_output: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Process data for a specific month using files from the monthly coverage JSON.
    
    Args:
        month_input: Either an index (int) or month string (e.g., "2022-01")
        input_json: json file obtained from get_monthly_coverage()
        input_folder: Base folder containing the input parquet files
        output_folder: Base folder for output parquet files
        window: Number of timestamps to check
        interval: Time interval in minutes between consecutive timestamps
        columns: Optional list of column names to select
        n_workers: Number of parallel workers
    
    Returns:
        Tuple of (DataFrame containing the processed data for the specified month,
        Dictionary mapping filenames to their entry counts)
    """
    import json
    
    # Load the JSON file
    with open(input_json, 'r') as f:
        monthly_files = json.load(f)
    
    # Determine the month index and month string
    if isinstance(month_input, int):
        month_idx = month_input
        month_str = pd.date_range(start='2022-01-01', periods=len(monthly_files), freq='MS')[month_idx].strftime('%Y-%m')
    else:
        month_str = month_input
        month_idx = pd.date_range(start='2022-01-01', periods=len(monthly_files), freq='MS').get_loc(pd.to_datetime(month_str))
    
    # Get files for the month
    files = monthly_files[month_idx]
    
    # Create output path
    output_path = os.path.join(output_folder, month_str, f"{month_str}.parquet")
    
    # Get start timestamp for the month
    start_time = pd.to_datetime(month_str + "-01")
    
    # Process the data
    return get_data_in_timeframe(
        folder=input_folder,
        filenames=files,
        start_time=start_time,
        window=window,
        interval=interval,
        columns=columns,
        n_workers=n_workers,
        output_path=output_path,
        save_output=save_output
    )

def process_timeframe(
    start_time: Union[str, int],
    window: int,
    threshold: Union[int, float] = 0.8,
    interval: int = 720,
    input_folder: str = 'USD_720',
    output_folder: str = 'USD_720_PCR',
    columns: Optional[List[str]] = None,
    n_workers: Optional[int] = None,
    save_output: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Extract data from each dataset (parquet) in input_folder that are within timeframe [start_time, start_time + window * interval)
    and save the data to output_folder.
    
    Args:
        start_time: Start time as string ("YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS") or unix timestamp
        window: Number of timestamps to check
        input_folder: Base folder containing the input parquet files
        output_folder: Base folder for output parquet files
        threshold: Minimum number of timestamps required (integer) or percentage (float <= 1)
        interval: Time interval in minutes between consecutive timestamps
        columns: Optional list of column names to select
        n_workers: Number of parallel workers
        save_output: Whether to save the output DataFrame
    
    Returns:
        Tuple of (DataFrame containing the processed data,
        Dictionary mapping filenames to their entry counts)
    """
    # Convert start_time to unix timestamp if it's a string
    if isinstance(start_time, str):
        if len(start_time.split()) == 1:  # Only date provided
            start_time = pd.to_datetime(start_time + " 00:00:00")
        else:
            start_time = pd.to_datetime(start_time)
        start_ts = int(start_time.timestamp())
    else:
        start_ts = start_time
    
    # Step 1: Check timestamp coverage to get valid files
    valid_files = check_timestamp_coverage(
        folder=input_folder,
        start_time=start_ts,
        window=window,
        threshold=threshold,
        interval=interval
    )
    
    if not valid_files:
        raise ValueError(f"No files found meeting the threshold criteria for the given timeframe")
    
    # Step 2: Create output path
    if save_output:
        # Create a timestamp string for the output filename in YYYY-MM format
        if isinstance(start_time, str):
            time_str = pd.to_datetime(start_time).strftime('%Y-%m')
        else:
            time_str = pd.to_datetime(start_time, unit='s').strftime('%Y-%m')
        output_path = os.path.join(output_folder, f"{time_str}_L{window}.parquet")
    else:
        output_path = None
    
    # Step 3: Get data in timeframe
    df, entry_counts = get_data_in_timeframe(
        folder=input_folder,
        filenames=valid_files,
        start_time=start_time,
        window=window,
        interval=interval,
        columns=columns,
        n_workers=n_workers,
        output_path=output_path,
        save_output=save_output
    )
    
    # Step 4: Print sorted entry counts
    print(f"\nProcessing complete. Found {len(valid_files)} valid files.")
    print("\nEntries per file (sorted by count):")
    sorted_items = sorted(entry_counts.items(), key=lambda x: x[1])
    for filename, count in sorted_items:
        print(f"{filename}: {count} entries")
    
    return df, entry_counts

if __name__ == "__main__":
    ############### Usage 1: check how many datasets are 'good' for a month #########################
    # df, dic = process_month_data("2022-01", save_output=False)
    # print(df.head())
    # sorted_items = sorted(dic.items(), key=lambda x: x[1])
    # for key, value in sorted_items:
    #     print(f"{key}: {value}")

    ############### Usage 2: create a PCR dataset for a month starting at start_time ########################
    
    for i in range(1, 10):
        df, counts = process_timeframe(
            start_time=f"2023-0{i}-01",
            window=60,
            save_output=True
        )

    # df = pd.read_parquet('USD_720_PCR/2023-01_L60.parquet')
    # print(df.head())
    # print(len(df))
 