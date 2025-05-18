import pandas as pd
import numpy as np
from multiprocessing import Pool
import os


def add_colnames(input_data, column_names=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']):
    """
    Add column names to a DataFrame without a header, from a CSV file or an existing DataFrame.
    input_data: String (CSV file path) or pandas DataFrame.
    """
    # Check if input_data is a string (CSV path)
    if isinstance(input_data, str):
        df = pd.read_csv(input_data, header=None)
    # Check if input_data is a DataFrame
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()  # Avoid modifying the original
    # Validate number of columns
    num_cols = df.shape[1]
    num_expected = len(column_names)
    
    if num_cols != num_expected:
        raise ValueError(f"DataFrame has {num_cols} columns, but {num_expected} column names were provided: {column_names}")
    df.columns = column_names
    return df

def add_return_logreturn_volume(df, timestep = 3600):
    """
    Input: df with 'timestamp', 'close', 'open', and 'volume' columns.
    Adds:
      - 'log_return': log(P_t / P_{t-1})
      - 'return': (P_t - P_{t-1}) / P_{t-1}
      - 'volume_change': (V_t - V_{t-1}) / V_{t-1}
    Uses open price if previous close is missing.
    """
    df = df.copy()
    df['log_return'] = np.nan
    df['return'] = np.nan
    df['volume_change'] = np.nan

    # First row: use open price for return, volume_change remains NaN
    df.loc[0, 'log_return'] = np.log(df.loc[0, 'close'] / df.loc[0, 'open'])
    df.loc[0, 'return'] = df.loc[0, 'close'] / df.loc[0, 'open'] - 1

    for i in range(1, len(df)):
        prev_close = df.loc[i-1, 'close']
        curr_close = df.loc[i, 'close']
        prev_vol = df.loc[i-1, 'volume']
        curr_vol = df.loc[i, 'volume']

        # Compute return
        if df.loc[i, 'timestamp'] == df.loc[i-1, 'timestamp'] + timestep and not np.isnan(prev_close) and not np.isnan(curr_close):
            df.loc[i, 'log_return'] = np.log(curr_close / prev_close)
            df.loc[i, 'return'] = curr_close / prev_close - 1
        elif not np.isnan(df.loc[i, 'open']):
            df.loc[i, 'log_return'] = np.log(curr_close / df.loc[i, 'open'])
            df.loc[i, 'return'] = curr_close / df.loc[i, 'open'] - 1

        # Compute volume change
        if not np.isnan(prev_vol) and not np.isnan(curr_vol) and prev_vol != 0:
            df.loc[i, 'volume_change'] = curr_vol / prev_vol - 1

    return df


def close_to_high_ratio(close: pd.Series, high: pd.Series) -> pd.Series:
    return close / high

def close_to_low_ratio(close: pd.Series, low: pd.Series) -> pd.Series:
    return close / low

def log_price_range(high: pd.Series, low: pd.Series) -> pd.Series:
    return np.log(high / low)

def amihud_illiquidity(log_return: pd.Series, volume: pd.Series) -> pd.Series:
    return np.abs(log_return) / volume

def process_df(csv_path: str, folder: str = None) -> pd.DataFrame:
    """
    Process a CSV file by:
    1. Adding column names
    2. Adding returns, log returns, and volume changes
    3. Adding technical indicators (close/high ratio, close/low ratio, log price range, Amihud)
    4. Saving to parquet format
    
    Args:
        csv_path: Path to input CSV file
        folder: Optional folder to save output parquet file. If None, saves in same location as input.
    """
    # Read and add column names
    df = add_colnames(csv_path)
    
    # Add returns and volume changes
    df = add_return_logreturn_volume(df)
    
    # Add technical indicators
    df['close_to_high'] = close_to_high_ratio(df['close'], df['high'])
    df['close_to_low'] = close_to_low_ratio(df['close'], df['low'])
    df['log_price_range'] = log_price_range(df['high'], df['low'])
    df['amihud'] = amihud_illiquidity(df['log_return'], df['volume'])
    
    # Save to parquet
    if folder is not None:
        os.makedirs(folder, exist_ok=True)
    output_path = csv_path.replace('.csv', '.parquet') if folder is None else f"{folder}/{csv_path.split('/')[-1].replace('.csv', '.parquet')}"
    df.to_parquet(output_path, index=False)
    
    return df

def process_all_files(filepaths: list, folder: str = None, n_workers: int = None):
    """
    Process all CSV files in parallel
    
    Args:
        filepaths: List of CSV file paths
        folder: Optional output folder for parquet files
        n_workers: Number of parallel workers. If None, uses CPU count
    """
    if n_workers is None:
        n_workers = os.cpu_count()
    
    # Create output folder if specified
    if folder is not None:
        os.makedirs(folder, exist_ok=True)
    
    # Process files in parallel
    with Pool(n_workers) as pool:
        pool.starmap(process_df, [(fp, folder) for fp in filepaths])


if __name__ == "__main__":
    # Read filepaths from the text file
    with open("USD_60_2022_01_01-2025_03_31_filenames.txt", "r") as file:
        filepaths = [line.strip() for line in file if line.strip()]
    
    # process_df(filepaths[0], folder="processed_data")
    # Process all files in parallel
    process_all_files(filepaths, folder="USD_60") 