import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os

def compute_beta(asset_returns, market_returns, window=240, min_periods=100):
    """
    Compute market beta using rolling window calculations on aligned return series.
    Ensure inputs asset_returns and market_returns are pd.Series of log returns that have timestamp as indices (set_index('timestamp')) 
    """
    # Align the series to asset_returns' index
    market_returns = market_returns.reindex(asset_returns.index)
    
    # Compute rolling covariance and variance directly on the aligned series
    rolling_cov = asset_returns.rolling(window=window, min_periods=min_periods).cov(market_returns)
    rolling_var = market_returns.rolling(window=window, min_periods=min_periods).var(ddof=1)
    
    # Compute beta, handling division by zero
    beta = rolling_cov / rolling_var.where(rolling_var != 0, np.nan)
    
    return beta

def process_df(file_path, df_market, df_btc=None, df_eth=None, window=720, min_periods=240, output_folder=None):
    """
    Process a single parquet file to compute and add beta columns.
    
    Parameters:
        file_path (str): Path to the parquet file
        df_market (pd.DataFrame): Market data with timestamp index and log_return column
        df_btc (pd.DataFrame): BTC data with timestamp index and log_return column
        df_eth (pd.DataFrame): ETH data with timestamp index and log_return column
        window (int): Number of periods in rolling window
        min_periods (int): Minimum required non-NaN observations within window
        output_folder (str, optional): Folder to save the processed file. If None, saves in place.
        
    Returns:
        Saves the processed file in place or in output_folder
    """
    # Load the asset data
    df_asset = pd.read_parquet(file_path)
    df_asset.set_index('timestamp', inplace=True)
    
    # Compute betas
    market_beta = compute_beta(df_asset['log_return'], df_market['log_return'], window, min_periods)
    #btc_beta = compute_beta(df_asset['log_return'], df_btc['log_return'], window, min_periods)
    #eth_beta = compute_beta(df_asset['log_return'], df_eth['log_return'], window, min_periods)
    
    # Add beta columns to the dataframe
    df_asset[['beta_market', 'beta_btc', 'beta_eth']] = pd.DataFrame({
        'beta_market': market_beta,
        #'beta_btc': btc_beta,
        #'beta_eth': eth_beta
    })
    
    # Determine output path
    if output_folder is not None:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        # Get the original filename
        original_filename = os.path.basename(file_path)
        # Create new output path
        output_path = os.path.join(output_folder, original_filename)
    else:
        output_path = file_path
    
    # Save the processed file
    df_asset.to_parquet(output_path)

def process_dfs(file_list, window=60, min_periods=30, max_workers=None, output_folder=None):
    """
    Process multiple parquet files in parallel to compute and add beta columns.
    
    Parameters:
        file_list (list): List of paths to parquet files
        window (int): Number of periods in rolling window
        min_periods (int): Minimum required non-NaN observations within window
        max_workers (int): Maximum number of parallel workers (default: number of CPU cores)
        output_folder (str, optional): Folder to save the processed files. If None, saves in place.
    """
    # Load market data once
    print("Loading market data...")
    df_market = pd.read_csv('mcap_processed.csv')
    df_btc = pd.read_parquet('USD_60_indicators/XBTUSD_60.parquet')
    df_eth = pd.read_parquet('USD_60_indicators/ETHUSD_60.parquet')
    
    # Set timestamp as index for all dataframes
    df_market.set_index('timestamp', inplace=True)
    df_btc.set_index('timestamp', inplace=True)
    df_eth.set_index('timestamp', inplace=True)
    
    if max_workers is None:
        max_workers = os.cpu_count()
    
    print(f"Processing {len(file_list)} files using {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all files for processing
        futures = [
            executor.submit(process_df, file_path, df_market, df_btc, df_eth, window, min_periods, output_folder)
            for file_path in file_list
        ]
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()
    
    print("Processing complete!")

if __name__ == "__main__":
    # Example usage
    with open('filenames_parquet.txt', 'r') as f:
        filepaths = f.read().splitlines()

    filepaths = ['USD_60_indicators/' + f for f in filepaths]
    # Process files and save to a new folder
    process_dfs(filepaths, output_folder='USD_60_betas') 