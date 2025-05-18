import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def fill_parquet(input_path: str, folder: str = None, timestep: int = 3600) -> pd.DataFrame:
    """
    Fill missing rows with NaN, using timestamp as index with timestep as gap.
    Use unix timestamps as input.

    Args:
        input_path: Path to input parquet file
        folder: Optional output folder for parquet file
        timestep: Time step in seconds between consecutive timestamps (default: 3600)
    """
    # Read input file
    df = pd.read_parquet(input_path)
    
    # Set timestamp as index
    df = df.set_index('timestamp')
    
    # Create full timestamp range
    full_timestamps = pd.RangeIndex(start=df.index.min(), stop=df.index.max() + timestep, step=timestep)
    df_filled = df.reindex(full_timestamps)
    df_filled.index.name = 'timestamp'
    df_filled = df_filled.reset_index()
    
    if folder is not None: # save to folder
        os.makedirs(folder, exist_ok=True)
        output_path = f"{folder}/{input_path.split('/')[-1].replace('.parquet', '_filled.parquet')}"
        df_filled.to_parquet(output_path, index=False)
    
    return df_filled

def infer_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    If 1 or 2 rows have NaN open and close values, infer them based on surrounding data.
    Then compute return and log_return if these columns exist.
    """
    def handle_single_missing(row_idx):
        open_prev = df.loc[row_idx - 1, 'close']
        close_next = df.loc[row_idx + 1, 'open']
        
        df.loc[row_idx, 'open'] = open_prev
        df.loc[row_idx, 'close'] = close_next
        if 'return' in df.columns and 'log_return' in df.columns:
            df.loc[row_idx, 'return'] = close_next / open_prev - 1
            df.loc[row_idx, 'log_return'] = np.log(close_next / open_prev)

    def handle_double_missing(row_idx):
        open_m2 = df.loc[row_idx - 2, 'close']
        open_p1 = df.loc[row_idx + 1, 'open']
        close_m2 = df.loc[row_idx - 2, 'close']
        
        df.loc[row_idx - 1, 'open'] = close_m2
        df.loc[row_idx, 'close'] = open_p1

        avg = (close_m2 + open_p1) / 2
        df.loc[row_idx - 1, 'close'] = avg
        df.loc[row_idx, 'open'] = avg

        if 'return' in df.columns and 'log_return' in df.columns:
            df.loc[row_idx - 1, 'return'] = avg / close_m2 - 1 
            df.loc[row_idx - 1, 'log_return'] = np.log(avg / close_m2)
            df.loc[row_idx, 'return'] = open_p1 / avg - 1
            df.loc[row_idx, 'log_return'] = np.log(open_p1 / avg)

    last_existing_idx = None
    
    for i in range(1, len(df) - 1):
        if not pd.isna(df.loc[i, 'close']):
            last_existing_idx = i
            continue
        if last_existing_idx is None or last_existing_idx >= i:
            continue    
        if last_existing_idx == i-1 and i+1 < len(df) and not pd.isna(df.loc[i + 1, 'close']):
            handle_single_missing(i)  
        elif last_existing_idx == i-1 and i+2 < len(df) and not pd.isna(df.loc[i + 2, 'close']):
            handle_double_missing(i+1)
            last_existing_idx = i + 2
       
    return df

# Basic price-based indicators (no time window needed)
def close_to_high_ratio(close: pd.Series, high: pd.Series) -> pd.Series:
    return close / high

def close_to_low_ratio(close: pd.Series, low: pd.Series) -> pd.Series:
    return close / low

def log_price_range(high: pd.Series, low: pd.Series) -> pd.Series:
    return np.log(high / low)

# Moving average and z-score
def compute_ma_zscore(close: pd.Series, window: int = 24) -> Tuple[pd.Series, pd.Series]:
    ma = close.rolling(window=window, min_periods=1).mean()
    std = close.rolling(window=window, min_periods=1).std()
    zscore = np.where(std != 0, (close - ma) / std, np.nan)
    return zscore

# EMA-based indicators
def compute_ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, min_periods=1).mean()


def price_ema_diff(price: pd.Series, window: int = 24) -> pd.Series:
    """
    Computes (Price - EMA) / Price
    """
    ema = price.ewm(span=window, adjust=False).mean()
    return (price - ema) / price

def ema_diff(price: pd.Series, fast: int, slow: int, normalize_by: str = 'price') -> pd.Series:
    """
    Computes (EMA_fast - EMA_slow) / normalizer
    normalize_by: 'price' or 'ema_slow'
    """
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    
    if normalize_by == 'price':
        return (ema_fast - ema_slow) / price
    elif normalize_by == 'ema_slow':
        return (ema_fast - ema_slow) / ema_slow
    else:
        raise ValueError("normalize_by must be 'price' or 'ema_slow'")

def compute_macd(price: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """
    Computes MACD = EMA_fast - EMA_slow
    """
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def macd_hist(price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """
    Computes MACD Histogram = MACD - Signal line
    """
    macd_line = compute_macd(price, fast, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

# Volatility and volume-based indicators
def compute_rolling_std(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window=window, min_periods=1).std()

def compute_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 24) -> pd.Series:
    typical_price = (high + low + close) / 3
    volume_sum = volume.rolling(window=window, min_periods=1).sum()
    return np.where(volume_sum != 0, 
                   (typical_price * volume).rolling(window=window, min_periods=1).sum() / volume_sum,
                   0)

def volume_zscore(volume: pd.Series, window: int) -> pd.Series:
    mean = volume.rolling(window=window, min_periods=1).mean()
    std = volume.rolling(window=window, min_periods=1).std()
    return np.where(std != 0, (volume - mean) / std, 0)

def compute_volatility(returns: pd.Series, volume: pd.Series, window: int = 24) -> Tuple[pd.Series, pd.Series]:
    squared_returns_weighted = (returns ** 2) * volume
    sum_returns_weighted = squared_returns_weighted.rolling(window=window, min_periods=1).sum()
    sum_volume = volume.rolling(window=window, min_periods=1).sum()
    weighted_vol = np.sqrt(sum_returns_weighted / sum_volume)
    
    sum_returns = (returns ** 2).rolling(window=window, min_periods=1).sum()
    realized_vol = np.sqrt(sum_returns)

    return realized_vol, weighted_vol

def amihud_illiquidity(log_return: pd.Series, volume: pd.Series, window: int = 12) -> pd.Series:
    illiquidity = np.where(volume != 0, np.abs(log_return) / volume, np.nan)
    return pd.Series(illiquidity).rolling(window=window, min_periods=1).mean()

# Momentum and oscillator indicators
def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 12) -> pd.Series:
    lowest_low = low.rolling(window=window, min_periods=1).min()
    highest_high = high.rolling(window=window, min_periods=1).max()
    denom = highest_high - lowest_low
    
    percent_k = np.where(denom == 0, 50, ((close - lowest_low) / denom) * 100) 
    return percent_k

def compute_momentum(close: pd.Series, a: int, b: int) -> pd.Series:
    """ 
    Compute momentum r_a,b = P_{t-b}/P_{t-a} - 1, which is the return of period [t-a, t-b].
    Args:
        close: Series of closing prices
        t-a: start time, t-b: end time (e.g. r_2,1  r_12,2  r_12,7  r_36,6)
    """
    return close.shift(b) / close.shift(a) - 1

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    close_filled = close.ffill().bfill()
    delta = close_filled.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = pd.Series(index=close.index, dtype=float)
    mask = (avg_gain == 0) & (avg_loss == 0)
    rs[~mask] = avg_gain[~mask] / avg_loss[~mask]
    rs[mask] = 100
    rsi = 100 - (100 / (1 + rs))
    rsi[close.isna()] = pd.NA
    return rsi

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators efficiently and add them to the DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns: timestamp, open, high, low, close, volume, trades, log_return, return
        config (Dict): Configuration dictionary for indicator parameters. If None, uses default values.
        
    Returns:
        pd.DataFrame: Original DataFrame with added technical indicators
    """
    # Preload frequently used columns into local variables for better performance
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    returns = df['return']
    log_returns = df['log_return']
    
    
    config = {
        'momentum': [(24, 4), (168, 24)],  # (a, b) pairs for momentum
        'reversal': [24, 168],    # lookback window for ST_Rev and LT_Rev
        'ema_diff': [(12, 48), (24, 120)],  # (fast, slow) pairs for ema_diff
        #'price_ema': [24],                  # windows for price_ema_diff
        #'stoch_osc': [12],              # windows for stochastic oscillator
        #'amihud': [12],                     # windows for amihud illiquidity
        #'volatility': [24],                 # windows for realized and weighted volatility
        #'ma_zscore': [24],                  # windows for MA and z-score
        #'rsi': [14],                        # windows for RSI
        #'macd': [(12, 26)]              # (fast, slow, signal) tuples for MACD
    }
    
    window1, window2 = 12, 24 # default window size
    
    # Compute volatility metrics first
    realized_vol, weighted_vol = compute_volatility(returns, volume, window2)
    
    # Create dictionary to store all indicators
    indicators_dict = {
        # Basic price-based indicators (no time window needed)
        'close_to_high': close_to_high_ratio(close, high),
        'close_to_low': close_to_low_ratio(close, low),
        'log_price_range': log_price_range(high, low),
        
        # MA z-score
        f'ma_zscore_{window1}': compute_ma_zscore(close, window1),
        
        # Momentum and reversal
        **{f'mom_{a}_{b}': compute_momentum(close, a, b) 
           for a, b in config['momentum']},
        f'strev_{config["reversal"][0]}': -compute_momentum(close, config['reversal'][0], 1), 
        f'ltrev_{config["reversal"][1]}': -compute_momentum(close, config['reversal'][1], 1), 
        
        # EMA-based indicators
        **{f'ema_diff_norm_{fast}_{slow}': ema_diff(close, fast, slow, 'price') for fast, slow in config['ema_diff']},
        f'price_ema_diff_{window1}': price_ema_diff(close, window1),
        f'macd_hist': macd_hist(close),
        
        # Volatility and volume-based indicators
        f'realized_vol_{window2}': realized_vol,
        f'weighted_vol_{window2}': weighted_vol,
        f'volume_zscore_{window2}': volume_zscore(volume, window2),
        f'vwap_{window2}': compute_vwap(high, low, close, volume, window2),
        
        # Stochastic oscillator
        f'sto_osc_{window1}': stochastic_oscillator(high, low, close, window1),
        
        # RSI
        f'rsi': compute_rsi(close),
        
        # Amihud illiquidity
        f'amihud_{window1}': amihud_illiquidity(log_returns, volume, window1)
    }
    
    # Add indicators directly to the DataFrame using assign
    return df.assign(**indicators_dict)

def process_single_file(filepath: str, output_folder: str = None) -> pd.DataFrame:
    """
    Process a single parquet file.
    """
    # Read and fill missing data
    df = fill_parquet(filepath).drop(columns=['amihud'])
    df = infer_missing_data(df)
    # Compute all indicators
    df = compute_all_indicators(df)
    # Save to parquet if output folder is specified
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        output_path = f"{output_folder}/{filepath.split('/')[-1]}"
        df.to_parquet(output_path, index=False)
    return df

def process_files_parallel(filepaths: List[str], output_folder: str = None, max_workers: int = None) -> None:
    """
    Process multiple parquet files in parallel and save results to output folder.
    Does not return any results.
    """
    process_func = partial(process_single_file, output_folder=output_folder)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_func, filepaths)


if __name__ == "__main__":
    
    with open('USD_60_filenames_parquet.txt', 'r') as f:
        filepaths = f.read().splitlines()
    
    # df = process_single_file(filepaths[0], output_folder="USD_60_indicators")
    # print(df.head(20))
    process_files_parallel(filepaths, output_folder="USD_60_indicators") 