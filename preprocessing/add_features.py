import pandas as pd
import numpy as np
from typing import Tuple, List

CONFIG = {
    "ret_features": {
        "close_to_high": True,
        "close_to_low": True,
        "ret_from_prev_high": True,
        "ret_from_prev_low": True,
        "log_price_range": True,
        "ret_from_today_high": True,
        "ret_from_today_low": True
    },
    "ma_zscore": {"window": 12},
    "momentum": {"pairs": [(28, 4), (120, 28)], "strev": [28], "ltrev": [120]},
    "ema_diff_norm": {"pairs": [(12, 28), (28, 120)], "normalize_by": "price"},
    "price_ema_diff": {"window": 12},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "volatility": {"window": 12},
    "vwap": {"window": 12},
    "amihud": {"window": 12},
    "stochastic_oscillator": {"window": 12},
    "rsi": {"window": 14},
    "pos_frac": {"windows": ["1d", "3d", "7d", "30d"]},
    "skew_kurt": {"windows": ["7d", "30d"]},
    "fng": {"ma_windows": ["30d", "90d"], "ret_periods": [28, 120]},
    "rolling_beta": {"windows": ["3d", "7d", "30d", "90d"], "min_periods": 4}
}


def add_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    
    def compute_ma_zscore(series: pd.Series, window: int) -> pd.Series:
        ma  = series.rolling(window=window, min_periods=1).mean()
        std = series.rolling(window=window, min_periods=1).std()
        return pd.Series(np.where(std != 0, (series - ma) / std, np.nan), index=series.index)

    def compute_momentum(series: pd.Series, a: int, b: int) -> pd.Series:
        return series.shift(b) / series.shift(a) - 1

    def ema_diff(price: pd.Series, fast: int, slow: int, normalize_by: str = "price") -> pd.Series:
        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        if normalize_by == "price":
            return (ema_fast - ema_slow) / price
        elif normalize_by == "ema_slow":
            return (ema_fast - ema_slow) / ema_slow
        else:
            raise ValueError("normalize_by must be 'price' or 'ema_slow'")

    def price_ema_diff(series: pd.Series, window: int) -> pd.Series:
        ema = series.ewm(span=window, adjust=False).mean()
        return (series - ema) / series

    def compute_macd_hist(series: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - signal_line

    def compute_volatility(returns: pd.Series, volume: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
        squared_weighted = (returns ** 2) * volume
        sum_rw = squared_weighted.rolling(window=window, min_periods=1).sum()
        sum_vol = volume.rolling(window=window, min_periods=1).sum()
        weighted_vol = pd.Series(np.sqrt(sum_rw / sum_vol), index=returns.index)

        sum_sq = (returns ** 2).rolling(window=window, min_periods=1).sum()
        realized_vol = pd.Series(np.sqrt(sum_sq), index=returns.index)
        return realized_vol, weighted_vol

    def compute_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
        typical_price = (high + low + close) / 3
        vol_sum = volume.rolling(window=window, min_periods=1).sum()
        tp_vol  = (typical_price * volume).rolling(window=window, min_periods=1).sum()
        return pd.Series(np.where(vol_sum != 0, tp_vol / vol_sum, 0), index=close.index)

    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        lowest_low  = low.rolling(window=window, min_periods=1).min()
        highest_high = high.rolling(window=window, min_periods=1).max()
        denom = highest_high - lowest_low
        percent_k = pd.Series(np.where(denom == 0, 50, ((close - lowest_low) / denom) * 100), index=close.index)
        return percent_k

    def compute_rsi(series: pd.Series, window: int) -> pd.Series:
        filled = series.ffill().bfill()
        delta = filled.diff()
        gain  = delta.where(delta > 0,  0)
        loss  = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = pd.Series(index=series.index, dtype=float)
        mask = (avg_gain == 0) & (avg_loss == 0)
        rs[~mask] = avg_gain[~mask] / avg_loss[~mask]
        rs[mask] = 100
        rsi_val = pd.Series(100 - (100 / (1 + rs)), index=series.index)
        rsi_val[series.isna()] = pd.NA
        return rsi_val

    def get_rolling_beta_and_corr(asset_ret: pd.Series, mcap_ret: pd.Series,window: str, min_periods: int) -> Tuple[pd.Series, pd.Series]:
        rolling_cov = asset_ret.rolling(window, min_periods=min_periods).cov(mcap_ret)
        rolling_var_asset = asset_ret.rolling(window, min_periods=min_periods).var(ddof=1)
        rolling_var_mcap  = mcap_ret.rolling(window, min_periods=min_periods).var(ddof=1)

        beta = rolling_cov / rolling_var_asset.where(rolling_var_asset != 0, np.nan)
        corr = rolling_cov / np.sqrt(
            rolling_var_asset * rolling_var_mcap
        ).where((rolling_var_asset != 0) & (rolling_var_mcap != 0), np.nan)
        return beta, corr

    # 1) Day of week & normalized date
    df['day_of_week'] = df.index.dayofweek
    df['date'] = df.index.normalize()

    # 2) Return‐based features
    rf = config["ret_features"]
    if rf.get("close_to_high"):
        df['close_to_high'] = df['close'] / df['high']
    if rf.get("close_to_low"):
        df['close_to_low'] = df['close'] / df['low']
    if rf.get("ret_from_prev_high"):
        df['ret_from_prev_high'] = df['close'] / df['high'].shift(1) - 1
    if rf.get("ret_from_prev_low"):
        df['ret_from_prev_low'] = df['close'] / df['low'].shift(1) - 1
    if rf.get("log_price_range"):
        df['log_price_range'] = np.log(df['high'] / df['low'])
    if rf.get("ret_from_today_high"):
        df['ret_from_today_high'] = df['close'] / df.groupby('date')['high'].cummax() - 1
    if rf.get("ret_from_today_low"):
        df['ret_from_today_low'] = df['close'] / df.groupby('date')['low'].cummin() - 1

    # 3) Moving‐average z‐score
    ma_cfg = config["ma_zscore"]
    df['ma_zscore_24'] = compute_ma_zscore(df['close'], window=ma_cfg["window"])

    # 4) Momentum, short‐term reversal, long‐term reversal
    mom_cfg = config["momentum"]
    for a, b in mom_cfg["pairs"]:
        df[f'mom_{a}_{b}'] = compute_momentum(df['close'], a=a, b=b)
    for a in mom_cfg["strev"]:
        df[f'strev_{a}'] = -compute_momentum(df['close'], a=a, b=1)
    for a in mom_cfg["ltrev"]:
        df[f'ltrev_{a}'] = -compute_momentum(df['close'], a=a, b=1)

    # 5) EMA differences (normalized)
    ema_cfg = config["ema_diff_norm"]
    for fast, slow in ema_cfg["pairs"]:
        df[f'ema_diff_norm_{fast}_{slow}'] = ema_diff(df['close'], fast=fast, slow=slow, normalize_by=ema_cfg["normalize_by"])

    # 6) Price minus EMA (normalized)
    ped_cfg = config["price_ema_diff"]
    df[f'price_ema_diff_{ped_cfg["window"]}'] = price_ema_diff(df['close'], window=ped_cfg["window"])

    # 7) MACD histogram
    macd_cfg = config["macd"]
    df['macd_hist'] = compute_macd_hist(df['close'], fast=macd_cfg["fast"], slow=macd_cfg["slow"], signal=macd_cfg["signal"])

    # 8) Volatility: realized and volume‐weighted
    vol_cfg = config["volatility"]
    df['realized_vol_24'], df['weighted_vol_24'] = compute_volatility(df['return'], df['volume'], window=vol_cfg["window"])

    # 9) VWAP over rolling window
    vwap_cfg = config["vwap"]
    df['vwap_24'] = compute_vwap(df['high'], df['low'], df['close'], df['volume'], window=vwap_cfg["window"])

    # 10) Amihud illiquidity measure
    ami_cfg = config["amihud"]
    df['amihud_illiquidity_12'] = (
        df['log_return'].div(df['volume'])
                 .where(df['volume'] != 0)
                 .rolling(window=ami_cfg["window"], min_periods=1)
                 .mean()
    )

    # 11) Stochastic oscillator
    sto_cfg = config["stochastic_oscillator"]
    df['stochastic_oscillator_12'] = stochastic_oscillator(df['high'], df['low'], df['close'], window=sto_cfg["window"])

    # 12) RSI
    rsi_cfg = config["rsi"]
    df['rsi_14'] = compute_rsi(df['close'], window=rsi_cfg["window"])

    # 13) Fraction of positive returns over various lookbacks
    pf_cfg = config["pos_frac"]
    for w in pf_cfg["windows"]:
        df[f'pos_frac_{w}'] = df['return'].gt(0).rolling(window=w).mean()

    # 14) Skewness and Kurtosis over lookback windows
    sk_cfg = config["skew_kurt"]
    df['skew_7d']  = df['return'].rolling(window=sk_cfg["windows"][0]).skew()
    df['kurt_7d']  = df['return'].rolling(window=sk_cfg["windows"][0]).kurt()
    df['skew_30d'] = df['return'].rolling(window=sk_cfg["windows"][1]).skew()
    df['kurt_30d'] = df['return'].rolling(window=sk_cfg["windows"][1]).kurt()

    # 15) Fear & Greed Index moving averages and returns
    fng_cfg = config["fng"]
    for w in fng_cfg["ma_windows"]:
        df[f'fng_ma_{w}'] = df['fng_index'].rolling(window=w).mean()
    for period in fng_cfg["ret_periods"]:
        df[f'fng_ret_{period}'] = df['fng_index'].pct_change(period)

    # 16) Rolling beta and correlation with market‐cap returns
    rb_cfg = config["rolling_beta"]
    for w in rb_cfg["windows"]:
        beta_col = f'rolling_beta_{w}'
        corr_col = f'rolling_correlation_{w}'
        df[beta_col], df[corr_col] = get_rolling_beta_and_corr(
            df['return'], df['return_mcap'],
            window=w,
            min_periods=rb_cfg["min_periods"]
        )

    df.dropna(inplace=True)
    return df

def prepare_df(df_fname: str, rfr_fname: str = "../DGS10.csv", fng_ind_fname: str = "../data/fng_index.csv", mcap_fname: str = "../mcap_processed.csv", impute: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(df_fname)

    # Imputing missing values by forward filling
    if impute:
        df_cts = pd.DataFrame({
            "timestamp": pd.date_range(start=df["timestamp"].min(), end=df["timestamp"].max(), freq="360min")
        })
        df_cts = pd.merge(df_cts, df, on = "timestamp", how = "left")
        df_cts.close.fillna(method='ffill', inplace=True)
        df_cts.volume.fillna(value = 0, inplace = True)
        df_cts.trades.fillna(value = 0, inplace = True)
        df_cts.loc[df_cts.volume == 0, "high"] = df_cts.loc[df_cts.volume == 0, "close"]
        df_cts.loc[df_cts.volume == 0, "low"] = df_cts.loc[df_cts.volume == 0, "close"]
        df_cts.loc[df_cts.volume == 0, "open"] = df_cts.loc[df_cts.volume == 0, "close"]
        df = df_cts.copy()

    df["date"] = df["timestamp"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    # Risk free rate
    rfr = pd.read_csv(rfr_fname)
    rfr["observation_date"] = pd.to_datetime(rfr["observation_date"])
    # Merge on df.date == rfr.observation_date
    df = pd.merge(
        df,
        rfr,
        left_on="date",
        right_on="observation_date",
        how="left"
    )

    # Fear & Greed Index
    fng_ind = pd.read_csv(fng_ind_fname, header=None)
    fng_ind.columns = ["date", "fng_index", "name"]
    fng_ind.drop(columns=["name"], inplace=True)
    fng_ind["date"] = pd.to_datetime(fng_ind["date"], format="%d-%m-%Y")
    df = pd.merge(
        df,
        fng_ind,
        on="date",
        how="left"
    )

    # Clean up
    df.drop(columns=["date", "observation_date"], inplace=True)
    df.rename(columns={"DGS10": "risk_free_rate"}, inplace=True)
    df["risk_free_rate"] /= 100
    df.set_index("timestamp", inplace=True)
    df["risk_free_rate"] = df["risk_free_rate"].ffill()
    df["fng_index"] = df["fng_index"].ffill()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["return"] = df["close"].pct_change()
    df.dropna(inplace=True)

    # Broader Market cap data
    mcap = pd.read_csv(mcap_fname)
    mcap["timestamp"] = pd.to_datetime(mcap["timestamp"], unit="s")
    mcap = mcap.resample("6H", on="timestamp").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "top_tier_volume": "sum",
    })
    mcap["return_mcap"] = mcap["close"].pct_change()
    mcap["log_return_mcap"] = np.log(mcap["close"] / mcap["close"].shift(1))
    mcap.dropna(inplace=True)
    
    df = pd.merge(
        df,
        mcap,
        left_index=True,
        right_index=True,
        how="right",
        suffixes=("", "_mcap")
    ).dropna()

    return df
