import numpy as np
import pandas as pd
from typing import Dict, List


def _rolling_z(series: pd.Series, window: int, min_periods: int = 20) -> pd.Series:
    """Return rolling z-score using only past samples.

    Args:
        series: Input numeric series.
        window: Rolling window length.
        min_periods: Minimum periods to start statistics.

    Returns:
        Rolling z-score with NaNs replaced by 0.0.
    """
    m = series.rolling(window, min_periods=min_periods).mean()
    s = series.rolling(window, min_periods=min_periods).std()
    z = (series - m) / s.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rolling_mean(series: pd.Series, window: int, min_periods: int = 20) -> pd.Series:
    """Return rolling mean."""
    return series.rolling(window, min_periods=min_periods).mean()


def _rolling_std(series: pd.Series, window: int, min_periods: int = 20) -> pd.Series:
    """Return rolling std."""
    return series.rolling(window, min_periods=min_periods).std()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def compute_macd_features(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9, lookback: int = 288) -> pd.DataFrame:
    """Compute MACD features and their rolling z-scores.

    Returns columns: macd, macd_signal, macd_hist, macd_z, macd_hist_z.
    """
    close = close.astype(float)
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = _ema(macd, signal)
    macd_hist = macd - macd_signal

    feats = pd.DataFrame({
        'macd': macd.astype(np.float32),
        'macd_signal': macd_signal.astype(np.float32),
        'macd_hist': macd_hist.astype(np.float32),
        'macd_z': _rolling_z(macd, lookback).astype(np.float32),
        'macd_hist_z': _rolling_z(macd_hist, lookback).astype(np.float32),
    }, index=close.index)
    return feats


def compute_liquidity_features(df: pd.DataFrame, lookback: int = 288) -> pd.DataFrame:
    """Compute liquidity proxies using available columns.

    Includes:
    - dollar_volume: preferred quote_volume else close*volume (log1p-z)
    - amihud: |log_return| / (dollar_volume + eps) (z)
    - volume_imbalance: (buy - sell)/(buy + sell) (z)
    - trades_z: z-score of trade count if available
    - hl_spread_proxy: (high - low)/close (z)
    """
    idx = df.index
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df.get('volume', pd.Series(0.0, index=idx)).astype(float)
    buy_vol = df.get('buy_volume', pd.Series(0.0, index=idx)).astype(float)
    sell_vol = df.get('sell_volume', pd.Series(0.0, index=idx)).astype(float)
    quote_volume = df.get('quote_volume', (close * volume)).astype(float)
    trades = df.get('trades', pd.Series(0.0, index=idx)).astype(float)

    dollar_volume = quote_volume
    log_close = np.log(np.clip(close, 1e-12, None))
    log_ret_1 = log_close.diff().fillna(0.0).astype(float)
    amihud = (np.abs(log_ret_1) / (dollar_volume.replace(0.0, np.nan))).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    vol_imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-12)
    hl_spread = ((high - low) / np.clip(close, 1e-12, None)).fillna(0.0)

    feats = pd.DataFrame({
        'dollar_volume_log_z': _rolling_z(np.log1p(np.clip(dollar_volume, 0.0, None)), lookback).astype(np.float32),
        'amihud_z': _rolling_z(amihud, lookback).astype(np.float32),
        'vol_imbalance_z': _rolling_z(vol_imbalance, lookback).astype(np.float32),
        'trades_z': _rolling_z(trades, lookback).astype(np.float32),
        'hl_spread_z': _rolling_z(hl_spread, lookback).astype(np.float32),
    }, index=idx)
    return feats


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.Series(
        np.maximum.reduce([
            (high - low).values,
            np.abs(high - prev_close).fillna(0.0).values,
            np.abs(low - prev_close).fillna(0.0).values,
        ]), index=close.index
    )
    return tr.rolling(period, min_periods=5).mean().fillna(0.0)


def compute_smc_features(df: pd.DataFrame, lookback: int = 288) -> pd.DataFrame:
    """Compute simplified Smart Money Concepts (SMC) proxies.

    Features:
    - structure_trend: -1 bear, 0 neutral, 1 bull via swing high/low progression
    - sweep_up/down: recent liquidity sweep flags
    - dist_to_swing_high/low: distance to recent swings (normalized)
    - displacement_z: body/ATR magnitude z-score
    """
    idx = df.index
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    open_ = df.get('open', df['close']).astype(float)
    close = df['close'].astype(float)

    atr = _atr(high, low, close)
    body = (close - open_).astype(float)
    body_ratio = (body.abs() / (atr.replace(0.0, np.nan))).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    left = 2
    right = 2
    swing_high = (high.shift(1).rolling(window=left, min_periods=1).max() < high) & (high > high.shift(-1).rolling(window=right, min_periods=1).max())
    swing_low = (low.shift(1).rolling(window=left, min_periods=1).min() > low) & (low < low.shift(-1).rolling(window=right, min_periods=1).min())

    recent_high = high.where(swing_high).ffill().bfill()
    recent_low = low.where(swing_low).ffill().bfill()

    dist_to_swing_high = (close - recent_high) / np.clip(close, 1e-12, None)
    dist_to_swing_low = (close - recent_low) / np.clip(close, 1e-12, None)

    hh = (recent_high.diff() > 0).astype(int)
    hl = (recent_low.diff() > 0).astype(int)
    lh = (recent_high.diff() < 0).astype(int)
    ll = (recent_low.diff() < 0).astype(int)
    bull = ((hh + hl) >= 1).astype(int)
    bear = ((lh + ll) >= 1).astype(int)
    structure_trend = (bull - bear).clip(-1, 1)

    N = max(10, lookback // 12)
    recent_max = high.rolling(N, min_periods=5).max()
    recent_min = low.rolling(N, min_periods=5).min()
    sweep_up = ((high > recent_max.shift(1)) & (close < recent_max.shift(1))).astype(int)
    sweep_down = ((low < recent_min.shift(1)) & (close > recent_min.shift(1))).astype(int)

    feats = pd.DataFrame({
        'smc_structure_trend': structure_trend.astype(np.float32),
        'smc_sweep_up': sweep_up.astype(np.float32),
        'smc_sweep_down': sweep_down.astype(np.float32),
        'smc_dist_to_swing_high': _rolling_z(dist_to_swing_high, lookback).astype(np.float32),
        'smc_dist_to_swing_low': _rolling_z(dist_to_swing_low, lookback).astype(np.float32),
        'smc_displacement_z': _rolling_z(body_ratio, lookback).astype(np.float32),
    }, index=idx)
    return feats


def compute_rolling_profile_features(df: pd.DataFrame, window: int = 288) -> pd.DataFrame:
    """Compute Rolling Volume Profile (Market Profile Proxy) Features."""
    close = df['close']
    volume = df['volume']
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    
    # 1. Rolling VWAP
    pv = typical_price * volume
    roll_pv = pv.rolling(window=window, min_periods=1).sum()
    roll_vol = volume.rolling(window=window, min_periods=1).sum()
    vwap = roll_pv / roll_vol.replace(0.0, np.nan)
    
    # 2. Rolling VWSD (Volume Weighted Standard Deviation)
    p2v = (typical_price ** 2) * volume
    roll_p2v = p2v.rolling(window=window, min_periods=1).sum()
    mean_p2 = roll_p2v / roll_vol.replace(0.0, np.nan)
    
    variance = mean_p2 - vwap ** 2
    variance = variance.clip(lower=0.0)
    vwsd = np.sqrt(variance)
    
    poc = vwap
    vah = vwap + vwsd
    val = vwap - vwsd
    
    z_score = (close - vwap) / vwsd.replace(0.0, 1.0)
    density = np.exp(-0.5 * z_score**2) 
    
    poc = poc.fillna(close)
    vah = vah.fillna(close * 1.01)
    val = val.fillna(close * 0.99)
    density = density.fillna(0.0)
    
    feats = pd.DataFrame({
        'profile_poc': poc.astype(np.float32),
        'profile_vah': vah.astype(np.float32),
        'profile_val': val.astype(np.float32),
        'profile_density': density.astype(np.float32)
    }, index=df.index)
    
    return feats

def compute_long_term_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute multi-timescale features for state vector (e.g. 5-day trend, 30-day vol regime)."""
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    
    # 5 days = 5 * 288 = 1440 steps (approx)
    # 30 days = 30 * 288 = 8640 steps
    
    # 1. Long-term Returns (5 days)
    # Log return over 5 days
    log_close = np.log(np.clip(close, 1e-12, None))
    ret_5d = log_close.diff(1440).fillna(0.0)
    
    # Normalize by long-term vol? Or just clip?
    # Let's use a rolling z-score over 30 days
    ret_5d_z = _rolling_z(ret_5d, window=8640)
    
    # 2. Volatility Regime (30 days)
    # Calculate daily range or returns std
    log_ret = log_close.diff().fillna(0.0)
    vol_30d = log_ret.rolling(window=8640, min_periods=288).std()
    
    # Rank of current vol against last year (approx 100k steps)
    # To save compute, we can use a shorter rank window or z-score
    vol_regime_long = _rolling_z(vol_30d, window=8640 * 3) # 3 months context
    
    # 3. Realized Volatility (5 days)
    vol_5d = log_ret.rolling(window=1440, min_periods=288).std()
    vol_5d_z = _rolling_z(vol_5d, window=8640)
    
    feats = pd.DataFrame({
        'lt_ret_5d_z': ret_5d_z.astype(np.float32),
        'lt_vol_regime': vol_regime_long.astype(np.float32),
        'lt_vol_5d_z': vol_5d_z.astype(np.float32)
    }, index=df.index)
    
    return feats

def compute_multi_timeframe_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market bias for multiple timeframes (15m, 1h, 1d).
    
    Returns:
        DataFrame with columns 'bias_15m', 'bias_1h', 'bias_1d' (normalized z-scores or ratios).
        Values are reindexed to match original df index (5m).
    """
    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        # Fallback if integer index: treat as 5min steps
        # Create dummy datetime index starting from 2000-01-01
        # This is just for resampling, won't replace original index
        temp_index = pd.date_range(start='2000-01-01', periods=len(df), freq='5min')
        series_close = pd.Series(df['close'].values, index=temp_index)
    else:
        series_close = df['close']

    biases = {}
    
    # Define timeframes and their window sizes for bias calc (e.g. EMA 20)
    # TF: frequency str
    timeframes = {
        '15m': '15min',
        '1h': '1h',
        '1d': '1d'
    }
    
    for tf_name, freq in timeframes.items():
        # Resample to higher TF
        # label='right', closed='right' ensures we use data up to time T
        resampled = series_close.resample(freq, label='right', closed='right').last().dropna()
        
        if len(resampled) < 20:
            # Not enough data
            bias = pd.Series(0.0, index=resampled.index)
        else:
            # Calculate Bias Indicator: (Close - EMA(20)) / ATR(20) approx
            # Or simpler: (Close - EMA(20)) / EMA(20) normalized
            ema = resampled.ewm(span=20, adjust=False).mean()
            diff = (resampled - ema) / ema
            
            # Rolling Z-score of this diff to normalize to roughly [-2, 2]
            # Use a window of say 50 bars of that TF
            bias = _rolling_z(diff, window=50, min_periods=10)
            
            # Clip to reasonable range [-3, 3]
            bias = bias.clip(-3, 3)
            
        # Reindex back to original 5m index with forward fill
        # We use reindex with method='ffill' to propagate the last known bias
        aligned_bias = bias.reindex(series_close.index, method='ffill').fillna(0.0)
        biases[f'bias_{tf_name}'] = aligned_bias.values
        
    return pd.DataFrame(biases, index=df.index).astype(np.float32)


def compute_market_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute shape features for CNN input (price_seq), fully normalized.
    
    Included Features (base F=7, plus lightweight multi-scale returns):
    1. ret_t: log(close_t / close_{t-1}) / rolling_std
    2. range_t: ((high_t - low_t) / close_{t-1}) / rolling_mean_range
    3. body_t: ((close_t - open_t) / close_{t-1}) / rolling_mean_range
    4. vol_t: log(volume_t / vol_mean_recent)
    5. vol_regime_t: Percentile rank of current range.
    6. spread_t: (ask-bid)/mid / mean_spread
    7. density_t: rolling profile density (short window)

    Extra (low-cost multi-scale, avoids duplicating all channels):
    8. ret_15m_t: rolling-sum log return over 3 bars (approx 15m) / rolling std
    9. ret_1h_t: rolling-sum log return over 12 bars (approx 1h) / rolling std
    """
    close = df['close'].astype(float)
    open_ = df.get('open', close).astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df.get('volume', pd.Series(0.0, index=df.index)).astype(float)
    
    prev_close = close.shift(1).bfill().fillna(0.0)
    
    # 0. Base Components
    log_close = np.log(np.clip(close, 1e-12, None))
    raw_ret = log_close.diff().fillna(0.0)
    
    raw_range_pct = (high - low) / np.clip(prev_close, 1e-12, None)
    raw_body_pct = (close - open_) / np.clip(prev_close, 1e-12, None)
    
    # Rolling Stats (Lookback 288 for robust normalization)
    norm_window = 288
    ret_std = _rolling_std(raw_ret, norm_window).replace(0.0, 1.0)
    range_mean = _rolling_mean(raw_range_pct, norm_window).replace(0.0, 1e-4) # avoid div 0
    
    # 1. Normalized Log Return (Z-score)
    ret_norm = raw_ret / ret_std
    ret_norm = ret_norm.clip(-5, 5)

    # 1b. Multi-scale returns (lightweight channels)
    # Use rolling sum of log-returns to approximate higher timeframe returns.
    # Normalize by rolling std (same norm_window) to keep scale stable.
    raw_ret_15m = raw_ret.rolling(window=3, min_periods=1).sum()
    raw_ret_1h = raw_ret.rolling(window=12, min_periods=1).sum()
    ret_15m_std = _rolling_std(raw_ret_15m, norm_window).replace(0.0, 1.0)
    ret_1h_std = _rolling_std(raw_ret_1h, norm_window).replace(0.0, 1.0)
    ret_15m_norm = (raw_ret_15m / ret_15m_std).clip(-5, 5)
    ret_1h_norm = (raw_ret_1h / ret_1h_std).clip(-5, 5)
    
    # 2. Relative Range
    range_norm = raw_range_pct / range_mean
    range_norm = range_norm.clip(0, 10)
    
    # 3. Relative Body
    body_norm = raw_body_pct / range_mean
    body_norm = body_norm.clip(-10, 10)
    
    # 4. Volume (already relative log-ratio)
    vol_mean = volume.rolling(20, min_periods=1).mean().replace(0.0, 1.0)
    vol_val = np.log(np.clip(volume / vol_mean, 1e-8, None))
    vol_val = vol_val.clip(-5, 5)
    
    # 5. Volatility Regime (0-1)
    vol_regime = raw_range_pct.rolling(window=288, min_periods=1).rank(pct=True).fillna(0.5)

    # 6. Spread (Relative)
    if 'ask1' in df.columns and 'bid1' in df.columns:
         mid = (df['ask1'] + df['bid1']) / 2
         spread_raw = (df['ask1'] - df['bid1']) / np.clip(mid, 1e-12, None)
         spread_mean = spread_raw.rolling(288, min_periods=1).mean().replace(0.0, 1e-6)
         spread_val = spread_raw / spread_mean
         spread_val = spread_val.clip(0, 10)
    else:
         spread_val = pd.Series(0.0, index=df.index)
    
    # 7. Profile Density (from pre-calculated or simple calc here if light)
    prof_feats = compute_rolling_profile_features(df, window=288) # 1 day window for short-term density
    
    features_dict = {
        'ret': ret_norm,
        'ret_15m': ret_15m_norm,
        'ret_1h': ret_1h_norm,
        'range': range_norm,
        'body': body_norm,
        'vol': vol_val,
        'vol_regime': vol_regime,
        'spread': spread_val,
        'density': prof_feats['profile_density'] # Add density to CNN input
    }
    
    features = pd.DataFrame(features_dict, index=df.index)
    
    return features.fillna(0.0).astype(np.float32)


def compute_price_position_features(df: pd.DataFrame, window: int = 288) -> pd.DataFrame:
    """Compute price position in range and ATR Z-score.
    
    Features:
    - price_pos_in_range: (Close - Low_W) / (High_W - Low_W)
    - atr_z: Rolling Z-score of ATR
    """
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    
    # 1. Price Position in Range
    roll_low = low.rolling(window=window, min_periods=1).min()
    roll_high = high.rolling(window=window, min_periods=1).max()
    den = (roll_high - roll_low).replace(0.0, 1.0)
    price_pos = (close - roll_low) / den
    price_pos = price_pos.clip(0.0, 1.0)
    
    # 2. ATR Z-Score
    atr = _atr(high, low, close, period=14)
    atr_z = _rolling_z(atr, window=window).clip(-3, 3)
    
    return pd.DataFrame({
        'price_pos_in_range': price_pos.astype(np.float32),
        'atr_z_score': atr_z.astype(np.float32)
    }, index=df.index)


def build_all_features(df: pd.DataFrame, lookback: int = 288) -> pd.DataFrame:
    """Build and return all additional feature columns aligned to df index.

    Args:
        df: Numeric dataframe with at least open/high/low/close/volume; optional buy/sell/quote_volume/trades.
        lookback: Rolling lookback for standardization.

    Returns:
        DataFrame of engineered features (float32), NaN-safe.
    """
    macd_df = compute_macd_features(df['close'], lookback=lookback)
    liq_df = compute_liquidity_features(df, lookback=lookback)
    smc_df = compute_smc_features(df, lookback=lookback)
    
    # Add Profile Features (Longer window for macro context in state vector)
    # 1440 = 5 days (approx)
    prof_df = compute_rolling_profile_features(df, window=1440)
    
    # Add Long-term features
    lt_df = compute_long_term_features(df)
    
    # Add Multi-timeframe Bias
    mtf_df = compute_multi_timeframe_bias(df)
    
    # Add Price Position and ATR Z (New)
    pos_df = compute_price_position_features(df, window=lookback)
    
    extra = pd.concat([macd_df, liq_df, smc_df, prof_df, lt_df, mtf_df, pos_df], axis=1)
    return extra.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
