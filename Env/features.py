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


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def normalize_feature_frame(
    df: pd.DataFrame,
    lookback: int,
    *,
    preserve_binary: bool = True,
    binary_tol: float = 1e-6,
) -> pd.DataFrame:
    """Normalize feature columns to comparable scales using rolling z-score.

    Args:
        df: Feature dataframe aligned with price index.
        lookback: Rolling window length for normalization.
        preserve_binary: If True, keep pure binary/ternary categorical columns intact.
        binary_tol: Numerical tolerance when detecting discrete values.

    Returns:
        Normalized dataframe (float32), NaN safe.
    """

    normalized: Dict[str, pd.Series] = {}
    for col in df.columns:
        series = df[col].astype(float)
        unique = pd.unique(series[~np.isnan(series)])

        if preserve_binary and unique.size <= 3:
            # Detect discrete {-1,0,1} / {0,1} patterns.
            if np.all(np.isclose(unique, 0.0, atol=binary_tol)):
                normalized[col] = pd.Series(0.0, index=df.index, dtype=np.float32)
                continue
            if np.all(np.isin(np.round(unique), [-1.0, 0.0, 1.0])):
                normalized[col] = series.astype(np.float32)
                continue

        std = series.std()
        if std < binary_tol:
            normalized[col] = pd.Series(0.0, index=df.index, dtype=np.float32)
            continue

        normalized[col] = _rolling_z(series, lookback).astype(np.float32)

    return pd.DataFrame(normalized, index=df.index)


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
    """Compute Smart Money Concepts inspired features.

    The implementation mirrors LuxAlgo's Smart Money Concepts indicator while ensuring a
    causal (past-only) workflow suitable for RL training. It tracks market structure,
    displacement, fair value gaps, liquidity events, premium/discount positioning, and
    order blocks.

    Returned columns (float32):
        smc_structure_trend             : swing bias (-1 bear, 0 neutral, 1 bull)
        smc_internal_trend              : internal structure bias (-1, 0, 1)
        smc_swing_bos                   : swing BOS / CHoCH signal (1 up, -1 down, else 0)
        smc_internal_bos                : internal BOS / CHoCH signal (1 up, -1 down, else 0)
        smc_sweep_up                    : liquidity sweep above previous swing high (0/1)
        smc_sweep_down                  : liquidity sweep below previous swing low (0/1)
        smc_equal_high                  : equal swing / internal highs detected (0/1)
        smc_equal_low                   : equal swing / internal lows detected (0/1)
        smc_dist_to_swing_high          : z-score of distance to most recent swing high
        smc_dist_to_swing_low           : z-score of distance to most recent swing low
        smc_premium_discount            : relative location within swing range (-0.5 ~ 0.5)
        smc_displacement_z              : candle displacement (|body| / ATR) z-score
        smc_fvg_bull_active             : active bullish fair value gap flag (0/1)
        smc_fvg_bear_active             : active bearish fair value gap flag (0/1)
        smc_fvg_bull_size_z             : bullish FVG size (gap / ATR) z-score
        smc_fvg_bear_size_z             : bearish FVG size (gap / ATR) z-score
        smc_dist_to_bull_fvg_z          : z-score of distance from close to bullish FVG
        smc_dist_to_bear_fvg_z          : z-score of distance from close to bearish FVG
        smc_order_block_bull_active     : active bullish order block flag (0/1)
        smc_order_block_bear_active     : active bearish order block flag (0/1)
        smc_order_block_bull_strength_z : z-score of displacement creating last bullish OB
        smc_order_block_bear_strength_z : z-score of displacement creating last bearish OB
        smc_dist_to_bull_ob_z           : z-score of distance from close to bullish OB zone
        smc_dist_to_bear_ob_z           : z-score of distance from close to bearish OB zone
    """

    idx = df.index
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    open_ = df.get('open', df['close']).astype(float)
    close = df['close'].astype(float)

    atr_fast = _atr(high, low, close, period=14)
    atr_slow = _atr(high, low, close, period=200)
    body = (close - open_).astype(float)
    body_ratio = (body.abs() / atr_fast.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    n = len(df)
    swing_window = max(20, lookback // 6)
    internal_window = max(5, swing_window // 4)
    equal_threshold = 0.1
    min_gap_ratio = 0.1
    displacement_threshold = 1.0
    eps = 1e-8

    structure_trend = np.zeros(n, dtype=np.float64)
    internal_trend = np.zeros(n, dtype=np.float64)
    swing_bos = np.zeros(n, dtype=np.float64)
    internal_bos = np.zeros(n, dtype=np.float64)
    sweep_up = np.zeros(n, dtype=np.float64)
    sweep_down = np.zeros(n, dtype=np.float64)
    equal_high = np.zeros(n, dtype=np.float64)
    equal_low = np.zeros(n, dtype=np.float64)
    dist_high_vals = np.zeros(n, dtype=np.float64)
    dist_low_vals = np.zeros(n, dtype=np.float64)
    premium_vals = np.zeros(n, dtype=np.float64)

    fvg_bull_active = np.zeros(n, dtype=np.float64)
    fvg_bear_active = np.zeros(n, dtype=np.float64)
    dist_bull_fvg_vals = np.zeros(n, dtype=np.float64)
    dist_bear_fvg_vals = np.zeros(n, dtype=np.float64)
    fvg_bull_size_vals = np.zeros(n, dtype=np.float64)
    fvg_bear_size_vals = np.zeros(n, dtype=np.float64)

    ob_bull_active = np.zeros(n, dtype=np.float64)
    ob_bear_active = np.zeros(n, dtype=np.float64)
    dist_bull_ob_vals = np.zeros(n, dtype=np.float64)
    dist_bear_ob_vals = np.zeros(n, dtype=np.float64)
    ob_bull_strength_vals = np.zeros(n, dtype=np.float64)
    ob_bear_strength_vals = np.zeros(n, dtype=np.float64)

    last_internal_high = high.iloc[0]
    last_internal_low = low.iloc[0]
    prev_internal_high = last_internal_high
    prev_internal_low = last_internal_low
    internal_high_crossed = False
    internal_low_crossed = False

    last_swing_high = high.iloc[0]
    last_swing_low = low.iloc[0]
    prev_swing_high = last_swing_high
    prev_swing_low = last_swing_low
    swing_high_crossed = False
    swing_low_crossed = False

    current_bull_fvg: dict | None = None
    current_bear_fvg: dict | None = None
    current_bull_ob: dict | None = None
    current_bear_ob: dict | None = None

    for i in range(n):
        price_high = high.iat[i]
        price_low = low.iat[i]
        price_close = close.iat[i]
        atr_fast_val = float(max(atr_fast.iat[i], eps))
        atr_measure = atr_slow.iat[i] if not np.isnan(atr_slow.iat[i]) else atr_fast_val

        prev_internal_bias = internal_trend[i - 1] if i > 0 else 0.0
        prev_swing_bias = structure_trend[i - 1] if i > 0 else 0.0
        internal_bias = prev_internal_bias
        swing_bias = prev_swing_bias

        # Active FVG maintenance
        fvg_bull_size_val = 0.0
        fvg_bear_size_val = 0.0
        if current_bull_fvg is not None:
            if price_low <= current_bull_fvg['lower'] + eps:
                current_bull_fvg = None
            else:
                fvg_bull_active[i] = 1.0
                dist_bull_fvg_vals[i] = np.clip(
                    (price_close - current_bull_fvg['lower']) / atr_fast_val,
                    -10.0,
                    10.0,
                )
                fvg_bull_size_val = current_bull_fvg['size'] / atr_fast_val
        if current_bear_fvg is not None:
            if price_high >= current_bear_fvg['upper'] - eps:
                current_bear_fvg = None
            else:
                fvg_bear_active[i] = 1.0
                dist_bear_fvg_vals[i] = np.clip(
                    (current_bear_fvg['upper'] - price_close) / atr_fast_val,
                    -10.0,
                    10.0,
                )
                fvg_bear_size_val = current_bear_fvg['size'] / atr_fast_val

        if i >= internal_window:
            start = max(0, i - internal_window)
            prior_high = high.iloc[start:i]
            if not prior_high.empty and price_high > prior_high.max() + eps:
                prev_internal_high = last_internal_high
                last_internal_high = price_high
                internal_high_crossed = False
                if atr_measure > 0.0 and abs(last_internal_high - prev_internal_high) <= equal_threshold * atr_measure:
                    equal_high[i] = 1.0

            prior_low = low.iloc[start:i]
            if not prior_low.empty and price_low < prior_low.min() - eps:
                prev_internal_low = last_internal_low
                last_internal_low = price_low
                internal_low_crossed = False
                if atr_measure > 0.0 and abs(last_internal_low - prev_internal_low) <= equal_threshold * atr_measure:
                    equal_low[i] = 1.0

        if i >= swing_window:
            start = max(0, i - swing_window)
            prior_high = high.iloc[start:i]
            if not prior_high.empty and price_high > prior_high.max() + eps:
                prev_swing_high = last_swing_high
                last_swing_high = price_high
                swing_high_crossed = False
                if atr_measure > 0.0 and abs(last_swing_high - prev_swing_high) <= equal_threshold * atr_measure:
                    equal_high[i] = 1.0

            prior_low = low.iloc[start:i]
            if not prior_low.empty and price_low < prior_low.min() - eps:
                prev_swing_low = last_swing_low
                last_swing_low = price_low
                swing_low_crossed = False
                if atr_measure > 0.0 and abs(last_swing_low - prev_swing_low) <= equal_threshold * atr_measure:
                    equal_low[i] = 1.0

        if price_high > last_swing_high + eps and price_close < last_swing_high:
            sweep_up[i] = 1.0
        if price_low < last_swing_low - eps and price_close > last_swing_low:
            sweep_down[i] = 1.0

        if price_close > last_internal_high + eps and not internal_high_crossed:
            internal_bias = 1.0
            internal_bos[i] = 1.0
            internal_high_crossed = True
            internal_low_crossed = False
        elif price_close < last_internal_low - eps and not internal_low_crossed:
            internal_bias = -1.0
            internal_bos[i] = -1.0
            internal_low_crossed = True
            internal_high_crossed = False

        if price_close > last_swing_high + eps and not swing_high_crossed:
            swing_bias = 1.0
            swing_bos[i] = 1.0
            swing_high_crossed = True
            swing_low_crossed = False
        elif price_close < last_swing_low - eps and not swing_low_crossed:
            swing_bias = -1.0
            swing_bos[i] = -1.0
            swing_low_crossed = True
            swing_high_crossed = False

        price_range = max(last_swing_high - last_swing_low, atr_fast_val + eps)
        dist_high_vals[i] = (last_swing_high - price_close) / price_range
        dist_low_vals[i] = (price_close - last_swing_low) / price_range
        premium_vals[i] = ((price_close - last_swing_low) / price_range) - 0.5

        internal_trend[i] = internal_bias
        structure_trend[i] = swing_bias

        # Detect new fair value gaps
        if i >= 2:
            base_high = high.iat[i - 2]
            base_low = low.iat[i - 2]
            mid_low = low.iat[i - 1]
            mid_high = high.iat[i - 1]

            if mid_low > base_high + eps and price_low > base_high + eps:
                gap_upper = min(mid_low, price_low)
                gap_size = gap_upper - base_high
                if gap_size > min_gap_ratio * atr_fast_val:
                    current_bull_fvg = {
                        'lower': base_high,
                        'upper': gap_upper,
                        'size': gap_size,
                    }
                    fvg_bull_active[i] = 1.0
                    dist_bull_fvg_vals[i] = np.clip(
                        (price_close - base_high) / atr_fast_val,
                        -10.0,
                        10.0,
                    )
                    fvg_bull_size_val = gap_size / atr_fast_val

            if mid_high < base_low - eps and price_high < base_low - eps:
                gap_lower = max(mid_high, price_high)
                gap_size = base_low - gap_lower
                if gap_size > min_gap_ratio * atr_fast_val:
                    current_bear_fvg = {
                        'lower': gap_lower,
                        'upper': base_low,
                        'size': gap_size,
                    }
                    fvg_bear_active[i] = 1.0
                    dist_bear_fvg_vals[i] = np.clip(
                        (base_low - price_close) / atr_fast_val,
                        -10.0,
                        10.0,
                    )
                    fvg_bear_size_val = gap_size / atr_fast_val

        if fvg_bull_size_val != 0.0:
            fvg_bull_size_vals[i] = fvg_bull_size_val
        if fvg_bear_size_val != 0.0:
            fvg_bear_size_vals[i] = fvg_bear_size_val

        # Order block maintenance
        if current_bull_ob is not None:
            if price_close < current_bull_ob['low'] - eps:
                current_bull_ob = None
            else:
                ob_bull_active[i] = 1.0
                dist_bull_ob_vals[i] = np.clip(
                    (price_close - current_bull_ob['mid']) / atr_fast_val,
                    -10.0,
                    10.0,
                )
                ob_bull_strength_vals[i] = current_bull_ob['strength']
                if price_low <= current_bull_ob['high'] + eps:
                    current_bull_ob['mitigated'] = True

        if current_bear_ob is not None:
            if price_close > current_bear_ob['high'] + eps:
                current_bear_ob = None
            else:
                ob_bear_active[i] = 1.0
                dist_bear_ob_vals[i] = np.clip(
                    (current_bear_ob['mid'] - price_close) / atr_fast_val,
                    -10.0,
                    10.0,
                )
                ob_bear_strength_vals[i] = current_bear_ob['strength']
                if price_high >= current_bear_ob['low'] - eps:
                    current_bear_ob['mitigated'] = True

        if swing_bos[i] == 1.0 and body_ratio.iat[i] >= displacement_threshold:
            search_start = max(0, i - swing_window)
            j = i - 1
            while j >= search_start:
                if close.iat[j] < open_.iat[j] - eps:
                    zone_high = max(open_.iat[j], close.iat[j])
                    zone_low = min(low.iat[j], open_.iat[j])
                    current_bull_ob = {
                        'high': zone_high,
                        'low': zone_low,
                        'mid': (zone_high + zone_low) / 2.0,
                        'strength': float(min(body_ratio.iat[i], 6.0)),
                        'mitigated': False,
                    }
                    ob_bull_active[i] = 1.0
                    dist_bull_ob_vals[i] = np.clip(
                        (price_close - current_bull_ob['mid']) / atr_fast_val,
                        -10.0,
                        10.0,
                    )
                    ob_bull_strength_vals[i] = current_bull_ob['strength']
                    break
                j -= 1

        if swing_bos[i] == -1.0 and body_ratio.iat[i] >= displacement_threshold:
            search_start = max(0, i - swing_window)
            j = i - 1
            while j >= search_start:
                if close.iat[j] > open_.iat[j] + eps:
                    zone_high = max(high.iat[j], open_.iat[j])
                    zone_low = min(open_.iat[j], close.iat[j])
                    current_bear_ob = {
                        'high': zone_high,
                        'low': zone_low,
                        'mid': (zone_high + zone_low) / 2.0,
                        'strength': float(min(body_ratio.iat[i], 6.0)),
                        'mitigated': False,
                    }
                    ob_bear_active[i] = 1.0
                    dist_bear_ob_vals[i] = np.clip(
                        (current_bear_ob['mid'] - price_close) / atr_fast_val,
                        -10.0,
                        10.0,
                    )
                    ob_bear_strength_vals[i] = current_bear_ob['strength']
                    break
                j -= 1

    dist_high_series = pd.Series(dist_high_vals, index=idx)
    dist_low_series = pd.Series(dist_low_vals, index=idx)
    premium_series = pd.Series(premium_vals, index=idx)
    dist_bull_fvg_series = pd.Series(dist_bull_fvg_vals, index=idx)
    dist_bear_fvg_series = pd.Series(dist_bear_fvg_vals, index=idx)
    fvg_bull_size_series = pd.Series(fvg_bull_size_vals, index=idx)
    fvg_bear_size_series = pd.Series(fvg_bear_size_vals, index=idx)
    dist_bull_ob_series = pd.Series(dist_bull_ob_vals, index=idx)
    dist_bear_ob_series = pd.Series(dist_bear_ob_vals, index=idx)
    ob_bull_strength_series = pd.Series(ob_bull_strength_vals, index=idx)
    ob_bear_strength_series = pd.Series(ob_bear_strength_vals, index=idx)

    feats = pd.DataFrame({
        'smc_structure_trend': structure_trend.astype(np.float32),
        'smc_internal_trend': internal_trend.astype(np.float32),
        'smc_swing_bos': swing_bos.astype(np.float32),
        'smc_internal_bos': internal_bos.astype(np.float32),
        'smc_sweep_up': sweep_up.astype(np.float32),
        'smc_sweep_down': sweep_down.astype(np.float32),
        'smc_equal_high': equal_high.astype(np.float32),
        'smc_equal_low': equal_low.astype(np.float32),
        'smc_dist_to_swing_high': _rolling_z(dist_high_series, lookback).astype(np.float32),
        'smc_dist_to_swing_low': _rolling_z(dist_low_series, lookback).astype(np.float32),
        'smc_premium_discount': premium_series.astype(np.float32),
        'smc_displacement_z': _rolling_z(body_ratio, lookback).astype(np.float32),
        'smc_fvg_bull_active': fvg_bull_active.astype(np.float32),
        'smc_fvg_bear_active': fvg_bear_active.astype(np.float32),
        'smc_fvg_bull_size_z': _rolling_z(fvg_bull_size_series, lookback).astype(np.float32),
        'smc_fvg_bear_size_z': _rolling_z(fvg_bear_size_series, lookback).astype(np.float32),
        'smc_dist_to_bull_fvg_z': _rolling_z(dist_bull_fvg_series, lookback).astype(np.float32),
        'smc_dist_to_bear_fvg_z': _rolling_z(dist_bear_fvg_series, lookback).astype(np.float32),
        'smc_order_block_bull_active': ob_bull_active.astype(np.float32),
        'smc_order_block_bear_active': ob_bear_active.astype(np.float32),
        'smc_order_block_bull_strength_z': _rolling_z(ob_bull_strength_series, lookback).astype(np.float32),
        'smc_order_block_bear_strength_z': _rolling_z(ob_bear_strength_series, lookback).astype(np.float32),
        'smc_dist_to_bull_ob_z': _rolling_z(dist_bull_ob_series, lookback).astype(np.float32),
        'smc_dist_to_bear_ob_z': _rolling_z(dist_bear_ob_series, lookback).astype(np.float32),
    }, index=idx)
    return feats


def build_all_features(df: pd.DataFrame, lookback: int = 288) -> pd.DataFrame:
    """Build and return all additional feature columns aligned to df index.

    Args:
        df: Numeric dataframe with at least open/high/low/close/volume; optional buy/sell/quote_volume/trades.
        lookback: Rolling lookback for standardization.

    Returns:
        DataFrame of engineered features (float32), NaN-safe.
    """
    macd_df = compute_macd_features(df['close'], lookback=lookback)#MACD指標
    liq_df = compute_liquidity_features(df, lookback=lookback)#流動性指標
    smc_df = compute_smc_features(df, lookback=lookback)#SMC指標
    extra = pd.concat([macd_df, liq_df, smc_df], axis=1)
    extra = extra.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return normalize_feature_frame(extra, lookback=lookback)


