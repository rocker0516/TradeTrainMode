"""
圖表歷史標記：B/S 與 GATE A/C 變化標籤。

供 `live_trading_loop` 與訓練後 holdout 視覺化共用，避免邏輯分叉。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def trade_bs_from_delta(
    *, prev_final_pos_pct: float, new_final_pos_pct: float, eps: float
) -> Optional[str]:
    """依淨倉位變化決定價格圖上的 B/S 標記（不區分翻倉語意）。

    Args:
        prev_final_pos_pct: 本步前之 effective 倉位比例。
        new_final_pos_pct: 本步後之 effective 倉位比例。
        eps: 門檻；若 ``|Δ| < eps`` 則不標記。

    Returns:
        ``\"B\"``、``\"S\"`` 或 ``None``。
    """
    delta = float(new_final_pos_pct) - float(prev_final_pos_pct)
    if delta > float(eps):
        return "B"
    if delta < -float(eps):
        return "S"
    return None


def gate_ac_change_labels(
    prev: Optional[Tuple[float, float, float]],
    curr: Optional[Tuple[float, float, float]],
) -> List[str]:
    """僅在 gate_A（index 0）或 gate_C（index 2）變化時產生文字標籤；忽略 gate_B。

    Args:
        prev: 上一輪 ``gate_flags``；``None`` 時不產生標籤（避免首步洗版）。
        curr: 本輪 ``gate_flags``。

    Returns:
        例如 ``[\"A:0→1\", \"C:0→-1\"]``。
    """
    if prev is None or curr is None:
        return []

    def _i_gate_a(v: float) -> int:
        return int(round(float(v)))

    def _i_gate_c(v: float) -> int:
        return int(round(float(v)))

    labels: List[str] = []
    if abs(float(curr[0]) - float(prev[0])) > 1e-6:
        labels.append(f"A:{_i_gate_a(prev[0])}→{_i_gate_a(curr[0])}")
    if abs(float(curr[2]) - float(prev[2])) > 1e-6:
        labels.append(f"C:{_i_gate_c(prev[2])}→{_i_gate_c(curr[2])}")
    return labels


def gate_flags_to_regime_indicator(
    gate_flags: Optional[Union[Tuple[float, ...], List[float], np.ndarray]],
) -> float:
    """
    與 `Env/Rewards/reward.py` regime_dir 一致：Gate A 多=+1、Gate C 空=-1、否則 0（中性）。

    Args:
        gate_flags: ``(gate_A, gate_B, gate_C)``；缺漏時視為 0。

    Returns:
        -1.0、0.0 或 1.0。
    """
    if gate_flags is None:
        return 0.0
    arr = np.asarray(gate_flags, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return 0.0
    gate_a = float(arr[0])
    gate_c = float(arr[2])
    if gate_a >= 0.5:
        return 1.0
    if gate_c <= -0.5:
        return -1.0
    return 0.0


def trend_tanh_signed_from_trend_score(trend_score: float, *, scale: float) -> float:
    """
    與 reward 使用相同縮放：``tanh(scale * clip(trend_score, -5, 5))``（帶正負號，反映 5m 多空傾向）。

    Args:
        trend_score: ``MarketData.get_market_metrics`` 的 ``trend_score``（5m 收盤上 (MA50-MA200)/MA200）。
        scale: ``conviction_trend_score_scale``（須與訓練 env 一致）。

    Returns:
        (-1, 1) 的 signed tanh。
    """
    scaled = float(scale) * float(np.clip(float(trend_score), -5.0, 5.0))
    return float(np.tanh(scaled))


def conviction_strength_from_trend_score(trend_score: float, *, scale: float) -> float:
    """
    與 `ConvictionTrendRewardCalculator` 一致：``strength = |tanh(scale * clip(trend_score, -5, 5))|``。

    Args:
        trend_score: ``MarketData.get_market_metrics`` 的 ``trend_score``（MA 趨勢小數比）。
        scale: ``conviction_trend_score_scale``（須與訓練 env 一致）。

    Returns:
        [0, 1] 的趨勢強度。
    """
    return float(abs(trend_tanh_signed_from_trend_score(trend_score, scale=scale)))


def build_sideway_mask_from_series(
    close_values: Sequence[float],
    atr_ratio_values: Sequence[float],
    *,
    lookback_l: int = 24,
    theta_er: float = 0.30,
    rw_max: float = 3.8,
    rp_low: float = 0.2,
    rp_high: float = 0.8,
    slope_k: int = 6,
    slope_abs_max: float = 0.30,
    eps: float = 1e-12,
) -> List[bool]:
    """
    依 E2 `build_sideway_labels` 同公式，回傳每根 bar 是否為盤整（True=盤整）。

    條件（同時滿足）：
    1) ER_L < theta_er
    2) RW_L < rw_max
    3) rp_low < RP_L < rp_high
    4) |Slope| < slope_abs_max
    """
    close_s = pd.Series(np.asarray(list(close_values), dtype=np.float64))
    atr_ratio_s = pd.Series(np.asarray(list(atr_ratio_values), dtype=np.float64))
    if close_s.empty or atr_ratio_s.empty:
        return []
    if len(close_s) != len(atr_ratio_s):
        n = min(len(close_s), len(atr_ratio_s))
        close_s = close_s.iloc[:n]
        atr_ratio_s = atr_ratio_s.iloc[:n]

    abs_diff = close_s.diff().abs().fillna(0.0)
    vol_l = abs_diff.rolling(window=int(lookback_l), min_periods=int(lookback_l)).sum()
    direction_l = (close_s - close_s.shift(int(lookback_l))).abs()
    er_l = direction_l / (vol_l + float(eps))

    hh_l = close_s.rolling(window=int(lookback_l), min_periods=int(lookback_l)).max()
    ll_l = close_s.rolling(window=int(lookback_l), min_periods=int(lookback_l)).min()
    atr_price = (atr_ratio_s * close_s.abs()).replace([np.inf, -np.inf], np.nan)
    atr_l = atr_price.rolling(window=int(lookback_l), min_periods=int(lookback_l)).mean()
    rw_l = (hh_l - ll_l) / (atr_l + float(eps))
    rp_l = (close_s - ll_l) / ((hh_l - ll_l) + float(eps))

    ema12 = close_s.ewm(span=12, adjust=False).mean()
    slope = (ema12 - ema12.shift(int(slope_k))) / (atr_l + float(eps))

    mask = (
        (er_l < float(theta_er))
        & (rw_l < float(rw_max))
        & (rp_l > float(rp_low))
        & (rp_l < float(rp_high))
        & (slope.abs() < float(slope_abs_max))
    )
    mask = mask.fillna(False)
    return [bool(v) for v in mask.tolist()]


def atr_ratio_from_ohlc(
    high_values: Sequence[float],
    low_values: Sequence[float],
    close_values: Sequence[float],
    *,
    window: int = 14,
) -> List[float]:
    """
    由 OHLC 估算 ATR/close，供 sideway 規則使用。
    """
    h = pd.Series(np.asarray(list(high_values), dtype=np.float64))
    l = pd.Series(np.asarray(list(low_values), dtype=np.float64))
    c = pd.Series(np.asarray(list(close_values), dtype=np.float64))
    n = min(len(h), len(l), len(c))
    if n <= 0:
        return []
    h = h.iloc[:n]
    l = l.iloc[:n]
    c = c.iloc[:n]
    prev_close = c.shift(1)
    tr = pd.concat(
        [
            (h - l).abs(),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=int(max(1, window)), min_periods=1).mean()
    ratio = atr / np.maximum(c.abs(), 1e-12)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return [float(v) for v in ratio.tolist()]
