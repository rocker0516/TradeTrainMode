"""
圖表歷史標記：B/S 與 GATE A/C 變化標籤。

供 `live_trading_loop` 與訓練後 holdout 視覺化共用，避免邏輯分叉。
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np


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
