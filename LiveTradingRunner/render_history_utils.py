"""
圖表歷史標記：B/S 與 GATE A/C 變化標籤。

供 `live_trading_loop` 與訓練後 holdout 視覺化共用，避免邏輯分叉。
"""

from __future__ import annotations

from typing import List, Optional, Tuple


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
