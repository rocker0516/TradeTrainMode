"""
評估相關的工具函數。
"""
from __future__ import annotations

import numpy as np
from typing import Any, List, Optional


def to_float_or_none(v: Any) -> Optional[float]:
    """將值轉換為 float 或 None。"""
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def to_int_or_none(v: Any) -> Optional[int]:
    """將值轉換為 int 或 None。"""
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    """計算有效值的平均值，若無有效值則返回 None。"""
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def fmt_float(v: Optional[float], digits: int = 6) -> str:
    """格式化 float 值為字串。"""
    if v is None:
        return "NA"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def fmt_pct(v: Optional[float], digits: int = 2) -> str:
    """格式化百分比值為字串。"""
    if v is None:
        return "NA"
    try:
        return f"{float(v) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "NA"


def fmt_int(v: Optional[int]) -> str:
    """格式化 int 值為字串。"""
    if v is None:
        return "NA"
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return "NA"


def format_episode_line(prefix: str, current_ts: int, ep_idx: int, ep: Any) -> str:
    """
    格式化單一 episode 的評估結果為一行字串。

    Args:
        prefix: 前綴字串（例如 "[EVAL]"）
        current_ts: 當前時間步
        ep_idx: episode 索引
        ep: EpisodeEval 實例

    Returns:
        格式化後的字串
    """
    # 新增顯示：start_step, fee%, pos (final pos)
    # 格式：[EVAL] ts=... ep=... PASS/FAIL ...
    pass_str = "PASS" if ep.passed else "FAIL"
    term_str = ep.termination_reason or 'NA'
    
    return (
        f"{prefix} ts={current_ts} ep={ep_idx} [{pass_str}] "
        f"term={term_str} start={ep.episode_start_timestamp or 'NA'}(idx={fmt_int(ep.episode_start_step)}) "
        f"steps={fmt_int(ep.episode_steps)} "
        f"ret={fmt_pct(ep.ret, 2)} dd={fmt_float(ep.episode_max_dd, 4)} "
        f"fee%={fmt_float(ep.total_fees_ratio, 4)} "
        f"cost={fmt_float(ep.mean_cost, 6)} "
        f"sl={fmt_int(ep.stop_loss_count)} "
        f"tr/s={fmt_float(ep.trades_per_step, 4)} "
        f"hold={fmt_float(ep.holding_ratio, 2)} "
        f"bal={fmt_float(ep.final_balance, 2)} "
        f"pos={fmt_float(ep.final_position_size, 4)}"
    )

