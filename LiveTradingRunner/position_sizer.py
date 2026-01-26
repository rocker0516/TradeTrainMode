"""Position sizing utilities for live runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PositionSizingResult:
    """倉位計算結果。"""

    target_pos_pct: float
    target_position_qty: float
    delta_qty: float


def compute_delta_position_qty(
    *,
    equity_usdt: float,
    last_price: float,
    leverage: float,
    target_pos_pct: float,
    current_position_qty: float,
) -> PositionSizingResult:
    """以 equity/price/leverage 計算目標持倉與差量。

    公式與你原本 RealTrading 的核心一致：
        target_qty = equity_usdt * target_pos_pct * leverage / last_price

    Args:
        equity_usdt: 目前帳戶 equity（USDT）。
        last_price: 最新已收盤 bar 的 close。
        leverage: 槓桿倍數（會與交易所側 leverage 保持一致）。
        target_pos_pct: 模型輸出的目標曝險比例（-1~1，通常會先被 clip）。
        current_position_qty: 目前持倉數量（正=多、負=空）。
    """
    eq = float(max(0.0, equity_usdt))
    p = float(max(1e-12, last_price))
    lev = float(max(1.0, leverage))
    pct = float(np.clip(float(target_pos_pct), -1.0, 1.0))
    cur = float(current_position_qty)

    target_qty = (eq * pct * lev) / p if eq > 0 else 0.0
    delta = float(target_qty - cur)
    return PositionSizingResult(target_pos_pct=pct, target_position_qty=float(target_qty), delta_qty=float(delta))


