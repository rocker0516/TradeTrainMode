"""Market snapshot utilities for environment step processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketSnapshot:
    """Aggregated market-related values required for一步交易流程."""

    current_price: float
    current_high: float
    current_low: float
    last_equity: float
    atr_ratio: float
    atr_est: float
    window_start: int
    window_end: int
    window_high: float
    window_low: float


class MarketSnapshotBuilder:
    """Builds :class:`MarketSnapshot` for the current environment step."""

    def build(
        self,
        *,
        df: pd.DataFrame,
        obs_features: pd.DataFrame,
        current_step: int,
        window_size: int,
        executor: Any,
        default_atr_ratio: float = 0.02,
    ) -> MarketSnapshot:
        """Create a snapshot of市場資料與風險量測。

        Args:
            df: 原始數據表。
            obs_features: 已預先計算的特徵表。
            current_step: 當前時間步索引。
            window_size: 觀察窗口長度。
            executor: 交易執行器，用於計算權益。
            default_atr_ratio: 初始 ATR 比例缺值時的 fallback。

        Returns:
            封裝當前價格、ATR 與窗口統計的 snapshot。
        """

        candle = df.iloc[current_step]
        current_price = float(candle['close'])
        current_high = float(candle['high'])
        current_low = float(candle['low'])
        last_equity = float(executor.equity(current_price))

        atr_ratio = float(
            obs_features['atr_ratio'].iloc[current_step - 1]
            if current_step > 0
            else default_atr_ratio
        )
        atr_est = float(max(1e-8, atr_ratio * current_price))

        window_start = max(0, current_step - window_size)
        window_end = current_step
        window_slice = df.iloc[window_start:window_end]
        window_high = float(np.max(window_slice['high'])) if not window_slice.empty else current_high
        window_low = float(np.min(window_slice['low'])) if not window_slice.empty else current_low

        return MarketSnapshot(
            current_price=current_price,
            current_high=current_high,
            current_low=current_low,
            last_equity=last_equity,
            atr_ratio=atr_ratio,
            atr_est=atr_est,
            window_start=window_start,
            window_end=window_end,
            window_high=window_high,
            window_low=window_low,
        )

