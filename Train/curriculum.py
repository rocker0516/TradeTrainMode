from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class WinRateFeeCurriculum:
    """
    依「勝率」觸發的手續費課程（stage-based）。

    規則：
    - 初始 fee_mult = initial_mult（預設 0.0，等同零手續費）
    - 若 rolling win_rate >= threshold，且累積 episode 數 >= min_episodes，
      且距離上次升階 >= min_episodes_between_advances，則 fee_mult += step_mult
    - fee_mult 上限為 1.0（手續費回到 base_fee_rate）

    注意：
    - base_fee_rate 使用專案既有定義：Config.TRANSACTION_FEE（單位為「百分比」值，例如 0.005 代表 0.005%）
    - 這裡只產生「應該設定到 env 的 fee_rate」，不直接操作 env。

    Args:
        base_fee_rate: 完整手續費（100%）時的 fee_rate（例如 0.005 代表 0.005%）
        step_mult: 每次升階增加的倍率（例如 0.1 代表 +10%）
        threshold: 勝率門檻（例如 0.5 代表 50%）
        avg_profit_threshold_pct: rolling 平均 Profit 門檻（單位 %，與 dashboard 顯示一致）。
            例如 50.0 代表最近窗口平均收益率 >= +50% 才能升階；若為 None 則不檢查。
        min_episodes: 至少累積多少 episodes 才開始啟用課程（避免早期樣本太少亂跳）
        min_episodes_between_advances: 每次升階之間至少間隔多少 episodes（避免同一段 win_rate 一直 > threshold 時狂升）
        initial_mult: 初始倍率（預設 0.0）
    """

    base_fee_rate: float
    step_mult: float = 0.1
    threshold: float = 0.5
    avg_profit_threshold_pct: Optional[float] = None
    min_episodes: int = 50
    min_episodes_between_advances: int = 50
    initial_mult: float = 0.0

    def __post_init__(self) -> None:
        self.base_fee_rate = float(self.base_fee_rate)
        self.step_mult = float(self.step_mult)
        self.threshold = float(self.threshold)
        self.avg_profit_threshold_pct = None if self.avg_profit_threshold_pct is None else float(self.avg_profit_threshold_pct)
        self.min_episodes = int(self.min_episodes)
        self.min_episodes_between_advances = int(self.min_episodes_between_advances)
        self.initial_mult = float(self.initial_mult)

        if self.base_fee_rate < 0.0:
            raise ValueError("base_fee_rate must be >= 0")
        if not (0.0 <= self.initial_mult <= 1.0):
            raise ValueError("initial_mult must be within [0, 1]")
        if self.step_mult <= 0.0:
            raise ValueError("step_mult must be > 0")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be within [0, 1]")
        if self.avg_profit_threshold_pct is not None and not np.isfinite(self.avg_profit_threshold_pct):
            raise ValueError("avg_profit_threshold_pct must be finite when provided")
        if self.min_episodes < 0:
            raise ValueError("min_episodes must be >= 0")
        if self.min_episodes_between_advances < 0:
            raise ValueError("min_episodes_between_advances must be >= 0")

        self._fee_mult: float = float(self.initial_mult)
        self._last_advance_episode: int = 0

    @property
    def fee_mult(self) -> float:
        """目前手續費倍率（0~1）。"""
        return float(self._fee_mult)

    @property
    def fee_rate(self) -> float:
        """目前應該套用到 env 的 fee_rate。"""
        return float(self.base_fee_rate * self._fee_mult)

    def maybe_advance(self, *, win_rate: float, avg_profit_pct: float, total_episodes: int) -> Optional[float]:
        """
        依當前 win_rate 與 total_episodes 判斷是否升階。

        Args:
            win_rate: rolling window 的勝率（0~1）
            avg_profit_pct: rolling window 的平均 Profit（單位 %，與 dashboard 顯示一致）
            total_episodes: 訓練至今累積的 episode 數

        Returns:
            若升階，回傳新的 fee_rate；否則回傳 None。
        """
        total_episodes = int(total_episodes)
        win_rate = float(win_rate)
        avg_profit_pct = float(avg_profit_pct)

        if self._fee_mult >= 1.0:
            return None
        if total_episodes < self.min_episodes:
            return None
        if win_rate < self.threshold:
            return None
        if self.avg_profit_threshold_pct is not None and avg_profit_pct < self.avg_profit_threshold_pct:
            return None
        if (total_episodes - self._last_advance_episode) < self.min_episodes_between_advances:
            return None

        new_mult = min(1.0, self._fee_mult + self.step_mult)
        if new_mult <= self._fee_mult + 1e-12:
            return None

        self._fee_mult = float(new_mult)
        self._last_advance_episode = int(total_episodes)
        return self.fee_rate


