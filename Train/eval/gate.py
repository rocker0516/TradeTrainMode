"""
訓練端 episode window gate，用於判斷是否可開始評估。
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque

from Train.eval.config import TrainEvalStartGateConfig


class TrainEpisodeWindowGate:
    """維護最近 N 個訓練回合，判斷是否可開始 eval。"""

    def __init__(self, *, config: TrainEvalStartGateConfig) -> None:
        self._cfg = config
        self._is_max_steps_hist: Deque[bool] = deque(maxlen=int(max(1, config.window_size)))
        self._max_steps_count: int = 0

    def update_from_infos(self, infos: Any) -> None:
        """
        從 SB3 callback locals 的 infos 更新 episode window。

        Args:
            infos: 一般是 List[dict]（VecEnv 每個 env 一個 info）
        """
        if not bool(getattr(self._cfg, "enabled", True)):
            return
        if not isinstance(infos, (list, tuple)):
            return

        for info in infos:
            if not isinstance(info, dict):
                continue

            # 檢查是否為 episode 結束（VecMonitor 會注入 "episode" key）
            if "episode" not in info:
                continue

            # 檢查 termination_reason
            term_reason = info.get("termination_reason", "")
            is_max_steps = str(term_reason).strip() == "max_steps_reached"

            # 更新 window
            if len(self._is_max_steps_hist) == self._is_max_steps_hist.maxlen:
                # 移除最舊的
                if self._is_max_steps_hist[0]:
                    self._max_steps_count -= 1
            self._is_max_steps_hist.append(is_max_steps)
            if is_max_steps:
                self._max_steps_count += 1

    def can_start_eval(self) -> bool:
        """
        判斷是否可開始 eval。

        Returns:
            True 若 gate 未啟用，或已達到 min_max_steps_reached_count。
        """
        if not bool(getattr(self._cfg, "enabled", False)):
            return True
        return int(self._max_steps_count) >= int(self._cfg.min_max_steps_reached_count)

    def is_ready(self) -> bool:
        """
        判斷是否已達到可開始 eval 的條件（別名方法，保持向後兼容）。

        Returns:
            True 若 gate 未啟用，或已達到 min_max_steps_reached_count。
        """
        if not bool(getattr(self._cfg, "enabled", True)):
            return True
        if len(self._is_max_steps_hist) < int(self._cfg.window_size):
            return False
        return int(self._max_steps_count) >= int(self._cfg.min_max_steps_reached_count)

