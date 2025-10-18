from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class RewardConfig:
    reward_scale: float = 100.0        # 對數權益報酬縮放
    reward_clip_min: float = -5.0
    reward_clip_max: float = 5.0
    use_vol_norm: bool = False
    vol_window: int = 200

    safety_threshold: float = 0.02     # 與強平價距離門檻（2%）
    safety_lambda: float = 0.7         # 接近強平連續懲罰權重
    dd_lambda: float = 0.5             # 回撤增量懲罰權重
    friction_eta: float = 0.08         # 換手懲罰權重（預留）
    terminal_penalty: float = -10.0    # 強平/資金不足終局懲罰


class RewardCalculator:
    """計算交易環境的步進獎勵。

    - 主信號：對數權益報酬 × 固定縮放，再 clip
    - 風險：接近強平懲罰、回撤增量懲罰
    - 可選：波動正規化（Sharpe 近似）
    """

    def __init__(self, config: RewardConfig) -> None:
        self.config = config
        self.peak_equity: float = 0.0
        self.last_drawdown: float = 0.0
        self.ret_history: list[float] = []

    def reset(self, initial_equity: float) -> None:
        self.peak_equity = float(initial_equity)
        self.last_drawdown = 0.0
        self.ret_history.clear()

    def compute_step_reward(
        self,
        *,
        last_equity: float,
        new_equity: float,
        position_size: float,
        current_price: float,
        liquidation_price: Optional[float],
    ) -> float:
        cfg = self.config

        # 對數權益報酬
        base = 0.0
        if last_equity > 0 and new_equity > 0:
            base = np.log(new_equity / last_equity)

        # 可選：波動正規化（以收益歷史為基礎）
        self.ret_history.append(base)
        if cfg.use_vol_norm and len(self.ret_history) >= 2:
            window = self.ret_history[-cfg.vol_window:] if len(self.ret_history) > cfg.vol_window else self.ret_history
            vol = np.std(window) if len(window) >= 2 else 0.0
            if vol > 0:
                base = base / vol

        reward = cfg.reward_scale * base

        # 風險：距離強平懲罰
        if position_size != 0.0 and liquidation_price is not None and liquidation_price > 0.0 and current_price > 0.0:
            if position_size > 0:
                distance_ratio = max(0.0, (current_price - liquidation_price) / current_price)
            else:
                distance_ratio = max(0.0, (liquidation_price - current_price) / current_price)
            if distance_ratio < cfg.safety_threshold:
                reward -= cfg.safety_lambda * (cfg.safety_threshold - distance_ratio) / cfg.safety_threshold

        # 回撤增量懲罰
        self.peak_equity = max(self.peak_equity, new_equity)
        current_dd = 0.0 if self.peak_equity <= 0 else max(0.0, (self.peak_equity - new_equity) / self.peak_equity)
        delta_dd = max(0.0, current_dd - self.last_drawdown)
        if delta_dd > 0:
            reward -= cfg.dd_lambda * delta_dd
        self.last_drawdown = current_dd

        # 裁剪
        reward = float(np.clip(reward, cfg.reward_clip_min, cfg.reward_clip_max))
        return reward

    def terminal_fail_penalty(self) -> float:
        return self.config.terminal_penalty


