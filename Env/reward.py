from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np


RewardMode = Literal['delta_equity', 'pct', 'log']


@dataclass
class RewardCalculator:
    """
    獎勵計算策略，適用於交易環境。

    參數:
        mode: 獎勵模式。選項:
            - 'delta_equity': new_equity - last_equity (絕對浮動盈虧)
            - 'pct': (new_equity / last_equity - 1)
            - 'log': log(new_equity / last_equity)
        scale: 將獎勵乘以此比例以穩定訓練。
    """
    mode: RewardMode = 'pct'
    scale: float = 1.0
    failure_penalty: float = 0.0
    daily_bonus_scale: float = 0.0

    
    def compute(self, *, last_equity: float, new_equity: float, done: bool = False, termination_reason: str | None = None,
                day_return: float | None = None, is_day_end: bool = False) -> float:
        '''
            計算獎勵
            last_equity: 上一步的權益
            new_equity: 當前權益
            return: 獎勵
        '''
        if self.mode == 'pct':
            if last_equity > 0:
                reward = (new_equity / last_equity) - 1.0
            else:
                reward = 0.0
        elif self.mode == 'log':
            if last_equity > 0 and new_equity > 0:
                reward = float(np.log(new_equity / last_equity))
            else:
                reward = 0.0
        else:
            reward = float(new_equity - last_equity)

        shaped = float(reward * self.scale)
        # 失敗終止的額外懲罰（非資料用盡）
        if done and termination_reason is not None and termination_reason != 'data_exhausted' and self.failure_penalty > 0:
            shaped -= float(self.failure_penalty)

        # 每日績效塑形：在日結束時，依據日報酬加成/扣分
        if is_day_end and day_return is not None and self.daily_bonus_scale != 0.0:
            # 正報酬加分、負報酬扣分（線性），避免過大影響留給 scale 控制
            shaped += float(self.daily_bonus_scale * day_return)

        return shaped


