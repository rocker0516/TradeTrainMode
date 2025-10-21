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
        failure_penalty: 非資料用盡終止的扣分（爆倉/破產）
        daily_bonus_positive: 日終正報酬加成係數
        daily_penalty_negative: 日終負報酬懲罰係數
        margin_buffer_penalty: 保證金緩衝不足懲罰係數（越接近強平扣越多）
        turnover_cost: 倉位變動成本係數（抑制頻繁交易）
    """
    mode: RewardMode = 'log'
    scale: float = 1.0
    failure_penalty: float = 100.0
    daily_bonus_positive: float = 5.0
    daily_penalty_negative: float = 3.0
    margin_buffer_penalty: float = 2.0
    turnover_cost: float = 0.1

    
    def compute(
        self, *,
        last_equity: float,
        new_equity: float,
        done: bool = False,
        termination_reason: str | None = None,
        day_return: float | None = None,
        is_day_end: bool = False,
        margin_buffer: float | None = None,
        position_change: float | None = None,
    ) -> float:
        '''
            計算獎勵
            last_equity: 上一步的權益
            new_equity: 當前權益
            done: 是否終止
            termination_reason: 終止原因
            day_return: 當日累計報酬（日終時提供）
            is_day_end: 是否為日終
            margin_buffer: 保證金緩衝比例（距離強平的安全空間，0~1）
            position_change: 倉位變動絕對值（用於計算換手成本）
            return: 獎勵
        '''
        # 1. 基礎成長獎勵
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

        # 2. 保證金緩衝懲罰：越接近強平扣分越多
        if margin_buffer is not None and self.margin_buffer_penalty > 0:
            # margin_buffer 建議為 [0, 1]，0 表示即將強平，1 表示很安全
            # 當 buffer < 0.3 時開始懲罰，越低懲罰越重
            buffer_threshold = 0.3
            if margin_buffer < buffer_threshold:
                # 非線性懲罰：(0.3 - buffer)^2 放大接近強平時的懲罰
                deficit = float(buffer_threshold - margin_buffer)
                shaped -= float(self.margin_buffer_penalty * (deficit ** 2))

        # 3. 換手成本懲罰：抑制頻繁交易
        if position_change is not None and self.turnover_cost > 0:
            shaped -= float(self.turnover_cost * abs(position_change))

        # 4. 每日績效塑形：鼓勵每天為正
        if is_day_end and day_return is not None:
            if day_return > 0:
                # 正報酬：加成
                shaped += float(self.daily_bonus_positive * day_return)
            else:
                # 負報酬：懲罰（較輕，避免過度懲罰正常波動）
                shaped -= float(self.daily_penalty_negative * abs(day_return))

        # 5. 終止懲罰：非資料用盡終止給重罰
        if done and termination_reason is not None and termination_reason != 'data_exhausted' and self.failure_penalty > 0:
            # 爆倉/破產給極大懲罰，避免策略走向這條路
            shaped -= float(self.failure_penalty)

        return shaped


