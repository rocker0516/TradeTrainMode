from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np


RewardMode = Literal['delta_equity', 'pct', 'log']


@dataclass
class RewardCalculator:
    """
    獎勵計算策略，適用於交易環境。
    
    核心設計理念：
    1. 鼓勵持倉，大幅懲罰頻繁交易
    2. 獎勵權益增長，同時考慮風險控制
    3. 避免爆倉/破產

    參數:
        mode: 獎勵模式。選項:
            - 'delta_equity': new_equity - last_equity (絕對浮動盈虧)
            - 'pct': (new_equity / last_equity - 1)
            - 'log': log(new_equity / last_equity)（推薦：穩定、對稱）
        scale: 將獎勵乘以此比例以穩定訓練
        failure_penalty: 非資料用盡終止的扣分（爆倉/破產）
        trade_penalty: 每次交易（開倉/加倉/減倉）的固定懲罰
        turnover_penalty: 倉位變動比例懲罰係數（基於變動量）
        holding_reward: 持有倉位的時間獎勵（每步給予小額獎勵）
        profitable_holding_bonus: 持有獲利倉位的額外獎勵係數
        margin_buffer_penalty: 保證金緩衝不足懲罰係數
        safe_zone_threshold: 保證金安全區閾值
    """
    mode: RewardMode = 'log'
    scale: float = 1000.0  # 大幅提高，讓權益增長信號更強
    failure_penalty: float = 10000.0  # 極高懲罰，確保避免爆倉
    trade_penalty: float = 5.0  # 每次交易固定懲罰（抑制頻繁交易）
    turnover_penalty: float = 50.0  # 大幅提高倉位變動懲罰
    holding_reward: float = 0.1  # 持倉時的基礎時間獎勵
    profitable_holding_bonus: float = 2.0  # 持有獲利倉位的額外係數
    margin_buffer_penalty: float = 100.0  # 保證金不足懲罰
    safe_zone_threshold: float = 0.6  # 提高安全區閾值

    
    def compute(
        self, *,
        last_equity: float,
        new_equity: float,
        done: bool = False,
        termination_reason: str | None = None,
        margin_buffer: float | None = None,
        position_change: float | None = None,
        has_position: bool = False,
        unrealized_pnl: float | None = None,
        traded: bool = False,
    ) -> float:
        '''
        計算獎勵（簡化版，專注於：減少交易、鼓勵持倉、保護資金）
        
        Args:
            last_equity: 上一步的權益
            new_equity: 當前權益
            done: 是否終止
            termination_reason: 終止原因
            margin_buffer: 保證金緩衝比例（0=即將強平，1=安全）
            position_change: 倉位變動絕對值（用於換手懲罰）
            has_position: 是否持有倉位（用於持倉獎勵）
            unrealized_pnl: 未實現損益（用於持倉獎勵）
            traded: 本步是否發生交易（用於交易次數懲罰）
            
        Returns:
            float: 塑形後的獎勵值
        '''
        # 1. 基礎獎勵：權益變化（使用 log 模式，穩定且對稱）
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

        # 2. 交易懲罰：每次交易都給固定懲罰（強力抑制頻繁交易）
        if traded and self.trade_penalty > 0:
            shaped -= float(self.trade_penalty)

        # 3. 倉位變動懲罰：基於變動量的額外懲罰（雙重抑制交易）
        if position_change is not None and self.turnover_penalty > 0:
            # position_change 是絕對值，需要正規化
            # 假設 position_change 是以 BTC 數量計算
            # 轉為相對於權益的比例懲罰
            if last_equity > 0:
                turnover_ratio = abs(position_change) / (last_equity / 50000.0)  # 假設 BTC 約 50000
                shaped -= float(self.turnover_penalty * turnover_ratio)

        # 4. 持倉獎勵：有倉位時給予時間獎勵（鼓勵持有）
        if has_position and self.holding_reward > 0:
            shaped += float(self.holding_reward)
            
            # 額外獎勵：如果持有獲利倉位，額外加成
            if unrealized_pnl is not None and unrealized_pnl > 0:
                pnl_ratio = unrealized_pnl / max(last_equity, 1.0)
                shaped += float(self.profitable_holding_bonus * min(pnl_ratio, 0.05))

        # 5. 保證金安全防護：低緩衝時給予懲罰（避免接近強平）
        if margin_buffer is not None and self.margin_buffer_penalty > 0:
            if margin_buffer < self.safe_zone_threshold:
                if margin_buffer >= 0.3:
                    # 警戒區：線性懲罰
                    deficit = float(self.safe_zone_threshold - margin_buffer)
                    shaped -= float(self.margin_buffer_penalty * deficit)
                else:
                    # 危險區：指數懲罰
                    deficit = float(0.3 - margin_buffer)
                    shaped -= float(self.margin_buffer_penalty * 10.0 * (deficit ** 2))

        # 6. 終止懲罰：爆倉/破產給予極大懲罰（確保避免失敗）
        if done and termination_reason is not None and termination_reason != 'data_exhausted':
            shaped -= float(self.failure_penalty)

        return shaped


