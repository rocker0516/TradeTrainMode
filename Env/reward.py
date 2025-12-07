"""
Log Return Reward Calculator (純主線版本)

設計目標：
- 主線只保留資產對數報酬與終局懲罰。
- 行為/風險懲罰移至成本線 (Lagrangian) 處理。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class RewardCalculator:
    """
    主線獎勵計算器：純 Log Return + 終局懲罰
    """
    c_liq: float = 10.0
    fee_limit_penalty: float = 2.0
    
    def compute(
        self,
        *,
        last_equity: float,
        new_equity: float,
        done: bool = False,
        termination_reason: str | None = None,
        episode_steps: int | None = None,
        episode_max_steps: int | None = None,
        **kwargs
    ) -> float:
        """
        計算主線獎勵：僅包含 log return 以及終局懲罰
        """
        # 1. 計算基礎 Log Return
        safe_last = max(last_equity, 1e-8)
        safe_new = max(new_equity, 1e-8)
        
        log_ret = np.log(safe_new / safe_last)
        
        reward = float(log_ret)
        
        # 2. 處理終局 (Liquidation / Balance Insufficient)
        if done and termination_reason in ('liq_triggered', 'balance_insufficient'):
            # 應用強平懲罰公式
            # r_T -= C_liq * (1 + (T_max - T) / T_max)
            
            curr_step = episode_steps if episode_steps is not None else 0
            max_step = episode_max_steps if episode_max_steps is not None and episode_max_steps > 0 else 1000
            
            # 剩餘時間比例 (越早死，剩越多，懲罰越重)
            # (T_max - T) / T_max
            # 若 curr_step >= max_step, term = 0
            remaining_ratio = max(0.0, float(max_step - curr_step)) / float(max_step)
            
            penalty = self.c_liq * (1.0 + remaining_ratio)
            
            reward -= penalty
            
        # Fee limit termination -> apply explicit penalty
        if done and termination_reason == 'fee_limit':
            penalty = self.fee_limit_penalty
            reward -= penalty
    
        return reward
    
    def get_info(self) -> dict:
        return {
            'type': 'log_return_only',
            'c_liq': self.c_liq
        }

# 工廠函數
def create_default_calculator(c_liq: float = 10.0, fee_limit_penalty: float = 2.0) -> RewardCalculator:
    return RewardCalculator(c_liq=c_liq, fee_limit_penalty=fee_limit_penalty)
