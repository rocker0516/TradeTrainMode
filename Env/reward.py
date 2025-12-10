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
    主線獎勵計算器：僅使用「淨」對數報酬。
    - Equity 已內含手續費、滑點、利息。
    - 終局懲罰改移至成本線（death cost）處理，不再在主線扣分。
    """
    c_liq: float = 0.0
    fee_limit_penalty: float = 0.0
    
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
        計算主線獎勵：僅包含淨 log return
        """
        # 1. 計算基礎 Log Return
        safe_last = max(last_equity, 1e-8)
        safe_new = max(new_equity, 1e-8)
        
        log_ret = np.log(safe_new / safe_last)
        
        reward = float(log_ret)
        return reward
    
    def get_info(self) -> dict:
        return {
            'type': 'log_return_only',
            'c_liq': self.c_liq
        }

# 工廠函數
def create_default_calculator(c_liq: float = 10.0, fee_limit_penalty: float = 2.0) -> RewardCalculator:
    # 兼容舊接口，但默認不再使用終局懲罰
    return RewardCalculator(c_liq=0.0, fee_limit_penalty=0.0)
