"""
Log Return Reward Calculator (Main Line Only)

Concepts:
- Reward = Log Return of Equity: ln(E_t / E_{t-1})
- Terminal Penalty (Liquidation/Bankruptcy): Extra penalty based on remaining time.
  r_T -= C_liq * (1 + (T_max - T) / T_max)
- Turnover Penalty (Action Cost): Explicit penalty for position changes to encourage sparsity.
  r_t -= alpha_turn * |Delta_Pos_Norm|
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class RewardCalculator:
    """
    主線獎勵計算器：Log Return + Turnover Penalty
    """
    # 強平懲罰係數 C_liq
    c_liq: float = 10.0
    # 換手懲罰係數 alpha_turn
    turnover_penalty: float = 0.0
    
    def compute(
        self,
        *,
        last_equity: float,
        new_equity: float,
        done: bool = False,
        termination_reason: str | None = None,
        episode_steps: int | None = None,
        episode_max_steps: int | None = None,
        position_change_norm: float = 0.0, # Normalize position change (0~1 usually)
        **kwargs
    ) -> float:
        """
        計算 Log Return 獎勵
        
        r = ln(new_equity / last_equity)
        r -= turnover_penalty * position_change_norm
        if liquidation: r -= C_liq * (1 + (T_max - T)/T_max)
        """
        # 1. 計算基礎 Log Return
        # 保護 log(0) 或負值 (雖然 equity 應該 > 0，但為了數值穩定)
        safe_last = max(last_equity, 1e-8)
        safe_new = max(new_equity, 1e-8)
        
        # 若本步破產 (new_equity <= min_balance or ~0)，log return 會是負大值
        # 但為了避免 -inf，我們用 safe_new 截斷，這本身就是一個懲罰
        log_ret = np.log(safe_new / safe_last)
        
        reward = float(log_ret)

        # 2. 加上換手懲罰 (Turnover Penalty)
        if self.turnover_penalty > 0 and position_change_norm > 0:
            reward -= self.turnover_penalty * position_change_norm
        
        # 3. 處理終局 (Liquidation / Balance Insufficient)
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
            
        # 正常結束 (Data Exhausted) 或 止損 (Stop Loss 不終止) 不加額外懲罰，
        # 僅反映在 Log Return (止損會導致 equity 下降，自然產生負 log return)

        return reward
    
    def get_info(self) -> dict:
        return {
            'type': 'log_return_plus_turnover',
            'c_liq': self.c_liq,
            'turnover_penalty': self.turnover_penalty
        }

# 工廠函數
def create_default_calculator(turnover_penalty: float = 0.0) -> RewardCalculator:
    return RewardCalculator(turnover_penalty=turnover_penalty)
