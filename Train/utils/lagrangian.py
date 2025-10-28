"""
Lagrangian 控制器（用於約束優化）

實現在線 Lagrangian 乘子更新（梯度上升），確保約束滿足：
- E[cost_prob] ≤ δ（機率違規約束）
- CVaR_α[loss] ≤ b_cvar（尾損約束）

參考：
- Constrained Policy Optimization (CPO)
- SAC-Lagrangian
- CMDP with Lagrangian Methods
"""

from __future__ import annotations
from typing import Dict
import numpy as np


class LagrangianController:
    """
    Lagrangian 乘子控制器（在線更新）
    
    使用梯度上升更新 λ，確保約束滿足：
        λ ← λ + lr * (constraint_violation)
        λ ← max(0, λ)  # 投影到非負空間
    
    Args:
        lambda_prob_init: 機率違規 λ 初始值
        lambda_cvar_init: CVaR 尾損 λ 初始值
        lr_prob: 機率違規 λ 學習率
        lr_cvar: CVaR 尾損 λ 學習率
        target_prob: 機率違規目標上界（δ）
        target_cvar: CVaR 尾損目標上界（b_cvar）
        lambda_min: λ 最小值（避免為 0）
        lambda_max: λ 最大值（避免過大）
    """
    
    def __init__(
        self,
        *,
        lambda_prob_init: float = 1.0,
        lambda_cvar_init: float = 1.0,
        lambda_risk_init: float = 0.0,
        lambda_struct_init: float = 0.0,
        lambda_step_cost_init: float = 0.0,
        lambda_trade_count_init: float = 0.0,
        lambda_entry_init: float = 0.0,
        lambda_drawdown_init: float = 0.0,
        lambda_stop_streak_init: float = 0.0,
        lr_prob: float = 0.0001,
        lr_cvar: float = 0.0001,
        lr_risk: float = 0.01,
        lr_struct: float = 0.01,
        lr_step_cost: float = 0.01,
        lr_trade_count: float = 0.01,
        lr_entry: float = 0.01,
        lr_drawdown: float = 0.01,
        lr_stop_streak: float = 0.01,
        target_prob: float = 0.03,  # 3% 止損/強平率
        target_cvar: float = 0.01,  # 1% CVaR 上界
        target_risk: float = 0.0,
        target_struct: float = 0.0,
        target_step_cost: float = 0.0,
        target_trade_count: float = 0.0,
        target_entry_cost: float = 0.0,
        target_drawdown_cost: float = 0.0,
        target_stop_streak_cost: float = 0.0,
        lambda_min: float = 0.00001,
        lambda_max: float = 100.0,
    ) -> None:
        self.lambda_prob = float(lambda_prob_init)
        self.lambda_cvar = float(lambda_cvar_init)
        self.lambda_risk = float(lambda_risk_init)
        self.lambda_struct = float(lambda_struct_init)
        self.lambda_step_cost = float(lambda_step_cost_init)
        self.lambda_trade_count = float(lambda_trade_count_init)
        self.lambda_entry = float(lambda_entry_init)
        self.lambda_drawdown = float(lambda_drawdown_init)
        self.lambda_stop_streak = float(lambda_stop_streak_init)
        self.lr_prob = float(lr_prob)
        self.lr_cvar = float(lr_cvar)
        self.lr_risk = float(lr_risk)
        self.lr_struct = float(lr_struct)
        self.lr_step_cost = float(lr_step_cost)
        self.lr_trade_count = float(lr_trade_count)
        self.lr_entry = float(lr_entry)
        self.lr_drawdown = float(lr_drawdown)
        self.lr_stop_streak = float(lr_stop_streak)
        self.target_prob = float(target_prob)
        self.target_cvar = float(target_cvar)
        self.target_risk = float(target_risk)
        self.target_struct = float(target_struct)
        self.target_step_cost = float(target_step_cost)
        self.target_trade_count = float(target_trade_count)
        self.target_entry_cost = float(target_entry_cost)
        self.target_drawdown_cost = float(target_drawdown_cost)
        self.target_stop_streak_cost = float(target_stop_streak_cost)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        
        # 滾動平均（用於穩定更新）
        self.ema_prob = 0.0
        self.ema_cvar = 0.0
        self.ema_risk = 0.0
        self.ema_struct = 0.0
        self.ema_step_cost = 0.0
        self.ema_trade_count = 0.0
        self.ema_entry_cost = 0.0
        self.ema_drawdown_cost = 0.0
        self.ema_stop_streak_cost = 0.0
        self.ema_alpha = 0.95
        
        # 統計
        self.total_updates = 0
    
    def update(
        self,
        *,
        cost_prob_batch: np.ndarray,
        loss_cvar_batch: np.ndarray,
        risk_cost_batch: np.ndarray | None = None,
        struct_cost_batch: np.ndarray | None = None,
        step_cost_batch: np.ndarray | None = None,
        trade_cost_batch: np.ndarray | None = None,
        entry_cost_batch: np.ndarray | None = None,
        drawdown_cost_batch: np.ndarray | None = None,
        stop_streak_cost_batch: np.ndarray | None = None,
    ) -> Dict[str, float]:
        """
        更新 Lagrangian 乘子（基於批次成本樣本）
        
        Args:
            cost_prob_batch: 機率違規批次（shape: [batch_size, 1]）
            loss_cvar_batch: CVaR 尾損批次（shape: [batch_size, 1]）
            
        Returns:
            更新統計字典
        """
        # 計算批次平均成本
        mean_cost_prob = float(np.mean(cost_prob_batch))
        mean_loss_cvar = float(np.mean(loss_cvar_batch))
        mean_risk = float(np.mean(risk_cost_batch)) if risk_cost_batch is not None else 0.0
        mean_struct = float(np.mean(struct_cost_batch)) if struct_cost_batch is not None else 0.0
        mean_step_cost = float(np.mean(step_cost_batch)) if step_cost_batch is not None else 0.0
        mean_trade = float(np.mean(trade_cost_batch)) if trade_cost_batch is not None else 0.0
        mean_entry = float(np.mean(entry_cost_batch)) if entry_cost_batch is not None else 0.0
        mean_drawdown = float(np.mean(drawdown_cost_batch)) if drawdown_cost_batch is not None else 0.0
        mean_stop_streak = float(np.mean(stop_streak_cost_batch)) if stop_streak_cost_batch is not None else 0.0
        
        # 更新 EMA
        self.ema_prob = self.ema_alpha * self.ema_prob + (1 - self.ema_alpha) * mean_cost_prob
        self.ema_cvar = self.ema_alpha * self.ema_cvar + (1 - self.ema_alpha) * mean_loss_cvar
        self.ema_risk = self.ema_alpha * self.ema_risk + (1 - self.ema_alpha) * mean_risk
        self.ema_struct = self.ema_alpha * self.ema_struct + (1 - self.ema_alpha) * mean_struct
        self.ema_step_cost = self.ema_alpha * self.ema_step_cost + (1 - self.ema_alpha) * mean_step_cost
        self.ema_trade_count = self.ema_alpha * self.ema_trade_count + (1 - self.ema_alpha) * mean_trade
        self.ema_entry_cost = self.ema_alpha * self.ema_entry_cost + (1 - self.ema_alpha) * mean_entry
        self.ema_drawdown_cost = self.ema_alpha * self.ema_drawdown_cost + (1 - self.ema_alpha) * mean_drawdown
        self.ema_stop_streak_cost = self.ema_alpha * self.ema_stop_streak_cost + (1 - self.ema_alpha) * mean_stop_streak
        
        # 計算約束違規（使用 EMA 穩定更新）
        violation_prob = self.ema_prob - self.target_prob
        violation_cvar = self.ema_cvar - self.target_cvar
        violation_risk = self.ema_risk - self.target_risk
        violation_struct = self.ema_struct - self.target_struct
        violation_step_cost = self.ema_step_cost - self.target_step_cost
        violation_trade_count = self.ema_trade_count - self.target_trade_count
        violation_entry_cost = self.ema_entry_cost - self.target_entry_cost
        violation_drawdown_cost = self.ema_drawdown_cost - self.target_drawdown_cost
        violation_stop_streak_cost = self.ema_stop_streak_cost - self.target_stop_streak_cost
        
        # 梯度上升更新 λ（投影到 [lambda_min, lambda_max]）
        self.lambda_prob += self.lr_prob * violation_prob
        self.lambda_prob = float(np.clip(self.lambda_prob, self.lambda_min, self.lambda_max))
        
        self.lambda_cvar += self.lr_cvar * violation_cvar
        self.lambda_cvar = float(np.clip(self.lambda_cvar, self.lambda_min, self.lambda_max))

        if risk_cost_batch is not None:
            self.lambda_risk += self.lr_risk * violation_risk
            self.lambda_risk = float(np.clip(self.lambda_risk, self.lambda_min, self.lambda_max))

        if struct_cost_batch is not None:
            self.lambda_struct += self.lr_struct * violation_struct
            self.lambda_struct = float(np.clip(self.lambda_struct, self.lambda_min, self.lambda_max))
        if step_cost_batch is not None:
            self.lambda_step_cost += self.lr_step_cost * violation_step_cost
            self.lambda_step_cost = float(np.clip(self.lambda_step_cost, self.lambda_min, self.lambda_max))
        if trade_cost_batch is not None:
            self.lambda_trade_count += self.lr_trade_count * violation_trade_count
            self.lambda_trade_count = float(np.clip(self.lambda_trade_count, self.lambda_min, self.lambda_max))
        if entry_cost_batch is not None:
            self.lambda_entry += self.lr_entry * violation_entry_cost
            self.lambda_entry = float(np.clip(self.lambda_entry, self.lambda_min, self.lambda_max))
        if drawdown_cost_batch is not None:
            self.lambda_drawdown += self.lr_drawdown * violation_drawdown_cost
            self.lambda_drawdown = float(np.clip(self.lambda_drawdown, self.lambda_min, self.lambda_max))
        if stop_streak_cost_batch is not None:
            self.lambda_stop_streak += self.lr_stop_streak * violation_stop_streak_cost
            self.lambda_stop_streak = float(np.clip(self.lambda_stop_streak, self.lambda_min, self.lambda_max))
        
        self.total_updates += 1
        
        return {
            'lambda_prob': self.lambda_prob,
            'lambda_cvar': self.lambda_cvar,
            'lambda_risk': self.lambda_risk,
            'lambda_struct': self.lambda_struct,
            'lambda_step_cost': self.lambda_step_cost,
            'lambda_trade_count': self.lambda_trade_count,
            'lambda_entry': self.lambda_entry,
            'lambda_drawdown': self.lambda_drawdown,
            'lambda_stop_streak': self.lambda_stop_streak,
            'mean_cost_prob': mean_cost_prob,
            'mean_loss_cvar': mean_loss_cvar,
            'mean_risk_cost': mean_risk,
            'mean_struct_cost': mean_struct,
            'mean_step_cost': mean_step_cost,
            'mean_trade_cost': mean_trade,
            'mean_entry_cost': mean_entry,
            'mean_drawdown_cost': mean_drawdown,
            'mean_stop_streak_cost': mean_stop_streak,
            'ema_cost_prob': self.ema_prob,
            'ema_loss_cvar': self.ema_cvar,
            'ema_risk_cost': self.ema_risk,
            'ema_struct_cost': self.ema_struct,
            'ema_step_cost': self.ema_step_cost,
            'ema_trade_cost': self.ema_trade_count,
            'ema_entry_cost': self.ema_entry_cost,
            'ema_drawdown_cost': self.ema_drawdown_cost,
            'ema_stop_streak_cost': self.ema_stop_streak_cost,
            'violation_prob': violation_prob,
            'violation_cvar': violation_cvar,
            'violation_risk': violation_risk,
            'violation_struct': violation_struct,
            'violation_step_cost': violation_step_cost,
            'violation_trade_count': violation_trade_count,
            'violation_entry_cost': violation_entry_cost,
            'violation_drawdown_cost': violation_drawdown_cost,
            'violation_stop_streak_cost': violation_stop_streak_cost,
            'total_updates': self.total_updates,
        }
    
    def get_lambdas(self) -> Dict[str, float]:
        """獲取當前 Lagrangian 乘子"""
        return {
            'lambda_prob': self.lambda_prob,
            'lambda_cvar': self.lambda_cvar,
            'lambda_risk': self.lambda_risk,
            'lambda_struct': self.lambda_struct,
            'lambda_step_cost': self.lambda_step_cost,
            'lambda_trade_count': self.lambda_trade_count,
            'lambda_entry': self.lambda_entry,
            'lambda_drawdown': self.lambda_drawdown,
            'lambda_stop_streak': self.lambda_stop_streak,
        }
    
    def set_learning_rates(self, lr_prob: float, lr_cvar: float, lr_risk: float, lr_struct: float, lr_step_cost: float, lr_trade_count: float, lr_entry: float, lr_drawdown: float, lr_stop_streak: float) -> None:
        """動態調整學習率"""
        self.lr_prob = float(lr_prob)
        self.lr_cvar = float(lr_cvar)
        self.lr_risk = float(lr_risk)
        self.lr_struct = float(lr_struct)
        self.lr_step_cost = float(lr_step_cost)
        self.lr_trade_count = float(lr_trade_count)
        self.lr_entry = float(lr_entry)
        self.lr_drawdown = float(lr_drawdown)
        self.lr_stop_streak = float(lr_stop_streak)
    
    def reset(
        self,
        lambda_prob: float = 1.0,
        lambda_cvar: float = 1.0,
        lambda_risk: float = 1.0,
        lambda_struct: float = 1.0,
        lambda_step_cost: float = 1.0,
        lambda_trade_count: float = 1.0,
        lambda_entry: float = 1.0,
        lambda_drawdown: float = 1.0,
        lambda_stop_streak: float = 1.0,
    ) -> None:
        """重置 Lagrangian 乘子（用於新回合或調試）"""
        self.lambda_prob = float(lambda_prob)
        self.lambda_cvar = float(lambda_cvar)
        self.lambda_risk = float(lambda_risk)
        self.lambda_struct = float(lambda_struct)
        self.lambda_step_cost = float(lambda_step_cost)
        self.lambda_trade_count = float(lambda_trade_count)
        self.lambda_entry = float(lambda_entry)
        self.lambda_drawdown = float(lambda_drawdown)
        self.lambda_stop_streak = float(lambda_stop_streak)
        self.ema_prob = 0.0
        self.ema_cvar = 0.0
        self.ema_risk = 0.0
        self.ema_struct = 0.0
        self.ema_step_cost = 0.0
        self.ema_trade_count = 0.0
        self.ema_entry_cost = 0.0
        self.ema_drawdown_cost = 0.0
        self.ema_stop_streak_cost = 0.0
        self.total_updates = 0
    
    def get_stats(self) -> Dict[str, float]:
        """獲取控制器統計信息"""
        return {
            'lambda_prob': self.lambda_prob,
            'lambda_cvar': self.lambda_cvar,
            'lambda_risk': self.lambda_risk,
            'lambda_struct': self.lambda_struct,
            'lambda_step_cost': self.lambda_step_cost,
            'lambda_trade_count': self.lambda_trade_count,
            'lambda_entry': self.lambda_entry,
            'lambda_drawdown': self.lambda_drawdown,
            'lambda_stop_streak': self.lambda_stop_streak,
            'ema_cost_prob': self.ema_prob,
            'ema_loss_cvar': self.ema_cvar,
            'ema_risk_cost': self.ema_risk,
            'ema_struct_cost': self.ema_struct,
            'ema_step_cost': self.ema_step_cost,
            'ema_trade_cost': self.ema_trade_count,
            'ema_entry_cost': self.ema_entry_cost,
            'ema_drawdown_cost': self.ema_drawdown_cost,
            'ema_stop_streak_cost': self.ema_stop_streak_cost,
            'target_prob': self.target_prob,
            'target_cvar': self.target_cvar,
            'target_risk': self.target_risk,
            'target_struct': self.target_struct,
            'target_step_cost': self.target_step_cost,
            'target_trade_count': self.target_trade_count,
            'target_entry_cost': self.target_entry_cost,
            'target_drawdown_cost': self.target_drawdown_cost,
            'target_stop_streak_cost': self.target_stop_streak_cost,
            'total_updates': self.total_updates,
        }

