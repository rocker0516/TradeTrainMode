"""
環境 Wrapper（用於 SAC-Lagrangian + RUDDER）

包含：
1. RiskPenaltyWrapper：外層 Lagrangian 懲罰 wrapper（當 HER 開啟時使用）
2. 其他輔助 wrapper
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np
import gymnasium as gym


class RiskPenaltyWrapper(gym.Wrapper):
    """
    風險懲罰 Wrapper（外層 Lagrangian 約束）
    
    當訓練端啟用 HER 或不使用獨立成本頭時，透過此 wrapper 施加懲罰：
        reward' = reward - λ_prob * cost_prob - λ_cvar * loss_cvar_sample
    
    λ 由外部 Lagrangian 控制器動態更新（在線學習）
    
    Args:
        env: 環境實例
        lambda_prob: 機率違規 Lagrangian 乘子（初始值）
        lambda_cvar: CVaR 尾損 Lagrangian 乘子（初始值）
    """
    
    def __init__(
        self,
        env: gym.Env,
        lambda_prob: float = 0.1,
        lambda_cvar: float = 0.1,
        lambda_risk: float = 0.0,
        lambda_struct: float = 0.0,
        lambda_step_cost: float = 0.0,
        lambda_trade_count: float = 0.0,
        lambda_entry: float = 0.0,
        lambda_drawdown: float = 0.0,
        lambda_stop_streak: float = 0.0,
    ) -> None:
        super().__init__(env)
        self.lambda_prob = float(lambda_prob)
        self.lambda_cvar = float(lambda_cvar)
        self.lambda_risk = float(lambda_risk)
        self.lambda_struct = float(lambda_struct)
        self.lambda_step_cost = float(lambda_step_cost)
        self.lambda_trade_count = float(lambda_trade_count)
        self.lambda_entry = float(lambda_entry)
        self.lambda_drawdown = float(lambda_drawdown)
        self.lambda_stop_streak = float(lambda_stop_streak)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """執行步驟並施加風險懲罰"""
        obs, reward, done, truncated, info = self.env.step(action)
        
        # 從 info 讀取成本
        cost_prob = info.get('cost_prob', 0.0)
        loss_cvar_sample = info.get('loss_cvar_sample', 0.0)
        risk_cost = info.get('constraint_risk_cost', 0.0)
        struct_cost = info.get('constraint_struct_cost', 0.0)
        step_cost = info.get('step_cost', 0.0)
        trade_cost = info.get('trade_count_cost', 0.0)
        entry_cost = info.get('entry_cost', 0.0)
        drawdown_cost = info.get('drawdown_cost', 0.0)
        stop_streak_cost = info.get('stop_loss_streak_cost', 0.0)
        
        risk_penalty = self.lambda_risk * risk_cost
        struct_penalty = self.lambda_struct * struct_cost
        step_penalty = self.lambda_step_cost * step_cost
        trade_penalty = self.lambda_trade_count * trade_cost
        entry_penalty = self.lambda_entry * entry_cost
        drawdown_penalty = self.lambda_drawdown * drawdown_cost
        stop_streak_penalty = self.lambda_stop_streak * stop_streak_cost
        base_penalty = self.lambda_prob * cost_prob + self.lambda_cvar * loss_cvar_sample
        penalty = (
            base_penalty
            + risk_penalty
            + struct_penalty
            + step_penalty
            + trade_penalty
            + entry_penalty
            + drawdown_penalty
            + stop_streak_penalty
        )
        reward_penalized = reward - penalty
        
        # 記錄原始 reward 與懲罰（用於監控）
        info['reward_original'] = float(reward)
        info['reward_penalty'] = float(penalty)
        info['reward_penalized'] = float(reward_penalized)
        info['reward_penalty_cost_prob'] = float(base_penalty)
        info['reward_penalty_risk'] = float(risk_penalty)
        info['reward_penalty_struct'] = float(struct_penalty)
        info['reward_penalty_step_cost'] = float(step_penalty)
        info['reward_penalty_trade_count'] = float(trade_penalty)
        info['reward_penalty_entry'] = float(entry_penalty)
        info['reward_penalty_drawdown'] = float(drawdown_penalty)
        info['reward_penalty_stop_streak'] = float(stop_streak_penalty)
        
        return obs, reward_penalized, done, truncated, info
    
    def update_lambdas(
        self,
        *,
        lambda_prob: float,
        lambda_cvar: float,
        lambda_risk: Optional[float] = None,
        lambda_struct: Optional[float] = None,
        lambda_step_cost: Optional[float] = None,
        lambda_trade_count: Optional[float] = None,
        lambda_entry: Optional[float] = None,
        lambda_drawdown: Optional[float] = None,
        lambda_stop_streak: Optional[float] = None,
    ) -> None:
        """更新 Lagrangian 乘子（由外部控制器調用）"""
        self.lambda_prob = float(lambda_prob)
        self.lambda_cvar = float(lambda_cvar)
        if lambda_risk is not None:
            self.lambda_risk = float(lambda_risk)
        if lambda_struct is not None:
            self.lambda_struct = float(lambda_struct)
        if lambda_step_cost is not None:
            self.lambda_step_cost = float(lambda_step_cost)
        if lambda_trade_count is not None:
            self.lambda_trade_count = float(lambda_trade_count)
        if lambda_entry is not None:
            self.lambda_entry = float(lambda_entry)
        if lambda_drawdown is not None:
            self.lambda_drawdown = float(lambda_drawdown)
        if lambda_stop_streak is not None:
            self.lambda_stop_streak = float(lambda_stop_streak)
    
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


class NormalizeRewardWrapper(gym.Wrapper):
    """
    獎勵正規化 Wrapper（可選，用於穩定訓練）
    
    使用滾動均值與標準差正規化 reward。
    
    Args:
        env: 環境實例
        gamma: 折扣因子（用於滾動更新）
        epsilon: 數值穩定常數
    """
    
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 滾動統計
        self.return_rms_mean = 0.0
        self.return_rms_var = 1.0
        self.return_val = 0.0
        self.count = 0
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """執行步驟並正規化 reward"""
        obs, reward, done, truncated, info = self.env.step(action)
        
        # 更新滾動 return
        self.return_val = reward + self.gamma * self.return_val * (1 - done)
        
        # 更新滾動統計
        self.count += 1
        delta = self.return_val - self.return_rms_mean
        self.return_rms_mean += delta / self.count
        self.return_rms_var += delta * (self.return_val - self.return_rms_mean)
        
        # 正規化 reward
        std = np.sqrt(self.return_rms_var / max(1, self.count - 1) + self.epsilon)
        reward_normalized = reward / max(std, self.epsilon)
        
        # 記錄
        info['reward_normalized'] = float(reward_normalized)
        info['reward_original'] = float(reward)
        
        if done:
            self.return_val = 0.0
        
        return obs, reward_normalized, done, truncated, info
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """重置環境"""
        self.return_val = 0.0
        return self.env.reset(**kwargs)


class InfoLoggerWrapper(gym.Wrapper):
    """
    Info 記錄 Wrapper（用於監控與調試）
    
    記錄關鍵 info 欄位的統計信息（episode 層級）
    """
    
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.reset_episode_stats()
    
    def reset_episode_stats(self) -> None:
        """重置 episode 統計"""
        self.episode_stats = {
            'entry_count': 0,
            'exit_count': 0,
            'reduce_count': 0,
            'stop_loss_count': 0,
            'liq_count': 0,
            'forced_close_count': 0,
            'total_outcome': 0.0,
            'total_step_cost': 0.0,
            'cost_prob_sum': 0.0,
            'risk_cost_sum': 0.0,
            'struct_cost_sum': 0.0,
            'trade_cost_sum': 0.0,
            'entry_cost_sum': 0.0,
            'drawdown_cost_sum': 0.0,
            'stop_streak_cost_sum': 0.0,
            'steps': 0,
        }
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """執行步驟並記錄統計"""
        obs, reward, done, truncated, info = self.env.step(action)
        
        # 更新統計
        self.episode_stats['steps'] += 1
        if info.get('is_entry', False):
            self.episode_stats['entry_count'] += 1
        if info.get('is_exit', False):
            self.episode_stats['exit_count'] += 1
            exit_reason = info.get('exit_reason', '')
            if exit_reason == 'stop_loss':
                self.episode_stats['stop_loss_count'] += 1
            elif exit_reason == 'liq':
                self.episode_stats['liq_count'] += 1
            elif exit_reason == 'forced_close_on_done':
                self.episode_stats['forced_close_count'] += 1
        if info.get('is_reduce', False):
            self.episode_stats['reduce_count'] += 1
        
        self.episode_stats['total_outcome'] += info.get('outcome_delta_to_entry', 0.0)
        self.episode_stats['total_step_cost'] += info.get('step_cost', 0.0)
        self.episode_stats['cost_prob_sum'] += info.get('cost_prob', 0.0)
        self.episode_stats['risk_cost_sum'] += info.get('constraint_risk_cost', 0.0)
        self.episode_stats['struct_cost_sum'] += info.get('constraint_struct_cost', 0.0)
        self.episode_stats['trade_cost_sum'] += info.get('trade_count_cost', 0.0)
        self.episode_stats['entry_cost_sum'] += info.get('entry_cost', 0.0)
        self.episode_stats['drawdown_cost_sum'] += info.get('drawdown_cost', 0.0)
        self.episode_stats['stop_streak_cost_sum'] += info.get('stop_loss_streak_cost', 0.0)
        
        # 在 episode 結束時，將統計加入 info
        if done:
            info['episode_stats'] = self.episode_stats.copy()
            # 計算平均值
            steps = max(1, self.episode_stats['steps'])
            info['episode_stats']['avg_outcome'] = self.episode_stats['total_outcome'] / max(1, self.episode_stats['exit_count'])
            info['episode_stats']['avg_step_cost'] = self.episode_stats['total_step_cost'] / steps
            info['episode_stats']['cost_prob_rate'] = self.episode_stats['cost_prob_sum'] / steps
            info['episode_stats']['risk_cost_avg'] = self.episode_stats['risk_cost_sum'] / steps
            info['episode_stats']['struct_cost_avg'] = self.episode_stats['struct_cost_sum'] / steps
            info['episode_stats']['trade_cost_rate'] = self.episode_stats['trade_cost_sum'] / steps
            info['episode_stats']['entry_cost_rate'] = self.episode_stats['entry_cost_sum'] / steps
            info['episode_stats']['drawdown_cost_avg'] = self.episode_stats['drawdown_cost_sum'] / steps
            info['episode_stats']['stop_streak_cost_avg'] = self.episode_stats['stop_streak_cost_sum'] / steps
            
            # 重置統計
            self.reset_episode_stats()
        
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """重置環境與統計"""
        self.reset_episode_stats()
        return self.env.reset(**kwargs)

