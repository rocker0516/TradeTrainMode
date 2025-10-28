"""
RUDDER Replay Buffer（支援 outcome 回填機制）

核心功能：
1. 追蹤每筆交易的進場步（trade_id → entry_step_idx）
2. 當出場事件發生時，將 outcome_delta 回填到進場步
3. 支援部分平倉（reduce）與完全平倉（close/stop_loss/liq/forced_close_on_done）
4. 相容 SAC-Lagrangian 的成本約束

參考：
- RUDDER: Return Decomposition for Delayed Rewards
- Constrained Policy Optimization with RUDDER
"""

from __future__ import annotations
from typing import Dict, Optional, List
import numpy as np
import torch
from collections import defaultdict


class RUDDERReplayBuffer:
    """
    RUDDER 經驗回放緩衝區（支援 outcome 回填）
    
    Args:
        capacity: 緩衝區容量
        observation_shape: 觀察空間形狀
        action_dim: 動作維度
        device: 訓練設備
    """
    
    def __init__(
        self,
        capacity: int,
        observation_shape: tuple,
        action_dim: int,
        device: str = 'cuda'
    ) -> None:
        """初始化 RUDDER 緩衝區"""
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        # 預分配記憶體（基礎 transition）
        self.states = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)  # shaping reward
        self.next_states = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        # RUDDER 專用欄位
        self.outcome_deltas = np.zeros((capacity, 1), dtype=np.float32)  # 回填的 outcome
        self.trade_ids = np.full((capacity,), -1, dtype=np.int32)  # 當前步的 trade_id
        
        # 成本約束欄位（Lagrangian）
        self.step_costs = np.zeros((capacity, 1), dtype=np.float32)  # 手續費 + 滑點
        self.cost_probs = np.zeros((capacity, 1), dtype=np.float32)  # 機率違規（0/1）
        self.loss_cvar_samples = np.zeros((capacity, 1), dtype=np.float32)  # CVaR 尾損樣本
        self.risk_costs = np.zeros((capacity, 1), dtype=np.float32)
        self.struct_costs = np.zeros((capacity, 1), dtype=np.float32)
        self.trade_costs = np.zeros((capacity, 1), dtype=np.float32)
        self.entry_costs = np.zeros((capacity, 1), dtype=np.float32)
        self.drawdown_costs = np.zeros((capacity, 1), dtype=np.float32)
        self.stop_streak_costs = np.zeros((capacity, 1), dtype=np.float32)
        
        # 交易追蹤（用於回填）
        # trade_id → entry_step_idx（在 buffer 中的位置）
        self._trade_entry_map: Dict[int, int] = {}
        
        # 統計
        self._total_outcomes_backfilled = 0
        self._total_forced_closes = 0
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,  # shaping reward only
        next_state: np.ndarray,
        done: bool,
        info: Dict,
    ) -> None:
        """
        添加一條經驗（並處理 RUDDER 回填）
        
        Args:
            state: 當前狀態
            action: 執行的動作
            reward: shaping reward（小額引導）
            next_state: 下一個狀態
            done: 是否結束
            info: 環境 info（包含 RUDDER 協議欄位）
        """
        # 基礎 transition
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        
        # 成本欄位
        self.step_costs[self.position] = info.get('step_cost', 0.0)
        self.cost_probs[self.position] = info.get('cost_prob', 0.0)
        self.loss_cvar_samples[self.position] = info.get('loss_cvar_sample', 0.0)
        self.risk_costs[self.position] = info.get('constraint_risk_cost', 0.0)
        self.struct_costs[self.position] = info.get('constraint_struct_cost', 0.0)
        self.trade_costs[self.position] = info.get('trade_count_cost', 0.0)
        self.entry_costs[self.position] = info.get('entry_cost', 0.0)
        self.drawdown_costs[self.position] = info.get('drawdown_cost', 0.0)
        self.stop_streak_costs[self.position] = info.get('stop_loss_streak_cost', 0.0)
        
        # trade_id（當前步的交易 ID）
        current_trade_id = info.get('trade_id', -1)
        self.trade_ids[self.position] = current_trade_id
        
        # 處理進場事件：註冊 trade_id → entry_step_idx
        if info.get('is_entry', False):
            entered_trade_id = info.get('entered_trade_id', -1)
            if entered_trade_id >= 0:
                self._trade_entry_map[entered_trade_id] = self.position
        
        # 處理出場事件：回填 outcome_delta 到進場步
        if info.get('is_exit', False) or info.get('is_reduce', False):
            exited_trade_id = info.get('exited_trade_id', -1)
            outcome_delta = info.get('outcome_delta_to_entry', 0.0)
            
            if exited_trade_id >= 0 and abs(outcome_delta) > 1e-8:
                # 回填到進場步
                entry_idx = self._trade_entry_map.get(exited_trade_id, None)
                if entry_idx is not None and 0 <= entry_idx < self.capacity:
                    # 累加（支援部分平倉多次回填）
                    self.outcome_deltas[entry_idx] += outcome_delta
                    self._total_outcomes_backfilled += 1
                    
                    # 若為完全平倉，移除映射
                    if info.get('is_exit', False):
                        del self._trade_entry_map[exited_trade_id]
                        # 統計強制平倉
                        if info.get('exit_reason', '') == 'forced_close_on_done':
                            self._total_forced_closes += 1
        
        # 更新位置與大小
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        # 清理過期的 trade_entry_map（避免記憶體洩漏）
        if self.size >= self.capacity:
            self._clean_expired_trade_map()
    
    def _clean_expired_trade_map(self) -> None:
        """清理過期的 trade_entry_map（當 buffer 循環覆蓋時）"""
        # 保留仍在 buffer 中的 trade_id
        valid_trade_ids = set(self.trade_ids[:self.size].tolist())
        expired_keys = [k for k in self._trade_entry_map if k not in valid_trade_ids]
        for k in expired_keys:
            del self._trade_entry_map[k]
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        隨機採樣批次數據（包含 RUDDER 回填的 outcome）
        
        Args:
            batch_size: 批次大小
            
        Returns:
            批次數據字典（包含 outcome_delta、step_cost、cost_prob 等）
        """
        if self.size < batch_size:
            raise ValueError(
                f"緩衝區數據不足：需要 {batch_size}，但只有 {self.size}"
            )
        
        # 隨機採樣索引
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # 轉換為 PyTorch 張量
        batch = {
            'state': torch.FloatTensor(self.states[indices]).to(self.device),
            'action': torch.FloatTensor(self.actions[indices]).to(self.device),
            'reward': torch.FloatTensor(self.rewards[indices]).to(self.device),  # shaping only
            'next_state': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'done': torch.FloatTensor(self.dones[indices]).to(self.device),
            # RUDDER 欄位
            'outcome_delta': torch.FloatTensor(self.outcome_deltas[indices]).to(self.device),
            # 成本約束欄位
            'step_cost': torch.FloatTensor(self.step_costs[indices]).to(self.device),
            'cost_prob': torch.FloatTensor(self.cost_probs[indices]).to(self.device),
            'loss_cvar_sample': torch.FloatTensor(self.loss_cvar_samples[indices]).to(self.device),
            'risk_cost': torch.FloatTensor(self.risk_costs[indices]).to(self.device),
            'struct_cost': torch.FloatTensor(self.struct_costs[indices]).to(self.device),
            'trade_cost': torch.FloatTensor(self.trade_costs[indices]).to(self.device),
            'entry_cost': torch.FloatTensor(self.entry_costs[indices]).to(self.device),
            'drawdown_cost': torch.FloatTensor(self.drawdown_costs[indices]).to(self.device),
            'stop_streak_cost': torch.FloatTensor(self.stop_streak_costs[indices]).to(self.device),
        }
        
        return batch
    
    def __len__(self) -> int:
        """返回當前緩衝區大小"""
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        """檢查是否有足夠數據進行採樣"""
        return self.size >= batch_size
    
    def clear(self) -> None:
        """清空緩衝區（包含 RUDDER 狀態）"""
        self.position = 0
        self.size = 0
        self.outcome_deltas.fill(0.0)
        self.trade_ids.fill(-1)
        self.step_costs.fill(0.0)
        self.cost_probs.fill(0.0)
        self.loss_cvar_samples.fill(0.0)
        self.risk_costs.fill(0.0)
        self.struct_costs.fill(0.0)
        self.trade_costs.fill(0.0)
        self.entry_costs.fill(0.0)
        self.drawdown_costs.fill(0.0)
        self.stop_streak_costs.fill(0.0)
        self._trade_entry_map.clear()
        self._total_outcomes_backfilled = 0
        self._total_forced_closes = 0
    
    def get_stats(self) -> Dict[str, float]:
        """獲取緩衝區統計信息（包含 RUDDER 特定統計）"""
        if self.size == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'usage': 0.0,
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'mean_outcome': 0.0,
                'std_outcome': 0.0,
                'mean_step_cost': 0.0,
                'cost_prob_rate': 0.0,
                'total_outcomes_backfilled': self._total_outcomes_backfilled,
                'total_forced_closes': self._total_forced_closes,
                'active_trades': len(self._trade_entry_map),
            }
        
        rewards = self.rewards[:self.size]
        outcomes = self.outcome_deltas[:self.size]
        step_costs = self.step_costs[:self.size]
        cost_probs = self.cost_probs[:self.size]
        risk_costs = self.risk_costs[:self.size]
        struct_costs = self.struct_costs[:self.size]
        trade_costs = self.trade_costs[:self.size]
        entry_costs = self.entry_costs[:self.size]
        drawdown_costs = self.drawdown_costs[:self.size]
        stop_streak_costs = self.stop_streak_costs[:self.size]
        
        return {
            'size': self.size,
            'capacity': self.capacity,
            'usage': self.size / self.capacity,
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'mean_outcome': float(np.mean(outcomes)),
            'std_outcome': float(np.std(outcomes)),
            'mean_step_cost': float(np.mean(step_costs)),
            'cost_prob_rate': float(np.mean(cost_probs)),
            'mean_risk_cost': float(np.mean(risk_costs)),
            'mean_struct_cost': float(np.mean(struct_costs)),
            'mean_trade_cost': float(np.mean(trade_costs)),
            'mean_entry_cost': float(np.mean(entry_costs)),
            'mean_drawdown_cost': float(np.mean(drawdown_costs)),
            'mean_stop_streak_cost': float(np.mean(stop_streak_costs)),
            'total_outcomes_backfilled': self._total_outcomes_backfilled,
            'total_forced_closes': self._total_forced_closes,
            'active_trades': len(self._trade_entry_map),
        }

