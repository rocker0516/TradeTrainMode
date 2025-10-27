"""
經驗回放緩衝區

實作高效的經驗儲存和採樣系統。
遵循單一職責原則 (SRP)。
"""

from __future__ import annotations
from typing import Dict
import numpy as np
import torch


class ReplayBuffer:
    """
    經驗回放緩衝區
    
    儲存和採樣經驗數據，用於離線學習。
    
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
        """初始化緩衝區"""
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        # 預分配記憶體（提高效率）
        self.states = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        添加一條經驗
        
        Args:
            state: 當前狀態
            action: 執行的動作
            reward: 獲得的獎勵
            next_state: 下一個狀態
            done: 是否結束
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        隨機採樣批次數據
        
        Args:
            batch_size: 批次大小
            
        Returns:
            批次數據字典
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
            'reward': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_state': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'done': torch.FloatTensor(self.dones[indices]).to(self.device),
        }
        
        return batch
    
    def __len__(self) -> int:
        """返回當前緩衝區大小"""
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        """
        檢查是否有足夠數據進行採樣
        
        Args:
            batch_size: 批次大小
            
        Returns:
            是否準備好
        """
        return self.size >= batch_size
    
    def clear(self) -> None:
        """清空緩衝區"""
        self.position = 0
        self.size = 0
    
    def get_stats(self) -> Dict[str, float]:
        """
        獲取緩衝區統計信息
        
        Returns:
            統計信息字典
        """
        if self.size == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'usage': 0.0,
                'mean_reward': 0.0,
                'std_reward': 0.0,
            }
        
        rewards = self.rewards[:self.size]
        
        return {
            'size': self.size,
            'capacity': self.capacity,
            'usage': self.size / self.capacity,
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
        }

