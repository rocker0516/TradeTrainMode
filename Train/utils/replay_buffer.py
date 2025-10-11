"""
经验回放缓冲区

用于存储和采样训练经验。
"""

import numpy as np
from typing import Tuple


class ReplayBuffer:
    """
    经验回放缓冲区
    
    使用循环缓冲区存储经验，支持高效采样。
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int, ...],
        action_dim: int,
    ) -> None:
        """
        初始化经验回放缓冲区
        
        Args:
            buffer_size: 缓冲区最大容量
            observation_shape: 观察空间形状
            action_dim: 动作维度
        """
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        
        # 预分配内存
        self.states = np.zeros((buffer_size, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, *observation_shape), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0  # 当前写入位置
        self.size = 0  # 当前缓冲区大小
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        """
        添加一条经验
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束（1.0 表示结束，0.0 表示未结束）
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        随机采样一批经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (states, actions, rewards, next_states, dones)
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )
    
    def __len__(self) -> int:
        """返回当前缓冲区大小"""
        return self.size
    
    def clear(self) -> None:
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0

