"""
強化學習模型抽象基類

定義所有 RL 模型必須實作的介面，遵循：
- 開放封閉原則 (OCP): 透過繼承擴展新模型
- 里氏替換原則 (LSP): 所有子類可替換基類
- 依賴反轉原則 (DIP): 訓練器依賴此抽象而非具體實作
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import torch
import numpy as np


class BaseRLModel(ABC):
    """
    強化學習模型抽象基類
    
    所有 RL 模型必須繼承此類並實作所有抽象方法。
    
    Args:
        observation_shape: 觀察空間形狀
        action_dim: 動作空間維度
        device: 訓練設備 ('cuda' 或 'cpu')
    """
    
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        device: str = 'cuda'
    ) -> None:
        """初始化模型"""
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.device = device
        self._training_mode = True
    
    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        evaluate: bool = False
    ) -> np.ndarray:
        """
        選擇動作
        
        Args:
            state: 當前狀態
            evaluate: 是否為評估模式（無探索噪音）
            
        Returns:
            選擇的動作
        """
        pass
    
    @abstractmethod
    def update(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        更新模型參數
        
        Args:
            batch: 批次數據字典，包含 state, action, reward, next_state, done
            
        Returns:
            損失字典，包含各項損失值
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路徑
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        加載模型
        
        Args:
            path: 模型路徑
        """
        pass
    
    def train(self) -> None:
        """設置為訓練模式"""
        self._training_mode = True
    
    def eval(self) -> None:
        """設置為評估模式"""
        self._training_mode = False
    
    @property
    def is_training(self) -> bool:
        """檢查是否為訓練模式"""
        return self._training_mode
    
    def to(self, device: str) -> BaseRLModel:
        """
        移動模型到指定設備
        
        Args:
            device: 目標設備
            
        Returns:
            self
        """
        self.device = device
        return self
    
    @abstractmethod
    def get_state_dict(self) -> Dict[str, Any]:
        """
        獲取模型狀態字典
        
        Returns:
            狀態字典
        """
        pass
    
    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        加載模型狀態字典
        
        Args:
            state_dict: 狀態字典
        """
        pass

