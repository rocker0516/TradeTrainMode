"""
训练器抽象基类

遵循 SRP (Single Responsibility Principle) 和 OCP (Open-Closed Principle)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import gymnasium as gym


class BaseTrainer(ABC):
    """
    训练器抽象基类
    
    所有训练器必须继承此类并实现训练逻辑。
    """
    
    def __init__(
        self,
        env: gym.Env,
        model: Any,
        config: Dict[str, Any],
    ) -> None:
        """
        初始化训练器
        
        Args:
            env: 训练环境
            model: 强化学习模型
            config: 训练配置
        """
        self.env = env
        self.model = model
        self.config = config
        
        # 训练统计
        self.episode_rewards: list = []
        self.episode_lengths: list = []
        self.total_steps: int = 0
        
    @abstractmethod
    def train(self, total_episodes: int, eval_interval: int = 10) -> None:
        """
        执行训练
        
        Args:
            total_episodes: 总训练回合数
            eval_interval: 评估间隔（每隔多少回合评估一次）
        """
        pass
    
    @abstractmethod
    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            num_episodes: 评估回合数
            
        Returns:
            评估指标字典
        """
        pass
    
    def save_checkpoint(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        保存训练检查点
        
        Args:
            filepath: 保存路径
            metadata: 额外的元数据
        """
        self.model.save(filepath)
        print(f"检查点已保存至: {filepath}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """
        记录训练指标
        
        Args:
            metrics: 指标字典
            step: 当前步数
        """
        print(f"Step {step}: {metrics}")

