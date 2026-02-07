"""抽象接口定义（遵循 DIP 原则）。

本模块定义所有抽象接口，确保高层模块依赖抽象而非具体实现。
"""
from __future__ import annotations

from typing import Protocol, Dict, Optional
import gymnasium as gym


class ILagrangianController(Protocol):
    """Lagrangian 控制器接口。
    
    所有 Lagrangian 控制器必须实现此接口，以支持依赖注入和测试。
    """
    
    @property
    def current_lambda(self) -> float:
        """获取当前 lambda 值（单通道模式）。
        
        Returns:
            当前 lambda 值
        """
        ...
    
    @property
    def current_lambdas(self) -> Dict[str, float]:
        """获取当前所有 lambda 值（多通道模式）。
        
        Returns:
            通道名到 lambda 值的映射
        """
        ...
    
    def update(self, avg_cost: float) -> float:
        """更新 lambda 值（单通道模式）。
        
        Args:
            avg_cost: 平均成本
            
        Returns:
            更新后的 lambda 值
        """
        ...
    
    def update_multi(self, avg_costs: Dict[str, float]) -> Dict[str, float]:
        """更新多个通道的 lambda 值（多通道模式）。
        
        Args:
            avg_costs: 通道名到平均成本的映射
            
        Returns:
            通道名到更新后 lambda 值的映射
        """
        ...


class IEnvironmentFactory(Protocol):
    """环境工厂接口。
    
    负责创建训练和评估环境实例。
    """
    
    def create_training_env(self, rank: int, seed: int) -> gym.Env:
        """创建训练环境实例。
        
        Args:
            rank: 环境排名（用于区分并行环境）
            seed: 随机种子
            
        Returns:
            配置好的 Gym 环境实例
        """
        ...
    
    def create_eval_env(self, rank: int, seed: int) -> gym.Env:
        """创建评估环境实例。
        
        Args:
            rank: 环境排名
            seed: 随机种子
            
        Returns:
            配置好的评估环境实例
        """
        ...


class IModelBuilder(Protocol):
    """模型构建器接口。
    
    负责创建和配置 SAC 模型。
    """
    
    def build(self, env: gym.Env) -> "SAC":  # type: ignore[name-defined]
        """构建 SAC 模型。
        
        Args:
            env: 训练环境
            
        Returns:
            配置好的 SAC 模型实例
        """
        ...


class ICallbackBuilder(Protocol):
    """回调构建器接口。
    
    负责创建和配置训练回调。
    """
    
    def build_training_callbacks(
        self,
        controller: ILagrangianController,
    ) -> list:
        """构建训练回调列表。
        
        Args:
            controller: Lagrangian 控制器
            
        Returns:
            回调列表
        """
        ...

