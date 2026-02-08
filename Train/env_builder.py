"""环境构建器（遵循 SRP 和 OCP 原则）。

负责创建训练和评估环境，支持通过配置灵活添加 wrapper。

效能：SubprocVecEnv 下可傳入預載的 (df_5m, df_1d)，避免每個子進程重複執行 load_data()（磁碟 I/O + 解析）。
"""
from __future__ import annotations

import gymnasium as gym
from typing import Dict, Any, Optional, Type, Tuple
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

import pandas as pd

from Env.trading_env import TradingEnvironment
from Env.wrappers import ActionRepeatWrapper, ActionClipWrapper
from Train.lagrangian import LagrangianRewardWrapper, MultiSharedLagrangianController
from Train.config_builder import EnvironmentConfig
from Train.interfaces import ILagrangianController
from Train.train_config import TrainConfig


class EnvironmentBuilder:
    """环境构建器（单一职责：环境创建）。
    
    支持通过配置灵活构建环境，避免硬编码 wrapper 顺序。
    """
    
    def __init__(
        self,
        config: EnvironmentConfig,
        controller: Optional[ILagrangianController] = None,
        max_position_pct: float = 0.7,
        action_repeat: int = 1,
        reward_scale: float = 1.0,
    ):
        """初始化环境构建器。
        
        Args:
            config: 环境配置
            controller: Lagrangian 控制器（可选）
            max_position_pct: 最大持仓百分比
            action_repeat: 动作重复次数
            reward_scale: 奖励缩放因子
        """
        self.config = config
        self.controller = controller
        self.max_position_pct = max_position_pct
        self.action_repeat = action_repeat
        self.reward_scale = reward_scale
    
    def build_base_env(
        self,
        rank: int,
        random_start: bool = True,
        df_5m: Optional[pd.DataFrame] = None,
        df_1d: Optional[pd.DataFrame] = None,
    ) -> gym.Env:
        """构建基础环境（不含 wrapper）。
        
        Args:
            rank: 环境排名
            random_start: 是否随机起点
            df_5m: 預載 5m 資料（可選，避免子進程重複 load_data）
            df_1d: 預載 1d 資料（可選）
            
        Returns:
            基础 TradingEnvironment 实例
        """
        env_kwargs = self.config.to_dict()
        env_kwargs["random_start"] = random_start
        if df_5m is not None and df_1d is not None:
            env_kwargs["df_5m"] = df_5m
            env_kwargs["df_1d"] = df_1d
        return TradingEnvironment(env_id=rank, **env_kwargs)
    
    def build_training_env(
        self,
        rank: int,
        seed: int,
        df_5m: Optional[pd.DataFrame] = None,
        df_1d: Optional[pd.DataFrame] = None,
    ) -> gym.Env:
        """构建训练环境（包含所有 wrapper）。
        
        Args:
            rank: 环境排名
            seed: 随机种子
            df_5m: 預載 5m 資料（可選）
            df_1d: 預載 1d 資料（可選）
            
        Returns:
            配置好的训练环境
        """
        # 1. 基础环境
        env = self.build_base_env(
            rank, random_start=True, df_5m=df_5m, df_1d=df_1d
        )
        
        # 2. Action Clip Wrapper
        env = ActionClipWrapper(env, max_position_pct=self.max_position_pct)
        
        # 3. Action Repeat Wrapper
        env = ActionRepeatWrapper(env, repeat=self.action_repeat)
        
        # 4. Lagrangian Reward Wrapper（如果提供控制器）
        if self.controller is not None:
            env = LagrangianRewardWrapper(
                env,
                self.controller,
                reward_scale=self.reward_scale,
            )
        
        return env
    
    def build_eval_env(self, rank: int, seed: int, symbol: str = "BTCUSDT") -> gym.Env:
        """构建评估环境（包含所有 wrapper）。
        
        Args:
            rank: 环境排名
            seed: 随机种子
            symbol: 交易标的符号（用于 render_dir）
            
        Returns:
            配置好的评估环境
        """
        # 使用评估配置
        eval_kwargs = self.config.to_eval_dict()
        eval_kwargs["random_start"] = TrainConfig.EVAL_RANDOM_START
        # 添加 render_dir
        import os
        eval_kwargs["render_dir"] = os.path.join("logs", "renders", f"eval_{symbol}")
        
        # 1. 基础环境
        env = TradingEnvironment(env_id=rank, **eval_kwargs)
        
        # 2. Action Clip Wrapper
        env = ActionClipWrapper(env, max_position_pct=self.max_position_pct)
        
        # 3. Action Repeat Wrapper
        env = ActionRepeatWrapper(env, repeat=self.action_repeat)
        
        # 注意：评估环境通常不使用 Lagrangian Wrapper
        # 因为评估时我们想看原始奖励表现
        
        return env
    
    def create_training_vec_env(
        self,
        n_envs: int,
        log_prefix: str = "logs/sac_lag",
        df_5m: Optional[pd.DataFrame] = None,
        df_1d: Optional[pd.DataFrame] = None,
    ) -> VecMonitor:
        """创建并行训练环境（VecEnv）。
        
        Args:
            n_envs: 并行环境数量
            log_prefix: 日志前缀
            df_5m: 預載 5m 資料（可選）；提供時可避免每個子進程重複 load_data()，大幅縮短啟動時間
            df_1d: 預載 1d 資料（可選）
            
        Returns:
            VecMonitor 包装的并行环境
        """
        # 注意：SubprocVecEnv 需要「可呼叫的工廠函式」列表，每個元素呼叫時才建立環境實例
        # 若傳入 [make_env_fn(0), make_env_fn(1), ...] 且 make_env_fn 回傳 env，則傳的是實例而非 callable，會觸發 'X object is not callable'
        def make_env_fn(rank: int):
            """回傳環境工廠 callable（用於 SubprocVecEnv），呼叫時才 build_training_env。"""
            def _init() -> gym.Env:
                return self.build_training_env(
                    rank, seed=rank, df_5m=df_5m, df_1d=df_1d
                )
            return _init

        # 傳入 [callable, callable, ...]，每個 callable() 在子進程中建立一個新環境
        vec_env = SubprocVecEnv([make_env_fn(i) for i in range(n_envs)])
        
        # 添加监控
        vec_env = VecMonitor(vec_env, filename=log_prefix)
        
        return vec_env
    
    def create_eval_vec_env(self, symbol: str = "BTCUSDT") -> DummyVecEnv:
        """创建评估环境（单环境 VecEnv）。
        
        Args:
            symbol: 交易标的符号（用于 render_dir）
            
        Returns:
            DummyVecEnv 包装的评估环境
        """
        def make_eval_env():
            """评估环境工厂函数。"""
            return self.build_eval_env(rank=9999, seed=0, symbol=symbol)
        
        return DummyVecEnv([make_eval_env])

