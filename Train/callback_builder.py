"""回调构建器（遵循 SRP 原则）。

负责创建和配置训练回调。
"""
from __future__ import annotations

import os
import logging
from typing import List, Optional

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv

from Train.lagrangian import LagrangianCallback, MultiSharedLagrangianController
from Train.eval import (
    ConstraintEvalCallback,
    EvalConfig as EvalCallbackConfig,
    EvalConstraints,
    TrainEvalStartGateConfig,
)
from Train.config_builder import TrainingConfig
from Train.interfaces import ILagrangianController, ICallbackBuilder

logger = logging.getLogger(__name__)


class CallbackBuilder(ICallbackBuilder):
    """回调构建器（单一职责：回调创建）。"""
    
    def __init__(self, config: TrainingConfig):
        """初始化回调构建器。
        
        Args:
            config: 训练配置
        """
        self.config = config
    
    def build_training_callbacks(
        self,
        controller: ILagrangianController,
        training_env: Optional[VecEnv] = None,
        eval_env: Optional[VecEnv] = None,
    ) -> List[BaseCallback]:
        """构建训练回调列表。
        
        Args:
            controller: Lagrangian 控制器
            training_env: 保留參數以相容介面，未使用（LagrangianCallback 從 BaseCallback.training_env 取得，即 model.get_env()）
            eval_env: 评估环境（可选）
            
        Returns:
            回调列表
        """
        callbacks: List[BaseCallback] = []
        
        # 1. Lagrangian Callback（training_env 由 SB3 在 callback 綁定 model 後經 BaseCallback.training_env 取得）
        lag_callback = self._build_lagrangian_callback(controller)
        callbacks.append(lag_callback)
        
        # 2. Checkpoint Callback
        checkpoint_callback = self._build_checkpoint_callback()
        callbacks.append(checkpoint_callback)
        
        # 3. Evaluation Callback（如果启用）
        if self.config.eval_config.enabled and eval_env is not None:
            eval_callback = self._build_eval_callback(eval_env)
            callbacks.append(eval_callback)
        
        return callbacks
    
    def _build_lagrangian_callback(
        self,
        controller: ILagrangianController,
    ) -> LagrangianCallback:
        """构建 Lagrangian 回调。
        
        Args:
            controller: Lagrangian 控制器
            
        Returns:
            LagrangianCallback 实例。
            訓練環境由 SB3 BaseCallback 的唯讀 property training_env（model.get_env()）提供，不可在此賦值。
        """
        return LagrangianCallback(
            controller=controller,
            update_freq=int(self.config.update_lambda_every_steps),
            log_freq=int(self.config.log_every_episodes),
            window_size=int(self.config.stats_window_episodes),
            reward_scale=float(self.config.reward_scale),
        )
    
    def _build_checkpoint_callback(self) -> CheckpointCallback:
        """构建检查点回调。
        
        Returns:
            CheckpointCallback 实例
        """
        save_path = f"{self.config.checkpoint_dir_prefix}_{self.config.symbol}"
        return CheckpointCallback(
            save_freq=int(self.config.checkpoint_save_freq),
            save_path=save_path,
            name_prefix="sac_lag",
        )
    
    def _build_eval_callback(self, eval_env: VecEnv) -> ConstraintEvalCallback:
        """构建评估回调。
        
        Args:
            eval_env: 评估环境
            
        Returns:
            ConstraintEvalCallback 实例
            
        Raises:
            ValueError: 评估配置无效
        """
        eval_cfg = self.config.eval_config
        
        # 构建最佳模型路径
        best_model_base_dir = f"{self.config.checkpoint_dir_prefix}_{self.config.symbol}"
        best_model_path = os.path.join(
            best_model_base_dir,
            "best_model",
            "best_model",
        )
        
        # 构建评估配置
        eval_config = EvalCallbackConfig(
            enabled=eval_cfg.enabled,
            eval_every_timesteps=int(eval_cfg.eval_every_timesteps),
            n_eval_episodes=int(eval_cfg.n_eval_episodes),
            deterministic=bool(eval_cfg.deterministic),
            reject_if_death_event=bool(eval_cfg.reject_if_death_event),
            constraints=EvalConstraints(
                max_dd_limit=float(eval_cfg.max_dd_limit),
                mean_cost_limit=float(eval_cfg.mean_cost_limit),
                min_mean_return=float(eval_cfg.min_mean_return),
            ),
            train_start_gate=TrainEvalStartGateConfig(
                enabled=bool(eval_cfg.gate_enabled),
                window_size=int(eval_cfg.gate_window_size),
                min_max_steps_reached_count=int(eval_cfg.gate_min_max_steps_reached_count),
            ),
            save_best_model=bool(eval_cfg.save_best_model),
            best_model_path=str(best_model_path),
            print_each_episode=bool(eval_cfg.print_each_episode),
            print_prefix=str(eval_cfg.print_prefix),
            render_each_episode=bool(eval_cfg.render_each_episode),
        )
        
        try:
            return ConstraintEvalCallback(
                eval_env=eval_env,
                eval_config=eval_config,
            )
        except Exception as e:
            logger.error(f"Failed to create eval callback: {e}")
            raise ValueError(f"Invalid eval configuration: {e}") from e

