"""SAC Lagrangian 训练器（遵循 SRP 和 DIP 原则）。

主训练类，负责协调整个训练流程。
"""
from __future__ import annotations

import logging
import multiprocessing
import os
from typing import Optional, Any

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecEnv

from Train.config_builder import TrainingConfig
from Train.lagrangian import MultiSharedLagrangianController
from Train.interfaces import (
    ILagrangianController,
    IVecEnvBuilder,
    IModelBuilder,
    ICallbackBuilder,
)
from Env.load_file import load_data

logger = logging.getLogger(__name__)


class SACLagrangianTrainer:
    """SAC Lagrangian 训练器（单一职责：协调训练流程）。
    
    遵循依赖反转原则，依赖抽象接口而非具体实现。
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        env_builder: IVecEnvBuilder,
        model_builder: IModelBuilder,
        callback_builder: ICallbackBuilder,
        controller: Optional[ILagrangianController] = None,
    ):
        """初始化训练器。
        
        Args:
            config: 训练配置
            env_builder: 环境构建器（依賴介面，符合 DIP）
            model_builder: 模型构建器
            callback_builder: 回调构建器
            controller: Lagrangian 控制器（如果为 None，将从配置创建）
        """
        self.config = config
        self.env_builder = env_builder
        self.model_builder = model_builder
        self.callback_builder = callback_builder
        
        # 创建或使用提供的控制器
        if controller is None:
            self.controller = self._create_controller()
        else:
            self.controller = controller
        
        # 训练环境（将在 train() 中创建）
        self.training_env: Optional[VecEnv] = None
        self.eval_env: Optional[VecEnv] = None
        self.model: Optional[SAC] = None
    
    def _create_controller(self) -> MultiSharedLagrangianController:
        """从配置创建 Lagrangian 控制器。
        
        Returns:
            MultiSharedLagrangianController 实例
        """
        channel_configs = self.config.lagrangian_config.to_channel_configs()
        return MultiSharedLagrangianController(channel_configs)
    
    def _create_training_env(
        self,
        df_5m: Optional[Any] = None,
        df_1d: Optional[Any] = None,
    ) -> VecEnv:
        """创建训练环境。
        
        Args:
            df_5m: 預載 5m 資料（可選）；提供時子進程不重複 load_data，縮短啟動時間
            df_1d: 預載 1d 資料（可選）
        
        Returns:
            VecEnv 训练环境
            
        Raises:
            RuntimeError: 环境创建失败
        """
        try:
            self.env_builder.set_controller(self.controller)
            log_prefix = f"{self.config.vec_monitor_log_prefix}_{self.config.symbol}"
            env = self.env_builder.create_training_vec_env(
                n_envs=self.config.n_envs,
                log_prefix=log_prefix,
                df_5m=df_5m,
                df_1d=df_1d,
            )
            logger.info(f"Training environment created ({self.config.n_envs} parallel envs)")
            return env
        except (OSError, RuntimeError) as e:
            logger.error(f"Failed to create training environment: {e}")
            raise RuntimeError(f"Environment creation failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error creating environment: {e}", exc_info=True)
            raise RuntimeError(f"Environment creation failed: {e}") from e
    
    def _create_eval_env(self) -> Optional[VecEnv]:
        """创建评估环境（如果启用）。
        
        Returns:
            VecEnv 评估环境，如果未启用则返回 None
        """
        if not self.config.eval_config.enabled:
            return None
        
        try:
            env = self.env_builder.create_eval_vec_env(symbol=self.config.symbol)
            logger.info("Evaluation environment created")
            return env
        except Exception as e:
            logger.warning(f"Failed to create eval environment: {e}, continuing without eval")
            return None
    
    def _resolve_load_model_path(self, path: str) -> str:
        """解析載入路徑：若為目錄則指向目錄內的 best_model（對應 best_model.zip）。"""
        path = os.path.abspath(path)
        if os.path.isdir(path):
            # 目錄如 models/sac_lag_BTCUSDT/best_model → 載入 best_model.zip
            return os.path.join(path, "best_model")
        # 若為檔案路徑，SB3 會自動加 .zip 尋找；若已含 .zip 則去掉讓 SB3 一致處理
        if path.endswith(".zip"):
            return path[:-4]
        return path

    def _create_model(self, env: VecEnv) -> SAC:
        """创建 SAC 模型（新建或從 load_model_path 載入）。
        
        Args:
            env: 训练环境
            
        Returns:
            SAC 模型实例
            
        Raises:
            RuntimeError: 模型创建失败
        """
        load_path = getattr(self.config, "load_model_path", None)
        if load_path and str(load_path).strip():
            try:
                resolved = self._resolve_load_model_path(str(load_path).strip())
                model = SAC.load(
                    resolved,
                    env=env,
                    device=self.config.model_config.device,
                )
                if self.config.sb3_verbose is not None:
                    model.verbose = self.config.sb3_verbose
                logger.info("SAC model loaded from %s (resolved: %s), continuing training", load_path, resolved)
                return model
            except Exception as e:
                logger.error("Failed to load model from %s: %s", load_path, e, exc_info=True)
                raise RuntimeError(f"Model load failed: {e}") from e

        try:
            model = self.model_builder.build(env)
            if self.config.sb3_verbose is not None:
                model.verbose = self.config.sb3_verbose
            logger.info("SAC model created successfully")
            return model
        except (ValueError, RuntimeError) as e:
            logger.error(f"Failed to create model: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating model: {e}", exc_info=True)
            raise RuntimeError(f"Model creation failed: {e}") from e
    
    def _create_callbacks(self) -> list:
        """创建训练回调。
        
        Returns:
            回调列表
        """
        callbacks = self.callback_builder.build_training_callbacks(
            controller=self.controller,
            training_env=self.training_env,
            eval_env=self.eval_env,
        )
        logger.info(f"Created {len(callbacks)} callbacks")
        return callbacks
    
    def _save_model(self) -> None:
        """保存最终模型。
        
        Raises:
            OSError: 文件保存失败
        """
        if self.model is None:
            logger.warning("No model to save")
            return
        
        try:
            save_path = f"models/sac_lag_{self.config.symbol}/final_model"
            self.model.save(save_path)
            logger.info(f"Final model saved to {save_path}")
        except OSError as e:
            logger.error(f"Failed to save model: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving model: {e}", exc_info=True)
            raise OSError(f"Model save failed: {e}") from e
    
    def train(self) -> None:
        """执行训练流程。
        
        功能：
        1. 创建训练环境
        2. 创建评估环境（如果启用）
        3. 创建 SAC 模型
        4. 创建回调
        5. 执行训练
        6. 保存模型
        
        Raises:
            RuntimeError: 训练流程中任何步骤失败
            KeyboardInterrupt: 用户中断训练
        """
        try:
            logger.info(
                f"Starting SAC-Lagrangian training on {self.config.symbol} "
                f"with {self.config.n_envs} envs..."
            )
            logger.info(
                f"Cost Limits (per step): "
                f"risk={self.config.lagrangian_config.risk_cost_limit}, "
                f"fric={self.config.lagrangian_config.fric_cost_limit}, "
                f"trade_freq={self.config.lagrangian_config.trade_freq_cost_limit} (window={self.config.lagrangian_config.trade_freq_window_steps} steps), "
                f"flat={self.config.lagrangian_config.flat_cost_limit} (window={self.config.lagrangian_config.flat_window_steps} steps)"
            )
            
            # 1. 載入市場資料一次，供所有子進程共用（避免 N 個 env 重複 load_data）
            df_5m, df_1d = load_data()
            # 2. 创建训练环境
            self.training_env = self._create_training_env(df_5m=df_5m, df_1d=df_1d)
            
            # 3. 创建评估环境（如果启用）
            self.eval_env = self._create_eval_env()
            
            # 4. 创建模型
            self.model = self._create_model(self.training_env)
            
            # 5. 创建回调
            callbacks = self._create_callbacks()
            
            # 6. 执行训练
            logger.info(f"Starting training for {self.config.total_timesteps} timesteps...")
            log_interval = int(self.config.sb3_log_interval)
            self.model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
                progress_bar=bool(self.config.show_progress_bar),
                log_interval=log_interval,
            )
            
            # 7. 保存模型
            self._save_model()
            
            logger.info("Training finished successfully")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            # 即使中断也尝试保存模型
            if self.model is not None:
                try:
                    self._save_model()
                except Exception as e:
                    logger.warning(f"Failed to save model after interrupt: {e}")
            raise
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise RuntimeError(f"Training failed: {e}") from e
        finally:
            # 清理资源
            self._cleanup()
    
    def _cleanup(self) -> None:
        """清理资源（关闭环境等）。"""
        try:
            if self.training_env is not None:
                self.training_env.close()
                logger.debug("Training environment closed")
        except Exception as e:
            logger.warning(f"Error closing training environment: {e}")
        
        try:
            if self.eval_env is not None:
                self.eval_env.close()
                logger.debug("Evaluation environment closed")
        except Exception as e:
            logger.warning(f"Error closing eval environment: {e}")

