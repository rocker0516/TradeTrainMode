"""模型构建器（遵循 SRP 原则）。

负责创建和配置 SAC 模型。
"""
from __future__ import annotations

import logging
from typing import Dict, Any

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecEnv

from Train.sb3_cnn_policy import DualCnnFeatureExtractor
from Train.optimized_dict_replay_buffer import OptimizedDictReplayBuffer
from Train.config_builder import ModelConfig
from Train.interfaces import IModelBuilder

logger = logging.getLogger(__name__)


class ModelBuilder(IModelBuilder):
    """SAC 模型构建器（单一职责：模型创建）。"""
    
    def __init__(self, config: ModelConfig):
        """初始化模型构建器。
        
        Args:
            config: 模型配置
        """
        self.config = config
    
    def build(self, env: VecEnv) -> SAC:
        """构建 SAC 模型。
        
        Args:
            env: 训练环境（VecEnv）
            
        Returns:
            配置好的 SAC 模型实例
            
        Raises:
            ValueError: 配置参数无效
            RuntimeError: 模型创建失败
        """
        try:
            # 构建 policy_kwargs
            policy_kwargs = dict(
                features_extractor_class=DualCnnFeatureExtractor,
                features_extractor_kwargs=dict(
                    emb_5m_target=self.config.emb_5m_target,
                    emb_5m_others=self.config.emb_5m_others,
                    emb_1d_target=self.config.emb_1d_target,
                    emb_1d_others=self.config.emb_1d_others,
                    emb_vec=self.config.emb_vec,
                    out_dim=self.config.out_dim,
                    use_cross_attention=self.config.use_cross_attention,
                ),
                net_arch=dict(
                    pi=list(self.config.pi_arch),
                    qf=list(self.config.qf_arch),
                ),
            )
            
            # 创建模型
            model = SAC(
                policy="MultiInputPolicy",
                env=env,
                policy_kwargs=policy_kwargs,
                learning_rate=float(self.config.learning_rate),
                buffer_size=int(self.config.buffer_size),
                optimize_memory_usage=True,  # 内存优化
                replay_buffer_class=OptimizedDictReplayBuffer,
                replay_buffer_kwargs={"handle_timeout_termination": False},
                batch_size=int(self.config.batch_size),
                ent_coef=self.config.ent_coef,
                train_freq=int(self.config.train_freq),
                gradient_steps=int(self.config.gradient_steps),
                device=self.config.device,
                verbose=0,  # 将在训练时设置
                tensorboard_log=str(self.config.tensorboard_log_dir),
            )
            
            logger.info(f"SAC model created successfully (device={self.config.device})")
            return model
            
        except ValueError as e:
            logger.error(f"Invalid model configuration: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"Failed to create SAC model: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating model: {e}", exc_info=True)
            raise RuntimeError(f"Model creation failed: {e}") from e

