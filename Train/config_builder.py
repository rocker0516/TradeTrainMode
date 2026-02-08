"""配置构建器（遵循 SRP 原则）。

负责从 TrainConfig 和 CLI 参数构建统一的训练配置。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, Optional

from Train.train_config import TrainConfig
from Train.lagrangian import LagrangianChannelConfig


@dataclass
class LagrangianConfig:
    """Lagrangian 配置（类型安全）。"""
    risk_cost_limit: float
    fric_cost_limit: float
    trade_freq_cost_limit: float = TrainConfig.TRADE_FREQ_COST_LIMIT
    trade_freq_window_steps: int = TrainConfig.TRADE_FREQ_WINDOW_STEPS
    flat_cost_limit: float = TrainConfig.FLAT_COST_LIMIT
    flat_window_steps: int = TrainConfig.FLAT_WINDOW_STEPS
    kp: float = TrainConfig.LAGRANGIAN_KP
    lambda_init: float = TrainConfig.LAGRANGIAN_LAMBDA_INIT
    lambda_min: float = TrainConfig.LAGRANGIAN_LAMBDA_MIN
    lambda_max: float = TrainConfig.LAGRANGIAN_LAMBDA_MAX

    def to_channel_configs(self) -> Dict[str, LagrangianChannelConfig]:
        """转换为通道配置字典。"""
        return {
            "risk": LagrangianChannelConfig(
                cost_limit=self.risk_cost_limit,
                kp=self.kp,
                lambda_init=self.lambda_init,
                lambda_min=self.lambda_min,
                lambda_max=self.lambda_max,
            ),
            "fric": LagrangianChannelConfig(
                cost_limit=self.fric_cost_limit,
                kp=self.kp,
                lambda_init=self.lambda_init,
                lambda_min=self.lambda_min,
                lambda_max=self.lambda_max,
            ),
            "trade_freq": LagrangianChannelConfig(
                cost_limit=self.trade_freq_cost_limit,
                kp=self.kp,
                lambda_init=self.lambda_init,
                lambda_min=self.lambda_min,
                lambda_max=self.lambda_max,
                window_steps=self.trade_freq_window_steps,
            ),
            "flat": LagrangianChannelConfig(
                cost_limit=self.flat_cost_limit,
                kp=self.kp,
                lambda_init=self.lambda_init,
                lambda_min=self.lambda_min,
                lambda_max=self.lambda_max,
                window_steps=self.flat_window_steps,
            ),
        }


@dataclass
class EnvironmentConfig:
    """环境配置（类型安全）。"""
    target_symbol: str
    window_size: int = TrainConfig.WINDOW_SIZE_5M
    window_size_1d: int = TrainConfig.WINDOW_SIZE_1D
    min_position_change: float = TrainConfig.MIN_POSITION_CHANGE
    no_trade_entry_threshold: float = TrainConfig.NO_TRADE_ENTRY_THRESHOLD
    no_trade_exit_threshold: float = TrainConfig.NO_TRADE_EXIT_THRESHOLD
    risk_base_update_steps: int = TrainConfig.WINDOW_SIZE_5M
    feature_symbols: tuple[str, ...] = TrainConfig.FEATURE_SYMBOLS
    data_split_enabled: bool = TrainConfig.DATA_SPLIT_ENABLED
    data_mode: str = "train"
    holdout_months: int = TrainConfig.HOLDOUT_MONTHS
    flat_threshold: float = TrainConfig.FLAT_THRESHOLD

    def to_dict(self) -> Dict[str, any]:
        """转换为字典（用于环境初始化）。"""
        return {
            "target_symbol": self.target_symbol,
            "window_size": self.window_size,
            "window_size_1d": self.window_size_1d,
            "min_position_change": self.min_position_change,
            "no_trade_entry_threshold": self.no_trade_entry_threshold,
            "no_trade_exit_threshold": self.no_trade_exit_threshold,
            "risk_base_update_steps": self.risk_base_update_steps,
            "feature_symbols": list(self.feature_symbols),
            "data_split_enabled": self.data_split_enabled,
            "data_mode": self.data_mode,
            "holdout_months": self.holdout_months,
            "flat_threshold": self.flat_threshold,
        }
    
    def to_eval_dict(self) -> Dict[str, any]:
        """转换为评估环境配置字典。"""
        base = self.to_dict()
        base.update({
            "random_start": TrainConfig.EVAL_RANDOM_START,
            "max_episode_steps": TrainConfig.EVAL_MAX_EPISODE_STEPS,
            "min_episode_steps": TrainConfig.EVAL_MAX_EPISODE_STEPS,
            "data_mode": "eval",
            "ensure_filled_obs": True,
            "render_enabled": True,
            "render_save": True,
            "render_show": True,
            "render_on_done": True,
        })
        return base


@dataclass
class ModelConfig:
    """模型配置（类型安全）。"""
    learning_rate: float = TrainConfig.LEARNING_RATE
    buffer_size: int = TrainConfig.BUFFER_SIZE
    batch_size: int = TrainConfig.BATCH_SIZE
    ent_coef: str | float = TrainConfig.ENT_COEF
    train_freq: int = TrainConfig.TRAIN_FREQ
    gradient_steps: int = TrainConfig.GRADIENT_STEPS
    device: str = TrainConfig.DEVICE
    tensorboard_log_dir: str = TrainConfig.TENSORBOARD_LOG_DIR
    
    # Policy 网络结构
    emb_5m_target: int = TrainConfig.EMB_5M_TARGET
    emb_5m_others: int = TrainConfig.EMB_5M_OTHERS
    emb_1d_target: int = TrainConfig.EMB_1D_TARGET
    emb_1d_others: int = TrainConfig.EMB_1D_OTHERS
    emb_vec: int = TrainConfig.EMB_VEC
    out_dim: int = TrainConfig.OUT_DIM
    pi_arch: tuple[int, ...] = TrainConfig.PI_ARCH
    qf_arch: tuple[int, ...] = TrainConfig.QF_ARCH
    use_cross_attention: bool = True
    
    def to_dict(self) -> Dict[str, any]:
        """转换为字典（用于模型初始化）。"""
        return {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "ent_coef": self.ent_coef,
            "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps,
            "device": self.device,
            "tensorboard_log": self.tensorboard_log_dir,
            "policy_kwargs": {
                "features_extractor_class": None,  # 将在 ModelBuilder 中设置
                "features_extractor_kwargs": {
                    "emb_5m_target": self.emb_5m_target,
                    "emb_5m_others": self.emb_5m_others,
                    "emb_1d_target": self.emb_1d_target,
                    "emb_1d_others": self.emb_1d_others,
                    "emb_vec": self.emb_vec,
                    "out_dim": self.out_dim,
                    "use_cross_attention": self.use_cross_attention,
                },
                "net_arch": {
                    "pi": list(self.pi_arch),
                    "qf": list(self.qf_arch),
                },
            },
        }


@dataclass
class EvalConfig:
    """评估配置（类型安全）。"""
    enabled: bool = TrainConfig.EVAL_ENABLED
    eval_every_timesteps: int = TrainConfig.EVAL_EVERY_TIMESTEPS
    n_eval_episodes: int = TrainConfig.EVAL_N_EVAL_EPISODES
    deterministic: bool = TrainConfig.EVAL_DETERMINISTIC
    reject_if_death_event: bool = TrainConfig.EVAL_REJECT_IF_DEATH_EVENT
    max_dd_limit: float = TrainConfig.EVAL_MAX_DD_LIMIT
    mean_cost_limit: float = TrainConfig.EVAL_MEAN_COST_LIMIT
    save_best_model: bool = TrainConfig.EVAL_SAVE_BEST_MODEL
    print_each_episode: bool = TrainConfig.EVAL_PRINT_EACH_EPISODE
    print_prefix: str = TrainConfig.EVAL_PRINT_PREFIX
    render_each_episode: bool = True
    
    # Gate 配置
    gate_enabled: bool = TrainConfig.EVAL_GATE_ENABLED
    gate_window_size: int = TrainConfig.EVAL_GATE_WINDOW_SIZE
    gate_min_max_steps_reached_count: int = TrainConfig.EVAL_GATE_MIN_MAX_STEPS_REACHED_COUNT


@dataclass
class TrainingConfig:
    """统一的训练配置（类型安全）。
    
    所有配置参数集中在此，便于管理和类型检查。
    """
    # 基本参数
    symbol: str
    total_timesteps: int
    n_envs: int
    device: str = TrainConfig.DEVICE
    show_progress_bar: bool = TrainConfig.SHOW_PROGRESS_BAR_DEFAULT
    sb3_verbose: Optional[int] = TrainConfig.SB3_VERBOSE_DEFAULT
    
    # 子配置
    lagrangian_config: LagrangianConfig = field(default_factory=lambda: LagrangianConfig(
        risk_cost_limit=TrainConfig.RISK_COST_LIMIT,
        fric_cost_limit=TrainConfig.FRIC_COST_LIMIT,
        trade_freq_cost_limit=TrainConfig.TRADE_FREQ_COST_LIMIT,
        trade_freq_window_steps=TrainConfig.TRADE_FREQ_WINDOW_STEPS,
        flat_cost_limit=TrainConfig.FLAT_COST_LIMIT,
        flat_window_steps=TrainConfig.FLAT_WINDOW_STEPS,
    ))
    env_config: Optional[EnvironmentConfig] = None
    model_config: ModelConfig = field(default_factory=ModelConfig)
    eval_config: EvalConfig = field(default_factory=EvalConfig)
    
    # Wrapper 配置
    max_position_pct: float = TrainConfig.MAX_POSITION_PCT
    action_repeat: int = TrainConfig.ACTION_REPEAT
    
    # 日志和检查点
    checkpoint_dir_prefix: str = TrainConfig.CHECKPOINT_DIR_PREFIX
    checkpoint_save_freq: int = TrainConfig.CHECKPOINT_SAVE_FREQ
    vec_monitor_log_prefix: str = TrainConfig.VEC_MONITOR_LOG_PREFIX
    
    # Lambda 更新频率
    update_lambda_every_steps: int = TrainConfig.UPDATE_LAMBDA_EVERY_STEPS
    log_every_episodes: int = TrainConfig.LOG_EVERY_EPISODES
    stats_window_episodes: int = TrainConfig.STATS_WINDOW_EPISODES
    reward_scale: float = TrainConfig.REWARD_SCALE
    cost_penalty_normalize: float = TrainConfig.COST_PENALTY_NORMALIZE_FACTOR


class TrainingConfigBuilder:
    """训练配置构建器（单一职责：配置构建）。"""
    
    @staticmethod
    def from_cli_args(args: argparse.Namespace) -> TrainingConfig:
        """从 CLI 参数构建配置。
        
        Args:
            args: 命令行参数
            
        Returns:
            统一的训练配置对象
        """
        # 确定进度条和 verbose
        show_progress_bar = TrainConfig.SHOW_PROGRESS_BAR_DEFAULT and not args.no_progress_bar
        sb3_verbose = (
            args.verbose if args.verbose is not None
            else (0 if show_progress_bar else 1)
        )
        
        # 构建环境配置
        env_config = EnvironmentConfig(
            target_symbol=args.symbol,
            window_size=TrainConfig.WINDOW_SIZE_5M,
            window_size_1d=TrainConfig.WINDOW_SIZE_1D,
            min_position_change=TrainConfig.MIN_POSITION_CHANGE,
            no_trade_entry_threshold=TrainConfig.NO_TRADE_ENTRY_THRESHOLD,
            no_trade_exit_threshold=TrainConfig.NO_TRADE_EXIT_THRESHOLD,
            risk_base_update_steps=TrainConfig.WINDOW_SIZE_5M,
            feature_symbols=TrainConfig.FEATURE_SYMBOLS,
            data_split_enabled=TrainConfig.DATA_SPLIT_ENABLED,
            data_mode="train",
            holdout_months=TrainConfig.HOLDOUT_MONTHS,
        )
        
        # 构建模型配置（train_freq / gradient_steps 可由 CLI 覆寫以優化 it/s）
        model_config = ModelConfig(
            device=args.device,
            train_freq=getattr(args, "train_freq", TrainConfig.TRAIN_FREQ),
            gradient_steps=getattr(args, "gradient_steps", TrainConfig.GRADIENT_STEPS),
        )
        
        # 构建评估配置
        eval_config = EvalConfig(
            gate_enabled=TrainConfig.EVAL_GATE_ENABLED,
            gate_window_size=TrainConfig.EVAL_GATE_WINDOW_SIZE,
            gate_min_max_steps_reached_count=TrainConfig.EVAL_GATE_MIN_MAX_STEPS_REACHED_COUNT,
        )
        
        # 构建 Lagrangian 配置
        lagrangian_config = LagrangianConfig(
            risk_cost_limit=float(args.risk_cost_limit),
            fric_cost_limit=float(args.fric_cost_limit),
            trade_freq_cost_limit=float(getattr(args, "trade_freq_cost_limit", TrainConfig.TRADE_FREQ_COST_LIMIT)),
            trade_freq_window_steps=int(getattr(args, "trade_freq_window_steps", TrainConfig.TRADE_FREQ_WINDOW_STEPS)),
            flat_cost_limit=float(getattr(args, "flat_cost_limit", TrainConfig.FLAT_COST_LIMIT)),
            flat_window_steps=int(getattr(args, "flat_window_steps", TrainConfig.FLAT_WINDOW_STEPS)),
            kp=TrainConfig.LAGRANGIAN_KP,
            lambda_init=TrainConfig.LAGRANGIAN_LAMBDA_INIT,
            lambda_min=TrainConfig.LAGRANGIAN_LAMBDA_MIN,
            lambda_max=TrainConfig.LAGRANGIAN_LAMBDA_MAX,
        )
        
        return TrainingConfig(
            symbol=args.symbol,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            device=args.device,
            show_progress_bar=show_progress_bar,
            sb3_verbose=sb3_verbose,
            lagrangian_config=lagrangian_config,
            env_config=env_config,
            model_config=model_config,
            eval_config=eval_config,
            max_position_pct=TrainConfig.MAX_POSITION_PCT,
            action_repeat=TrainConfig.ACTION_REPEAT,
            checkpoint_dir_prefix=TrainConfig.CHECKPOINT_DIR_PREFIX,
            checkpoint_save_freq=TrainConfig.CHECKPOINT_SAVE_FREQ,
            vec_monitor_log_prefix=TrainConfig.VEC_MONITOR_LOG_PREFIX,
            update_lambda_every_steps=args.update_lambda_every_steps,
            log_every_episodes=args.log_every_episodes,
            stats_window_episodes=TrainConfig.STATS_WINDOW_EPISODES,
            reward_scale=TrainConfig.REWARD_SCALE,
            cost_penalty_normalize=float(getattr(TrainConfig, "COST_PENALTY_NORMALIZE_FACTOR", 2000.0)),
        )

