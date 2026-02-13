from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
import time
from typing import Any

# ---------------------------
# 路徑修正（Windows 直跑腳本常見問題）
# ---------------------------
# 你使用：
#   & .conda/python.exe Train/run_sac_lag.py
# 時，Python 的 sys.path 不一定包含「專案根目錄」，導致 `import Env` 失敗。
# 這裡強制把專案根（Train/ 的上一層）加入 sys.path，確保可直接執行腳本。
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 線程控制：在 SubprocVecEnv 多進程下減少 oversubscription，提升 it/s
# 必須在 import numpy/torch 之前設定，子進程會繼承
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from Train.train_config import TrainConfig
from Train.config_builder import TrainingConfigBuilder
from Train.env_builder import EnvironmentBuilder
from Train.model_builder import ModelBuilder
from Train.callback_builder import CallbackBuilder
from Train.trainer import SACLagrangianTrainer
from Train.lagrangian import MultiSharedLagrangianController
from Env.load_file import load_data

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# 注意：EnvFactory 已移至 EnvironmentBuilder，这里保留空注释以保持兼容性


def _create_env_builder(config: Any) -> EnvironmentBuilder:
    """依 config 建立 EnvironmentBuilder，供 profile 與 main 共用，避免重複建構。"""
    builder = EnvironmentBuilder(
        config=config.env_config,
        max_position_pct=config.max_position_pct,
        action_repeat=config.action_repeat,
        reward_scale=config.reward_scale,
        cost_penalty_normalize=getattr(config, "cost_penalty_normalize", 2000.0),
        cost_penalty_normalize_per_channel=getattr(config, "cost_penalty_normalize_per_channel", None),
    )
    return builder


def run_profiling(config: Any, n_steps: int) -> None:
    """
    跑 n_steps 個 env step，量測 env.step() 與 model.predict() 的時間佔比，找出 it/s 瓶頸。
    不寫入 checkpoint、不跑 eval。
    """
    from stable_baselines3.common.vec_env import VecEnv
    logger.info("Profile mode: creating env and model (no callbacks)...")
    df_5m, df_1d = load_data()
    env_builder = _create_env_builder(config)
    env_builder.controller = MultiSharedLagrangianController(config.lagrangian_config.to_channel_configs())
    log_prefix = f"{config.vec_monitor_log_prefix}_{config.symbol}"
    vec_env: VecEnv = env_builder.create_training_vec_env(
        n_envs=config.n_envs, log_prefix=log_prefix, df_5m=df_5m, df_1d=df_1d
    )
    model_builder = ModelBuilder(config.model_config)
    model = model_builder.build(vec_env)
    reset_out = vec_env.reset()
    obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
    n_envs = vec_env.num_envs

    t_step_total = 0.0
    t_predict_total = 0.0
    steps_done = 0
    target_steps = max(n_envs, n_steps)
    try:
        while steps_done < target_steps:
            t0 = time.perf_counter()
            actions, _ = model.predict(obs, deterministic=False)
            t_predict_total += time.perf_counter() - t0
            t1 = time.perf_counter()
            obs, _, _, _ = vec_env.step(actions)
            t_step_total += time.perf_counter() - t1
            steps_done += n_envs
    finally:
        vec_env.close()

    total = t_step_total + t_predict_total
    inv_total = 1.0 / total if total > 0 else 0.0
    steps_per_sec = steps_done * inv_total
    pct_step = 100.0 * t_step_total * inv_total
    pct_predict = 100.0 * t_predict_total * inv_total
    logger.info("[Profile] %d steps in %.2f s => %.1f it/s", steps_done, total, steps_per_sec)
    logger.info("[Profile] env.step(): %.2f s (%.1f%%); model.predict(): %.2f s (%.1f%%)", t_step_total, pct_step, t_predict_total, pct_predict)
    if t_step_total > t_predict_total:
        logger.info("[Profile] 瓶頸在 env.step（可縮小 obs、減少每步分配、或檢查 SubprocVecEnv IPC）")
    else:
        logger.info("[Profile] 瓶頸在 policy predict（治本：--device cuda 或 --compile_policy）")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。
    
    Returns:
        解析后的命令行参数
    """
    parser = argparse.ArgumentParser(description="SAC Lagrangian Training")
    # 預設值全部由 TrainConfig 提供；CLI 參數只做覆寫
    parser.add_argument("--symbol", type=str, default=TrainConfig.SYMBOL, help="交易標的符號")
    parser.add_argument("--total_timesteps", type=int, default=TrainConfig.TOTAL_TIMESTEPS, help="總訓練步數")
    parser.add_argument("--n_envs", type=int, default=TrainConfig.N_ENVS, help="並行環境數量")
    parser.add_argument("--cost_limit", type=float, default=TrainConfig.COST_LIMIT, help="平均成本限制（單一 Lambda 模式，已棄用）")
    parser.add_argument("--risk_cost_limit", type=float, default=TrainConfig.RISK_COST_LIMIT, help="風險成本限制（死亡風險通道）")
    parser.add_argument("--fric_cost_limit", type=float, default=TrainConfig.FRIC_COST_LIMIT, help="摩擦成本限制（手續費通道）")
    parser.add_argument("--trade_freq_cost_limit", type=float, default=TrainConfig.TRADE_FREQ_COST_LIMIT, help="交易頻率成本限制（最近 N 步內交易比例上限，0~1）")
    parser.add_argument("--trade_freq_window_steps", type=int, default=TrainConfig.TRADE_FREQ_WINDOW_STEPS, help="交易頻率約束的窗口步數 N")
    parser.add_argument("--flat_cost_limit", type=float, default=TrainConfig.FLAT_COST_LIMIT, help="空倉成本限制（最近 N 步內空倉比例上限，0~1；鼓勵持倉、允許避險）")
    parser.add_argument("--flat_window_steps", type=int, default=TrainConfig.FLAT_WINDOW_STEPS, help="空倉比例約束的窗口步數 N")
    parser.add_argument("--device", type=str, default=TrainConfig.DEVICE, help="計算設備：'cuda' / 'cpu' / 'auto'")
    parser.add_argument("--log_every_episodes", type=int, default=TrainConfig.LOG_EVERY_EPISODES, help="每 N 回合輸出交易統計")
    parser.add_argument("--update_lambda_every_steps", type=int, default=TrainConfig.UPDATE_LAMBDA_EVERY_STEPS, help="每 N steps 更新一次 lambda")
    parser.add_argument("--train_freq", type=int, default=TrainConfig.TRAIN_FREQ, help="每 N 個 env step 做一次梯度更新")
    parser.add_argument("--gradient_steps", type=int, default=TrainConfig.GRADIENT_STEPS, help="每次更新時的梯度步數")
    parser.add_argument("--compile_policy", action="store_true", help="對 features_extractor 做 torch.compile（PyTorch 2+，瓶頸在 GPU 時可試）")
    # 進度條：預設開啟（避免你忘記加參數而覺得「沒有進度」）
    parser.add_argument("--no_progress_bar", action="store_true", help="關閉 SB3 進度條（預設會顯示）")
    parser.add_argument("--verbose", type=int, default=TrainConfig.SB3_VERBOSE_DEFAULT, help="SB3 verbose 等級（預設：開進度條時=0，否則=1）")
    parser.add_argument("--profile_steps", type=int, default=0, help="若 >0：僅跑此步數並輸出 env/predict 時間佔比，找出 it/s 瓶頸後即結束")
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="從此路徑載入模型繼續訓練（目錄如 models/sac_lag_BTCUSDT/best_model，或 .zip 路徑）",
    )
    return parser.parse_args()


def main() -> None:
    """
    SAC Lagrangian 训练主入口。
    
    功能：
    1. 解析命令行参数
    2. 构建训练配置
    3. 创建构建器（环境、模型、回调）
    4. 创建训练器
    5. 执行训练
    
    Raises:
        ValueError: 配置参数无效
        RuntimeError: 环境或模型创建失败
        KeyboardInterrupt: 用户中断训练
    """
    try:
        # 1. 解析命令行参数
        args = parse_args()
        
        # 2. 構建訓練配置
        config = TrainingConfigBuilder.from_cli_args(args)
        per_ch = getattr(config, "cost_penalty_normalize_per_channel", None)
        if per_ch:
            logger.info("Cost penalty per-channel norms (used by LagrangianRewardWrapper): %s", per_ch)
        profile_steps = getattr(args, "profile_steps", 0)
        if profile_steps > 0:
            run_profiling(config, profile_steps)
            return

        # 3. 主進程 PyTorch 線程數：多 env 時避免與子進程搶 CPU，有助提升 it/s
        try:
            import torch
            n_envs = config.n_envs
            n_cpu = multiprocessing.cpu_count()
            # 留出約一半 CPU 給 env 子進程，主進程用於 policy + 訓練
            ideal_threads = max(1, min(8, n_cpu // 2))
            if n_envs >= 8:
                torch.set_num_threads(ideal_threads)
        except Exception:
            pass
        
        # 4. 创建构建器
        env_builder = _create_env_builder(config)
        model_builder = ModelBuilder(config.model_config)
        callback_builder = CallbackBuilder(config)
        
        # 5. 创建训练器
        trainer = SACLagrangianTrainer(
            config=config,
            env_builder=env_builder,
            model_builder=model_builder,
            callback_builder=callback_builder,
        )
        
        # 6. 执行训练
        trainer.train()
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        raise
    except (ValueError, RuntimeError) as e:
        logger.error("%s: %s", type(e).__name__, e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise RuntimeError(f"Training failed: {e}") from e


if __name__ == "__main__":
    # Windows 下 multiprocessing 必須在 if __name__ == "__main__": 下執行
    multiprocessing.freeze_support()
    main()
