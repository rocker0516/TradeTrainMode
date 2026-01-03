from __future__ import annotations

import argparse
import os
import time
import multiprocessing
import sys
from typing import Any, Dict

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

import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from Env.trading_env import TradingEnvironment
from Env.wrappers import ActionRepeatWrapper, ActionSmoothClipWrapper
from Train.sb3_cnn_policy import DualCnnFeatureExtractor
from Train.lagrangian import SharedLagrangianController, LagrangianRewardWrapper, LagrangianCallback
from Train.train_config import TrainConfig


class EnvFactory:
    """
    環境工廠類別（Functor）。
    
    解決 Windows multiprocessing (spawn mode) 無法 pickle 巢狀函數 (Local Function) 的問題。
    將環境建置邏輯封裝在此類別中，SubprocVecEnv 可以安全地序列化此物件。
    """
    def __init__(
        self, 
        rank: int, 
        seed: int = 0, 
        config_overrides: Dict[str, Any] = None, 
        controller: SharedLagrangianController = None
    ):
        self.rank = rank
        self.seed = seed
        self.config_overrides = config_overrides or {}
        self.controller = controller

    def __call__(self) -> gym.Env:
        # 1. 基礎環境
        env = TradingEnvironment(env_id=self.rank, random_start=True, **self.config_overrides)
        
        # 2. 動作平滑與截斷 (Action Smooth & Clip)
        # 防止 Agent 輸出極端動作導致手續費暴增
        env = ActionSmoothClipWrapper(env, max_position_pct=1.0, smoothing_alpha=0.5)
        
        # 3. 動作重複 (Action Repeat)
        # 降低決策頻率，穩定訓練
        env = ActionRepeatWrapper(env, repeat=1)
        
        # 4. Lagrangian Reward Shaping
        # 如果有 Controller，套用 Lagrangian Wrapper 修改獎勵
        if self.controller is not None:
            # reward_scale=10.0: 放大原始獲利，讓 Agent 更有動力去賺錢，而不是只顧著避險
            env = LagrangianRewardWrapper(env, self.controller, reward_scale=10.0)
            
        return env


def make_env(
    rank: int, 
    seed: int = 0, 
    config_overrides: Dict[str, Any] = None,
    controller: SharedLagrangianController = None
):
    """
    輔助函式：回傳 EnvFactory 實例。
    SubprocVecEnv 接受 callable，EnvFactory 實作了 __call__，所以本身就是 callable。
    """
    return EnvFactory(rank, seed, config_overrides, controller)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAC Lagrangian Training")
    # 預設值全部由 TrainConfig 提供；CLI 參數只做覆寫
    parser.add_argument("--symbol", type=str, default=TrainConfig.SYMBOL)
    parser.add_argument("--total_timesteps", type=int, default=TrainConfig.TOTAL_TIMESTEPS)
    parser.add_argument("--n_envs", type=int, default=TrainConfig.N_ENVS, help="Number of parallel environments")
    parser.add_argument("--cost_limit", type=float, default=TrainConfig.COST_LIMIT, help="Average cost limit per step")
    parser.add_argument("--device", type=str, default=TrainConfig.DEVICE)
    parser.add_argument("--log_every_episodes", type=int, default=TrainConfig.LOG_EVERY_EPISODES, help="每 N 回合輸出交易統計")
    parser.add_argument("--update_lambda_every_steps", type=int, default=TrainConfig.UPDATE_LAMBDA_EVERY_STEPS, help="每 N steps 更新一次 lambda")
    # 進度條：預設開啟（避免你忘記加參數而覺得「沒有進度」）
    parser.add_argument("--no_progress_bar", action="store_true", help="關閉 SB3 進度條（預設會顯示）")
    parser.add_argument("--verbose", type=int, default=TrainConfig.SB3_VERBOSE_DEFAULT, help="SB3 verbose 等級（預設：開進度條時=0，否則=1）")
    args = parser.parse_args()

    # progress bar 預設開啟；除非你顯式指定 --no_progress_bar
    show_progress_bar = bool(TrainConfig.SHOW_PROGRESS_BAR_DEFAULT) and (not bool(args.no_progress_bar))
    # 進度條與 SB3 的表格 logger 會互相干擾，因此預設：開進度條時 verbose=0
    sb3_verbose = int(args.verbose) if args.verbose is not None else (0 if show_progress_bar else 1)

    # 1. 初始化 Shared Lagrangian Controller
    # 設定 cost_limit=0.05，代表我們容許每步平均產生 0.05 的風險成本 (約等於 5% 的風險程度)
    lag_controller = SharedLagrangianController(
        cost_limit=args.cost_limit,
        kp=0.1,          # P-gain
        lambda_init=0.0,
        lambda_max=5.0   # Clamp 上限，防躺平
    )
    
    # 2. 建立並行環境 (SubprocVecEnv)
    env_kwargs = {
        "target_symbol": args.symbol,
        "window_size": TrainConfig.WINDOW_SIZE_5M,
        "window_size_1d": TrainConfig.WINDOW_SIZE_1D,
        # 這裡可以覆寫 Env/config.py 的預設值
    }
    
    # 建立 N 個並行進程，使用 EnvFactory 以支援 Windows spawn
    env = SubprocVecEnv([
        make_env(i, seed=i, config_overrides=env_kwargs, controller=lag_controller) 
        for i in range(args.n_envs)
    ])
    
    # VecMonitor: 負責記錄每個 episode 的 reward/length 到 TensorBoard
    # 同時也會自動處理 Reset，讓 SubprocVecEnv 的輸出符合 Gym 介面
    env = VecMonitor(env, filename=f"logs/sac_lag_{args.symbol}")

    # 3. 建立 SAC 模型
    policy_kwargs = dict(
        features_extractor_class=DualCnnFeatureExtractor,
        features_extractor_kwargs=dict(
            emb_5m=TrainConfig.EMB_5M,
            emb_1d=TrainConfig.EMB_1D,
            emb_vec=TrainConfig.EMB_VEC,
            out_dim=TrainConfig.OUT_DIM,
        ),
        net_arch=dict(pi=list(TrainConfig.PI_ARCH), qf=list(TrainConfig.QF_ARCH)),
    )

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=float(TrainConfig.LEARNING_RATE),
        buffer_size=int(TrainConfig.BUFFER_SIZE),  # 經驗回放池大小
        batch_size=int(TrainConfig.BATCH_SIZE),
        ent_coef=TrainConfig.ENT_COEF,
        train_freq=int(TrainConfig.TRAIN_FREQ),
        gradient_steps=int(TrainConfig.GRADIENT_STEPS),
        device=args.device,
        verbose=sb3_verbose,
        tensorboard_log=str(TrainConfig.TENSORBOARD_LOG_DIR),
    )

    # 4. 設定 Callbacks
    # LagrangianCallback: 更新 λ 與顯示交易統計
    lag_callback = LagrangianCallback(
        controller=lag_controller,
        update_freq=int(args.update_lambda_every_steps),  # 每 N 步更新一次 λ
        log_freq=int(args.log_every_episodes)             # 每 N episodes 顯示一次交易狀態
    )
    
    # CheckpointCallback: 定期存檔
    checkpoint_callback = CheckpointCallback(
        save_freq=int(TrainConfig.CHECKPOINT_SAVE_FREQ),
        save_path=f"{TrainConfig.CHECKPOINT_DIR_PREFIX}_{args.symbol}",
        name_prefix="sac_lag"
    )

    print(f"Start training SAC-Lagrangian on {args.symbol} with {args.n_envs} envs...")
    print(f"Cost Limit: {args.cost_limit}, Lambda Max: {lag_controller.lambda_max}")
    
    model.learn(
        total_timesteps=args.total_timesteps, 
        callback=[lag_callback, checkpoint_callback],
        progress_bar=bool(show_progress_bar),
    )
    
    # 5. 存檔與關閉
    model.save(f"models/sac_lag_{args.symbol}/final_model")
    env.close()
    print("Training finished.")


if __name__ == "__main__":
    # Windows 下 multiprocessing 必須在 if __name__ == "__main__": 下執行
    multiprocessing.freeze_support()
    main()
