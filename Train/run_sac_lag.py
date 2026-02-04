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
from Train.sac_with_aux import SACWithAuxiliaryLoss
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from Env.trading_env import TradingEnvironment
from Env.wrappers import ActionRepeatWrapper, ActionClipWrapper
from Train.eval_callback import (
    ConstraintEvalCallback,
    EvalConfig,
    EvalConstraints,
    TrainEvalStartGateConfig,
)
from Train.optimized_dict_replay_buffer import OptimizedDictReplayBuffer
from Train.sb3_cnn_policy import DualCnnFeatureExtractor
from Train.lagrangian import (
    SharedLagrangianController,
    MultiSharedLagrangianController,
    LagrangianChannelConfig,
    LagrangianRewardWrapper,
    LagrangianCallback,
)
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
        controller: Any = None
    ):
        self.rank = rank
        self.seed = seed
        self.config_overrides = config_overrides or {}
        self.controller = controller

    def __call__(self) -> gym.Env:
        # 1. 基礎環境
        env = TradingEnvironment(env_id=self.rank, random_start=True, **self.config_overrides)
        
        # 2. 動作截斷 (Action Clip)
        # 防止 Agent 輸出極端動作導致手續費暴增
        # 重要：這裡「實際生效」的上限以 TrainConfig.MAX_POSITION_PCT 為主，
        # 因為 wrapper 是在訓練端建立並把參數顯式傳入。
        # Env.config.Config.MAX_POSITION_PCT 只是 env 層的「預設參考值」，除非你在別處用它來建 wrapper。
        env = ActionClipWrapper(env, max_position_pct=TrainConfig.MAX_POSITION_PCT)
        
        # 3. 動作重複 (Action Repeat)
        # 降低決策頻率，穩定訓練
        env = ActionRepeatWrapper(env, repeat=TrainConfig.ACTION_REPEAT)
        
        # 4. Lagrangian Reward Shaping
        # 如果有 Controller，套用 Lagrangian Wrapper 修改獎勵
        if self.controller is not None:
            # reward_scale=10.0: 放大原始獲利，讓 Agent 更有動力去賺錢，而不是只顧著避險
            env = LagrangianRewardWrapper(env, self.controller, reward_scale=TrainConfig.REWARD_SCALE)
            
        return env


def make_env(
    rank: int, 
    seed: int = 0, 
    config_overrides: Dict[str, Any] = None,
    controller: Any = None
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
    parser.add_argument("--cost_limit", type=float, default=TrainConfig.COST_LIMIT, help="Average cost limit per step (legacy single-lambda)")
    parser.add_argument("--risk_cost_limit", type=float, default=TrainConfig.RISK_COST_LIMIT, help="risk cost limit per step (death)")
    parser.add_argument("--fric_cost_limit", type=float, default=TrainConfig.FRIC_COST_LIMIT, help="freq channel cost limit per step (c_freq 0~1), suggest 0.05~0.20")
    parser.add_argument("--sl_buf_cost_limit", type=float, default=TrainConfig.SL_BUF_COST_LIMIT, help="sl_buf cost limit per step (stop buffer, 0~1)")
    parser.add_argument("--sl_event_cost_limit", type=float, default=TrainConfig.SL_EVENT_COST_LIMIT, help="sl_event cost limit per step (stop loss event)")
    parser.add_argument("--trade_freq_cost_limit", type=float, default=TrainConfig.TRADE_FREQ_COST_LIMIT, help="trade_freq cost limit per step (scaled 0~0.001; 0.0001 = 10%% ratio)")
    parser.add_argument("--device", type=str, default=TrainConfig.DEVICE)
    parser.add_argument("--log_every_episodes", type=int, default=TrainConfig.LOG_EVERY_EPISODES, help="每 N 回合輸出交易統計")
    parser.add_argument("--update_lambda_every_steps", type=int, default=TrainConfig.UPDATE_LAMBDA_EVERY_STEPS, help="每 N steps 更新一次 lambda")
    parser.add_argument("--kp", type=float, default=getattr(TrainConfig, "LAGRANGIAN_KP", 0.1), help="Lagrangian P-gain（全通道，越大 λ 反應越快）")
    parser.add_argument("--lambda_init", type=float, default=getattr(TrainConfig, "LAGRANGIAN_LAMBDA_INIT", 0.0), help="λ 初始值（全通道）")
    parser.add_argument("--lambda_max", type=float, default=getattr(TrainConfig, "LAGRANGIAN_LAMBDA_MAX", 5.0), help="λ 上限（全通道）")
    _default_cost_window = getattr(TrainConfig, "COST_WINDOW_STEPS", None)
    parser.add_argument("--cost_window_steps", type=int, default=(int(_default_cost_window) if _default_cost_window is not None else 0), help="cost 平均視窗步數（0=用 update_freq*n_envs，>0 則拉長視窗使 λ 更平滑）")
    # 進度條：預設開啟（避免你忘記加參數而覺得「沒有進度」）
    parser.add_argument("--no_progress_bar", action="store_true", help="關閉 SB3 進度條（預設會顯示）")
    parser.add_argument("--verbose", type=int, default=TrainConfig.SB3_VERBOSE_DEFAULT, help="SB3 verbose 等級（預設：開進度條時=0，否則=1）")
    args = parser.parse_args()

    # progress bar 預設開啟；除非你顯式指定 --no_progress_bar
    show_progress_bar = bool(TrainConfig.SHOW_PROGRESS_BAR_DEFAULT) and (not bool(args.no_progress_bar))
    # 進度條與 SB3 的表格 logger 會互相干擾，因此預設：開進度條時 verbose=0
    sb3_verbose = int(args.verbose) if args.verbose is not None else (0 if show_progress_bar else 1)

    # 1. 初始化 Multi Lagrangian Controller（分四條成本線：risk / fric / sl_buf / sl_event）
    # kp / lambda_init / lambda_max 可由 CLI 覆寫，便於調教 λ（見 docs/lambda_tuning_optimization.md）
    kp = float(args.kp)
    lam_init = float(args.lambda_init)
    lam_max = float(args.lambda_max)
    lag_controller: Any = MultiSharedLagrangianController(
        {
            "risk": LagrangianChannelConfig(cost_limit=float(args.risk_cost_limit), kp=kp, lambda_init=lam_init, lambda_max=lam_max),
            "fric": LagrangianChannelConfig(cost_limit=float(args.fric_cost_limit), kp=kp, lambda_init=lam_init, lambda_max=lam_max),
            "sl_buf": LagrangianChannelConfig(cost_limit=float(args.sl_buf_cost_limit), kp=kp, lambda_init=lam_init, lambda_max=lam_max),
            "sl_event": LagrangianChannelConfig(cost_limit=float(args.sl_event_cost_limit), kp=kp, lambda_init=lam_init, lambda_max=lam_max),
            "trade_freq": LagrangianChannelConfig(cost_limit=float(args.trade_freq_cost_limit), kp=kp, lambda_init=lam_init, lambda_max=lam_max),
        }
    )
    
    # 2. 建立並行環境 (SubprocVecEnv)
    env_kwargs = {
        "target_symbol": args.symbol,
        "window_size": TrainConfig.WINDOW_SIZE_5M,
        "window_size_1d": TrainConfig.WINDOW_SIZE_1D,
        # Deadband：抑制微小調倉（避免 action 抖動導致過度成交/手續費爆炸）
        # 注意：Env/config.py 預設可能是 0.0；訓練入口必須顯式傳入才會生效。
        "min_position_change": float(getattr(TrainConfig, "MIN_POSITION_CHANGE", 0.0)),
        # No-trade hysteresis（方案2：雙門檻），讓 0 倉位更穩定
        "no_trade_entry_threshold": float(getattr(TrainConfig, "NO_TRADE_ENTRY_THRESHOLD", 0.0)),
        "no_trade_exit_threshold": float(getattr(TrainConfig, "NO_TRADE_EXIT_THRESHOLD", 0.0)),
        "idle_penalty_per_step": float(getattr(TrainConfig, "IDLE_PENALTY_PER_STEP", 0.0)),
        # 交易頻率硬限制（與 cost_trade_freq 同視窗）
        "trade_freq_hard_limit": float(getattr(TrainConfig, "TRADE_FREQ_HARD_LIMIT", 0.15)),
        "trade_freq_recovery_ratio": float(getattr(TrainConfig, "TRADE_FREQ_RECOVERY_RATIO", 0.5)),
        # daily_risk_base 更新頻率（用於 max_step_pos_change 的「單步加倉上限」基準）
        # 你希望每 288 steps（一日 5m K 數）才更新一次 base 資金，這裡固定跟隨 WINDOW_SIZE_5M。
        "risk_base_update_steps": int(getattr(TrainConfig, "WINDOW_SIZE_5M", 288)),
        # 固定特徵 symbols：讓 obs 維度包含 ETH/SOL/DOGE/1000PEPE 的跨市場摘要（5m）
        "feature_symbols": list(TrainConfig.FEATURE_SYMBOLS),
        # 資料切分：訓練與評估資料分離（避免資料洩漏）
        "data_split_enabled": bool(getattr(TrainConfig, "DATA_SPLIT_ENABLED", False)),
        "data_mode": "train",
        "holdout_months": int(getattr(TrainConfig, "HOLDOUT_MONTHS", 3)),
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

    model = SACWithAuxiliaryLoss(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=float(TrainConfig.LEARNING_RATE),
        buffer_size=int(TrainConfig.BUFFER_SIZE),  # 經驗回放池大小
        # SB3 memory optimization: reduce replay buffer RAM
        # - Avoid storing next_obs in a separate buffer (saves significant memory for large Dict obs)
        optimize_memory_usage=True,
        replay_buffer_class=OptimizedDictReplayBuffer,
        # Required when optimize_memory_usage=True: disable timeout-specific handling.
        # (Same constraint as SB3 ReplayBuffer; avoids known bug with timeouts + optimized storage)
        replay_buffer_kwargs={"handle_timeout_termination": False},
        batch_size=int(TrainConfig.BATCH_SIZE),
        ent_coef=TrainConfig.ENT_COEF,
        train_freq=int(TrainConfig.TRAIN_FREQ),
        gradient_steps=int(TrainConfig.GRADIENT_STEPS),
        device=args.device,
        verbose=sb3_verbose,
        tensorboard_log=str(TrainConfig.TENSORBOARD_LOG_DIR),
        aux_coef=float(getattr(TrainConfig, "AUX_COEF", 0.1)),
    )

    # 4. 設定 Callbacks
    # LagrangianCallback: 更新 λ 與顯示交易統計；cost_window_steps>0 時拉長 cost 平均視窗使 λ 更平滑
    cost_window = int(args.cost_window_steps) if getattr(args, "cost_window_steps", 0) and int(getattr(args, "cost_window_steps", 0)) > 0 else None
    lag_callback = LagrangianCallback(
        controller=lag_controller,
        update_freq=int(args.update_lambda_every_steps),  # 每 N 步更新一次 λ
        log_freq=int(args.log_every_episodes),            # 每 N episodes 顯示一次交易狀態
        window_size=int(TrainConfig.STATS_WINDOW_EPISODES),# 統計視窗大小
        reward_scale=float(TrainConfig.REWARD_SCALE),     # 與 Env wrapper 一致
        cost_window_steps=cost_window,                     # None 則用 update_freq*n_envs；設大則 λ 更新更平滑
    )
    
    # CheckpointCallback: 定期存檔
    checkpoint_callback = CheckpointCallback(
        save_freq=int(TrainConfig.CHECKPOINT_SAVE_FREQ),
        save_path=f"{TrainConfig.CHECKPOINT_DIR_PREFIX}_{args.symbol}",
        name_prefix="sac_lag"
    )

    callbacks = [lag_callback, checkpoint_callback]

    # 4.1 Periodic Evaluation（每 N steps 驗證 + 保存 best model）
    eval_env = None
    if bool(getattr(TrainConfig, "EVAL_ENABLED", False)):
        def _make_eval_env() -> gym.Env:
            # 與訓練環境維持一致的 wrappers（clip/repeat），但固定起點避免指標抖動
            eval_kwargs = dict(env_kwargs)
            eval_kwargs.update(
                {
                    "random_start": bool(TrainConfig.EVAL_RANDOM_START),
                    "max_episode_steps": int(TrainConfig.EVAL_MAX_EPISODE_STEPS),
                    # 評估最短長度：觀測暖機(1d window) + 評估回合長度
                    "min_episode_steps": int(TrainConfig.EVAL_MAX_EPISODE_STEPS),
                    # eval 用最近 N 個月
                    "data_mode": "eval",
                    # eval 的 obs 必須填滿（起點 >= max(WINDOW_SIZE_5M, WINDOW_SIZE_1D*288)）
                    "ensure_filled_obs": True,
                    # ---- Render (EVAL) ----
                    "render_enabled": False,
                    "render_save": False,
                    "render_show": False,
                    # VecEnv 會在 done 時自動 reset；因此必須在「終止那一步」就 render，並把路徑塞回 info。
                    "render_on_done": False,
                    "render_dir": os.path.join("logs", "renders", f"eval_{args.symbol}"),
                }
            )
            e = TradingEnvironment(env_id=9999, **eval_kwargs)
            e = ActionClipWrapper(e, max_position_pct=TrainConfig.MAX_POSITION_PCT)
            e = ActionRepeatWrapper(e, repeat=TrainConfig.ACTION_REPEAT)
            return e

        eval_env = DummyVecEnv([_make_eval_env])

        best_model_base_dir = f"{TrainConfig.CHECKPOINT_DIR_PREFIX}_{args.symbol}"
        best_model_path = os.path.join(best_model_base_dir, str(TrainConfig.EVAL_BEST_MODEL_SUBDIR), "best_model")
        eval_cb = ConstraintEvalCallback(
            eval_env=eval_env,
            eval_config=EvalConfig(
                enabled=True,
                eval_every_timesteps=int(TrainConfig.EVAL_EVERY_TIMESTEPS),
                n_eval_episodes=int(TrainConfig.EVAL_N_EVAL_EPISODES),
                deterministic=bool(TrainConfig.EVAL_DETERMINISTIC),
                reject_if_death_event=bool(TrainConfig.EVAL_REJECT_IF_DEATH_EVENT),
                constraints=EvalConstraints(
                    max_dd_limit=float(TrainConfig.EVAL_MAX_DD_LIMIT),
                    mean_cost_limit=float(TrainConfig.EVAL_MEAN_COST_LIMIT),
                ),
                train_start_gate=TrainEvalStartGateConfig(
                    enabled=True,
                    window_size=100,
                    min_max_steps_reached_count=70,
                ),
                save_best_model=bool(TrainConfig.EVAL_SAVE_BEST_MODEL),
                best_model_path=str(best_model_path),
                print_each_episode=bool(getattr(TrainConfig, "EVAL_PRINT_EACH_EPISODE", True)),
                print_prefix=str(getattr(TrainConfig, "EVAL_PRINT_PREFIX", "[EVAL]")),
                render_each_episode=True,
            ),
        )
        callbacks.append(eval_cb)

    print(f"Start training SAC-Lagrangian on {args.symbol} with {args.n_envs} envs...")
    print(
        "Cost Limits (per step): "
        f"risk={args.risk_cost_limit}, fric={args.fric_cost_limit}, sl_buf={args.sl_buf_cost_limit}, sl_event={args.sl_event_cost_limit}, trade_freq={args.trade_freq_cost_limit} "
        f"(legacy cost_limit={args.cost_limit})"
    )
    
    model.learn(
        total_timesteps=args.total_timesteps, 
        callback=callbacks,
        progress_bar=bool(show_progress_bar),
    )
    
    # 5. 存檔與關閉
    model.save(f"models/sac_lag_{args.symbol}/final_model")
    env.close()
    if eval_env is not None:
        eval_env.close()
    print("Training finished.")


if __name__ == "__main__":
    # Windows 下 multiprocessing 必須在 if __name__ == "__main__": 下執行
    multiprocessing.freeze_support()
    main()
