from __future__ import annotations

"""
訓練端設定檔（TrainConfig）

說明：
- 你原本的 `Train/config.py` 是「相容性 shim」：只做 `from Env.config import Config`，不應塞訓練參數。
- 因此訓練腳本 `Train/run_sac_lag.py` 的參數預設值，集中放在此檔，避免 Env/Train 設定混雜。

使用方式：
    from Train.train_config import TrainConfig
    default_n_envs = TrainConfig.N_ENVS
"""


class TrainConfig:
    """SAC-Lagrangian 訓練參數預設值（可由 CLI 覆寫）。"""

    # ---- 基本訓練參數 ----
    SYMBOL: str = "BTCUSDT"
    TOTAL_TIMESTEPS: int = 10_000_000
    N_ENVS: int = 64
    DEVICE: str = "auto"  # "cuda" / "cpu" / "auto"

    # ---- Lagrangian / 約束 ----
    COST_LIMIT: float = 0.05
    UPDATE_LAMBDA_EVERY_STEPS: int = 1000
    LOG_EVERY_EPISODES: int = 100
    REWARD_SCALE: float = 10.0

    # ---- SB3 SAC 超參數 ----
    LEARNING_RATE: float = 3e-4
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 256
    ENT_COEF: str = "auto"
    TRAIN_FREQ: int = 1
    GRADIENT_STEPS: int = 1

    # ---- Policy / 網路結構 ----
    EMB_5M: int = 128
    EMB_1D: int = 64
    EMB_VEC: int = 128
    OUT_DIM: int = 256
    PI_ARCH: tuple[int, int] = (256, 256)
    QF_ARCH: tuple[int, int] = (256, 256)

    # ---- Wrapper（動作平滑/重複）----
    ACTION_SMOOTH_ALPHA: float = 0.5
    ACTION_REPEAT: int = 1
    MAX_POSITION_PCT: float = 0.8 # 最大持倉比例 1.0 代表 100%

    # ---- 環境參數 ----
    WINDOW_SIZE_5M: int = 288 * 3
    WINDOW_SIZE_1D: int = 7

    # ---- Log / Checkpoint ----
    TENSORBOARD_LOG_DIR: str = "logs/sac_lag_tb"
    VEC_MONITOR_LOG_PREFIX: str = "logs/sac_lag"
    CHECKPOINT_SAVE_FREQ: int = 50_000
    CHECKPOINT_DIR_PREFIX: str = "models/sac_lag"

    # ---- UI / Console ----
    # 進度條預設開啟；若你想讓 SB3 的表格輸出（verbose=1）更乾淨，可把預設 verbose 設 0
    SHOW_PROGRESS_BAR_DEFAULT: bool = True
    SB3_VERBOSE_DEFAULT: int | None = None  # None 代表由 run_sac_lag.py 依 progress bar 自動決定


