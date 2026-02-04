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
    # 特徵 symbols（固定順序、固定維度）：
    # - 用於 5m 跨市場摘要 +（後續可擴充）多幣 1d regime
    # - 注意：Gym observation_space 必須固定 shape，因此這裡用「固定清單」，而不是隨 Data 目錄動態增減。
    FEATURE_SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT")
    TOTAL_TIMESTEPS: int = 100_000_000
    N_ENVS: int = 64
    DEVICE: str = "auto"  # "cuda" / "cpu" / "auto"

    # ---- Lagrangian / 約束 ----
    # 新版 Cost 已正規化為 Cost/Equity。
    # 建議值 0.0005 (5bps) 代表容許每步平均損耗 0.05% 的權益 (含手續費與死亡風險攤提)
    COST_LIMIT: float = 0.00008
    # 新版：雙路徑成本限制（risk / friction）- 目前 Controller 尚未完全支援分開的 dual-lambda，
    # 但保留參數供未來擴充。邏輯同上，Risk 應趨近於 0，Fric 容許少量。
    RISK_COST_LIMIT: float = 0.001 # 0.000005 代表 0.0005% 死亡風險(容許極小風險)
    FRIC_COST_LIMIT: float = 0.00005 # 0.002：代表允許每步平均手續費佔權益 0.2%
    # Stop-Buffer Cost（0~1）：建議先設很小的平均步成本上限，因為「接近止損」應該是短暫狀態
    SL_BUF_COST_LIMIT: float = 0.01 # 0.1：代表允許每步平均止損緩衝成本佔權益 10%
    # Stop-Loss Event Cost（事件型）：當步觸發止損時才會出現的成本（獨立成本線，不歸類到 risk）。
    SL_EVENT_COST_LIMIT: float = 0.002 # 0.01：代表允許每步平均止損事件成本佔權益 1%
    
    UPDATE_LAMBDA_EVERY_STEPS: int = 1
    # 交易統計輸出：
    # - LOG_EVERY_EPISODES: 每 N 個 episode 刷新一次統計（建議：20）
    # - STATS_WINDOW_EPISODES: 統計最多取最近 M 個 episode（滾動視窗，建議：100）
    LOG_EVERY_EPISODES: int = 50
    STATS_WINDOW_EPISODES: int = 100
    REWARD_SCALE: float = 1 # 獎勵尺度 1.0 代表獎勵不放大

    # ---- SB3 SAC 超參數 ----
    LEARNING_RATE: float = 5e-5
    BUFFER_SIZE: int = 1_600_000
    BATCH_SIZE: int = 256 # 512 / 1_500_000 = 0.034% 
    ENT_COEF: str = "auto"
    TRAIN_FREQ: int = 1 # 1 代表每次更新參數時，只用一個 batch 的資料
    GRADIENT_STEPS: int = 1 # 1 代表每次更新參數時，只用一個 batch 的資料進行梯度下降

    # ---- Policy / 網路結構 ----
    # 新的双CNN架构参数
    EMB_5M_TARGET: int = 128  # Target 5m 特征维度
    EMB_5M_OTHERS: int = 64   # Others 5m 特征维度
    EMB_1D_TARGET: int = 64   # Target 1d 特征维度
    EMB_1D_OTHERS: int = 32   # Others 1d 特征维度
    EMB_VEC: int = 128        # 向量特征维度
    OUT_DIM: int = 256        # 输出维度
    # 兼容性：保留旧参数（用于向后兼容）
    EMB_5M: int = 256
    EMB_1D: int = 128
    PI_ARCH: tuple[int, int] = (256, 256) # P 網路結構
    QF_ARCH: tuple[int, int] = (128, 128) # Q 網路結構

    # ---- Wrapper（動作平滑/重複）----
    # 訓練時建議用「更強的降頻/降換手」設定，否則手續費與 turnover 會把主線 log-return 磨成長期負值。
    # 這些會由 Train/run_sac_lag.py 以 CLI 參數覆寫（不必動 Env/config.py 的全域預設）。
    ACTION_REPEAT: int = 1 # 5 代表 5 步一決策
    # 作用：訓練入口 `Train/run_sac_lag.py` 會用這個值建立 `ActionClipWrapper`，
    # 用來限制 agent 的「目標持倉百分比」在 [-MAX_POSITION_PCT, +MAX_POSITION_PCT]。
    # 優先順序：在 run_sac_lag 訓練流程中，此值會「覆蓋」Env.config.Config.MAX_POSITION_PCT（因為此處是顯式傳參）。
    MAX_POSITION_PCT: float = 0.7
    MIN_POSITION_CHANGE: float = 0.2

    # ---- No-trade hysteresis（方案2：雙門檻）----
    # 建議先用很小的值觀察 trade_steps 與 fees 是否下降：
    # - entry：空倉時需有明確訊號才進場
    # - exit ：有倉時較容易回到空倉（避免 0 附近抖動）
    # 注意：單位是目標倉位比例（position pct），並且動作會先被 clip 到 [-MAX_POSITION_PCT, +MAX_POSITION_PCT]。
    NO_TRADE_ENTRY_THRESHOLD: float = MIN_POSITION_CHANGE
    NO_TRADE_EXIT_THRESHOLD: float = 0.1

    # ---- 環境參數 ----
    WINDOW_SIZE_5M: int = 288
    WINDOW_SIZE_1D: int = 14

    # ---- 資料切分（Train/Eval 分離）----
    # 需求：評估資料使用「最近 N 個月」，剩餘資料作為訓練資料
    DATA_SPLIT_ENABLED: bool = True
    HOLDOUT_MONTHS: int = 3

    # ---- Log / Checkpoint ----
    TENSORBOARD_LOG_DIR: str = "logs/sac_lag_tb"
    VEC_MONITOR_LOG_PREFIX: str = "logs/sac_lag"
    CHECKPOINT_SAVE_FREQ: int = 1_000_000
    CHECKPOINT_DIR_PREFIX: str = "models/sac_lag"

    # ---- Periodic Evaluation / Validation（每 N steps 驗證）----
    # 目的：
    # - 訓練期間定期跑 eval episodes，輸出你關心的指標（止損/強平/交易頻率/進場/平均持倉/最終資金/收益率）
    # - 依規則挑選並保存 best model
    EVAL_ENABLED: bool = True
    # 以「訓練總 timesteps」為基準（使用 SB3 的 model.num_timesteps），確保在 VecEnv 下語意正確
    EVAL_EVERY_TIMESTEPS: int = 1_000_000
    EVAL_N_EVAL_EPISODES: int = 25
    EVAL_DETERMINISTIC: bool = True
    # Eval 環境設定：避免隨機起點使指標不穩定（可重現）
    EVAL_RANDOM_START: bool = True
    # 為了避免 eval 回合過長拖慢訓練：允許在 eval 端覆寫 episode 上限
    EVAL_MAX_EPISODE_STEPS: int = 288*14
    # Eval 每回合在 Terminal 顯示一行摘要
    EVAL_PRINT_EACH_EPISODE: bool = True
    EVAL_PRINT_PREFIX: str = "[EVAL]"

    # ---- Best model 保存 ----
    EVAL_SAVE_BEST_MODEL: bool = True
    # 實際保存路徑會在訓練入口用 symbol 組合（避免 TrainConfig 內直接格式化）
    EVAL_BEST_MODEL_SUBDIR: str = "best_model"

    # ---- Best model 規則：constraints_then_balance + 排除死亡事件 ----
    # 1) 任何「死亡事件」出現 => 整次 eval 不合格（不更新 best）
    # - termination_reason in {"liq_triggered", "balance_insufficient"}
    # - 或 episode_liq_count > 0
    EVAL_REJECT_IF_DEATH_EVENT: bool = True
    # 2) 只在通過約束時才允許更新 best
    EVAL_MAX_DD_LIMIT: float = 0.40 # 0.40 代表 40%
    EVAL_MEAN_COST_LIMIT: float = 0.01 # 0.01 代表 1%

    # ---- 指標口徑（固定）----
    # - holding_ratio = episode_holding_steps / episode_steps
    # - trades_per_step = episode_trade_count / episode_steps

    # ---- UI / Console ----
    # 進度條預設開啟；若你想讓 SB3 的表格輸出（verbose=1）更乾淨，可把預設 verbose 設 0
    SHOW_PROGRESS_BAR_DEFAULT: bool = True
    SB3_VERBOSE_DEFAULT: int | None = None  # None 代表由 run_sac_lag.py 依 progress bar 自動決定


