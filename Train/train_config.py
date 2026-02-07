from __future__ import annotations

"""
訓練端設定檔（TrainConfig）

說明：
- 原本的 `Train/config.py` 是「相容性 shim」：只做 `from Env.config import Config`，不應塞訓練參數。
- 因此訓練腳本 `Train/run_sac_lag.py` 的參數預設值，集中放在此檔，避免 Env/Train 設定混雜。

使用方式：
    from Train.train_config import TrainConfig
    default_n_envs = TrainConfig.N_ENVS
"""


class TrainConfig:
    """
    SAC-Lagrangian 訓練參數預設值（可由 CLI 覆寫）。
    
    所有參數都可在 `Train/run_sac_lag.py` 中透過命令行參數覆寫。
    """

    # ==================== 基本訓練參數 ====================
    SYMBOL: str = "BTCUSDT"
    """交易標的符號"""
    
    FEATURE_SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT")
    """
    特徵 symbols（固定順序、固定維度）
    - 用於 5m 跨市場摘要 +（後續可擴充）多幣 1d regime
    - 注意：Gym observation_space 必須固定 shape，因此這裡用「固定清單」，
      而不是隨 Data 目錄動態增減。
    """
    
    TOTAL_TIMESTEPS: int = 100_000_000
    """總訓練步數"""
    
    N_ENVS: int = 64
    """並行環境數量"""
    
    DEVICE: str = "auto"
    """計算設備：'cuda' / 'cpu' / 'auto'"""

    # ==================== Lagrangian / 約束 ====================
    COST_LIMIT: float = 0.00008
    """
    總成本限制（單一 Lambda 模式，已棄用）
    - 新版 Cost 已正規化為 Cost/Equity
    - 建議值 0.00008 (0.8bps) 代表容許每步平均損耗 0.008% 的權益
    - 注意：目前使用多通道模式（RISK_COST_LIMIT / FRIC_COST_LIMIT），此參數僅供相容性
    """
    
    RISK_COST_LIMIT: float = 0.001
    """
    風險成本限制（死亡風險通道）
    - 單位：每步平均死亡風險（正規化為 Cost/Equity）
    - 0.001 代表容許 0.1% 的死亡風險（建議設為接近 0 的值）
    """
    
    FRIC_COST_LIMIT: float = 0.00005
    """
    摩擦成本限制（手續費通道）
    - 單位：每步平均手續費佔權益比例（正規化為 Cost/Equity）
    - 0.00005 代表允許每步平均手續費佔權益 0.005%
    - 注意：目前 cost_fric 固定為 0.0，此參數供未來擴充使用
    """
    
    UPDATE_LAMBDA_EVERY_STEPS: int = 1_000
    """
    Lambda 更新頻率（每 N 個 global steps 更新一次）
    - 建議值：1000（在 n_envs=64 時，約每 15.6 個訓練回合更新一次）
    """
    
    LOG_EVERY_EPISODES: int = 50
    """每 N 個 episode 刷新一次統計輸出（建議：20-50）"""
    
    STATS_WINDOW_EPISODES: int = 100
    """統計視窗大小：最多取最近 M 個 episode（滾動視窗，建議：100）"""
    
    REWARD_SCALE: float = 1.0
    """獎勵尺度：1.0 代表獎勵不放大，>1.0 會放大獎勵信號"""

    # ==================== SB3 SAC 超參數 ====================
    LEARNING_RATE: float = 5e-5
    """學習率"""
    
    BUFFER_SIZE: int = 1_600_000
    """Replay Buffer 大小"""
    
    BATCH_SIZE: int = 256
    """訓練批次大小"""
    
    ENT_COEF: str = "auto"
    """熵係數：'auto' 表示自動調整，或指定數值（如 0.01）"""
    
    TRAIN_FREQ: int = 1
    """訓練頻率：1 代表每次更新參數時，只用一個 batch 的資料"""
    
    GRADIENT_STEPS: int = 1
    """梯度步數：1 代表每次更新參數時，只用一個 batch 的資料進行梯度下降"""

    # ==================== Policy / 網路結構 ====================
    # 新的雙 CNN 架構參數（DualCnnFeatureExtractor）
    EMB_5M_TARGET: int = 128
    """Target 5m 特徵維度（price_seq_target 的 CNN 輸出維度）"""
    
    EMB_5M_OTHERS: int = 64
    """Others 5m 特徵維度（price_seq_others 的 CNN 輸出維度）"""
    
    EMB_1D_TARGET: int = 64
    """Target 1d 特徵維度（price_seq_1d_target 的 CNN 輸出維度）"""
    
    EMB_1D_OTHERS: int = 32
    """Others 1d 特徵維度（price_seq_1d_others 的 CNN 輸出維度）"""
    
    EMB_VEC: int = 128
    """向量特徵維度（account_state + cost_state 的 MLP 輸出維度）"""
    
    OUT_DIM: int = 256
    """最終融合層輸出維度"""
    
    # 相容性：保留舊參數（用於向後相容，目前未使用）
    EMB_5M: int = 256
    """舊版 5m 特徵維度（已棄用）"""
    
    EMB_1D: int = 128
    """舊版 1d 特徵維度（已棄用）"""
    
    PI_ARCH: tuple[int, int] = (256, 256)
    """Policy 網路結構（隱藏層大小）"""
    
    QF_ARCH: tuple[int, int] = (128, 128)
    """Q 網路結構（隱藏層大小）"""

    # ==================== Wrapper（動作平滑/重複）====================
    ACTION_REPEAT: int = 1
    """
    動作重複次數（Frame Skipping）
    - 1 代表每步都決策
    - 5 代表 5 步一決策（重複執行相同動作 5 次）
    - 訓練時建議用「更強的降頻/降換手」設定，否則手續費與 turnover 會把主線 log-return 磨成長期負值
    """
    
    MAX_POSITION_PCT: float = 0.7
    """
    最大持倉百分比
    - 限制 agent 的「目標持倉百分比」在 [-MAX_POSITION_PCT, +MAX_POSITION_PCT]
    - 優先順序：在 run_sac_lag 訓練流程中，此值會「覆蓋」Env.config.Config.MAX_POSITION_PCT
    """
    
    MIN_POSITION_CHANGE: float = 0.2
    """
    最小持倉變化閾值（Deadband）
    - 抑制微小調倉，避免 action 抖動導致過度成交/手續費爆炸
    - 單位：目標倉位比例（position pct）
    """

    # ==================== No-trade hysteresis（方案2：雙門檻）====================
    NO_TRADE_ENTRY_THRESHOLD: float = 0.2
    """
    空倉進場閾值
    - 空倉時需有明確訊號（動作幅度 >= 此值）才進場
    - 單位：目標倉位比例（position pct）
    - 建議先用很小的值觀察 trade_steps 與 fees 是否下降
    """
    
    NO_TRADE_EXIT_THRESHOLD: float = 0.1
    """
    有倉出場閾值
    - 有倉時較容易回到空倉（避免 0 附近抖動）
    - 單位：目標倉位比例（position pct）
    - 注意：動作會先被 clip 到 [-MAX_POSITION_PCT, +MAX_POSITION_PCT]
    """

    # ==================== 環境參數 ====================
    WINDOW_SIZE_5M: int = 288
    """5 分鐘 K 線視窗大小（288 = 1 天）"""
    
    WINDOW_SIZE_1D: int = 14
    """1 日 K 線視窗大小（14 天）"""

    # ==================== 資料切分（Train/Eval 分離）====================
    DATA_SPLIT_ENABLED: bool = True
    """是否啟用資料切分：評估資料使用「最近 N 個月」，剩餘資料作為訓練資料"""
    
    HOLDOUT_MONTHS: int = 3
    """保留用於評估的月份數（從資料末尾往前取）"""

    # ==================== Log / Checkpoint ====================
    TENSORBOARD_LOG_DIR: str = "logs/sac_lag_tb"
    """TensorBoard 日誌目錄"""
    
    VEC_MONITOR_LOG_PREFIX: str = "logs/sac_lag"
    """VecMonitor 日誌前綴"""
    
    CHECKPOINT_SAVE_FREQ: int = 1_000_000
    """檢查點保存頻率（每 N 個 timesteps）"""
    
    CHECKPOINT_DIR_PREFIX: str = "models/sac_lag"
    """檢查點目錄前綴"""

    # ==================== Periodic Evaluation / Validation ====================
    EVAL_ENABLED: bool = True
    """
    是否啟用週期性評估
    - 訓練期間定期跑 eval episodes，輸出指標（止損/強平/交易頻率/進場/平均持倉/最終資金/收益率）
    - 依規則挑選並保存 best model
    """
    
    EVAL_EVERY_TIMESTEPS: int = 1_000_000
    """
    評估頻率（每 N 個 timesteps）
    - 以「訓練總 timesteps」為基準（使用 SB3 的 model.num_timesteps）
    - 確保在 VecEnv 下語意正確
    """
    
    EVAL_N_EVAL_EPISODES: int = 25
    """每次評估運行的 episode 數量"""
    
    EVAL_DETERMINISTIC: bool = True
    """評估時是否使用確定性策略（True 表示使用平均策略，False 表示採樣）"""
    
    EVAL_RANDOM_START: bool = True
    """Eval 環境設定：是否使用隨機起點（True 可增加評估多樣性，False 可重現）"""
    
    EVAL_MAX_EPISODE_STEPS: int = 288 * 14
    """Eval episode 最大步數（避免回合過長拖慢訓練）"""
    
    EVAL_PRINT_EACH_EPISODE: bool = True
    """是否在每個 eval episode 結束時在 Terminal 顯示一行摘要"""
    
    EVAL_PRINT_PREFIX: str = "[EVAL]"
    """Eval 輸出前綴"""

    # ==================== Best Model 保存 ====================
    EVAL_SAVE_BEST_MODEL: bool = True
    """是否保存最佳模型"""
    
    EVAL_BEST_MODEL_SUBDIR: str = "best_model"
    """最佳模型子目錄名稱（實際保存路徑會在訓練入口用 symbol 組合）"""

    # ==================== Best Model 規則 ====================
    EVAL_REJECT_IF_DEATH_EVENT: bool = True
    """
    是否拒絕包含死亡事件的評估結果
    - True：任何「死亡事件」出現 => 整次 eval 不合格（不更新 best）
    - 死亡事件定義：
      * termination_reason in {"liq_triggered", "balance_insufficient"}
      * 或 episode_liq_count > 0
    """
    
    EVAL_MAX_DD_LIMIT: float = 0.40
    """
    評估最大回撤限制
    - 0.40 代表 40%
    - 只在通過此約束時才允許更新 best model
    """
    
    EVAL_MEAN_COST_LIMIT: float = 0.01
    """
    評估平均成本限制
    - 0.01 代表 1%（每步平均成本佔權益比例）
    - 只在通過此約束時才允許更新 best model
    """

    # ==================== 指標口徑（固定）====================
    # 以下指標的計算方式固定，僅供參考：
    # - holding_ratio = episode_holding_steps / episode_steps
    # - trades_per_step = episode_trade_count / episode_steps

    # ==================== UI / Console ====================
    SHOW_PROGRESS_BAR_DEFAULT: bool = True
    """進度條預設是否開啟"""
    
    SB3_VERBOSE_DEFAULT: int | None = None
    """
    SB3 預設 verbose 等級
    - None：由 run_sac_lag.py 依 progress bar 自動決定
    - 0：僅顯示錯誤
    - 1：顯示表格輸出
    - 若想讓 SB3 的表格輸出更乾淨，可設為 0
    """


