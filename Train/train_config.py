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
    
    N_ENVS: int = 8*8
    """
    並行環境數量。
    效能：SubprocVecEnv 下主進程會先 load_data() 一次並傳入各 worker，避免 N 次磁碟 I/O。
    """
    
    DEVICE: str = "cuda"
    """計算設備：'cuda' / 'cpu' / 'auto'"""

    COMPILE_POLICY: bool = False
    """
    是否對 policy 的 features_extractor 做 torch.compile（PyTorch 2+）。
    - True：可加速 predict/forward，首次會多編譯時間；瓶頸在主進程/GPU 時可試。
    - False（預設）：不編譯，相容性最佳。
    """

    # ==================== Lagrangian / 約束 ====================
    COST_LIMIT: float = 0.00008
    """
    總成本限制（單一 Lambda 模式，已棄用）
    - 新版 Cost 已正規化為 Cost/Equity
    - 建議值 0.00008 (0.8bps) 代表容許每步平均損耗 0.008% 的權益
    - 注意：目前使用多通道模式（RISK_COST_LIMIT / FRIC_COST_LIMIT），此參數僅供相容性
    """
    
    RISK_COST_LIMIT: float = 0.000
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

    TRADE_FREQ_COST_LIMIT: float = 0.2
    """
    交易頻率成本限制（交易比例通道）
    - 單位：最近 N 步內「交易步數 / N」的上限（比例，0~1）
    - 0.1 代表允許最多 10% 的步數發生持倉變化（交易）
    """

    TRADE_FREQ_WINDOW_STEPS: int = 288
    """
    交易頻率約束的窗口寬度（步數）
    - 交易比例 = 窗口內交易次數 / min(窗口寬度, 已收集步數)
    """

    FLAT_COST_LIMIT: float = 0.6
    """
    空倉成本限制（鼓勵持倉、允許避險）
    - 單位：最近 N 步內「空倉步數 / N」的上限（比例，0~1）
    - 0.2 代表允許最多 20% 步數空倉；超過則 λ 上升、懲罰變大
    - 若訓練時 flat [VIOLATION] 且 Eval 仍 0 交易：可試著調低（如 0.1）以更強懲罰空倉，促使 policy 輸出較大 |action| 以超過 NO_TRADE_ENTRY_THRESHOLD
    """

    FLAT_WINDOW_STEPS: int = 288 * 3
    """空倉比例約束的窗口寬度（步數）"""

    FLAT_THRESHOLD: float = 0.2
    """視為空倉的持倉比例門檻：|final_pos_pct| < 此值則計為空倉（步級 cost_flat=1）"""
    
    # Lagrangian 控制器參數
    LAGRANGIAN_KP: float = 0.1
    """Lagrangian P-Control 增益係數"""
    
    LAGRANGIAN_LAMBDA_INIT: float = 0.0
    """Lagrangian Lambda 初始值"""
    
    LAGRANGIAN_LAMBDA_MIN: float = 0.0
    """Lagrangian Lambda 最小值"""
    
    LAGRANGIAN_LAMBDA_MAX: float = 5.0
    """Lagrangian Lambda 最大值"""
    
    UPDATE_LAMBDA_EVERY_STEPS: int = 1_000
    """
    Lambda 更新頻率（每 N 個 global steps 更新一次）
    - 建議值：1000（在 n_envs=64 時，約每 15.6 個訓練回合更新一次）
    """
    
    LOG_EVERY_EPISODES: int = 100
    """每 N 個 episode 刷新一次統計輸出（建議：20-50）"""
    
    STATS_WINDOW_EPISODES: int = 100
    """統計視窗大小：最多取最近 M 個 episode（滾動視窗，建議：100）"""
    
    REWARD_SCALE: float = 10.0
    """獎勵尺度：1.0 代表獎勵不放大，>1.0 會放大獎勵信號"""

    COST_PENALTY_NORMALIZE_FACTOR: float = 1.0
    """
    成本懲罰正規化「預設」係數：未在 COST_PENALTY_NORMALIZE_PER_CHANNEL 指定的通道皆用此值。
    每步該通道貢獻 penalty_k = (λ_k * cost_k) / max(1, norm_k)。
    """

    # 以下會經 config_builder → env_builder 傳入 LagrangianRewardWrapper，確實會生效。
    # 每條成本線獨立除數；鍵=通道名(risk|fric|trade_freq|flat)，值=norm_k。未列出者用 COST_PENALTY_NORMALIZE_FACTOR。
    COST_PENALTY_NORMALIZE_PER_CHANNEL: dict[str, float] = {
        "risk": 1.0,       # 死亡成本 0/1，除小一點讓懲罰有感
        "fric": 1.0,         # 手續費比例
        "trade_freq":1000.0,   # 交易步 0/1
        "flat": 2000.0,         # 空倉比例 0~1
    }

    # ==================== SB3 SAC 超參數 ====================
    LEARNING_RATE: float = 3e-4
    """學習率"""
    
    BUFFER_SIZE: int = 1_500_000
    """Replay Buffer 大小"""
    
    BATCH_SIZE: int = 128
    """訓練批次大小"""
    
    ENT_COEF: str = "auto"
    """熵係數：'auto' 表示自動調整，或指定數值（如 0.01）"""
    
    TRAIN_FREQ: int = 2
    """
    訓練頻率：每 N 個 env step 做一次梯度更新。
    - 2（預設）：平衡 it/s 與學習效率，建議優先保證訓練效果
    - 1：每 step 都更新（學習最密、it/s 最低）
    - 4 或 8：瓶頸在 GPU 時可提升 it/s，但更新變疏、對訓練未必有益，僅在需要衝高吞吐時使用
    """
    
    GRADIENT_STEPS: int = 1
    """梯度步數：每次更新時做的梯度步數；建議與 TRAIN_FREQ 同值。過大僅提升 it/s，不利學習效率"""

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
    - 空倉時需有明確訊號（|action| >= 此值）才進場，否則強制 target=0（不交易）
    - 單位：目標倉位比例（position pct）
    - Eval 使用 deterministic 策略（輸出為 mean action）；若 policy 學到的 mean 落在 (-0.2, 0.2)，
      則 Eval 會永遠不進場（0 交易）。可嘗試調低至 0.05~0.1 讓較小動作也能進場，或加強空倉成本使 policy 輸出更大 |action|。
    """
    
    NO_TRADE_EXIT_THRESHOLD: float = 0.1
    """
    有倉出場閾值
    - 有倉時較容易回到空倉（避免 0 附近抖動）
    - 單位：目標倉位比例（position pct）
    - 注意：動作會先被 clip 到 [-MAX_POSITION_PCT, +MAX_POSITION_PCT]
    """

    # ==================== 環境參數 ====================
    # 規劃建議（Window Size vs Policy/網路結構）：
    # - CNN 使用 AdaptiveAvgPool1d(1)，輸出維度與序列長度 L 無關，故 EMB_* / OUT_DIM 不隨 window 線性綁定。
    # - 5m 視窗：L 大（如 288）→ 歷史長，可維持或略增 EMB_5M_* / OUT_DIM；L 小（如 36）→ 可維持或略減以防過擬合。
    # - 1d 視窗：L 小（7~14）→ 輕量即可；L 大（如 30）→ 可略增 EMB_1D_*，或保持不變先觀察。
    # - 經驗式比例（僅供參考）：OUT_DIM ≈ 1~2x(EMB_5M_TARGET+EMB_5M_OTHERS)；PI_ARCH 首層 ≥ OUT_DIM。
    WINDOW_SIZE_5M: int = 288 // 8  # 36（約 3 小時）；288 = 1 天
    """5 分鐘 K 線視窗大小（根數）。288 = 1 天。"""
    
    WINDOW_SIZE_1D: int = 21
    """1 日 K 線視窗大小（天數）。"""

    # ==================== 資料切分（Train/Eval 分離）====================
    DATA_SPLIT_ENABLED: bool = True
    """是否啟用資料切分：評估資料使用「最近 N 個月」，剩餘資料作為訓練資料"""
    
    HOLDOUT_MONTHS: int = 2
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
    """是否在每個 eval episode 結束時在 Terminal 顯示一行摘要（為 True 時會自動關閉進度條，避免 Windows 下與進度條衝突導致崩潰）"""
    
    EVAL_PRINT_PREFIX: str = "[EVAL]"
    """Eval 輸出前綴"""

    EVAL_RENDER_EACH_EPISODE: bool = False
    """
    Eval 時是否在每個 episode 結束後產出 render 圖檔。
    - True：每個 eval episode 結束時觸發 env render（存檔至 logs/renders/eval_<symbol>/）
    - 實際是否彈出視窗由 env 的 render_show 控制（eval 環境預設 render_show=True，可改 to_eval_dict）
    """

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
    
    EVAL_MIN_MEAN_RETURN: float = 0.20
    """
    評估最低平均收益率（門檻）
    - 0.20 代表 +20%（mean_return >= 0.20 才通過）
    - 未達此門檻視為 eval 失敗，不更新 best model
    """
    
    # ==================== Eval Gate 配置 ====================
    EVAL_GATE_ENABLED: bool = True
    """是否啟用評估門控（避免早期評估）"""
    
    EVAL_GATE_WINDOW_SIZE: int = 100
    """評估門控視窗大小（最近 N 個訓練回合）"""
    
    EVAL_GATE_MIN_MAX_STEPS_REACHED_COUNT: int = 90
    """評估門控最小達標回合數（視窗內需有 N 個回合達到 max_steps，100 回合中至少 90 回 max_steps_reached 才開啟 EVAL）"""

    # ==================== 指標口徑（固定）====================
    # 以下指標的計算方式固定，僅供參考：
    # - holding_ratio = episode_holding_steps / episode_steps
    # - trades_per_step = episode_trade_count / episode_steps

    # ==================== UI / Console ====================
    SHOW_PROGRESS_BAR_DEFAULT: bool = True
    """進度條預設是否開啟"""
    
    SB3_LOG_INTERVAL: int = 10_000
    """
    SB3 內建表格輸出的間隔（每 N 個 timestep 才 dump 一次）。
    - 預設 10000 可大幅減少重複的 rollout/time 表格刷屏
    - 自訂統計仍由 LagrangianCallback 依 LOG_EVERY_EPISODES 輸出
    """
    
    SB3_VERBOSE_DEFAULT: int | None = None
    """
    SB3 預設 verbose 等級
    - None：由 run_sac_lag.py 依 progress bar 自動決定
    - 0：僅顯示錯誤（不印 SB3 內建表格）
    - 1：顯示表格輸出（受 log_interval 控制頻率）
    """

    # ==================== 結果不佳時的調參建議 ====================
    # 若訓練後出現：高死亡率(balance_insufficient)、flat 長期 VIOLATION、λ_flat=5 頂滿、
    # Cost Penalty 主導 Total Reward、Win Rate 極低、Stop Loss 佔絕大多數出場，
    # 可依序嘗試下列調整（改 train_config 預設或 CLI）：
    #
    # 1) 放寬空倉約束，避免 λ_flat 頂滿壓過主線獎勵
    #    FLAT_COST_LIMIT: 0.4 → 0.55 或 0.6（讓目前 55% 空倉率先不 violation，再慢慢收緊）
    #
    # 2) 降低 flat 通道懲罰權重（讓「存活」與「持倉」有機會學到信號）
    #    COST_PENALTY_NORMALIZE_PER_CHANNEL["flat"]: 1500 → 3000 或 5000（每步懲罰變小）
    #
    # 3) 提高死亡懲罰存在感（risk 通道 norm=1 已夠大，可提高 λ 上限或 kp）
    #    LAGRANGIAN_LAMBDA_MAX: 5.0 → 10.0（僅 risk 需更強時可單獨調 risk 的 channel config）
    #    或 LAGRANGIAN_KP: 0.1 → 0.15（λ 上升更快，約束更硬）
    #
    # 4) 止損過緊導致大量 SL 出場 → 空倉多 → flat violation；可適度放寬止損距離
    #    Env/config.py: STOP_LOSS_ATR 1.5 → 2.0（需在環境建構時傳入）
    #
    # 5) 先讓 agent 學會「少死」再收緊 flat：可暫時放寬 FLAT_COST_LIMIT 到 0.6，
    #    等 death rate 下降後再改回 0.4～0.45


