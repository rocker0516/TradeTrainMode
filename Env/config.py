"""
Env 專用設定檔。

目的：
- 將 `Env/trading_env.py` 依賴的參數集中在 `Env/` 下，方便環境側調整與維護。
- 同時保留與舊路徑 `Train.config.Config` 的相容性（由 `Train/config.py` 轉匯入）。

注意：
- 本專案其他模組仍可能透過 kwargs 覆寫設定（`TradingEnvironment(..., **kwargs)`）。
"""

from __future__ import annotations


class Config:
    """交易環境的預設參數集合（類似常數容器）。"""

    # ---- 基本資金與交易成本 ----
    INITIAL_BALANCE: float = 10_000.0
    TRANSACTION_FEE: float = 0.01  # (%) 手續費百分比（例如 0.04 代表 0.04%）

    # ---- 視窗大小 ----
    WINDOW_SIZE: int = 288 * 3  # 5m * 288 = 1 day
    WINDOW_SIZE_1D: int = 30

    # ---- 交易參數 ----
    LEVERAGE: float = 10.0
    MIN_BALANCE: float = INITIAL_BALANCE * 0.5 # 最小餘額 0. 5 代表 50%
    MIN_EPISODE_STEPS: int = 288 * 31 # 最小步數 288 * 31 = 8928 步
    MAX_EPISODE_STEPS: int = 288 * 31 # 最大步數 288 * 31 = 8928 步
    # 最小調倉幅度（Deadband, 0.0 ~ 1.0）
    # 預設使用 0.0：讓「單步倉位變化限制(max_step_pos_change_pct)」可以逐步累積倉位，
    # 需要抑制微小調倉刷手續費時，再由外部 kwargs 覆寫（例如 0.2 代表 20%）。
    MIN_POSITION_CHANGE: float = 0.0 # 最小調倉幅度（預設不啟用 deadband）
    MAX_STEP_POS_CHANGE_PCT: float = 0.5 # 最大單步持倉比例變化 0.5 代表 50%

    # ---- 訓練/執行 Wrapper 參數（單一來源）----
    # ActionClipWrapper：硬限制最大目標倉位（不做 action smoothing）
    # 作用：提供「環境側/通用」的預設 clip 上限（-P~P）。
    #
    # 重要：
    # - `TradingEnvironment` 本身不會自動使用這個值做 clip；只有你在外部建立 `ActionClipWrapper` 並把
    #   `max_position_pct=Config.MAX_POSITION_PCT` 傳進去時才會生效。
    # - 若你用 `Train/run_sac_lag.py` 訓練，實際生效的是 `TrainConfig.MAX_POSITION_PCT`
    #   （訓練端建立 wrapper 時顯式傳參），因此訓練端優先。
    MAX_POSITION_PCT: float = 0.8  # 最大目標持倉比例（-P~P）0.8 代表 80%
    # ActionRepeatWrapper：降低決策頻率（Frame Skip）
    ACTION_REPEAT: int = 3

    # ---- 手續費限制 ----
    FEE_LIMIT_ENABLED: bool = False
    FEE_LIMIT_RATIO: float = 0.05
    FEE_ROLLING_WINDOW: int = 288

    # ---- 止損 / 清算提醒（供 Observer 或外部使用）----
    # STOP_LOSS_ATR：止損距離的「ATR 倍數」。
    # - Executor 設定止損時：stop_distance = ATR * STOP_LOSS_ATR
    # - 直覺：越大 => 止損越遠（更不容易被洗出場；但單次虧損可能更大）
    # - 建議範圍（5m 常見）：1.5 ~ 3.0；2.0 屬於中庸值
    STOP_LOSS_ATR: float = 2
    STOP_LOSS_LIQ_BUFFER_PCT: float = 0.2 # 止損相對強平價的安全緩衝（比例） 0.2 代表 20%
    STOP_LOSS_COOLDOWN_STEPS: int = 30 / 5 # 止損冷卻步數
    # 止損事件成本（事件型，非密集）：當本 step 觸發止損時，額外給一個固定成本（0~1）。
    #
    # 語義：
    # - sl_buf：只罰「持倉時貼近止損卻不撤」（密集型）
    # - stop_loss_event_cost：罰「真的被止損打掉」（事件型）
    #
    # 建議：
    # - 先用小值（例如 0.01~0.05），避免其主導 reward；再看統計調整。
    STOP_LOSS_EVENT_COST: float = 0.02 # 0.02 代表 2%

    # ---- Stop-Buffer Cost（止損安全緩衝成本線）----
    # 目標：不是罰虧損，而是罰「你把倉位放在快撞止損的地方還不撤」。
    #
    # 定義（ATR 正規化、無量綱）：
    #   d_t = |P_t - SL_t| / ATR_t
    #
    # 成本（0~1，線性爬升 + clip）：
    #   c_sl_buf = clip( max(0, d_min - d_t) / d_scale, 0, 1 )
    #
    # 怎麼選參數（不用猜）：
    # - 先決定你想在距離止損多少 ATR 開始罰：d_min
    # - 再決定你想在距離止損多少 ATR 視為「非常危險（cost=1）」：d_crit
    #   則：d_scale = d_min - d_crit
    #
    # 你選擇「更近一點」的版本（更不干擾主線）：
    # - d_min=0.3：只有當「距離止損 < 0.3 ATR」才開始被罰 
    # - d_scale=0.3：罰得較溫和;越大代表 cost 變化越慢；理論上要到 d_t≈0（幾乎撞到止損）才會接近 cost=1 
    STOP_BUFFER_D_MIN: float = 0.4
    STOP_BUFFER_D_SCALE: float = 0.6
    
    # 這些閾值保留供 Observation 特徵使用 (near_liq, near_stop)，但不參與 Cost 計算
    LIQUIDATION_WARN_PCT: float = 0.3 # 清算警告比例  0.05 代表 5%
    STOP_LOSS_WARN_PCT: float = 0.6 # 止損警告比例 0.1 代表 10%

    # ---- Cost (Lagrangian Constraints) - REFACTORED ----
    # 舊的固定權重已移除 (COST_W_LIQ_EVENT 等)。
    # 新版 Cost 計算完全正規化 (Cost / Equity)，無須在此設定絕對值權重。
    # 相關邏輯請見 Env/Costs/cost.py

    # ---- 逐倉維持保證金 ----
    MAINTENANCE_MARGIN_RATE: float = 0.005

    # ---- Log ----
    STEP_LOG_ENABLED: bool = False
    STEP_LOG_DIR: str = "step_logs"
    STEP_LOG_EVERY_N: int = 1

    # ---- Market State（相容性用）----
    # 舊版程式/測試可能會期待此屬性存在；目前 MarketData 會自行從 df 欄位推導 features。
    MARKET_STATE_COLS: list[str] = []

    # ---- Observation dtype（訓練 RAM 優化）----
    # 注意：
    # - Gym/SB3 需要 observation_space dtype 與實際 obs dtype 一致。
    # - 為了降低 replay buffer RAM，我們允許輸出 float16 obs；
    #   訓練端 feature extractor 會將 tensor cast 回 float32 做卷積/MLP，維持穩定。
    OBS_DTYPE: str = "float16"  # "float16" | "float32"

    # ---- 其他（測試/相容性用）----
    TURNOVER_NOTIONAL_SCALE: float = 1.0
