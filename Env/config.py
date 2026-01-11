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
    INITIAL_BALANCE: float = 10000.0
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
    ACTION_REPEAT: int = 1

    # ---- 手續費限制 ----
    FEE_LIMIT_ENABLED: bool = False
    FEE_LIMIT_RATIO: float = 0.05
    FEE_ROLLING_WINDOW: int = 288

    # ---- 止損 / 清算提醒（供 Observer 或外部使用）----
    STOP_LOSS_ATR: float = 2
    STOP_LOSS_LIQ_BUFFER_PCT: float = 0.05 # 止損相對強平價的安全緩衝（比例） 0.05 代表 5%
    STOP_LOSS_COOLDOWN_STEPS: int = 30 / 5 # 止損冷卻步數

    # ---- Stop-Buffer Cost（止損安全緩衝成本線）----
    # 定義：d_t = |P_t - SL_t| / ATR_t（無量綱）
    # 成本：c_sl_buf = clip( max(0, d_min - d_t) / d_scale, 0, 1 )
    #
    # 建議：
    # - d_min: 安全緩衝門檻（常見量級 0.2~0.5 ATR）
    # - d_scale: 線性縮放（建議先用 d_min，讓 d_t=0 時成本=1）
    STOP_BUFFER_D_MIN: float = 0.5
    STOP_BUFFER_D_SCALE: float = 0.2 # 0.2 代表 20% 
    
    # 這些閾值保留供 Observation 特徵使用 (near_liq, near_stop)，但不參與 Cost 計算
    LIQUIDATION_WARN_PCT: float = 0.3 # 清算警告比例  0.05 代表 5%
    STOP_LOSS_WARN_PCT: float = 0.2 # 止損警告比例 0.02 代表 2%

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

    # ---- 其他（測試/相容性用）----
    TURNOVER_NOTIONAL_SCALE: float = 1.0
