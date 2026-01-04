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
    WINDOW_SIZE: int = 288  # 5m * 288 = 1 day
    WINDOW_SIZE_1D: int = 30

    # ---- 交易參數 ----
    LEVERAGE: float = 10.0
    MIN_BALANCE: float = INITIAL_BALANCE * 0.2 # 最小餘額 0.2 代表 20%
    MIN_EPISODE_STEPS: int = 288 * 31 # 最小步數 288 * 31 = 8928 步
    MAX_EPISODE_STEPS: int = 288 * 31 # 最大步數 288 * 31 = 8928 步
    # 最小調倉幅度（Deadband, 0.0 ~ 1.0）
    # 預設使用 0.0：讓「單步倉位變化限制(max_step_pos_change_pct)」可以逐步累積倉位，
    # 需要抑制微小調倉刷手續費時，再由外部 kwargs 覆寫（例如 0.2 代表 20%）。
    MIN_POSITION_CHANGE: float = 0.2 # 最小調倉幅度 0.2 代表 20%
    MAX_STEP_POS_CHANGE_PCT: float = 0.5 # 最大單步持倉比例變化 0.5 代表 50%

    # ---- 訓練/執行 Wrapper 參數（單一來源）----
    # ActionSmoothClipWrapper：先硬限制最大目標倉位，再做動作平滑，抑制高頻翻倉刷手續費
    MAX_POSITION_PCT: float = 0.8  # 最大目標持倉比例（-P~P）
    ACTION_SMOOTH_ALPHA: float = 0.5  # 0~1；越小越平滑
    # ActionRepeatWrapper：降低決策頻率（Frame Skip）
    ACTION_REPEAT: int = 1

    # ---- 風險 / Flip 預算 ----
    FLIP_BUDGET_MAX: float = 1.0
    FLIP_COST: float = 0.25
    FLIP_THRESHOLD: float = 0.0
    FLIP_RECOVERY_RATE: float = 0.01
    FLIP_PROFIT_RECOVERY_RATE: float = 0.1

    # ---- 手續費限制 ----
    FEE_LIMIT_ENABLED: bool = False
    FEE_LIMIT_RATIO: float = 0.05
    FEE_ROLLING_WINDOW: int = 288

    # ---- 止損 / 清算提醒（供 Observer 或外部使用；目前 trading_env 主要用 STOP_LOSS_ATR）----
    STOP_LOSS_ATR: float = 2 
    STOP_LOSS_LIQ_BUFFER_PCT: float = 0.05 # 止損相對強平價的安全緩衝（比例）
    STOP_LOSS_COOLDOWN_STEPS: int = 30 / 5 # 止損冷卻步數
    LIQUIDATION_WARN_PCT: float = 0.2 # 清算警告比例  0.05 代表 5%
    STOP_LOSS_WARN_PCT: float = 0.1 # 止損警告比例 0.02 代表 2%

    # ---- Lagrangian / Cost（成本線） - REFACTORED ----
    # 設計理念：Lagrangian 約束僅用於「生存邊界」與「極端異常」。
    # 交易損耗（手續費、停損、回撤）應由 Main Reward (Log Return) 負責，
    # 避免雙重懲罰導致 Agent 為了不觸發成本而放棄交易。

    COST_W_FEE: float = 0.0 # 停用（改由 Main Reward 內扣手續費）
    
    # 交易摩擦：保留少量換手懲罰，抑制高頻刷單，但不應過大
    COST_W_FEE_EQUITY: float = 0.0 # 停用
    COST_W_TURNOVER: float = 0.1   # 降低權重
    COST_W_TRADE_EVENT: float = 0.0 # 停用（交易本身不是罪）

    # 生存約束（Risk）：這些是真正的紅線
    COST_W_LIQ_PROXIMITY: float = 1.0 # 接近爆倉：危險
    COST_W_MARGIN_PROXIMITY: float = 1.0 # 保證金不足：危險
    COST_W_BALANCE_PROXIMITY: float = 5.0 # 接近破產：極度危險（拉高權重）
    
    # 風控紀律（Risk）：
    COST_W_STOP_MISSING: float = 1.0 # 未設停損：違規（保留）
    
    # 下列項目屬於「交易結果」而非「違規」，移除以避免誤導 Agent
    COST_W_DD: float = 0.0           # 回撤由 Log Return 負責
    COST_W_STOP_PROXIMITY: float = 0.0 # 接近停損是市場波動，不罰
    COST_W_STOP_EVENT: float = 0.0     # 觸發停損是正確風控，不罰！
    
    COST_W_LIQ_EVENT: float = 5.0 # 實際爆倉：嚴重違規

    # ---- Cost proximity 曲線與警戒帶（越小越嚴格）----
    # - BALANCE_WARN_UP_RATIO: 當 equity <= (1+ratio)*min_balance 時開始拉高 balance_proximity_cost（線性到 1）
    # - PROX_CURVE_POWER: proximity 類成本的非線性倍率（>1 更「末端敏感」）
    BALANCE_WARN_UP_RATIO: float = 0.5
    PROX_CURVE_POWER: float = 2.0

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

 