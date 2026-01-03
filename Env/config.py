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
    TRANSACTION_FEE: float = 0.04  # (%) 手續費百分比（例如 0.04 代表 0.04%）

    # ---- 視窗大小 ----
    WINDOW_SIZE: int = 288  # 5m * 288 = 1 day
    WINDOW_SIZE_1D: int = 30

    # ---- 交易參數 ----
    LEVERAGE: float = 10.0
    MIN_BALANCE: float = INITIAL_BALANCE * 0.5
    MIN_EPISODE_STEPS: int = 288 * 31
    MAX_EPISODE_STEPS: int = 288 * 31
    # 最小調倉幅度（Deadband, 0.0 ~ 1.0）
    # 預設使用 0.0：讓「單步倉位變化限制(max_step_pos_change_pct)」可以逐步累積倉位，
    # 需要抑制微小調倉刷手續費時，再由外部 kwargs 覆寫（例如 0.2 代表 20%）。
    MIN_POSITION_CHANGE: float = 0.0
    MAX_STEP_POS_CHANGE_PCT: float = 0.5 # 最大單步持倉比例變化 0.5 代表 50%

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
    STOP_LOSS_ATR: float = 3.0 
    STOP_LOSS_LIQ_BUFFER_PCT: float = 0.0 # 止損相對強平價的安全緩衝（比例）
    STOP_LOSS_COOLDOWN_STEPS: int = 30 / 5 # 止損冷卻步數
    LIQUIDATION_WARN_PCT: float = 0.05 # 清算警告比例
    STOP_LOSS_WARN_PCT: float = 0.02 # 止損警告比例

    # ---- Lagrangian / Cost（成本線）----
    # cost 的定義在 `Env/Costs/cost.py`，此處僅提供可調權重（避免硬編碼散落各處）
    COST_W_FEE: float = 1.0
    COST_W_LIQ_PROXIMITY: float = 1.0
    COST_W_MARGIN_PROXIMITY: float = 0.5
    COST_W_DD: float = 0.2
    COST_W_STOP_MISSING: float = 0.5
    COST_W_STOP_PROXIMITY: float = 0.2
    COST_W_LIQ_EVENT: float = 5.0
    COST_W_STOP_EVENT: float = 1.0

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

 