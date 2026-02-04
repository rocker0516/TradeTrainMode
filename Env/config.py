"""
Env 專用設定檔。

目的：
- 將 `Env/trading_env.py` 依賴的參數集中在 `Env/` 下，方便環境側調整與維護。
- 其他模組可透過 kwargs 覆寫（例如 TradingEnvironment(..., **kwargs)）。
"""

from __future__ import annotations


class Config:
    """交易環境的預設參數集合（常數容器）。"""

    # -------------------------------------------------------------------------
    # 資金與交易成本
    # -------------------------------------------------------------------------
    INITIAL_BALANCE: float = 1000.0
    TRANSACTION_FEE: float = 0.01  # 手續費百分比（例：0.04 代表 0.04%）

    # -------------------------------------------------------------------------
    # 視窗與步數
    # -------------------------------------------------------------------------
    WINDOW_SIZE: int = 288 * 1.5  # 5m 根數，432 ≈ 1.5 天
    WINDOW_SIZE_1D: int = 30
    MIN_EPISODE_STEPS: int = 288 * 31 * 3
    MAX_EPISODE_STEPS: int = 288 * 31 * 3
    RISK_BASE_UPDATE_STEPS: int = 288  # 每 N steps 更新 daily_risk_base（用於單步倉位變化上限）

    # -------------------------------------------------------------------------
    # 槓桿、餘額與倉位限制
    # -------------------------------------------------------------------------
    LEVERAGE: float = 10.0
    MIN_BALANCE: float = INITIAL_BALANCE * 0.5  # 最小餘額比例（0.5 = 50%）
    MIN_POSITION_CHANGE: float = 0.0  # 最小調倉幅度 deadband（0 = 不啟用）
    MAX_STEP_POS_CHANGE_PCT: float = 0.5  # 單步最大持倉比例變化（0.5 = 50%）
    MAX_POSITION_PCT: float = 0.8  # 最大目標持倉比例（供 ActionClipWrapper 等使用）

    # No-trade 雙門檻（hysteresis）：空倉時 |action| < ENTRY 不進場；有倉時 |action| < EXIT 易回空倉
    NO_TRADE_ENTRY_THRESHOLD: float = 0.0
    NO_TRADE_EXIT_THRESHOLD: float = 0.0

    # -------------------------------------------------------------------------
    # 手續費與 Fee Limit
    # -------------------------------------------------------------------------
    FEE_ROLLING_WINDOW: int = 288

    # -------------------------------------------------------------------------
    # 止損與清算
    # -------------------------------------------------------------------------
    STOP_LOSS_ATR: float = 2.0  # 止損距離的 ATR 倍數
    STOP_LOSS_LIQ_BUFFER_PCT: float = 0.2  # 止損相對強平價的安全緩衝（比例）
    STOP_LOSS_COOLDOWN_STEPS: int = 6  # 止損後冷卻步數（30/5）
    STOP_LOSS_EVENT_COST: float = 0.02  # 觸發止損時的額外事件成本（比例）

    # Stop-Buffer Cost：罰「持倉接近止損卻不撤」（ATR 正規化）
    STOP_BUFFER_D_MIN: float = 0.4   # 距離止損 < d_min ATR 開始罰
    STOP_BUFFER_D_SCALE: float = 0.6 # 罰的尺度

    # 觀察用閾值（near_liq / near_stop 特徵），不參與 Cost 計算
    LIQUIDATION_WARN_PCT: float = 0.3
    STOP_LOSS_WARN_PCT: float = 0.6
    


    # -------------------------------------------------------------------------
    # 保證金與 Wrapper 預設
    # -------------------------------------------------------------------------
    MAINTENANCE_MARGIN_RATE: float = 0.005
    ACTION_REPEAT: int = 3  # ActionRepeatWrapper 預設（Frame Skip）

    # -------------------------------------------------------------------------
    # Step Log
    # -------------------------------------------------------------------------
    STEP_LOG_ENABLED: bool = False
    STEP_LOG_DIR: str = "step_logs"
    STEP_LOG_EVERY_N: int = 1

    # -------------------------------------------------------------------------
    # Observation 與相容性
    # -------------------------------------------------------------------------
    OBS_DTYPE: str = "float16"  # "float16" | "float32"（訓練 RAM 優化）
    MARKET_STATE_COLS: list[str] = []  # 相容性；MarketData 從 df 推導 features
