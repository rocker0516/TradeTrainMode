"""
Live runner 與 `TradingEnvironment`／訓練端對齊的預設常數。

用途：
- 集中 `live_trading_loop._build_config` 內與模擬環境一致的參數，避免與業務 CLI 混寫。
- 5m/1d 視窗預設（14/12）可能與 `Env.config.Config` 的 WINDOW_SIZE / WINDOW_SIZE_1D 不同；
  Live 對齊的是訓練／Eval 慣用約定，非全域 Env 預設。
- 手續費、單步加倉上限、no-trade 門檻等與 `Env.config.Config` 對齊，避免與模擬環境漂移。
"""

from __future__ import annotations

from Env.config import Config


class LiveRunnerEnvConfig:
    """Live 與 Env 對齊的常數容器（可由 `Eval.train_config.TrainConfig` 選擇性覆寫）。"""

    WINDOW_SIZE_5M_DEFAULT: int = 72
    WINDOW_SIZE_1D_DEFAULT: int = 12
    ACTION_REPEAT: int = 1

    MAX_STEP_POS_CHANGE_PCT: float = float(Config.MAX_STEP_POS_CHANGE_PCT)
    MIN_POSITION_CHANGE: float = float(Config.MIN_POSITION_CHANGE)
    NO_TRADE_ENTRY_THRESHOLD: float = float(Config.NO_TRADE_ENTRY_THRESHOLD)
    NO_TRADE_EXIT_THRESHOLD: float = float(Config.NO_TRADE_EXIT_THRESHOLD)
    RISK_BASE_UPDATE_STEPS: int = int(Config.RISK_BASE_UPDATE_STEPS)

    DEFAULT_TRANSACTION_FEE_PCT: float = float(Config.TRANSACTION_FEE)
