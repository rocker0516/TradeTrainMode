"""
Phase A/B 訓練用 `TradingEnvironment` kwargs 預設配方。

說明：
- 數值與 `Env.config.Config` 全域預設可能不同，屬 Phase AB 實驗約定；
  調整請改此檔或於 `get_phase_ab_env_kwargs` 呼叫端覆寫參數。
"""

from __future__ import annotations


class PhaseABEnvConfig:
    """Phase A/B 傳入 `TradingEnvironment` 的固定 kwargs 常數（不含 symbol／動態 holdout 等）。"""

    ENV_ID: int = 0
    RANDOM_START: bool = True
    WINDOW_SIZE: int = 72
    WINDOW_SIZE_1D: int = 12
    DEFAULT_MAX_EPISODE_STEPS: int = 288 * 31

    NO_TRADE_ENTRY_THRESHOLD: float = 0.3
    NO_TRADE_EXIT_THRESHOLD: float = 0.1
    MAX_STEP_POS_CHANGE_PCT: float = 0.4
    MIN_POSITION_CHANGE: float = 0.1

    TRADE_FREQ_WINDOW_STEPS: None | int = None
    TRADE_FREQ_COST_LIMIT: None | float = None

    # §5c：中性區仍成交時主線固定扣分（與 regime_alignment 分開）；0=關閉
    NEUTRAL_TRADE_PENALTY_WEIGHT: float = 0.01

    EVAL_MIN_EPISODE_STEPS: int = 1
