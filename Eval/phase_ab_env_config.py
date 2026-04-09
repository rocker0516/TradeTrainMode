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
    WINDOW_SIZE: int = 24
    WINDOW_SIZE_1D: int = 12
    DEFAULT_MAX_EPISODE_STEPS: int = 288 * 31

    NO_TRADE_ENTRY_THRESHOLD: float = 0.2
    NO_TRADE_EXIT_THRESHOLD: float = 0.1
    MAX_STEP_POS_CHANGE_PCT: float = 0.4
    MIN_POSITION_CHANGE: float = 0.2
    # 目標持倉比例上限 |w|<=MAX_POSITION_PCT（與 ActionProcessor clip、action_space 一致）；Phase A/B 訓練／eval 由 get_phase_ab_env_kwargs 傳入。
    MAX_POSITION_PCT: float = 0.8

    TRADE_FREQ_WINDOW_STEPS: None | int = None
    TRADE_FREQ_COST_LIMIT: None | float = None

    # §5c：中性區仍成交時主線固定扣分（與 regime_alignment 分開）；0=關閉
    NEUTRAL_TRADE_PENALTY_WEIGHT: float = 0.005

    EVAL_MIN_EPISODE_STEPS: int = 288 * 7
    # Guardrail baseline（可由 CLI 覆寫）
    DEFAULT_BASELINE_REPORT_PATH: str = (
        "logs/phase_B_lr10_lb001_rs100_rb3e-05_cb003_ntp0005_lf007_fas3000000_cfs100000_fg10_fw2000000_eval.json"
    )
    DEFAULT_GUARDRAIL_MIN_PROFIT_RATIO: float = 0.9
    DEFAULT_GUARDRAIL_MAX_DD_RATIO: float = 1.1
    DEFAULT_GUARDRAIL_MAX_TRADE_COUNT_RATIO: float = 0.7
