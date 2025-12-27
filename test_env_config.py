"""
Unit tests for Env.config / Train.config compatibility.

目的：
- 確保 `Env.config.Config` 可被 import，且具備 trading_env 需要的屬性。
- 確保 `Train.config.Config` 仍可用（向後相容），並與 `Env.config.Config` 指向同一個 Config 類別。
"""

from __future__ import annotations


def test_env_config_importable_and_has_required_attributes() -> None:
    """Env Config 應可匯入，且具備環境/測試所需屬性。"""
    from Env.config import Config

    required_attrs = [
        "INITIAL_BALANCE",
        "TRANSACTION_FEE",
        "WINDOW_SIZE",
        "WINDOW_SIZE_1D",
        "LEVERAGE",
        "MIN_BALANCE",
        "MIN_EPISODE_STEPS",
        "MIN_POSITION_CHANGE",
        "MAX_STEP_POS_CHANGE_PCT",
        "FEE_LIMIT_ENABLED",
        "FEE_LIMIT_RATIO",
        "FEE_ROLLING_WINDOW",
        "STOP_LOSS_ATR",
        "LIQUIDATION_WARN_PCT",
        "STOP_LOSS_WARN_PCT",
        "FLIP_BUDGET_MAX",
        "FLIP_COST",
        "FLIP_THRESHOLD",
        "FLIP_RECOVERY_RATE",
        "FLIP_PROFIT_RECOVERY_RATE",
        "STEP_LOG_ENABLED",
        "STEP_LOG_DIR",
        "STEP_LOG_EVERY_N",
        "MARKET_STATE_COLS",
        "MAINTENANCE_MARGIN_RATE",
        "MAX_EPISODE_STEPS",
        "STOP_LOSS_LIQ_BUFFER_PCT",
        "STOP_LOSS_COOLDOWN_STEPS",
        "COST_LIQ_PENALTY",
        "REWARD_LOG_RET_WEIGHT",
        "REWARD_CONVICTION_TREND_BONUS_WEIGHT",
        "REWARD_CONVICTION_TREND_MIN_STRENGTH",
        "REWARD_CONVICTION_MIN_ABS_POS",
        "TURNOVER_NOTIONAL_SCALE",
    ]

    for a in required_attrs:
        assert hasattr(Config, a), f"Env.config.Config missing attribute: {a}"

    # Basic type sanity checks
    assert isinstance(Config.INITIAL_BALANCE, float)
    assert isinstance(Config.TRANSACTION_FEE, float)
    assert isinstance(Config.WINDOW_SIZE, int)
    assert isinstance(Config.WINDOW_SIZE_1D, int)
    assert isinstance(Config.LEVERAGE, float)
    assert isinstance(Config.STEP_LOG_DIR, str)
    assert isinstance(Config.MARKET_STATE_COLS, list)


def test_train_config_is_compatible_shim() -> None:
    """Train.config 應轉匯入 Env.config 的 Config，維持舊 import 路徑可用。"""
    from Env.config import Config as EnvConfig
    from Train.config import Config as TrainConfig

    assert TrainConfig is EnvConfig


