"""
Unit tests for Train.config.

目的：確保 `Config` 可被 import，且 `Env/trading_env.py` 依賴的關鍵屬性存在。
"""

from __future__ import annotations


def test_config_importable_and_has_required_attributes() -> None:
    """`Config` 應可匯入，且具備環境所需屬性。"""
    from Train.config import Config

    required_attrs = [
        "INITIAL_BALANCE",
        "TRANSACTION_FEE",
        "WINDOW_SIZE",
        "LEVERAGE",
        "MIN_BALANCE",
        "MIN_EPISODE_STEPS",
        "MIN_POSITION_CHANGE",
        "MAX_STEP_POS_CHANGE_PCT",
        "FEE_ROLLING_WINDOW",
        "STOP_LOSS_ATR",
        "LIQUIDATION_WARN_PCT",
        "STOP_LOSS_WARN_PCT",
        "STEP_LOG_ENABLED",
        "STEP_LOG_DIR",
        "STEP_LOG_EVERY_N",
        "MAINTENANCE_MARGIN_RATE",
        "MAX_EPISODE_STEPS",
        "STOP_LOSS_LIQ_BUFFER_PCT",
        "STOP_LOSS_COOLDOWN_STEPS",
    ]

    for a in required_attrs:
        assert hasattr(Config, a), f"Config missing attribute: {a}"

    # Basic type sanity checks
    assert isinstance(Config.INITIAL_BALANCE, float)
    assert isinstance(Config.TRANSACTION_FEE, float)
    assert isinstance(Config.WINDOW_SIZE, int)
    assert isinstance(Config.LEVERAGE, float)
    assert isinstance(Config.STEP_LOG_DIR, str)


