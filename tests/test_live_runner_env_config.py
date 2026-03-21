"""
契約測試：`LiveRunnerEnvConfig` 欄位完整，且與 `Env.config.Config` 對齊項不漂移；
`_build_config` 在未覆寫 CLI 時沿用 `LiveRunnerEnvConfig`。
"""

from __future__ import annotations

import sys
from typing import Any

# 與 live_trading_loop 相同：確保以檔案路徑執行時可 import LiveTradingRunner
_PROJECT_ROOT = __import__("os").path.abspath(
    __import__("os").path.join(__import__("os").path.dirname(__file__), "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Env.config import Config  # noqa: E402
from LiveTradingRunner.live_runner_env_config import LiveRunnerEnvConfig  # noqa: E402
from LiveTradingRunner.live_trading_loop import _build_config, _parse_args  # noqa: E402

_LIVE_ENV_ALIGN_FIELDS: tuple[tuple[str, type[Any]], ...] = (
    ("WINDOW_SIZE_5M_DEFAULT", int),
    ("WINDOW_SIZE_1D_DEFAULT", int),
    ("ACTION_REPEAT", int),
    ("MAX_STEP_POS_CHANGE_PCT", float),
    ("MIN_POSITION_CHANGE", float),
    ("NO_TRADE_ENTRY_THRESHOLD", float),
    ("NO_TRADE_EXIT_THRESHOLD", float),
    ("RISK_BASE_UPDATE_STEPS", int),
    ("DEFAULT_TRANSACTION_FEE_PCT", float),
)


def test_live_runner_env_config_fields_exist_and_types() -> None:
    """完整性：對齊用常數皆存在且型別正確。"""
    for name, expected_type in _LIVE_ENV_ALIGN_FIELDS:
        assert hasattr(LiveRunnerEnvConfig, name), f"missing {name}"
        val = getattr(LiveRunnerEnvConfig, name)
        assert isinstance(val, expected_type), f"{name}={val!r} is not {expected_type}"


def test_live_runner_env_config_matches_env_config() -> None:
    """從 Config 轉引的欄位與 `Config` 本體一致。"""
    assert LiveRunnerEnvConfig.MAX_STEP_POS_CHANGE_PCT == float(Config.MAX_STEP_POS_CHANGE_PCT)
    assert LiveRunnerEnvConfig.MIN_POSITION_CHANGE == float(Config.MIN_POSITION_CHANGE)
    assert LiveRunnerEnvConfig.NO_TRADE_ENTRY_THRESHOLD == float(Config.NO_TRADE_ENTRY_THRESHOLD)
    assert LiveRunnerEnvConfig.NO_TRADE_EXIT_THRESHOLD == float(Config.NO_TRADE_EXIT_THRESHOLD)
    assert LiveRunnerEnvConfig.RISK_BASE_UPDATE_STEPS == int(Config.RISK_BASE_UPDATE_STEPS)
    assert LiveRunnerEnvConfig.DEFAULT_TRANSACTION_FEE_PCT == float(Config.TRANSACTION_FEE)


def test_build_config_uses_live_runner_env_defaults() -> None:
    """端到端：空 CLI 時 Env 對齊欄位與 `LiveRunnerEnvConfig` 一致。"""
    cfg = _build_config(_parse_args([]))
    assert cfg.window_size_5m == LiveRunnerEnvConfig.WINDOW_SIZE_5M_DEFAULT
    assert cfg.window_size_1d == LiveRunnerEnvConfig.WINDOW_SIZE_1D_DEFAULT
    assert cfg.action_repeat == LiveRunnerEnvConfig.ACTION_REPEAT
    assert cfg.max_step_pos_change_pct == LiveRunnerEnvConfig.MAX_STEP_POS_CHANGE_PCT
    assert cfg.min_position_change == LiveRunnerEnvConfig.MIN_POSITION_CHANGE
    assert cfg.no_trade_entry_threshold == LiveRunnerEnvConfig.NO_TRADE_ENTRY_THRESHOLD
    assert cfg.no_trade_exit_threshold == LiveRunnerEnvConfig.NO_TRADE_EXIT_THRESHOLD
    assert cfg.risk_base_update_steps == LiveRunnerEnvConfig.RISK_BASE_UPDATE_STEPS
