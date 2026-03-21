"""post_train_holdout_rollout：步數上限與 unwrap 契約（不開 GUI）。"""

from __future__ import annotations

import sys

import gymnasium as gym
import pytest

_PROJECT_ROOT = __import__("os").path.abspath(
    __import__("os").path.join(__import__("os").path.dirname(__file__), "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from Eval.post_train_holdout_rollout import (  # noqa: E402
    _ohlc_slice_for_live_render,
    post_train_max_episode_steps_cap,
    unwrap_to_trading_env,
)


def test_post_train_max_episode_steps_cap_split_vs_full() -> None:
    assert post_train_max_episode_steps_cap(data_split=True, holdout_months=3) == 1_000_000_000
    assert post_train_max_episode_steps_cap(data_split=False, holdout_months=2) == 288 * 30 * 2
    assert post_train_max_episode_steps_cap(data_split=False, holdout_months=1) >= 288


def test_ohlc_slice_prefixed_columns_matches_merged_csv_layout() -> None:
    """合併後 df_5m 為 BTCUSDT_close 等形式時仍可組出標準 OHLC。"""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="5min"),
            "BTCUSDT_open": [1.0, 1.1, 1.2, 1.3, 1.4],
            "BTCUSDT_high": [1.05, 1.15, 1.25, 1.35, 1.45],
            "BTCUSDT_low": [0.95, 1.05, 1.15, 1.25, 1.35],
            "BTCUSDT_close": [1.02, 1.12, 1.22, 1.32, 1.42],
        }
    )
    out = _ohlc_slice_for_live_render(df, target_symbol="BTCUSDT", i0=1, i1=3)
    assert out is not None
    assert list(out.columns) == ["timestamp", "open", "high", "low", "close"]
    assert len(out) == 3
    assert float(out["close"].iloc[-1]) == pytest.approx(1.32)


def test_ohlc_slice_open_fallback_to_close_when_prefixed_open_missing() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="5min"),
            "ETHUSDT_high": [2.0, 2.1, 2.2],
            "ETHUSDT_low": [1.9, 2.0, 2.1],
            "ETHUSDT_close": [1.95, 2.05, 2.15],
        }
    )
    out = _ohlc_slice_for_live_render(df, target_symbol="ETHUSDT", i0=0, i1=2)
    assert out is not None
    assert out["open"].tolist() == out["close"].tolist()


def test_unwrap_to_trading_env_typeerror_on_plain_env() -> None:
    class PlainEnv(gym.Env):
        """無內層 TradingEnvironment 的假 env。"""

        def __init__(self) -> None:
            super().__init__()
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float)
            self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float)

    with pytest.raises(TypeError, match="Cannot unwrap"):
        unwrap_to_trading_env(PlainEnv())
