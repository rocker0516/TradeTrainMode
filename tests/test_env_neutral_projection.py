"""Neutral regime projection tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from Env.trading_env import TradingEnvironment


def _make_env_for_neutral_projection(*, current_pos_pct: float) -> TradingEnvironment:
    env = object.__new__(TradingEnvironment)
    env.current_step = 0
    env.leverage = 10.0
    env.market_data = SimpleNamespace(
        get_gate_flags=lambda step_idx: np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )
    current_price = 100.0
    last_equity = 1000.0
    size = current_pos_pct * last_equity * env.leverage / current_price
    env.executor = SimpleNamespace(position=SimpleNamespace(size=size))
    return env


def test_neutral_projection_blocks_entry_when_flat() -> None:
    env = _make_env_for_neutral_projection(current_pos_pct=0.0)

    projected = env._apply_regime_action_projection(
        np.array([0.7], dtype=np.float32),
        current_price=100.0,
        last_equity=1000.0,
    )

    assert float(projected[0]) == 0.0


def test_neutral_projection_long_allows_hold_reduce_close_only() -> None:
    env = _make_env_for_neutral_projection(current_pos_pct=0.4)

    projected_add = env._apply_regime_action_projection(
        np.array([0.8], dtype=np.float32),
        current_price=100.0,
        last_equity=1000.0,
    )
    projected_reduce = env._apply_regime_action_projection(
        np.array([0.2], dtype=np.float32),
        current_price=100.0,
        last_equity=1000.0,
    )
    projected_reverse = env._apply_regime_action_projection(
        np.array([-0.6], dtype=np.float32),
        current_price=100.0,
        last_equity=1000.0,
    )

    assert float(projected_add[0]) == np.float32(0.4)
    assert float(projected_reduce[0]) == np.float32(0.2)
    assert float(projected_reverse[0]) == 0.0


def test_neutral_projection_short_allows_hold_reduce_close_only() -> None:
    env = _make_env_for_neutral_projection(current_pos_pct=-0.35)

    projected_add = env._apply_regime_action_projection(
        np.array([-0.9], dtype=np.float32),
        current_price=100.0,
        last_equity=1000.0,
    )
    projected_reduce = env._apply_regime_action_projection(
        np.array([-0.1], dtype=np.float32),
        current_price=100.0,
        last_equity=1000.0,
    )
    projected_reverse = env._apply_regime_action_projection(
        np.array([0.6], dtype=np.float32),
        current_price=100.0,
        last_equity=1000.0,
    )

    assert float(projected_add[0]) == np.float32(-0.35)
    assert float(projected_reduce[0]) == np.float32(-0.1)
    assert float(projected_reverse[0]) == 0.0
