"""
Tests for mark-to-market reward logic.

This module verifies that the environment's main reward reflects
the path return across bars (mark-to-market), not only immediate
transaction effects at the same price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from Env.trading_env import TradingEnvironment
from Env.reward import create_default_calculator


def _make_linear_price_df(n: int = 200, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    """Create a minimal OHLCV dataframe with deterministic rising prices."""
    closes = np.array([start + i * step for i in range(n)], dtype=np.float64)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.0001,
            "low": closes * 0.9999,
            "close": closes,
            "volume": np.full(n, 1000, dtype=np.int64),
        }
    )


def test_reward_is_mark_to_market_log_return_when_fee_zero() -> None:
    """
    With fee=0 and no stop-loss, reward should equal log(equity_{t+1}/equity_t).

    We open a long position at t and expect a positive reward when price rises at t+1.
    """
    df = _make_linear_price_df(n=400, start=100.0, step=1.0)
    env = TradingEnvironment(
        df=df,
        env_id=0,
        initial_balance=10_000.0,
        transaction_fee=0.0,
        leverage=1.0,
        window_size=20,
        random_start=False,
        min_episode_steps=50,
        min_position_change=0.0,
        max_step_pos_change_pct=1.0,
        # Disable ATR-based stop-loss to avoid intrabar forced closes in this unit test.
        stop_loss_atr=0.0,
        # Disable optional conviction/trend shaping so reward equals pure log-return.
        reward_calculator=create_default_calculator(
            base_log_ret_weight=1.0,
            conviction_trend_bonus_weight=0.0,
        ),
    )

    obs, _ = env.reset()
    assert isinstance(obs, dict)

    # At reset with random_start=False, current_step == window_size.
    t_idx = int(env.current_step)
    p_t = float(env.df.iloc[t_idx]["close"])
    p_tp1 = float(env.df.iloc[t_idx + 1]["close"])

    # No position at start: equity_t equals initial balance.
    equity_t = float(env.executor.equity(p_t))
    assert np.isclose(equity_t, env.initial_balance)

    # Open a 50% long position.
    action = np.array([0.5], dtype=np.float32)
    _, reward, done, _, info = env.step(action)
    assert done is False
    assert "equity" in info

    # After the step, the reward should be mark-to-market across to p_{t+1}.
    equity_tp1 = float(env.executor.equity(p_tp1))
    expected = float(np.log(max(equity_tp1, 1e-8) / max(equity_t, 1e-8)))
    assert np.isclose(float(reward), expected, atol=1e-6)


