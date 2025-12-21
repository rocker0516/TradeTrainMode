"""
Inventory metrics tests.

This module ensures TradingEnvironment provides position exposure diagnostics
in the per-step info dict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from Env.trading_env import TradingEnvironment


def _make_df(n: int = 200, price: float = 100.0) -> pd.DataFrame:
    close = np.full(n, price, dtype=np.float64)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.0001,
            "low": close * 0.9999,
            "close": close,
            "volume": np.full(n, 1000, dtype=np.int64),
        }
    )


def test_step_info_contains_position_exposure_fields() -> None:
    df = _make_df(n=400, price=100.0)
    env = TradingEnvironment(
        df=df,
        env_id=0,
        initial_balance=10_000.0,
        transaction_fee=0.0,
        leverage=2.0,
        window_size=20,
        random_start=False,
        min_episode_steps=50,
        max_step_pos_change_pct=1.0,
        stop_loss_atr=0.0,
    )
    env.reset()

    # Open some position so exposure is non-zero.
    action = np.array([0.5], dtype=np.float32)
    _, _, _, _, info = env.step(action)

    assert "position_pct" in info
    assert "abs_position_pct" in info
    assert "position_notional" in info

    pos_pct = float(info["position_pct"])
    abs_pos_pct = float(info["abs_position_pct"])
    assert -1.0 <= pos_pct <= 1.0
    assert 0.0 <= abs_pos_pct <= 1.0
    assert np.isclose(abs_pos_pct, abs(pos_pct), atol=1e-6)


