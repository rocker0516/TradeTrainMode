import numpy as np
import pandas as pd

from Env.trading_env import TradingEnvironment
from Train.curriculum import WinRateFeeCurriculum


def _make_dummy_ohlcv(n: int = 300) -> pd.DataFrame:
    # Minimal columns required by Env/features.py: open/high/low/close/volume
    close = np.linspace(100.0, 120.0, n).astype(np.float32)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = np.full(n, 1000.0, dtype=np.float32)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_win_rate_fee_curriculum_advances_with_episode_spacing():
    cur = WinRateFeeCurriculum(
        base_fee_rate=0.005,
        initial_mult=0.0,
        step_mult=0.1,
        threshold=0.5,
        avg_profit_threshold_pct=None,
        min_episodes=50,
        min_episodes_between_advances=50,
    )

    # Too early: should not advance
    assert cur.maybe_advance(win_rate=0.6, avg_profit_pct=0.0, total_episodes=10) is None
    assert cur.fee_mult == 0.0

    # Meets threshold and min episodes: first advance
    fee_1 = cur.maybe_advance(win_rate=0.6, avg_profit_pct=0.0, total_episodes=50)
    assert fee_1 == 0.005 * 0.1
    assert cur.fee_mult == 0.1

    # Still above threshold but not enough episode spacing: no advance
    assert cur.maybe_advance(win_rate=0.9, avg_profit_pct=0.0, total_episodes=80) is None
    assert cur.fee_mult == 0.1

    # Enough episode spacing: second advance
    fee_2 = cur.maybe_advance(win_rate=0.9, avg_profit_pct=0.0, total_episodes=100)
    assert fee_2 == 0.005 * 0.2
    assert cur.fee_mult == 0.2


def test_win_rate_fee_curriculum_requires_avg_profit_threshold_when_set():
    cur = WinRateFeeCurriculum(
        base_fee_rate=0.005,
        initial_mult=0.0,
        step_mult=0.1,
        threshold=0.5,
        avg_profit_threshold_pct=50.0,
        min_episodes=50,
        min_episodes_between_advances=50,
    )

    # Win rate OK but avg profit below threshold -> should NOT advance
    assert cur.maybe_advance(win_rate=0.6, avg_profit_pct=10.0, total_episodes=50) is None
    assert cur.fee_mult == 0.0

    # Both conditions satisfied -> advance
    fee_1 = cur.maybe_advance(win_rate=0.6, avg_profit_pct=60.0, total_episodes=50)
    assert fee_1 == 0.005 * 0.1
    assert cur.fee_mult == 0.1


def test_env_set_fee_rate_updates_executor():
    df = _make_dummy_ohlcv(300)
    env = TradingEnvironment(
        df=df,
        env_id=0,
        random_start=False,
        window_size=20,
        min_episode_steps=50,
        initial_balance=10_000,
        min_balance=1.0,
        transaction_fee=0.005,
    )

    env.set_fee_rate(0.0)
    assert env.get_fee_rate() == 0.0
    assert env.executor.get_fee_rate() == 0.0

    env.set_fee_rate(0.001)
    assert env.get_fee_rate() == 0.001
    assert env.executor.get_fee_rate() == 0.001


