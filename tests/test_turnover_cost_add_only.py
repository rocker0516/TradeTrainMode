from __future__ import annotations

import pytest

from Env.trading_env import TradingEnvironment


def _make_min_env_for_reward_features() -> TradingEnvironment:
    """
    建立最小可用的 TradingEnvironment 物件（不跑 __init__），
    只用來測試 _compute_reward_features 的 turnover_ratio 行為。
    """
    env = TradingEnvironment.__new__(TradingEnvironment)
    env._last_position_size = 0.0
    env.current_step = 0
    env.last_trade_step = 0
    env.daily_risk_base = 1000.0
    env.leverage = 10.0
    env.max_equity_so_far = 10_000.0
    # turnover 分母改為 rolling anchor（與 last_equity 解耦）；測試沿用舊 last_equity 尺度
    env.turnover_anchor_equity = 10_000.0
    return env


def test_turnover_ratio_penalizes_only_exposure_increase() -> None:
    env = _make_min_env_for_reward_features()

    # 減碼：|pos| 從 10 -> 5，不應產生 turnover_ratio
    env._last_position_size = 10.0
    pos_change, traded, _norm, turnover_ratio, _tr_full, _dd = env._compute_reward_features(
        current_price=100.0,
        last_equity=10_000.0,
        new_equity=10_000.0,
        new_size=5.0,
    )
    assert traded is True
    assert pos_change == pytest.approx(5.0)
    assert turnover_ratio == pytest.approx(0.0)

    # 加碼：|pos| 從 10 -> 15，應產生 turnover_ratio
    env._last_position_size = 10.0
    pos_change, traded, _norm, turnover_ratio, _tr_full, _dd = env._compute_reward_features(
        current_price=100.0,
        last_equity=10_000.0,
        new_equity=10_000.0,
        new_size=15.0,
    )
    assert traded is True
    assert pos_change == pytest.approx(5.0)
    # exposure_increase=5 => notional=500; scale=anchor_equity*lev=100000 => ratio=0.005
    assert turnover_ratio == pytest.approx(0.005)


def test_turnover_ratio_no_penalty_when_flip_reduces_exposure() -> None:
    env = _make_min_env_for_reward_features()

    # 翻向但縮小曝險：|-10| -> |+5|，不應產生 turnover_ratio
    env._last_position_size = -10.0
    pos_change, traded, _norm, turnover_ratio, _tr_full, _dd = env._compute_reward_features(
        current_price=100.0,
        last_equity=10_000.0,
        new_equity=10_000.0,
        new_size=5.0,
    )
    assert traded is True
    assert pos_change == pytest.approx(15.0)
    assert turnover_ratio == pytest.approx(0.0)


def test_turnover_ratio_uses_anchor_equity_not_last_equity() -> None:
    """last_equity 再大也不稀釋 turnover_ratio，分母僅跟 turnover_anchor_equity。"""
    env = _make_min_env_for_reward_features()
    env.turnover_anchor_equity = 1000.0
    env._last_position_size = 0.0
    pos_change, traded, _norm, turnover_ratio, _tr_full, _dd = env._compute_reward_features(
        current_price=100.0,
        last_equity=500_000.0,
        new_equity=500_000.0,
        new_size=5.0,
    )
    assert traded is True
    scale = 1000.0 * 10.0
    expected = (5.0 * 100.0) / scale
    assert turnover_ratio == pytest.approx(expected)

