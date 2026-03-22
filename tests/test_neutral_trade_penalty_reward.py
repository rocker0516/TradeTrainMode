"""
§5c：中性區（regime_dir=0）且本步 traded 時，RewardCalculator 扣 neutral_trade_penalty_weight。
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

_PROJECT_ROOT = __import__("os").path.abspath(
    __import__("os").path.join(__import__("os").path.dirname(__file__), "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Env.Rewards.reward import (  # noqa: E402
    ConvictionTrendRewardCalculator,
    RewardCalculator,
    create_default_calculator,
)


def _base_kwargs(
    *,
    traded: bool,
    gate_a: float,
    gate_c: float,
    last_equity: float = 100.0,
    new_equity: float = 100.0,
) -> dict:
    return dict(
        last_equity=last_equity,
        new_equity=new_equity,
        margin_buffer=1.0,
        position_change=0.1,
        position_change_norm=0.1,
        turnover_ratio=0.0,
        dist_to_extreme_atr=0.0,
        mae_atr=0.0,
        leverage_ratio=0.0,
        has_position=True,
        unrealized_pnl=0.0,
        traded=traded,
        realized_pnl_step=0.0,
        episode_steps=1,
        episode_max_steps=100,
        stop_loss_triggered=False,
        done=False,
        termination_reason=None,
        step_fee_ratio=0.0,
        current_dd=0.0,
        fee_budget_ratio=1.0,
        position_pct=0.2,
        abs_position_pct=0.2,
        trend_score=0.01,
        gate_flags=np.array([gate_a, 0.0, gate_c], dtype=np.float64),
        regime_score=np.array([0.0, 0.0, 0.5], dtype=np.float64),
    )


def test_neutral_regime_traded_applies_penalty() -> None:
    calc = RewardCalculator(
        neutral_trade_penalty_weight=0.05,
        base_log_ret_weight=1.0,
    )
    kw = _base_kwargs(traded=True, gate_a=0.4, gate_c=0.0)
    r = calc.compute(**kw)
    assert calc.last_neutral_trade_penalty == pytest.approx(-0.05)
    assert r == pytest.approx(-0.05)


def test_neutral_regime_no_trade_no_penalty() -> None:
    calc = RewardCalculator(neutral_trade_penalty_weight=0.05, base_log_ret_weight=1.0)
    kw = _base_kwargs(traded=False, gate_a=0.4, gate_c=0.0)
    r = calc.compute(**kw)
    assert calc.last_neutral_trade_penalty == 0.0
    assert r == pytest.approx(0.0)


def test_gate_a_active_no_neutral_penalty_even_if_traded() -> None:
    calc = RewardCalculator(neutral_trade_penalty_weight=0.05, base_log_ret_weight=1.0)
    kw = _base_kwargs(traded=True, gate_a=0.6, gate_c=0.0)
    r = calc.compute(**kw)
    assert calc.last_neutral_trade_penalty == 0.0
    assert r == pytest.approx(0.0)


def test_gate_c_active_no_neutral_penalty_even_if_traded() -> None:
    calc = RewardCalculator(neutral_trade_penalty_weight=0.05, base_log_ret_weight=1.0)
    kw = _base_kwargs(traded=True, gate_a=0.0, gate_c=-0.6)
    r = calc.compute(**kw)
    assert calc.last_neutral_trade_penalty == 0.0
    assert r == pytest.approx(0.0)


def test_weight_zero_disables_neutral_penalty() -> None:
    calc = RewardCalculator(neutral_trade_penalty_weight=0.0, base_log_ret_weight=1.0)
    kw = _base_kwargs(traded=True, gate_a=0.4, gate_c=0.0)
    r = calc.compute(**kw)
    assert calc.last_neutral_trade_penalty == 0.0
    assert r == pytest.approx(0.0)


def test_create_default_calculator_passes_neutral_weight() -> None:
    calc = create_default_calculator(
        conviction_trend_bonus_weight=0.0,
        neutral_trade_penalty_weight=0.03,
    )
    assert isinstance(calc, RewardCalculator)
    assert float(getattr(calc, "neutral_trade_penalty_weight")) == pytest.approx(0.03)


def test_conviction_calculator_neutral_penalty_via_super() -> None:
    calc = ConvictionTrendRewardCalculator(
        conviction_trend_bonus_weight=0.0,
        neutral_trade_penalty_weight=0.04,
        base_log_ret_weight=1.0,
    )
    kw = _base_kwargs(traded=True, gate_a=0.3, gate_c=-0.2)
    r = calc.compute(**kw)
    assert calc.last_neutral_trade_penalty == pytest.approx(-0.04)
    assert r == pytest.approx(-0.04)
