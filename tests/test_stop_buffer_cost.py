from __future__ import annotations

import pytest

from Env.Costs.cost import CostCalculator


def test_stop_buffer_cost_flat_is_zero() -> None:
    calc = CostCalculator()
    out = calc.compute(
        liq_triggered=False,
        equity=10_000.0,
        min_balance=1.0,
        step_fee=0.0,
        has_position=False,
        current_price=100.0,
        stop_loss_price=90.0,
        atr=5.0,
        stop_buffer_d_min=0.3,
        stop_buffer_d_scale=0.3,
    )
    assert out["cost_sl_buf"] == pytest.approx(0.0)
    assert out["cost_breakdown"]["sl_buf_cost"] == pytest.approx(0.0)
    assert out["cost_breakdown"]["stop_missing_cost"] == pytest.approx(0.0)


def test_stop_buffer_cost_missing_stop_is_one_when_in_position() -> None:
    calc = CostCalculator()
    out = calc.compute(
        liq_triggered=False,
        equity=10_000.0,
        min_balance=1.0,
        step_fee=0.0,
        has_position=True,
        current_price=100.0,
        stop_loss_price=0.0,  # missing
        atr=5.0,
        stop_buffer_d_min=0.3,
        stop_buffer_d_scale=0.3,
    )
    assert out["cost_sl_buf"] == pytest.approx(1.0)
    assert out["cost_breakdown"]["stop_missing_cost"] == pytest.approx(1.0)
    assert out["cost_breakdown"]["sl_buf_cost"] == pytest.approx(0.0)


def test_stop_buffer_cost_formula_clip_and_threshold() -> None:
    calc = CostCalculator()
    # d_t = |P-SL|/ATR = |100-99|/5 = 0.2
    # d_min = 0.3, d_scale = 0.3 => cost = (0.3-0.2)/0.3 = 1/3
    out = calc.compute(
        liq_triggered=False,
        equity=10_000.0,
        min_balance=1.0,
        step_fee=0.0,
        has_position=True,
        current_price=100.0,
        stop_loss_price=99.0,
        atr=5.0,
        stop_buffer_d_min=0.3,
        stop_buffer_d_scale=0.3,
    )
    assert out["cost_breakdown"]["stop_missing_cost"] == pytest.approx(0.0)
    assert out["cost_breakdown"]["sl_buf_cost"] == pytest.approx(1.0 / 3.0)
    assert out["cost_sl_buf"] == pytest.approx(1.0 / 3.0)

    # 距離足夠安全：d_t >= d_min -> 0
    out2 = calc.compute(
        liq_triggered=False,
        equity=10_000.0,
        min_balance=1.0,
        step_fee=0.0,
        has_position=True,
        current_price=100.0,
        stop_loss_price=98.0,  # d_t=0.4
        atr=5.0,
        stop_buffer_d_min=0.3,
        stop_buffer_d_scale=0.3,
    )
    assert out2["cost_sl_buf"] == pytest.approx(0.0)


