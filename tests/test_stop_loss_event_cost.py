from __future__ import annotations

import pytest

from Env.Costs.cost import CostCalculator


def test_stop_loss_event_cost_is_added_to_sl_event_channel_only() -> None:
    """
    止損事件成本應該：
    - 出現在 cost_breakdown["stop_loss_event_cost"]
    - 影響 cost_sl_event（而不是 cost_risk / cost_sl_buf）
    """
    calc = CostCalculator()

    out = calc.compute(
        liq_triggered=False,
        equity=10_000.0,
        min_balance=1.0,
        step_fee=0.0,
        # event
        stop_loss_triggered=True,
        stop_loss_event_cost=0.02,
        # sl_buf inputs（此案例設為 flat，確保 sl_buf=0）
        has_position=False,
        current_price=100.0,
        stop_loss_price=90.0,
        atr=5.0,
        stop_buffer_d_min=0.3,
        stop_buffer_d_scale=0.3,
    )

    assert out["cost_risk"] == pytest.approx(0.0)
    assert out["cost_sl_event"] == pytest.approx(0.02)
    assert out["cost_sl_buf"] == pytest.approx(0.0)
    assert out["cost_breakdown"]["stop_loss_event_cost"] == pytest.approx(0.02)


def test_stop_loss_event_cost_is_zero_when_not_triggered() -> None:
    calc = CostCalculator()
    out = calc.compute(
        liq_triggered=False,
        equity=10_000.0,
        min_balance=1.0,
        step_fee=0.0,
        stop_loss_triggered=False,
        stop_loss_event_cost=0.02,
        has_position=False,
        current_price=100.0,
        stop_loss_price=90.0,
        atr=5.0,
        stop_buffer_d_min=0.3,
        stop_buffer_d_scale=0.3,
    )
    assert out["cost_breakdown"]["stop_loss_event_cost"] == pytest.approx(0.0)
    assert out["cost_risk"] == pytest.approx(0.0)
    assert out["cost_sl_event"] == pytest.approx(0.0)


def test_stop_loss_event_cost_does_not_exceed_risk_one_on_death_step() -> None:
    """
    若同一步是死亡事件（liq 或 balance_insufficient），risk 通道應維持語義為 1.0，
    不因 stop_loss_event_cost 疊加超過 1。
    """
    calc = CostCalculator()
    out = calc.compute(
        liq_triggered=True,
        equity=0.0,
        min_balance=1.0,
        step_fee=0.0,
        stop_loss_triggered=True,
        stop_loss_event_cost=0.5,
        has_position=True,
        current_price=100.0,
        stop_loss_price=99.0,
        atr=5.0,
        stop_buffer_d_min=0.3,
        stop_buffer_d_scale=0.3,
    )
    assert out["cost_breakdown"]["death_cost"] == pytest.approx(1.0)
    assert out["cost_risk"] == pytest.approx(1.0)
    assert out["cost_sl_event"] == pytest.approx(0.0)
