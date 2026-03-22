"""
CostCalculator 現行版本不再區分 stop-loss 事件通道；相關成本由環境／executor 處理。
此檔保留為「死亡 + fric + dense」與舊測試路徑相容的 smoke 測試。
"""

from __future__ import annotations

import pytest

from Env.Costs.cost import CostCalculator


def test_liq_triggered_matches_equity_death_semantics() -> None:
    calc = CostCalculator()
    out = calc.compute(
        liq_triggered=True,
        equity=1.0,
        min_balance=10.0,
        step_fee=0.0,
        episode_steps=5,
        episode_max_steps=100,
        initial_balance=10_000.0,
    )
    assert out["cost_risk"] >= 1.0
    assert out["cost_breakdown"]["death_cost"] == pytest.approx(out["cost_risk"])


def test_kwargs_stop_loss_flags_do_not_change_cost_shape() -> None:
    """額外 kwargs 不應讓 compute 丟錯；舊版 stop_loss_event 參數會被忽略。"""
    calc = CostCalculator()
    out = calc.compute(
        liq_triggered=False,
        equity=5_000.0,
        min_balance=1.0,
        step_fee=0.0,
        initial_balance=10_000.0,
        stop_loss_triggered=True,
        stop_loss_event_cost=0.5,
        has_position=True,
        current_price=100.0,
        stop_loss_price=99.0,
        atr=5.0,
    )
    assert "cost" in out and "cost_breakdown" in out
    assert out["cost_risk"] == pytest.approx(0.0)
