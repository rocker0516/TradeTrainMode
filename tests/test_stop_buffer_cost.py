"""
CostCalculator 契約測試（對齊 `Env/Costs/cost.py` 現行 API）。

舊版曾輸出 cost_sl_buf / stop_missing_cost 等通道；目前已收斂為
death + fric（合併為 cost）以及獨立的 cost_risk_dense。
"""

from __future__ import annotations

import pytest

from Env.Costs.cost import CostCalculator


def test_alive_flat_equity_dense_and_fric_channels() -> None:
    """存活、遠離死亡線：cost_risk=0，dense 為 (1-buffer)^2，fric 依手續費。"""
    calc = CostCalculator()
    out = calc.compute(
        liq_triggered=False,
        equity=10_000.0,
        min_balance=1.0,
        step_fee=10.0,
        initial_balance=10_000.0,
        cost_fric_scale=1.0,
    )
    assert out["cost_risk"] == pytest.approx(0.0)
    assert out["cost_fric"] == pytest.approx(10.0 / 10_000.0)
    assert out["cost"] == pytest.approx(out["cost_risk"] + out["cost_fric"])
    assert "dense_buffer_cost" in out["cost_breakdown"]
    assert "death_cost" in out["cost_breakdown"]
    assert "fric_cost" in out["cost_breakdown"]
    assert out["cost_breakdown"]["death_cost"] == pytest.approx(0.0)
    assert out["cost_breakdown"]["fric_cost"] == pytest.approx(out["cost_fric"])


def test_death_sets_cost_risk_and_death_breakdown() -> None:
    """權益低於 min_balance：死亡成本為正，且寫入 cost_breakdown["death_cost"]。"""
    calc = CostCalculator()
    out = calc.compute(
        liq_triggered=False,
        equity=0.5,
        min_balance=1.0,
        step_fee=0.0,
        episode_steps=0,
        episode_max_steps=100,
        initial_balance=10_000.0,
    )
    assert out["cost_risk"] >= 1.0
    assert out["cost_breakdown"]["death_cost"] == pytest.approx(out["cost_risk"])
    assert out["cost"] == pytest.approx(out["cost_risk"] + out["cost_fric"])


def test_cost_fric_uses_cost_fric_scale() -> None:
    calc = CostCalculator()
    out = calc.compute(
        liq_triggered=False,
        equity=1_000.0,
        min_balance=1.0,
        step_fee=5.0,
        initial_balance=10_000.0,
        cost_fric_scale=100.0,
    )
    expected = (5.0 / 1_000.0) * 100.0
    assert out["cost_fric"] == pytest.approx(expected)
    assert out["cost_breakdown"]["fric_cost"] == pytest.approx(expected)
