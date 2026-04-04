"""
CostCalculator 契約測試（對齊 `Env/Costs/cost.py` 現行 API）。

輸出為 death（cost / cost_risk）與獨立的 cost_risk_dense；摩擦已自成本通道移除。
"""

from __future__ import annotations

import pytest

from Env.Costs.cost import CostCalculator


def test_alive_flat_equity_dense_channel() -> None:
    """存活、遠離死亡線：cost_risk=0，cost=0，dense 為 (1-buffer)^2。"""
    calc = CostCalculator()
    out = calc.compute(
        liq_triggered=False,
        equity=10_000.0,
        min_balance=1.0,
        initial_balance=10_000.0,
    )
    assert out["cost_risk"] == pytest.approx(0.0)
    assert out["cost"] == pytest.approx(0.0)
    assert out["cost_risk_dense"] == pytest.approx(0.0, abs=1e-6)
    assert "dense_buffer_cost" in out["cost_breakdown"]
    assert "death_cost" in out["cost_breakdown"]
    assert "fric_cost" not in out["cost_breakdown"]
    assert out["cost_breakdown"]["death_cost"] == pytest.approx(0.0)
    assert out["cost_breakdown"]["dense_buffer_cost"] == pytest.approx(out["cost_risk_dense"])


def test_death_sets_cost_risk_and_death_breakdown() -> None:
    """權益低於 min_balance：死亡成本為正，且寫入 cost_breakdown["death_cost"]。"""
    calc = CostCalculator()
    out = calc.compute(
        liq_triggered=False,
        equity=0.5,
        min_balance=1.0,
        episode_steps=0,
        episode_max_steps=100,
        initial_balance=10_000.0,
    )
    assert out["cost_risk"] >= 1.0
    assert out["cost_breakdown"]["death_cost"] == pytest.approx(out["cost_risk"])
    assert out["cost"] == pytest.approx(out["cost_risk"])
