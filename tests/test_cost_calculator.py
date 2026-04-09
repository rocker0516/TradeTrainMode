from __future__ import annotations

import pytest

from Env.Costs.cost import CostCalculator


def test_cost_calculator_turnover_stays_linear_below_threshold() -> None:
    calc = CostCalculator(
        turnover_quadratic_coef=10.0,
        turnover_quadratic_threshold=0.02,
    )

    out = calc.compute(
        liq_triggered=False,
        equity=1000.0,
        min_balance=100.0,
        initial_balance=1000.0,
        turnover_ratio=0.01,
        turnover_ratio_full=0.03,
    )

    assert float(out["cost_turnover"]) == pytest.approx(0.01)
    assert float(out["cost_breakdown"]["turnover_linear_cost"]) == pytest.approx(0.01)
    assert float(out["cost_breakdown"]["turnover_quadratic_cost"]) == pytest.approx(0.0)


def test_cost_calculator_turnover_uses_nonlinear_penalty_above_threshold() -> None:
    calc = CostCalculator(
        turnover_quadratic_coef=10.0,
        turnover_quadratic_threshold=0.02,
    )

    out = calc.compute(
        liq_triggered=False,
        equity=1000.0,
        min_balance=100.0,
        initial_balance=1000.0,
        turnover_ratio=0.12,
        turnover_ratio_full=0.35,
    )

    expected_quadratic = 10.0 * ((0.12 - 0.02) ** 2)
    expected_total = 0.12 + expected_quadratic

    assert float(out["cost_turnover"]) == pytest.approx(expected_total)
    assert float(out["cost_breakdown"]["turnover_cost"]) == pytest.approx(expected_total)
    assert float(out["cost_breakdown"]["turnover_linear_cost"]) == pytest.approx(0.12)
    assert float(out["cost_breakdown"]["turnover_quadratic_cost"]) == pytest.approx(expected_quadratic)


def test_cost_calculator_turnover_can_penalize_reduction() -> None:
    calc = CostCalculator(
        turnover_quadratic_coef=10.0,
        turnover_quadratic_threshold=0.02,
    )

    out = calc.compute(
        liq_triggered=False,
        equity=1000.0,
        min_balance=100.0,
        initial_balance=1000.0,
        turnover_ratio=0.0,
        turnover_ratio_full=0.18,
        penalize_turnover_reduction=True,
    )

    expected_quadratic = 10.0 * ((0.18 - 0.02) ** 2)
    expected_total = 0.18 + expected_quadratic
    assert float(out["cost_turnover"]) == pytest.approx(expected_total)
    assert float(out["cost_breakdown"]["turnover_cost"]) == pytest.approx(expected_total)


def test_cost_calculator_total_cost_remains_death_only() -> None:
    calc = CostCalculator(
        turnover_quadratic_coef=10.0,
        turnover_quadratic_threshold=0.02,
    )

    out = calc.compute(
        liq_triggered=False,
        equity=1000.0,
        min_balance=100.0,
        initial_balance=1000.0,
        turnover_ratio=0.25,
        turnover_ratio_full=0.25,
        penalize_turnover_reduction=True,
    )

    assert float(out["cost"]) == pytest.approx(0.0)
    assert float(out["cost_risk"]) == pytest.approx(0.0)
    assert float(out["cost_turnover"]) > 0.25
