from __future__ import annotations

import os
import sys


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Eval.post_eval_judgment import (  # noqa: E402
    JudgmentStatus,
    extract_judgment_metrics,
    format_judgment_summary,
    judge_payload,
)


def _payload(
    *,
    profit_mean: float,
    profit_per_trade_mean: float,
    trade_count_mean: float,
    total_fees_mean: float,
    episode_steps_mean: float = 2016.0,
    episode_max_dd_mean: float = 0.15,
    episode_max_dd_std: float = 0.05,
    cost_risk_dense_mean: float = 0.05,
    guardrails_enabled: bool = True,
    profit_guardrail_pass: bool = True,
    max_dd_guardrail_pass: bool = True,
    trade_count_guardrail_pass: bool = True,
    results: list[dict] | None = None,
    termination_reason_counts: dict[str, int] | None = None,
) -> dict:
    return {
        "episodes": len(results or []),
        "summary": {
            "profit": {"mean": profit_mean, "std": 0.0, "min": 0.0, "max": 0.0},
            "profit_per_trade": {
                "mean": profit_per_trade_mean,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            },
            "episode_trade_count": {
                "mean": trade_count_mean,
                "std": 10.0,
                "min": 0.0,
                "max": 0.0,
            },
            "episode_steps": {
                "mean": episode_steps_mean,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            },
            "total_fees": {"mean": total_fees_mean, "std": 0.0, "min": 0.0, "max": 0.0},
            "episode_max_dd": {
                "mean": episode_max_dd_mean,
                "std": episode_max_dd_std,
                "min": 0.0,
                "max": 0.0,
            },
            "episode_cost_risk_dense_sum": {
                "mean": cost_risk_dense_mean,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            },
            "episode_cost_risk_sum": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "override_rate": {"mean": 0.01, "std": 0.0, "min": 0.0, "max": 0.0},
            "execution_rate": {"mean": 0.2, "std": 0.0, "min": 0.0, "max": 0.0},
        },
        "guardrails": {
            "enabled": guardrails_enabled,
            "checks": {
                "profit_guardrail": {"pass": profit_guardrail_pass},
                "max_dd_guardrail": {"pass": max_dd_guardrail_pass},
                "trade_count_guardrail": {"pass": trade_count_guardrail_pass},
            },
        },
        "termination_reason_counts": termination_reason_counts or {},
        "results": results or [],
    }


def test_extract_judgment_metrics_counts_low_and_high_churn_examples() -> None:
    payload = _payload(
        profit_mean=-10.0,
        profit_per_trade_mean=-0.1,
        trade_count_mean=250.0,
        total_fees_mean=30.0,
        results=[
            {"profit": 20.0, "episode_trade_count": 20, "episode_max_dd": 0.1},
            {"profit": 10.0, "episode_trade_count": 30, "episode_max_dd": 0.1},
            {"profit": -30.0, "episode_trade_count": 500, "episode_max_dd": 0.4},
            {"profit": -40.0, "episode_trade_count": 600, "episode_max_dd": 0.5},
        ],
    )

    metrics = extract_judgment_metrics(payload)

    assert metrics["positive_profit_episode_count"] == 2
    assert metrics["negative_profit_episode_count"] == 2
    assert metrics["low_churn_positive_episode_count"] >= 1
    assert metrics["high_churn_negative_episode_count"] >= 1
    assert metrics["trade_rate"] > 0.0
    assert metrics["trades_per_day"] > 0.0


def test_judge_payload_marks_p0_and_objective_fail_when_high_churn_dominates() -> None:
    payload = _payload(
        profit_mean=-50.0,
        profit_per_trade_mean=-0.2,
        trade_count_mean=400.0,
        total_fees_mean=80.0,
        profit_guardrail_pass=False,
        trade_count_guardrail_pass=False,
        results=[
            {"profit": 15.0, "episode_trade_count": 30, "episode_max_dd": 0.1},
            {"profit": -100.0, "episode_trade_count": 700, "episode_max_dd": 0.5},
            {"profit": -80.0, "episode_trade_count": 650, "episode_max_dd": 0.4},
        ],
    )

    result = judge_payload(payload)

    assert result.p0_economics.status == JudgmentStatus.FAIL
    assert result.p1_objective.status == JudgmentStatus.FAIL
    assert result.should_block_reward_redesign is True
    assert "reward redesign" in result.recommended_next_action


def test_judge_payload_marks_p2_fail_when_results_missing() -> None:
    payload = _payload(
        profit_mean=0.0,
        profit_per_trade_mean=0.0,
        trade_count_mean=0.0,
        total_fees_mean=0.0,
        results=[],
        guardrails_enabled=False,
    )

    result = judge_payload(payload)

    assert result.p2_observability.status == JudgmentStatus.FAIL
    assert result.should_request_more_diagnostics is True


def test_format_judgment_summary_renders_all_statuses() -> None:
    payload = _payload(
        profit_mean=5.0,
        profit_per_trade_mean=0.5,
        trade_count_mean=20.0,
        total_fees_mean=3.0,
        results=[
            {"profit": 5.0, "episode_trade_count": 15, "episode_max_dd": 0.05},
            {"profit": 6.0, "episode_trade_count": 25, "episode_max_dd": 0.06},
        ],
        guardrails_enabled=False,
    )

    summary = format_judgment_summary(judge_payload(payload))

    assert "[Judgment] P0=" in summary
    assert "Recommended next action:" in summary


def test_judge_payload_warns_when_trades_per_day_is_very_high_even_if_absolute_trade_count_is_moderate() -> None:
    payload = _payload(
        profit_mean=20.0,
        profit_per_trade_mean=0.2,
        trade_count_mean=420.0,
        total_fees_mean=5.0,
        episode_steps_mean=1440.0,  # 5 天 => 84 trades/day
        guardrails_enabled=False,
        results=[
            {"profit": 20.0, "episode_trade_count": 420, "episode_max_dd": 0.08},
            {"profit": 18.0, "episode_trade_count": 410, "episode_max_dd": 0.07},
        ],
    )

    result = judge_payload(payload)

    assert result.p1_objective.status == JudgmentStatus.WARN
    assert "頻繁成交" in result.p1_objective.reason
