from __future__ import annotations

import os
import sys

import pytest


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Train.run_sac_phase_ab import PhaseABEvaluationTriggerCallback  # noqa: E402


class _FakeEvaluator:
    """回傳固定 payload 的假 evaluator。"""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


def test_phase_ab_eval_callback_dispatches_post_eval_payload_on_training_end() -> None:
    payload = {"summary": {"profit": {"mean": 1.0}}, "results": [], "guardrails": {}}
    evaluator = _FakeEvaluator(payload=payload)
    received: list[dict] = []

    callback = PhaseABEvaluationTriggerCallback(
        evaluator=evaluator,
        trigger=None,
        log_freq=1,
        eval_on_train_end=True,
        post_eval_callback=received.append,
        verbose=0,
    )
    callback.model = object()
    callback.num_timesteps = 123
    callback.n_calls = 123

    callback._on_training_end()

    assert len(evaluator.calls) == 1
    assert received == [payload]


def test_phase_ab_eval_callback_collects_episode_death_metrics() -> None:
    payload = {"summary": {}, "results": [], "guardrails": {}}
    evaluator = _FakeEvaluator(payload=payload)
    callback = PhaseABEvaluationTriggerCallback(
        evaluator=evaluator,
        trigger=None,
        log_freq=1,
        eval_on_train_end=False,
        verbose=0,
    )
    callback.model = object()
    callback.num_timesteps = 64000
    callback.n_calls = 1
    callback.locals = {
        "infos": [
            {
                "episode_log_return_sum": -0.5,
                "final_balance": 500.0,
                "episode_cost_risk_sum": 1.9,
                "episode_cost_risk_dense_sum": 300.0,
                "episode_cost_turnover_sum": 0.4,
                "episode_steps": 500,
                "episode_liq_count": 1,
                "episode_stop_loss_count": 0,
                "action_overridden_flag": 1.0,
                "last_action_raw": 0.8,
                "last_final_pos_pct": 0.6,
                "trade_executed_flag": 1.0,
            },
            {
                "episode_log_return_sum": 0.1,
                "final_balance": 1100.0,
                "episode_cost_risk_sum": 0.0,
                "episode_cost_risk_dense_sum": 20.0,
                "episode_cost_turnover_sum": 0.1,
                "episode_steps": 1200,
                "episode_liq_count": 0,
                "episode_stop_loss_count": 2,
                "termination_reason": "max_steps_reached",
                "action_overridden_flag": 0.0,
                "last_action_raw": -0.4,
                "last_final_pos_pct": -0.1,
                "trade_executed_flag": 0.0,
            },
        ]
    }

    assert callback._on_step() is True

    assert callback._latest_metrics["cost_risk_sum_mean"] == pytest.approx(0.95)
    assert callback._latest_metrics["episode_death_rate_mean"] == pytest.approx(0.5)
    assert callback._latest_metrics["episode_steps_mean"] == pytest.approx(850.0)
    assert callback._latest_metrics["episode_liq_count_mean"] == pytest.approx(0.5)
    assert callback._latest_metrics["episode_stop_loss_count_mean"] == pytest.approx(1.0)
