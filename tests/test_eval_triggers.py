from __future__ import annotations

from Eval.eval_triggers import (
    CompositeTrigger,
    EvalTriggerContext,
    ExpressionTrigger,
    MetricRule,
    MetricTrigger,
    StepTrigger,
)


def test_step_trigger_supports_at_steps_and_every_n() -> None:
    trigger = StepTrigger(at_steps=(1000,), every_n_steps=500, min_interval_steps=0)

    # 500 bucket 觸發
    assert trigger.should_evaluate(EvalTriggerContext(step=500)) is True
    # 1000 at_steps 只觸發一次
    assert trigger.should_evaluate(EvalTriggerContext(step=1000)) is True
    assert trigger.should_evaluate(EvalTriggerContext(step=1000)) is False


def test_metric_trigger_all_mode() -> None:
    trigger = MetricTrigger(
        rules=(
            MetricRule(metric_name="log_return_sum_mean", op=">=", threshold=0.2),
            MetricRule(metric_name="tracking_error", op="<=", threshold=0.15),
        ),
        mode="all",
    )
    assert (
        trigger.should_evaluate(
            EvalTriggerContext(
                step=10,
                metrics={"log_return_sum_mean": 0.3, "tracking_error": 0.12},
            )
        )
        is True
    )
    assert (
        trigger.should_evaluate(
            EvalTriggerContext(
                step=10,
                metrics={"log_return_sum_mean": 0.3, "tracking_error": 0.2},
            )
        )
        is False
    )


def test_expression_trigger_uses_context_metrics() -> None:
    trigger = ExpressionTrigger("step>=5000 and log_return_sum_mean>0 and training_end==False")
    assert (
        trigger.should_evaluate(
            EvalTriggerContext(step=4999, metrics={"log_return_sum_mean": 0.1}, training_end=False)
        )
        is False
    )
    assert (
        trigger.should_evaluate(
            EvalTriggerContext(step=5000, metrics={"log_return_sum_mean": 0.1}, training_end=False)
        )
        is True
    )


def test_composite_trigger_any_mode() -> None:
    composite = CompositeTrigger(
        triggers=(
            MetricTrigger(rules=(MetricRule("log_return_sum_mean", ">=", 1.0),), mode="all"),
            ExpressionTrigger("tracking_error < 0.1"),
        ),
        mode="any",
    )
    assert (
        composite.should_evaluate(
            EvalTriggerContext(step=100, metrics={"log_return_sum_mean": 0.2, "tracking_error": 0.08})
        )
        is True
    )
