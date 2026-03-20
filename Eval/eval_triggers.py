from __future__ import annotations

"""
Phase A/B 訓練評估觸發器。

此模組提供可擴展的評估觸發規則，支援：
- 指定步數與固定間隔觸發
- 內建 metric 門檻規則
- 自訂 Python 條件式
- 多規則組合（ANY/ALL）
"""

from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol, Sequence


@dataclass(frozen=True)
class EvalTriggerContext:
    """評估觸發判斷所需上下文。"""

    step: int
    metrics: Mapping[str, float] = field(default_factory=dict)
    eval_count: int = 0
    training_end: bool = False
    last_eval_step: Optional[int] = None


class EvalTrigger(Protocol):
    """評估觸發器介面。"""

    def should_evaluate(self, context: EvalTriggerContext) -> bool:
        """回傳是否應觸發評估。"""


@dataclass
class StepTrigger:
    """
    以步數條件觸發評估。

    - `at_steps`: 達到指定步數後觸發一次（只觸發一次）
    - `every_n_steps`: 每 N 步觸發一次
    - `min_interval_steps`: 兩次評估最小間隔
    """

    at_steps: Sequence[int] = field(default_factory=tuple)
    every_n_steps: int = 0
    min_interval_steps: int = 0
    include_training_end: bool = True
    _fired_at_steps: set[int] = field(default_factory=set, init=False, repr=False)
    _last_every_bucket: int = field(default=-1, init=False, repr=False)

    def should_evaluate(self, context: EvalTriggerContext) -> bool:
        if context.training_end and not self.include_training_end:
            return False
        if self.min_interval_steps > 0 and context.last_eval_step is not None:
            if context.step - int(context.last_eval_step) < self.min_interval_steps:
                return False

        step = int(context.step)

        at_hit = False
        for target in sorted(int(s) for s in self.at_steps if int(s) >= 0):
            if step >= target and target not in self._fired_at_steps:
                self._fired_at_steps.add(target)
                at_hit = True
                break

        every_hit = False
        if self.every_n_steps and self.every_n_steps > 0:
            bucket = step // int(self.every_n_steps)
            if bucket > 0 and bucket > self._last_every_bucket:
                self._last_every_bucket = bucket
                every_hit = True

        return at_hit or every_hit


@dataclass(frozen=True)
class MetricRule:
    """單一 metric 規則。"""

    metric_name: str
    op: str
    threshold: float

    def evaluate(self, metrics: Mapping[str, float]) -> bool:
        if self.metric_name not in metrics:
            return False
        value = float(metrics[self.metric_name])
        threshold = float(self.threshold)
        if self.op == ">":
            return value > threshold
        if self.op == ">=":
            return value >= threshold
        if self.op == "<":
            return value < threshold
        if self.op == "<=":
            return value <= threshold
        if self.op == "==":
            return value == threshold
        if self.op == "!=":
            return value != threshold
        raise ValueError(f"Unsupported operator: {self.op}")


@dataclass
class MetricTrigger:
    """以內建 metrics 條件觸發評估。"""

    rules: Sequence[MetricRule] = field(default_factory=tuple)
    mode: str = "all"
    include_training_end: bool = True

    def should_evaluate(self, context: EvalTriggerContext) -> bool:
        if context.training_end and not self.include_training_end:
            return False
        if not self.rules:
            return False
        results = [rule.evaluate(context.metrics) for rule in self.rules]
        if self.mode.lower() == "any":
            return any(results)
        return all(results)


@dataclass
class ExpressionTrigger:
    """
    以自訂 Python 條件式觸發評估。

    可用變數：
    - `step`、`eval_count`、`training_end`
    - metrics key（僅限合法 Python 識別字）
    """

    expression: str
    include_training_end: bool = True

    def should_evaluate(self, context: EvalTriggerContext) -> bool:
        if context.training_end and not self.include_training_end:
            return False
        expr = (self.expression or "").strip()
        if not expr:
            return False

        safe_globals = {"__builtins__": {}}
        local_vars: dict[str, float | int | bool] = {
            "step": int(context.step),
            "eval_count": int(context.eval_count),
            "training_end": bool(context.training_end),
        }
        for key, value in context.metrics.items():
            if key.isidentifier():
                local_vars[key] = float(value)
        try:
            result = eval(expr, safe_globals, local_vars)
        except Exception:
            return False
        return bool(result)


@dataclass
class CompositeTrigger:
    """組合多個觸發器，支援 any/all。"""

    triggers: Sequence[EvalTrigger] = field(default_factory=tuple)
    mode: str = "any"

    def should_evaluate(self, context: EvalTriggerContext) -> bool:
        if not self.triggers:
            return False
        results = [trigger.should_evaluate(context) for trigger in self.triggers]
        if self.mode.lower() == "all":
            return all(results)
        return any(results)
