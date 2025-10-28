"""Episode termination evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TerminationResult:
    """Aggregated終止資訊。"""

    done: bool
    reason: Optional[str]
    data_exhausted: bool
    balance_insufficient: bool
    liq_triggered: bool


class TerminationEvaluator:
    """Determines whether episode should end."""

    def evaluate(
        self,
        *,
        current_step: int,
        data_len: int,
        new_equity: float,
        min_balance: float,
        liq_triggered: bool,
    ) -> TerminationResult:
        data_exhausted = current_step >= data_len - 1
        balance_insufficient = new_equity <= min_balance

        if data_exhausted:
            return TerminationResult(
                done=True,
                reason='data_exhausted',
                data_exhausted=True,
                balance_insufficient=balance_insufficient,
                liq_triggered=liq_triggered,
            )

        if liq_triggered:
            return TerminationResult(
                done=True,
                reason='liq_triggered',
                data_exhausted=False,
                balance_insufficient=balance_insufficient,
                liq_triggered=True,
            )

        if balance_insufficient:
            return TerminationResult(
                done=True,
                reason='balance_insufficient',
                data_exhausted=False,
                balance_insufficient=True,
                liq_triggered=liq_triggered,
            )

        return TerminationResult(
            done=False,
            reason=None,
            data_exhausted=False,
            balance_insufficient=False,
            liq_triggered=liq_triggered,
        )

