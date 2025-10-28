"""Reward computation helpers for environment steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .execution import TradeExecutionResult


@dataclass
class PotentialState:
    """Stores previous potential variables used by RUDDER shaping."""

    margin_buffer: Optional[float]
    mae_atr: Optional[float]
    dist_to_extreme_atr: Optional[float]
    has_position: bool


@dataclass(frozen=True)
class RewardComputation:
    """封裝獎勵結果、PBRS 狀態與附帶資訊。"""

    reward: float
    updated_state: PotentialState
    details: Dict[str, float]


class RewardAdapter:
    """Calculates reward based on當前設定並更新 PBRS 狀態。"""

    def compute(
        self,
        *,
        execution: TradeExecutionResult,
        use_rudder: bool,
        shaping_calc: Any,
        outcome_calc: Any,
        cost_calc: Any,
        reward_calculator: Any,
        last_equity: float,
        new_equity: float,
        realized_pnl_step: float,
        episode_steps: int,
        termination_reason: Optional[str],
        previous_state: PotentialState,
        initial_equity: float,
    ) -> RewardComputation:
        """Return reward and updated potential狀態，並提供拆解資訊。"""

        if use_rudder and shaping_calc is not None:
            breakdown = shaping_calc.compute_with_breakdown(
                margin_buffer=execution.margin_buffer,
                mae_atr=execution.mae_atr,
                dist_to_extreme_atr=execution.dist_to_extreme_atr,
                has_position=execution.has_position,
                prev_margin_buffer=previous_state.margin_buffer,
                prev_mae_atr=previous_state.mae_atr,
                prev_dist_to_extreme_atr=previous_state.dist_to_extreme_atr,
                prev_has_position=previous_state.has_position,
                entry_happened=execution.entry_happened,
                entry_streak_count=execution.entry_streak_count,
                last_equity=last_equity,
                new_equity=new_equity,
                initial_equity=initial_equity,
            )

            reward = float(breakdown.total)
            updated_state = PotentialState(
                margin_buffer=execution.margin_buffer,
                mae_atr=execution.mae_atr,
                dist_to_extreme_atr=execution.dist_to_extreme_atr,
                has_position=execution.has_position,
            )
            details: Dict[str, float] = {
                'reward_total': reward,
                'reward_shaping_total': reward,
                'reward_shaping_pbrs': float(breakdown.pbrs),
                'reward_shaping_entry': float(breakdown.entry),
                'reward_shaping_return': float(breakdown.return_component),
                'reward_shaping_survival': float(breakdown.survival_component),
                'reward_shaping_risk_penalty': float(breakdown.risk_penalty),
                'reward_shaping_struct_penalty': float(breakdown.struct_penalty),
                'constraint_risk_cost': float(breakdown.risk_cost),
                'constraint_struct_cost': float(breakdown.struct_cost),
                'constraint_entry_cost': 0.0,
                'constraint_trade_count': 0.0,
            }
            return RewardComputation(reward=reward, updated_state=updated_state, details=details)

        if reward_calculator is None:
            details = {'reward_total': 0.0}
            return RewardComputation(reward=0.0, updated_state=previous_state, details=details)

        reward = float(
            reward_calculator.compute(
                last_equity=last_equity,
                new_equity=new_equity,
                margin_buffer=execution.margin_buffer,
                position_change=execution.position_change,
                turnover_ratio=execution.turnover_ratio,
                dist_to_extreme_atr=execution.dist_to_extreme_atr,
                mae_atr=execution.mae_atr,
                leverage_ratio=execution.leverage_ratio,
                has_position=execution.has_position,
                unrealized_pnl=execution.unrealized_pnl,
                traded=execution.position_change > 1e-8,
                entry_happened=execution.entry_happened,
                entry_streak_count=execution.entry_streak_count,
                entry_streak_side=execution.entry_streak_side,
                realized_pnl_step=realized_pnl_step,
                episode_steps=episode_steps,
                stop_loss_triggered=execution.stop_loss_triggered,
                done=termination_reason is not None,
                termination_reason=termination_reason,
            )
        )
        details = {'reward_total': reward}
        return RewardComputation(reward=reward, updated_state=previous_state, details=details)

