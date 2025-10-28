"""Step info builder for RUDDER + Lagrangian integration.

遵循單一職責原則：此模組專責構建環境 step 結果的 `info` 字典，
封裝交易事件、風險統計與成本等欄位，供上層回調與後處理使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .trade_executor import TradeExecutor


@dataclass(frozen=True)
class StepContext:
    """Immutable context for building step info.

    Args:
        executor: 目前交易執行器實例。
        has_position: 是否持有倉位。
        stop_loss_triggered: 是否於本步觸發止損。
        liq_triggered: 是否於本步觸發強平。
        current_price: 當前最新價格，用於缺省時估算名目金額。
        use_rudder: 是否啟用 RUDDER 獎勵系統。
        outcome_calc: RUDDER outcome 計算器，可為 None。
        cost_calc: RUDDER cost 計算器，可為 None。
        margin_buffer: 本步 margin buffer（0-1）。
        mae_atr: 最大不利移動（ATR 單位）。
        dist_to_extreme_atr: 距極值距離（ATR 單位）。
        traded: 是否於本步執行交易。
        entry_happened: 是否於本步產生新進場。
        entry_streak_count: 同向連續進場次數。
        peak_equity: 當前回合歷史最高權益，用於計算回撤成本。
        current_equity: 當前權益。
        consecutive_stop_losses: 目前連續止損次數。
        cooldown_active: 是否處於停損冷靜期。
        cooldown_remaining: 冷靜期剩餘步數。
        cooldown_prevent_entry: 是否因冷靜期而阻止本步進場。
    """

    executor: TradeExecutor
    has_position: bool
    stop_loss_triggered: bool
    liq_triggered: bool
    current_price: float
    use_rudder: bool
    outcome_calc: Any | None
    cost_calc: Any | None
    traded: bool
    margin_buffer: float | None
    mae_atr: float | None
    dist_to_extreme_atr: float | None
    entry_happened: bool
    entry_streak_count: Optional[int]
    peak_equity: float
    current_equity: float
    consecutive_stop_losses: int
    cooldown_active: bool
    cooldown_remaining: int
    cooldown_prevent_entry: bool


class StepInfoBuilder:
    """Builds `info` dictionaries for each environment step.

    提供高階 `build()` 介面，並透過內部私有方法組合交易旗標、交易 ID、
    outcome 變化與成本統計，維持程式結構清晰且方便測試。
    """

    def build(self, context: StepContext) -> Dict[str, Any]:
        """Construct the info dict for the provided step context."""

        info: Dict[str, Any] = {}
        exit_snapshot = self._snapshot_exit_info(context.executor)

        self._populate_entry_flags(info, context)
        self._populate_exit_flags(info, context, exit_snapshot)
        self._populate_trade_identifier(info, context)
        self._populate_cost_metrics(info, context, exit_snapshot)

        info['cooldown_active'] = bool(context.cooldown_active)
        info['cooldown_remaining'] = int(context.cooldown_remaining)
        info['cooldown_prevent_entry'] = bool(context.cooldown_prevent_entry)

        return info

    def _snapshot_exit_info(self, executor: TradeExecutor) -> Optional[Dict[str, Any]]:
        last_exit = getattr(executor, 'last_exit_info', None)
        return dict(last_exit) if last_exit else None

    def _populate_entry_flags(self, info: Dict[str, Any], context: StepContext) -> None:
        is_entry = bool(context.executor.entry_happened)
        if context.has_position and is_entry:
            info['is_entry'] = True
            info['entered_trade_id'] = int(context.executor.position.trade_id)
        else:
            info['is_entry'] = False

    def _populate_exit_flags(
        self,
        info: Dict[str, Any],
        context: StepContext,
        exit_snapshot: Optional[Dict[str, Any]],
    ) -> None:
        if not exit_snapshot:
            info['is_reduce'] = False
            info['is_exit'] = False
            info['outcome_delta_to_entry'] = 0.0
            return

        exit_reason = exit_snapshot.get('exit_reason', 'close')
        exited_trade_id = exit_snapshot.get('trade_id', -1)

        if exit_reason == 'reduce':
            info['is_reduce'] = True
            info['is_exit'] = False
        else:
            info['is_reduce'] = False
            info['is_exit'] = True

        info['exited_trade_id'] = int(exited_trade_id)
        info['exit_reason'] = exit_reason
        info['outcome_delta_to_entry'] = self._compute_outcome_delta(
            context=context,
            exit_snapshot=exit_snapshot,
            exit_reason=exit_reason,
        )

    def _populate_trade_identifier(self, info: Dict[str, Any], context: StepContext) -> None:
        if context.has_position:
            info['trade_id'] = int(context.executor.position.trade_id)
        else:
            info['trade_id'] = -1

    def _populate_cost_metrics(
        self,
        info: Dict[str, Any],
        context: StepContext,
        exit_snapshot: Optional[Dict[str, Any]],
    ) -> None:
        if not (context.use_rudder and context.cost_calc is not None):
            info['step_cost'] = 0.0
            info['cost_prob'] = 0.0
            info['loss_cvar_sample'] = 0.0
            info.setdefault('constraint_risk_cost', 0.0)
            info.setdefault('constraint_struct_cost', 0.0)
            info.setdefault('trade_count_cost', 0.0)
            info.setdefault('entry_cost', 0.0)
            info.setdefault('drawdown_cost', 0.0)
            info.setdefault('stop_loss_streak_cost', 0.0)
            return

        fee = exit_snapshot.get('fee', 0.0) if exit_snapshot else 0.0
        step_cost = context.cost_calc.compute_step_cost(fee=fee, slippage=0.0, funding=0.0)
        info['step_cost'] = float(step_cost)

        cost_prob = context.cost_calc.compute_cost_prob(
            stop_loss_triggered=context.stop_loss_triggered,
            liq_triggered=context.liq_triggered,
        )
        info['cost_prob'] = float(cost_prob)

        outcome_delta = float(info.get('outcome_delta_to_entry', 0.0))
        loss_cvar_sample = context.cost_calc.compute_cvar_loss_sample(outcome_delta=outcome_delta)
        info['loss_cvar_sample'] = float(loss_cvar_sample)

        risk_cost = context.cost_calc.compute_risk_cost(
            margin_buffer=context.margin_buffer,
            has_position=context.has_position,
        )
        struct_cost = context.cost_calc.compute_struct_cost(
            mae_atr=context.mae_atr,
            dist_to_extreme_atr=context.dist_to_extreme_atr,
            has_position=context.has_position,
        )
        trade_cost = context.cost_calc.compute_trade_count_cost(traded=context.traded)
        entry_cost = context.cost_calc.compute_entry_cost(
            entry_happened=context.entry_happened,
            entry_streak_count=context.entry_streak_count,
        )
        drawdown_cost = context.cost_calc.compute_drawdown_cost(
            peak_equity=context.peak_equity,
            current_equity=context.current_equity,
        )
        stop_streak_cost = context.cost_calc.compute_stop_loss_streak_cost(
            consecutive_stop_losses=context.consecutive_stop_losses,
        )
        info['constraint_risk_cost'] = float(risk_cost)
        info['constraint_struct_cost'] = float(struct_cost)
        info['trade_count_cost'] = float(trade_cost)
        info['entry_cost'] = float(entry_cost)
        info['drawdown_cost'] = float(drawdown_cost)
        info['stop_loss_streak_cost'] = float(stop_streak_cost)

    def _compute_outcome_delta(
        self,
        *,
        context: StepContext,
        exit_snapshot: Dict[str, Any],
        exit_reason: str,
    ) -> float:
        if not (context.use_rudder and context.outcome_calc is not None):
            return 0.0

        realized_pnl = exit_snapshot.get('realized_pnl', 0.0)
        size = exit_snapshot.get('size', 0.0)
        price = exit_snapshot.get('price', context.current_price)
        notional = abs(float(size) * float(price))

        outcome_delta = context.outcome_calc.compute_outcome_delta(
            realized_pnl=realized_pnl,
            notional=notional,
            exit_reason=exit_reason,
            atr_multiple=context.mae_atr,
        )
        return float(outcome_delta)

