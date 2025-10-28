"""Trade execution processing for environment steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .market import MarketSnapshot


@dataclass(frozen=True)
class TradeExecutionResult:
    """Outcome statistics after執行單一步交易。"""

    new_equity: float
    realized_pnl_step: float
    margin_buffer: Optional[float]
    position_change: float
    turnover_ratio: Optional[float]
    unrealized_pnl: float
    has_position: bool
    stop_loss_triggered: bool
    dist_to_extreme_atr: Optional[float]
    mae_atr: Optional[float]
    leverage_ratio: Optional[float]
    updated_last_position_size: float
    entry_happened: bool
    entry_streak_count: int
    entry_streak_side: int
    cooldown_active: bool
    cooldown_remaining: int
    cooldown_prevent_entry: bool


class TradeExecutionProcessor:
    """Handles trade execution and related metrics aggregation."""

    def __init__(
        self,
        *,
        stop_loss_cooldown_window: int = 6,
        stop_loss_cooldown_length: int = 12,
        stop_loss_cooldown_limit: int = 2,
        cooldown_hold_ratio: float = 0.0,
    ) -> None:
        """Initialise the processor and configure stop-loss cooldown defaults.

        Args:
            stop_loss_cooldown_window: Sliding window (in steps) used to aggregate recent stop-loss events.
            stop_loss_cooldown_length: Number of steps to pause new entries once the cooldown triggers.
            stop_loss_cooldown_limit: Stop-loss count threshold within the window that activates the cooldown.
            cooldown_hold_ratio: Target position ratio while in cooldown (typically 0 to stay flat).
        """

        if stop_loss_cooldown_window <= 0:
            raise ValueError("stop_loss_cooldown_window must be positive")
        if stop_loss_cooldown_length <= 0:
            raise ValueError("stop_loss_cooldown_length must be positive")
        if stop_loss_cooldown_limit <= 0:
            raise ValueError("stop_loss_cooldown_limit must be positive")

        self.stop_loss_cooldown_window = int(stop_loss_cooldown_window)
        self.stop_loss_cooldown_length = int(stop_loss_cooldown_length)
        self.stop_loss_cooldown_limit = int(stop_loss_cooldown_limit)
        self.cooldown_hold_ratio = float(cooldown_hold_ratio)

    def execute(
        self,
        *,
        executor: Any,
        action: float,
        snapshot: MarketSnapshot,
        leverage: float,
        last_position_size: float,
    ) -> TradeExecutionResult:
        """Execute the trade action and回傳統計。"""

        prev_wallet_balance = float(executor.wallet_balance)
        current_step = int(getattr(snapshot, 'window_end', 0))

        # 停損冷靜期參數（可由 executor 覆寫）
        cooldown_window = int(getattr(executor, 'stop_loss_cooldown_window', self.stop_loss_cooldown_window))#停損冷靜期窗口 
        cooldown_length = int(getattr(executor, 'stop_loss_cooldown_length', self.stop_loss_cooldown_length))#停損冷靜期長度 = 2 * cooldown_window
        cooldown_limit = int(getattr(executor, 'stop_loss_cooldown_limit', self.stop_loss_cooldown_limit))#停損冷靜期限制
        cooldown_hold_ratio = float(getattr(executor, 'cooldown_hold_ratio', self.cooldown_hold_ratio))
        executor.stop_loss_cooldown_window = cooldown_window#停損冷靜期窗口
        executor.stop_loss_cooldown_length = cooldown_length
        executor.stop_loss_cooldown_limit = cooldown_limit
        executor.cooldown_hold_ratio = cooldown_hold_ratio

        if not hasattr(executor, 'cooldown_until_step'):
            executor.cooldown_until_step = -1
        if not hasattr(executor, 'stop_loss_steps'):
            executor.stop_loss_steps = []

        in_cooldown = current_step < getattr(executor, 'cooldown_until_step', -1)
        cooldown_prevent_entry = False
        action_to_use = action
        if in_cooldown and abs(float(getattr(executor.position, 'size', 0.0))) <= 1e-8:
            action_to_use = cooldown_hold_ratio
            cooldown_prevent_entry = True

        executor.execute(
            position_percent=action_to_use,
            current_price=snapshot.current_price,
            high=snapshot.current_high,
            low=snapshot.current_low,
            equity=snapshot.last_equity,
            atr=snapshot.atr_est,
        )

        new_equity = float(executor.equity(snapshot.current_price))
        realized_pnl_step = float(executor.wallet_balance - prev_wallet_balance)

        margin_buffer = self._compute_margin_buffer(
            executor=executor,
            leverage=leverage,
            current_price=snapshot.current_price,
            new_equity=new_equity,
        )

        position_change = abs(float(executor.position.size - last_position_size))
        turnover_ratio = self._compute_turnover_ratio(
            position_change=position_change,
            current_price=snapshot.current_price,
            new_equity=new_equity,
        )

        unrealized_pnl = float(executor.unrealized_pnl(snapshot.current_price))
        has_position = abs(executor.position.size) > 1e-8
        stop_loss_triggered = bool(executor.stop_loss_triggered)

        if stop_loss_triggered:
            history = executor.stop_loss_steps
            history.append(current_step)
            cutoff = current_step - cooldown_window
            history = [step for step in history if current_step - step <= cooldown_window]
            executor.stop_loss_steps = history
            if len(history) >= cooldown_limit:
                window_range = history[-1] - history[-cooldown_limit]
                if window_range <= cooldown_window:
                    executor.cooldown_until_step = max(
                        executor.cooldown_until_step,
                        current_step + cooldown_length,
                    )
        else:
            executor.stop_loss_steps = [
                step for step in executor.stop_loss_steps
                if current_step - step <= cooldown_window
            ]

        cooldown_active = current_step < getattr(executor, 'cooldown_until_step', -1)#停損冷靜期是否激活
        cooldown_remaining = max(0, getattr(executor, 'cooldown_until_step', -1) - current_step)#停損冷靜期剩余时间

        entry_happened = bool(getattr(executor, 'entry_happened', False))#是否發生入場
        entry_streak_count = int(getattr(executor, 'entry_streak_count', 0))#入場連續次數
        entry_streak_side = int(getattr(executor, 'entry_streak_side', 0))#入場連續方向

        dist_to_extreme_atr, mae_atr, leverage_ratio = self._compute_position_metrics(
            executor=executor,
            snapshot=snapshot,
            has_position=has_position,
        )

        return TradeExecutionResult(
            new_equity=new_equity,
            realized_pnl_step=realized_pnl_step,
            margin_buffer=margin_buffer,
            position_change=position_change,
            turnover_ratio=turnover_ratio,
            unrealized_pnl=unrealized_pnl,
            has_position=has_position,
            stop_loss_triggered=stop_loss_triggered,
            dist_to_extreme_atr=dist_to_extreme_atr,
            mae_atr=mae_atr,
            leverage_ratio=leverage_ratio,
            updated_last_position_size=float(executor.position.size),
            entry_happened=entry_happened,
            entry_streak_count=entry_streak_count,
            entry_streak_side=entry_streak_side,
            cooldown_active=cooldown_active,
            cooldown_remaining=int(cooldown_remaining),
            cooldown_prevent_entry=bool(cooldown_prevent_entry),
        )

    def _compute_margin_buffer(
        self,
        *,
        executor: Any,
        leverage: float,
        current_price: float,
        new_equity: float,
    ) -> Optional[float]:
        try:
            position_value = abs(float(executor.position.size * current_price))
            if new_equity <= 0:
                return None
            leverage_ratio = position_value / new_equity
            safe_leverage = float(leverage) * 0.8
            if leverage_ratio >= safe_leverage:
                return 0.0
            return float(np.clip(1.0 - (leverage_ratio / safe_leverage), 0.0, 1.0))
        except Exception:
            return 1.0

    def _compute_turnover_ratio(
        self,
        *,
        position_change: float,
        current_price: float,
        new_equity: float,
    ) -> Optional[float]:
        try:
            notional_change = position_change * current_price
            if new_equity <= 0:
                return None
            return float(abs(notional_change) / new_equity)
        except Exception:
            return None

    def _compute_position_metrics(
        self,
        *,
        executor: Any,
        snapshot: MarketSnapshot,
        has_position: bool,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        dist_to_extreme_atr: Optional[float] = None
        mae_atr: Optional[float] = None
        leverage_ratio: Optional[float] = None

        if not has_position:
            return dist_to_extreme_atr, mae_atr, leverage_ratio

        try:
            atr_est = snapshot.atr_est
            if executor.position.size > 0:
                dist = max(0.0, snapshot.window_high - snapshot.current_price)
                adverse_move = max(0.0, float(executor.position.entry_price) - snapshot.window_low)
            else:
                dist = max(0.0, snapshot.current_price - snapshot.window_low)
                adverse_move = max(0.0, snapshot.window_high - float(executor.position.entry_price))

            if atr_est > 0:
                dist_to_extreme_atr = float(dist / atr_est)
                mae_atr = float(adverse_move / atr_est)

            position_value = abs(float(executor.position.size * snapshot.current_price))
            equity = float(executor.equity(snapshot.current_price))
            if equity > 0:
                leverage_ratio = float(position_value / equity)
        except Exception:
            pass

        return dist_to_extreme_atr, mae_atr, leverage_ratio

