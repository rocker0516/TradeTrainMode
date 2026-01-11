from __future__ import annotations

import numpy as np

from Env.Components.action_processor import ActionProcessor
from Env.Executors.trade_executor import TradeExecutor


def _make_executor(*, leverage: float = 10.0) -> TradeExecutor:
    return TradeExecutor(
        initial_balance=1000.0,
        fee_rate=0.0,
        leverage=leverage,
        min_trade_qty=0.001,
        maintenance_margin_rate=0.005,
        margin_mode="isolated",
        stop_loss_atr=0.0,
        stop_loss_liq_buffer_pct=0.0,
        min_position_change=0.0,
    )


def test_action_processor_flip_spends_budget_when_available() -> None:
    ex = _make_executor()
    ap = ActionProcessor(
        leverage=10.0,
        max_step_pos_change_pct=1.0,
        min_position_change=0.0,
    )

    # 先讓 executor 有多單（以 executor 自身交易建立現倉）
    ex.execute(
        position_percent=1.0,
        current_price=100.0,
        high=101.0,
        low=99.0,
        equity=ex.equity(100.0),
        atr=0.0,
        risk_base=1000.0,
    )

    target, is_flip = ap.process_action(np.array([-1.0], dtype=np.float32), ex, 100.0)

    assert is_flip is True
    assert target == -1.0


def test_action_processor_max_step_change_limits_position_build_up() -> None:
    ex = _make_executor()
    ap = ActionProcessor(
        leverage=10.0,
        max_step_pos_change_pct=0.05,  # 單步最多 5% capacity
        min_position_change=0.0,
    )

    price = 100.0
    equity = ex.equity(price)
    final_pct = ap.calculate_effective_action(1.0, ex, price, risk_base=1000.0)

    # 目標滿倉，但單步限制會讓 final_pct < 1
    assert 0.0 < final_pct < 1.0

    # 同樣對「關倉」應放寬（closing relax），讓 final_pct 能接近 0（從空倉本來就是 0）
    ex.execute(
        position_percent=final_pct,
        current_price=price,
        high=101.0,
        low=99.0,
        equity=equity,
        atr=0.0,
        risk_base=1000.0,
    )
    final_pct_close = ap.calculate_effective_action(0.0, ex, price, risk_base=1000.0)
    assert abs(final_pct_close) <= abs(final_pct)


