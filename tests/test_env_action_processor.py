from __future__ import annotations

import numpy as np
import pytest

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


def test_action_processor_no_trade_hysteresis_blocks_entry_when_flat() -> None:
    ex = _make_executor()
    ap = ActionProcessor(
        leverage=10.0,
        max_step_pos_change_pct=1.0,
        min_position_change=0.0,
        no_trade_entry_threshold=0.06,
        no_trade_exit_threshold=0.03,
    )

    # 空倉時：小於 entry threshold 的 action 應被吸附到 0（不進場）
    target, is_flip = ap.process_action(np.array([0.05], dtype=np.float32), ex, 100.0)
    assert is_flip is False
    assert float(target) == 0.0


def test_action_processor_no_trade_hysteresis_allows_entry_above_threshold() -> None:
    ex = _make_executor()
    ap = ActionProcessor(
        leverage=10.0,
        max_step_pos_change_pct=1.0,
        min_position_change=0.0,
        no_trade_entry_threshold=0.06,
        no_trade_exit_threshold=0.03,
    )

    target, is_flip = ap.process_action(np.array([0.07], dtype=np.float32), ex, 100.0)
    assert is_flip is False
    assert float(target) == pytest.approx(0.07, abs=1e-8)


def test_action_processor_no_trade_hysteresis_exits_when_has_position_and_action_below_exit_threshold() -> None:
    ex = _make_executor()
    ap = ActionProcessor(
        leverage=10.0,
        max_step_pos_change_pct=1.0,
        min_position_change=0.0,
        no_trade_entry_threshold=0.06,
        no_trade_exit_threshold=0.03,
    )

    # 先建立一個小倉位（讓 has_pos=True）
    ex.execute(
        position_percent=0.2,
        current_price=100.0,
        high=101.0,
        low=99.0,
        equity=ex.equity(100.0),
        atr=0.0,
        risk_base=1000.0,
    )

    # 有倉時：小於 exit threshold 的 action 應被吸附到 0（更容易回到空倉）
    target, is_flip = ap.process_action(np.array([0.02], dtype=np.float32), ex, 100.0)
    assert is_flip is False
    assert float(target) == 0.0


def test_action_processor_respects_max_position_pct_clip() -> None:
    ex = _make_executor()
    ap = ActionProcessor(
        leverage=10.0,
        max_step_pos_change_pct=1.0,
        min_position_change=0.0,
        max_position_pct=0.8,
    )
    target, _ = ap.process_action(np.array([1.0], dtype=np.float32), ex, 100.0)
    assert float(target) == pytest.approx(0.8, abs=1e-6)
    target_neg, _ = ap.process_action(np.array([-1.0], dtype=np.float32), ex, 100.0)
    assert float(target_neg) == pytest.approx(-0.8, abs=1e-6)


def test_action_processor_no_trade_hysteresis_validates_threshold_order() -> None:
    # entry 必須 >= exit（否則 hysteresis 會失去意義）
    with pytest.raises(ValueError):
        ActionProcessor(
            leverage=10.0,
            max_step_pos_change_pct=1.0,
            min_position_change=0.0,
            no_trade_entry_threshold=0.02,
            no_trade_exit_threshold=0.03,
        )


