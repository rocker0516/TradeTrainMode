"""單元測試：`LiveTradingRunner/live_obs_align` 與訓練 env 對齊的輔助邏輯。"""

from __future__ import annotations

from LiveTradingRunner.live_obs_align import (
    append_flat_window,
    build_account_metrics_live,
    default_last_action_effects,
    estimate_expected_fee_usdt,
    merge_completed_last_action_effects,
    paper_position_qty,
    sync_executor_snapshot_for_obs_cache,
    update_fee_rolling_live,
)
from Env.Executors.trade_executor import TradeExecutor


def test_default_last_action_effects_has_trade_freq_blocked() -> None:
    d = default_last_action_effects()
    assert "trade_freq_blocked" in d
    assert d["last_action_raw"] == 0.0


def test_paper_position_qty_zero_price() -> None:
    assert paper_position_qty(equity_usdt=1000.0, leverage=10.0, pos_pct=0.5, price=0.0) == 0.0


def test_estimate_expected_fee_matches_no_change() -> None:
    fee = estimate_expected_fee_usdt(
        final_pos_pct=0.1,
        last_equity=1000.0,
        current_price=50000.0,
        current_size=paper_position_qty(
            equity_usdt=1000.0, leverage=10.0, pos_pct=0.1, price=50000.0
        ),
        leverage=10.0,
        fee_rate_pct=0.05,
    )
    assert abs(fee) < 1e-9


def test_fee_rolling_window_drops_old() -> None:
    hist: list[tuple[int, float]] = [(1, 1.0), (2, 2.0)]
    h2, roll, last = update_fee_rolling_live(
        fee_history=hist,
        rolling_fee_sum=3.0,
        decision_step=400,
        step_fee=1.0,
        fee_window=288,
    )
    assert last == 1.0
    assert roll >= 0.0
    assert all(ds >= 400 - 288 for ds, _ in h2)


def test_append_flat_window_ratio() -> None:
    h, r = append_flat_window([], is_flat=True, max_len=4)
    assert len(h) == 1 and r == 1.0
    h2, r2 = append_flat_window([1.0, 1.0, 0.0, 0.0], is_flat=False, max_len=4)
    assert len(h2) == 4
    assert 0.0 <= r2 <= 1.0


def test_build_account_metrics_steps_since_trade_fallback() -> None:
    am = build_account_metrics_live(
        session_initial_balance=1000.0,
        max_equity_so_far=1000.0,
        episode_stop_loss_count=0,
        episode_liq_count=0,
        risk_budget=1.0,
        decision_step_completed=0,
        episode_max_steps_cap=1000,
        last_trade_decision_step=-999999,
        position_entry_decision_step=None,
        last_step_fee=0.0,
        rolling_fee_sum=0.0,
        cooldown_remaining=0.0,
        min_balance=600.0,
        window_size_5m=24,
        recent_flat_ratio=0.5,
        trade_freq_blocked_last=0.0,
    )
    # 與 `TradingEnvironment.reset()` 相同：`last_trade_step=-999999` 仍大於 -1e8，故不走路徑 fallback
    assert am["steps_since_trade"] == 999999.0
    assert am["episode_steps"] == 0


def test_merge_completed_last_action_effects_updates_keys() -> None:
    ex = TradeExecutor(
        initial_balance=1000.0,
        fee_rate=0.05,
        leverage=10.0,
        min_trade_qty=0.001,
        margin_mode="isolated",
    )
    sync_executor_snapshot_for_obs_cache(
        ex,
        wallet_balance=1000.0,
        position_qty=0.01,
        entry_price=50000.0,
    )
    out = merge_completed_last_action_effects(
        base=default_last_action_effects(),
        executor=ex,
        expected_fee=1.0,
        current_price=50000.0,
        action_raw=0.5,
        action_used_scalar=0.5,
        target_pos_pct=0.4,
        final_pos_pct=0.4,
        traded=True,
        action_overridden_flag=False,
        trade_freq_blocked=False,
        cooldown_remaining_norm=0.0,
    )
    assert out["last_action_raw"] == 0.5
    assert out["trade_executed_flag"] == 1.0
    assert "predicted_used_margin_after_action" in out
