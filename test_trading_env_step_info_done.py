"""
Unit tests for Env.trading_env helper methods.

目的：
- 確認 step() 的終止條件分類（terminated / truncated）符合 Gymnasium 語意。
- 確認 info dict 的組裝邏輯被抽離後仍維持關鍵欄位/行為一致。
"""

from __future__ import annotations


def _make_env_shallow() -> "TradingEnvironment":
    """
    建立一個不跑 __init__ 的 TradingEnvironment，用於測試純 helper 方法。

    注意：只填入 _build_step_info 所需的最小屬性，避免測試依賴 CSV/外部資料。
    """
    from Env.trading_env import TradingEnvironment

    env = TradingEnvironment.__new__(TradingEnvironment)
    env.initial_balance = 1000.0
    env.risk_budget = 0.5
    env.episode_stop_loss_count = 0
    env.episode_liq_count = 0

    # _build_step_info(done=True) 會讀取 executor / position 的最小欄位。
    class _DummyPos:
        size = 0.0
        entry_price = 0.0
        stop_loss_price = 0.0

    class _DummyExecutor:
        total_fees = 0.0
        long_entry_count = 0
        short_entry_count = 0
        long_close_count = 0
        short_close_count = 0
        position = _DummyPos()

    env.executor = _DummyExecutor()
    return env


def test_determine_termination_terminated_on_liq() -> None:
    """爆倉應視為 terminated，且不應同時標成 truncated。"""
    env = _make_env_shallow()

    terminated, truncated, reason = env._determine_termination(
        data_exhausted=False,
        max_steps_reached=False,
        balance_insufficient=False,
        liq_triggered=True,
    )

    assert terminated is True
    assert truncated is False
    assert reason == "liq_triggered"


def test_determine_termination_terminated_on_balance_insufficient() -> None:
    """權益不足應視為 terminated。"""
    env = _make_env_shallow()

    terminated, truncated, reason = env._determine_termination(
        data_exhausted=False,
        max_steps_reached=False,
        balance_insufficient=True,
        liq_triggered=False,
    )

    assert terminated is True
    assert truncated is False
    assert reason == "balance_insufficient"


def test_determine_termination_truncated_on_max_steps() -> None:
    """達到回合步數上限應視為 truncated。"""
    env = _make_env_shallow()

    terminated, truncated, reason = env._determine_termination(
        data_exhausted=False,
        max_steps_reached=True,
        balance_insufficient=False,
        liq_triggered=False,
    )

    assert terminated is False
    assert truncated is True
    assert reason == "max_steps_reached"


def test_determine_termination_truncated_on_data_exhausted() -> None:
    """數據走完應視為 truncated（屬於資料截斷）。"""
    env = _make_env_shallow()

    terminated, truncated, reason = env._determine_termination(
        data_exhausted=True,
        max_steps_reached=False,
        balance_insufficient=False,
        liq_triggered=False,
    )

    assert terminated is False
    assert truncated is True
    assert reason == "data_exhausted"


def test_build_step_info_contains_required_keys_and_done_fields() -> None:
    """結束回合時，info 應包含 termination_reason / final_balance / terminated / truncated。"""
    env = _make_env_shallow()

    info = env._build_step_info(
        new_equity=1100.0,
        stop_loss_triggered=False,
        liq_triggered=False,
        step_fee=1.25,
        is_flip=True,
        current_dd=0.1,
        episode_max_dd=0.25,
        episode_turnover_notional=123.0,
        episode_holding_steps=10,
        episode_trade_count=3,
        episode_flat_steps=5,
        terminated=False,
        truncated=True,
        termination_reason="max_steps_reached",
    )

    assert info["equity"] == 1100.0
    assert info["profit"] == 100.0
    assert info["is_flip"] is True
    assert info["risk_budget"] == 0.5
    assert info["termination_reason"] == "max_steps_reached"
    assert info["final_balance"] == 1100.0
    assert info["episode_max_dd"] == 0.25
    assert info["episode_turnover_notional"] == 123.0
    assert info["episode_holding_steps"] == 10
    assert info["episode_trade_count"] == 3
    assert info["episode_flat_steps"] == 5
    assert "fees_to_equity_ratio" in info
    assert info["terminated"] is False
    assert info["truncated"] is True


