from __future__ import annotations

import json

import numpy as np

from LiveTradingRunner.live_trading_loop import LiveRunnerState, _load_state, _save_state
from LiveTradingRunner.live_obs_align import default_last_action_effects


def test_state_save_and_load_roundtrip(tmp_path):  # type: ignore[no-untyped-def]
    p = tmp_path / "state.json"
    lae = default_last_action_effects()
    lae["last_final_pos_pct"] = 0.25
    s0 = LiveRunnerState(
        last_processed_closed_ts=np.datetime64("2026-01-25T09:00:00"),
        last_action_effects=lae,
        obs_metrics_initialized=True,
        session_initial_balance_usdt=1000.0,
        max_equity_so_far_usdt=1050.0,
        last_trade_decision_step=3,
        position_entry_decision_step=2,
        rolling_fee_sum_usdt=1.5,
        fee_history_pairs=[[1, 0.1], [2, 0.2]],
        flat_window_flags=[1.0, 0.0],
        recent_flat_ratio=0.5,
    )
    _save_state(str(p), s0)

    s1 = _load_state(str(p))
    assert str(s1.last_processed_closed_ts) == str(s0.last_processed_closed_ts)
    assert float(s1.last_action_effects["last_final_pos_pct"]) == 0.25
    assert s1.obs_metrics_initialized is True
    assert float(s1.max_equity_so_far_usdt) == 1050.0
    assert s1.last_trade_decision_step == 3
    assert s1.position_entry_decision_step == 2
    assert s1.fee_history_pairs == [[1.0, 0.1], [2.0, 0.2]]


def test_load_state_with_garbage_returns_default(tmp_path):  # type: ignore[no-untyped-def]
    p = tmp_path / "state.json"
    p.write_text("{not valid json", encoding="utf-8")
    s = _load_state(str(p))
    assert s.last_processed_closed_ts is None


