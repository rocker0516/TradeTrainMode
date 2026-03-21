from __future__ import annotations

import csv

from LiveTradingRunner.live_trading_loop import (
    LiveRunnerState,
    TickDecision,
    _append_state_csv,
    _state_from_json,
    _state_to_json,
)


def _decision() -> TickDecision:
    return TickDecision(
        closed_bar_ts="2026-03-21 12:00:00",
        last_price=100.0,
        action_raw=0.2,
        action_clipped=0.2,
        target_pos_pct=0.2,
        final_pos_pct=0.2,
        equity_usdt=None,
        current_position_qty=None,
        target_position_qty=None,
        delta_qty=None,
        obs_account_context_named=None,
    )


def test_state_json_roundtrip_keeps_paper_fields() -> None:
    state = LiveRunnerState(
        decision_step=3,
        last_final_pos_pct=0.15,
        paper_equity_usdt=1024.5,
        last_mark_price=101.2,
        cumulative_fee_usdt=1.23,
    )
    payload = _state_to_json(state)
    restored = _state_from_json(payload)
    assert restored.decision_step == 3
    assert abs(restored.paper_equity_usdt - 1024.5) < 1e-9
    assert abs(float(restored.last_mark_price or 0.0) - 101.2) < 1e-9
    assert abs(restored.cumulative_fee_usdt - 1.23) < 1e-9


def test_append_state_csv_writes_header_once(tmp_path) -> None:
    csv_path = tmp_path / "state_history.csv"
    state = LiveRunnerState(paper_equity_usdt=1001.0, cumulative_fee_usdt=0.2, decision_step=1)
    _append_state_csv(csv_path=str(csv_path), decision=_decision(), state=state, fee_rate_pct=0.04)
    _append_state_csv(csv_path=str(csv_path), decision=_decision(), state=state, fee_rate_pct=0.04)

    with open(csv_path, "r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) == 2
    assert "paper_equity_usdt" in rows[0]
    assert "fee_rate_pct" in rows[0]

