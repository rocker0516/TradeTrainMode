from __future__ import annotations

import json

import numpy as np

from LiveTradingRunner.live_trading_loop import LiveRunnerState, _load_state, _save_state


def test_state_save_and_load_roundtrip(tmp_path):  # type: ignore[no-untyped-def]
    p = tmp_path / "state.json"
    s0 = LiveRunnerState(last_processed_closed_ts=np.datetime64("2026-01-25T09:00:00"))
    _save_state(str(p), s0)

    s1 = _load_state(str(p))
    assert str(s1.last_processed_closed_ts) == str(s0.last_processed_closed_ts)


def test_load_state_with_garbage_returns_default(tmp_path):  # type: ignore[no-untyped-def]
    p = tmp_path / "state.json"
    p.write_text("{not valid json", encoding="utf-8")
    s = _load_state(str(p))
    assert s.last_processed_closed_ts is None


