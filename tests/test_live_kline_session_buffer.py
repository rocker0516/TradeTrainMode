"""Tests for live session K-line buffer (API merged OHLC → state → render)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from LiveTradingRunner.live_trading_loop import LiveRunnerState
from LiveTradingRunner.runner_core import extract_closed_target_ohlc, update_kline_session_buffer


def _dummy_merged_plus(*, n: int = 5) -> pd.DataFrame:
    """建立含尾端 dummy 的 merged_plus 寬表（單一 BTCUSDT）。"""
    base = pd.date_range("2026-01-01", periods=n, freq="5min")
    rows = []
    for i, ts in enumerate(base):
        o = 100.0 + i
        rows.append(
            {
                "timestamp": ts,
                "BTCUSDT_open": o,
                "BTCUSDT_high": o + 1,
                "BTCUSDT_low": o - 1,
                "BTCUSDT_close": o + 0.5,
            }
        )
    df = pd.DataFrame(rows)
    dummy = df.iloc[[-1]].copy()
    dummy.loc[:, "timestamp"] = df["timestamp"].iloc[-1] + pd.Timedelta(minutes=5)
    return pd.concat([df, dummy], ignore_index=True)


def test_extract_closed_target_ohlc_excludes_dummy() -> None:
    merged = _dummy_merged_plus(n=4)
    out = extract_closed_target_ohlc(merged, "BTCUSDT")
    assert len(out) == len(merged) - 1
    assert set(out.columns) == {"timestamp", "open", "high", "low", "close"}


def test_update_kline_session_buffer_seeds_then_appends() -> None:
    merged = _dummy_merged_plus(n=6)
    state = LiveRunnerState()
    update_kline_session_buffer(
        state,
        df_5m=merged,
        symbol="BTCUSDT",
        window_size_5m=3,
        max_buffer_rows=100,
    )
    assert len(state.kline_session_rows) == 3
    last_ts = state.kline_session_rows[-1]["timestamp"]

    # 模擬下一根收盤：將 merged 最後真實列時間往後推 5m，dummy 再延後
    closed_only = merged.iloc[:-1].copy()
    closed_only.loc[closed_only.index[-1], "timestamp"] = closed_only["timestamp"].iloc[-1] + pd.Timedelta(minutes=5)
    last_ts_new = closed_only["timestamp"].iloc[-1]
    dummy = closed_only.iloc[[-1]].copy()
    dummy.loc[:, "timestamp"] = last_ts_new + pd.Timedelta(minutes=5)
    merged2 = pd.concat([closed_only, dummy], ignore_index=True)

    update_kline_session_buffer(
        state,
        df_5m=merged2,
        symbol="BTCUSDT",
        window_size_5m=3,
        max_buffer_rows=100,
    )
    assert state.kline_session_rows[-1]["timestamp"] == str(pd.Timestamp(last_ts_new))[:19]
    assert state.kline_session_rows[-1]["timestamp"] != last_ts


def test_state_roundtrip_keeps_kline_and_gate_flags(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from LiveTradingRunner.live_trading_loop import _save_state, _load_state

    p = tmp_path / "s.json"
    s0 = LiveRunnerState(
        last_processed_closed_ts=np.datetime64("2026-01-25T09:00:00"),
        kline_session_rows=[{"timestamp": "2026-01-01 00:00:00", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5}],
        last_gate_flags=(1.0, 0.0, -1.0),
    )
    _save_state(str(p), s0)
    s1 = _load_state(str(p))
    assert len(s1.kline_session_rows) == 1
    assert s1.kline_session_rows[0]["open"] == 1.0
    assert s1.last_gate_flags == (1.0, 0.0, -1.0)
