from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from services.data_fetch.incremental import resolve_start_time


def test_resolve_start_time_uses_fallback_when_file_missing(tmp_path) -> None:
    fallback = datetime(2026, 1, 1, 0, 0, 0)
    step = timedelta(minutes=5)
    missing_file = tmp_path / "missing.csv"

    got_binance = resolve_start_time(str(missing_file), "timestamp", fallback, step)
    got_coinglass = resolve_start_time(str(missing_file), "time", fallback, step)

    assert got_binance == fallback
    assert got_coinglass == fallback


def test_resolve_start_time_uses_next_bar_after_max_timestamp(tmp_path) -> None:
    csv_path = tmp_path / "binance.csv"
    pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:05:00",
                "2026-01-01 00:10:00",
            ]
        }
    ).to_csv(csv_path, index=False)

    fallback = datetime(2025, 12, 30, 0, 0, 0)
    step = timedelta(minutes=5)

    got = resolve_start_time(str(csv_path), "timestamp", fallback, step)
    assert got == datetime(2026, 1, 1, 0, 15, 0)


def test_resolve_start_time_never_earlier_than_fallback(tmp_path) -> None:
    csv_path = tmp_path / "coinglass.csv"
    pd.DataFrame({"time": ["2025-01-01 00:00:00"]}).to_csv(csv_path, index=False)

    fallback = datetime(2026, 2, 1, 0, 0, 0)
    step = timedelta(days=1)

    got = resolve_start_time(str(csv_path), "time", fallback, step)
    assert got == fallback

