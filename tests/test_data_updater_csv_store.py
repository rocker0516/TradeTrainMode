from __future__ import annotations

from pathlib import Path

import pandas as pd

from DataUpdaterService.csv_store import read_last_timestamp, upsert_dataframe_to_csv_atomic


def test_read_last_timestamp_returns_max(tmp_path: Path) -> None:
    p = tmp_path / "x.csv"
    df = pd.DataFrame({"timestamp": ["2020-01-01 00:00:00", "2020-01-03 00:00:00", "2020-01-02 00:00:00"]})
    df.to_csv(p, index=False)

    last = read_last_timestamp(p, time_col="timestamp")
    assert last is not None
    assert str(last)[:10] == "2020-01-03"


def test_upsert_dataframe_to_csv_atomic_overwrites_duplicate_key(tmp_path: Path) -> None:
    p = tmp_path / "x.csv"
    df1 = pd.DataFrame(
        {
            "timestamp": ["2020-01-01 00:00:00", "2020-01-02 00:00:00"],
            "v": [1, 2],
        }
    )
    upsert_dataframe_to_csv_atomic(df1, filename=p, key_cols=["timestamp"], parse_dates=["timestamp"])

    # duplicate timestamp should be overwritten by new_df (keep last)
    df2 = pd.DataFrame(
        {
            "timestamp": ["2020-01-02 00:00:00", "2020-01-03 00:00:00"],
            "v": [999, 3],
        }
    )
    combined = upsert_dataframe_to_csv_atomic(df2, filename=p, key_cols=["timestamp"], parse_dates=["timestamp"])

    assert len(combined) == 3
    row = combined[combined["timestamp"] == pd.Timestamp("2020-01-02 00:00:00")].iloc[0]
    assert int(row["v"]) == 999


