from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from DataUpdaterService.binance_5m_updater import Binance5mUpdater


class _FakeBinanceClient:
    """最小 fake client：模擬 futures_klines 分頁。"""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def futures_klines(
        self,
        *,
        symbol: str,
        interval: str,
        startTime: int,
        endTime: int,
        limit: int,
    ) -> Sequence[Sequence[Any]]:
        self.calls.append((startTime, endTime))
        # 只回傳很少 bars，避免測試跑太久
        step = 5 * 60 * 1000  # 5m
        rows: list[list[Any]] = []
        t = int(startTime)
        for i in range(3):
            open_time = t + i * step
            if open_time >= endTime:
                break
            rows.append(
                [
                    open_time,  # timestamp(ms)
                    "1",  # open
                    "2",  # high
                    "0.5",  # low
                    "1.5",  # close
                    "10",  # volume
                    open_time + step - 1,  # close_time
                    "100",  # quote_volume
                    "7",  # trades
                    "6",  # taker_buy_base
                    "60",  # taker_buy_quote
                    "0",  # ignore
                ]
            )
        return rows


def test_binance_5m_incremental_appends(tmp_path: Path) -> None:
    data_dir = tmp_path / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sym = "BTCUSDT"
    csv_path = data_dir / f"{sym}_futures_volume_5years_5min.csv"

    # 先放一筆「接近現在」的 timestamp，讓 updater 只抓很短的範圍
    last_ts = (pd.Timestamp.utcnow() - timedelta(minutes=10)).replace(tzinfo=None)
    pd.DataFrame({"timestamp": [last_ts], "open": [1]}).to_csv(csv_path, index=False)

    fake = _FakeBinanceClient()
    updater = Binance5mUpdater(client=fake)
    r = updater.update_symbol(data_dir=data_dir, symbol=sym, initial_backfill_days=30)
    assert r.csv_path == csv_path
    assert r.rows_fetched >= 1

    out = pd.read_csv(csv_path, parse_dates=["timestamp"])
    assert "volume_ratio" in out.columns
    assert out["timestamp"].max() >= pd.Timestamp(last_ts)


