"""Binance 5m CSV incremental updater.

設計目標：
- 依 CSV 最後 timestamp 增量抓取 futures 5m klines
- 產出欄位格式與舊版 `GetTradeData.py` 一致，避免下游處理改動
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional, Protocol, Sequence

import pandas as pd

from DataUpdaterService.csv_store import read_last_timestamp, upsert_dataframe_to_csv_atomic


class FuturesKlinesClient(Protocol):
    """抽象 Binance futures klines client（DIP：可在 pytest 注入 fake）。"""

    def futures_klines(
        self,
        *,
        symbol: str,
        interval: str,
        startTime: int,
        endTime: int,
        limit: int,
    ) -> Sequence[Sequence[Any]]: ...


def _get_public_binance_client() -> Any:
    """建立 Binance 公開端點 client（不需要 API key）。"""
    from binance.client import Client  # local import: 減少非必要依賴在單元測試被 import

    return Client(api_key=None, api_secret=None)


@dataclass(frozen=True)
class Binance5mUpdateResult:
    symbol: str
    rows_fetched: int
    csv_path: Path


class Binance5mUpdater:
    """負責更新 5m 的 futures volume CSV。"""

    def __init__(self, *, client: Optional[FuturesKlinesClient] = None) -> None:
        self._client: FuturesKlinesClient = client if client is not None else _get_public_binance_client()

    def update_symbol(
        self,
        *,
        data_dir: Path,
        symbol: str,
        initial_backfill_days: int = 30,
    ) -> Binance5mUpdateResult:
        """更新單一 symbol 的 5m CSV。

        Args:
            data_dir: 專案 Data 目錄
            symbol: 交易對（例：BTCUSDT）
            initial_backfill_days: 若 CSV 不存在或讀不到最後時間，回補天數

        Returns:
            Binance5mUpdateResult
        """
        data_dir.mkdir(parents=True, exist_ok=True)
        csv_path = data_dir / f"{symbol}_futures_volume_5years_5min.csv"

        last_ts = read_last_timestamp(csv_path, time_col="timestamp")
        now = datetime.now(timezone.utc)
        if last_ts is None:
            start_dt = now - timedelta(days=int(initial_backfill_days))
        else:
            # +1ms 避免重複
            start_dt = pd.Timestamp(last_ts).to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(milliseconds=1)

        end_dt = now
        rows = self._fetch_5m_klines(symbol=symbol, start_dt=start_dt, end_dt=end_dt)
        if not rows:
            return Binance5mUpdateResult(symbol=symbol, rows_fetched=0, csv_path=csv_path)

        df = self._rows_to_dataframe(rows)

        # 原子 upsert（用 timestamp 當 key）
        upsert_dataframe_to_csv_atomic(
            df,
            filename=csv_path,
            key_cols=["timestamp"],
            parse_dates=["timestamp"],
        )
        return Binance5mUpdateResult(symbol=symbol, rows_fetched=len(df), csv_path=csv_path)

    def _fetch_5m_klines(self, *, symbol: str, start_dt: datetime, end_dt: datetime) -> List[Sequence[Any]]:
        """分頁抓取 futures_klines。"""
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        out: List[Sequence[Any]] = []
        current = start_ms
        prev_current: Optional[int] = None

        while current < end_ms:
            batch = self._client.futures_klines(
                symbol=str(symbol),
                interval="5m",
                startTime=int(current),
                endTime=int(end_ms),
                limit=1000,
            )
            if not batch:
                break
            out.extend(batch)
            # 下一頁：以上一根 open time + 1ms
            next_current = int(batch[-1][0]) + 1

            # 若 API 回傳沒有推進（保險避免死迴圈）
            if prev_current is not None and next_current <= prev_current:
                break
            prev_current = next_current
            current = next_current
        return out

    @staticmethod
    def _rows_to_dataframe(rows: Sequence[Sequence[Any]]) -> pd.DataFrame:
        """將 Binance futures_klines 原始 rows 轉成標準 DataFrame（欄位與舊版一致）。"""
        df = pd.DataFrame(
            list(rows),
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )
        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=False)
        for col in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"):
            df[col] = df[col].astype(float)

        # 舊版 GetTradeData.py 計算方式（維持一致，不額外加 eps）
        df["buy_volume"] = df["taker_buy_base"]
        df["sell_volume"] = df["volume"] - df["taker_buy_base"]
        df["volume_ratio"] = df["buy_volume"] / df["sell_volume"]
        df["long_short_ratio"] = df["taker_buy_base"] / (df["volume"] - df["taker_buy_base"])

        return df[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "buy_volume",
                "sell_volume",
                "volume_ratio",
                "long_short_ratio",
                "trades",
                "quote_volume",
            ]
        ].copy()


