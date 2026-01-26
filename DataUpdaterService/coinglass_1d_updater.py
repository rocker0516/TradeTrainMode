"""CoinGlass 1d updater (futures + macro/index)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from DataUpdaterService.coinglass_client import CoinGlassClient
from DataUpdaterService.csv_store import read_last_timestamp, upsert_dataframe_to_csv_atomic


@dataclass(frozen=True)
class CoinGlass1dUpdateResult:
    name: str
    rows_fetched: int
    csv_path: Path


def _ensure_time_datetime(df: pd.DataFrame, *, col: str) -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns:
        return out
    # time/timestamp 可能是 ms int 或 ISO string；統一轉成 datetime（寫 CSV 時會變成字串）
    if pd.api.types.is_numeric_dtype(out[col]):
        out[col] = pd.to_datetime(out[col].astype("int64"), unit="ms")
    else:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _ts_to_ms(ts: pd.Timestamp) -> int:
    """Timestamp -> epoch ms（不受本機時區影響）。"""
    return int(pd.Timestamp(ts).value // 1_000_000)


class CoinGlass1dUpdater:
    """更新 CoinGlass 1d futures + macro/index CSV。"""

    def __init__(self, *, client: CoinGlassClient) -> None:
        self._client = client

    def update_symbols_futures_1d(
        self,
        *,
        data_dir: Path,
        symbols: Iterable[str],
        exchange: str,
        initial_backfill_days: int = 3650,
    ) -> Dict[str, CoinGlass1dUpdateResult]:
        results: Dict[str, CoinGlass1dUpdateResult] = {}
        for sym in symbols:
            results[str(sym)] = self.update_symbol_futures_1d(
                data_dir=data_dir,
                symbol=str(sym),
                exchange=exchange,
                initial_backfill_days=initial_backfill_days,
            )
        return results

    def update_symbol_futures_1d(
        self,
        *,
        data_dir: Path,
        symbol: str,
        exchange: str,
        initial_backfill_days: int = 3650,
    ) -> CoinGlass1dUpdateResult:
        """更新單一 symbol 的 coinglass 1d CSV（合併多個 endpoint）。"""
        data_dir.mkdir(parents=True, exist_ok=True)
        csv_path = data_dir / f"{symbol}_futures_volume_coinglass_5years_1d.csv"

        last_ts = read_last_timestamp(csv_path, time_col="time")
        now = datetime.now(timezone.utc)
        if last_ts is None:
            start_dt = now - timedelta(days=int(initial_backfill_days))
        else:
            start_dt = pd.Timestamp(last_ts).to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(milliseconds=1)

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(now.timestamp() * 1000)

        interval = "1d"
        limit = 1000

        # 先抓 price（作為 base，含 time），需要分頁避免 limit=1000 截斷
        base = self._fetch_futures_history_paged(
            endpoint="futures/price/history",
            params={"exchange": exchange, "symbol": symbol, "interval": interval, "limit": limit, "start_time": start_ms, "end_time": end_ms},
            time_col="time",
        )
        if base.empty:
            return CoinGlass1dUpdateResult(name=symbol, rows_fetched=0, csv_path=csv_path)

        base = _ensure_time_datetime(base, col="time")
        merged = base

        # 依舊版腳本，將各 endpoint 的欄位加前綴（time 不加）
        def merge_opt(endpoint: str, prefix: str, *, params: Dict[str, object]) -> None:
            nonlocal merged
            df = self._fetch_futures_history_paged(endpoint=endpoint, params=params, time_col="time")
            if df.empty:
                return
            df = _ensure_time_datetime(df, col="time")
            df = df.rename(columns=lambda c: f"{prefix}_{c}" if c != "time" else c)
            merged = merged.merge(df, on="time", how="left")

        common_params = {"exchange": exchange, "symbol": symbol, "interval": interval, "limit": limit, "start_time": start_ms, "end_time": end_ms}
        merge_opt("futures/open-interest/history", "open_interest", params=dict(common_params))
        merge_opt("futures/funding-rate/history", "funding_rate", params=dict(common_params))

        # 兩個 weight endpoint 的 symbol 需要去 USDT（沿用舊版行為）
        sym_no_usdt = symbol.replace("USDT", "")
        merge_opt(
            "futures/funding-rate/oi-weight-history",
            "funding_rate_oi_weight",
            params={"symbol": sym_no_usdt, "interval": interval, "limit": limit, "start_time": start_ms, "end_time": end_ms},
        )
        merge_opt(
            "futures/funding-rate/vol-weight-history",
            "funding_rate_vol_weight",
            params={"symbol": sym_no_usdt, "interval": interval, "limit": limit, "start_time": start_ms, "end_time": end_ms},
        )

        merge_opt("futures/global-long-short-account-ratio/history", "global_long_short_account_ratio", params=dict(common_params))
        merge_opt("futures/top-long-short-account-ratio/history", "top_long_short_account_ratio", params=dict(common_params))
        merge_opt("futures/top-long-short-position-ratio/history", "top_long_short_position_ratio", params=dict(common_params))
        merge_opt("futures/liquidation/history", "liquidation", params=dict(common_params))
        merge_opt(
            "futures/liquidation/aggregated-history",
            "liquidation_aggregated",
            params={"exchange_list": exchange, "symbol": sym_no_usdt, "interval": interval, "limit": limit, "start_time": start_ms, "end_time": end_ms},
        )
        merge_opt("futures/orderbook/ask-bids-history", "ask_bids", params=dict(common_params))
        merge_opt(
            "futures/orderbook/aggregated-ask-bids-history",
            "aggregated_ask_bids",
            params={"exchange_list": exchange, "symbol": sym_no_usdt, "interval": interval, "limit": limit, "start_time": start_ms, "end_time": end_ms},
        )

        merged = merged.sort_values("time").drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)
        merged = merged.dropna(how="all")

        upsert_dataframe_to_csv_atomic(
            merged,
            filename=csv_path,
            key_cols=["time"],
            parse_dates=["time"],
        )
        return CoinGlass1dUpdateResult(name=symbol, rows_fetched=len(merged), csv_path=csv_path)

    def _fetch_futures_history_paged(self, *, endpoint: str, params: Dict[str, object], time_col: str) -> pd.DataFrame:
        """用 start_time/end_time 分頁抓取 CoinGlass futures history，避免 limit=1000 截斷。

        假設回傳資料含 time_col 並可用其最大值作為下一頁起點。
        """
        limit = int(params.get("limit", 1000)) if params.get("limit") is not None else 1000
        end_time = int(params.get("end_time", 0)) if params.get("end_time") is not None else 0
        current = int(params.get("start_time", 0)) if params.get("start_time") is not None else 0

        frames: list[pd.DataFrame] = []
        prev_max_ms: Optional[int] = None
        max_pages = 10000  # safety

        for _ in range(max_pages):
            if end_time and current >= end_time:
                break
            p = dict(params)
            p["start_time"] = int(current)
            df = self._client.fetch_dataframe(endpoint=endpoint, params=p)
            if df.empty:
                break
            if time_col not in df.columns:
                # 沒時間欄位無法分頁：直接回傳當前資料（避免死迴圈）
                frames.append(df)
                break

            df = _ensure_time_datetime(df, col=time_col)
            frames.append(df)

            max_ts = pd.to_datetime(df[time_col], errors="coerce").max()
            if pd.isna(max_ts):
                break
            max_ms = _ts_to_ms(pd.Timestamp(max_ts))

            if prev_max_ms is not None and max_ms <= prev_max_ms:
                break
            prev_max_ms = max_ms

            # 下一頁：上一頁最大時間 + 1ms
            current = max_ms + 1

            # 若回傳筆數 < limit，通常代表已到底
            if len(df) < limit:
                break

        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        return out

    def update_macro_indexes_1d(self, *, data_dir: Path) -> Dict[str, CoinGlass1dUpdateResult]:
        """更新 macro/index 1d CSV（fear_greed / altcoin_season / sopr / macro_oscillator）。"""
        data_dir.mkdir(parents=True, exist_ok=True)
        results: Dict[str, CoinGlass1dUpdateResult] = {}

        # 1) Fear & Greed（time）
        fg_path = data_dir / "fear_greed_index_history_1d.csv"
        fg = self._client.fetch_fear_greed_history()
        if not fg.empty:
            fg = _ensure_time_datetime(fg, col="time").sort_values("time").drop_duplicates(subset=["time"], keep="last")
            fg_last = read_last_timestamp(fg_path, time_col="time")
            if fg_last is not None:
                fg = fg[fg["time"] > fg_last]
            if not fg.empty:
                upsert_dataframe_to_csv_atomic(fg, filename=fg_path, key_cols=["time"], parse_dates=["time"])
            results["fear_greed"] = CoinGlass1dUpdateResult("fear_greed", len(fg), fg_path)
        else:
            results["fear_greed"] = CoinGlass1dUpdateResult("fear_greed", 0, fg_path)

        # 2) Index endpoints（timestamp）
        def update_index(endpoint: str, out_name: str, out_file: str) -> CoinGlass1dUpdateResult:
            out_path = data_dir / out_file
            df = self._client.fetch_dataframe(endpoint=endpoint, params=None)
            if df.empty:
                return CoinGlass1dUpdateResult(out_name, 0, out_path)
            df = _ensure_time_datetime(df, col="timestamp").sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
            last_ts = read_last_timestamp(out_path, time_col="timestamp")
            if last_ts is not None:
                df = df[df["timestamp"] > last_ts]
            if not df.empty:
                upsert_dataframe_to_csv_atomic(df, filename=out_path, key_cols=["timestamp"], parse_dates=["timestamp"])
            return CoinGlass1dUpdateResult(out_name, len(df), out_path)

        results["altcoin_season"] = update_index("index/altcoin-season", "altcoin_season", "altcoin_season_index_history_1d.csv")
        results["bitcoin_sth_sopr"] = update_index("index/bitcoin-sth-sopr", "bitcoin_sth_sopr", "bitcoin_sth_sopr_index_history_1d.csv")
        # 修正舊版 bug：LTH SOPR 不應覆蓋 STH 檔案
        results["bitcoin_lth_sopr"] = update_index("index/bitcoin-lth-sopr", "bitcoin_lth_sopr", "bitcoin_lth_sopr_index_history_1d.csv")
        results["bitcoin_macro_oscillator"] = update_index(
            "index/bitcoin-macro-oscillator",
            "bitcoin_macro_oscillator",
            "bitcoin_macro_oscillator_index_history_1d.csv",
        )

        return results


