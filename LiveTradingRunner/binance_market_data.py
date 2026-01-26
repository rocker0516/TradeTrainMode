"""Binance market data fetcher for live runner (public endpoints).

本模組不需要 API Key（使用公開端點）。
用途：抓取「最新 N 根 5m K 線」並轉成與 Env/load_file.py 相容的欄位格式：
- timestamp (datetime)
- {SYMBOL}_open/high/low/close/volume/quote_volume/trades
- {SYMBOL}_buy_volume/{SYMBOL}_sell_volume/{SYMBOL}_volume_ratio/{SYMBOL}_long_short_ratio

注意：
- `long_short_ratio` 在此沿用你舊資料管線的計算方式（taker_buy_base / taker_sell_base）。
- 這不是 Binance 的真實多空比指標，但能維持與訓練資料一致的欄位語意。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LatestBarsResult:
    """抓取結果。"""

    df_5m: pd.DataFrame
    # 本次要用於決策的「最新已收盤 bar」時間（字串）
    latest_closed_bar_ts: str
    # 最新已收盤 bar 的價格（target symbol close）
    latest_closed_price: float


def _get_public_binance_client() -> Any:
    """建立 Binance 公開端點 client（不需要 key）。"""
    from binance.client import Client

    return Client(api_key=None, api_secret=None)


def _fetch_symbol_klines_5m(*, client: Any, symbol: str, limit: int) -> pd.DataFrame:
    """抓單一 symbol 的 futures 5m klines，並回傳標準化欄位。"""
    raw = client.futures_klines(symbol=str(symbol), interval="5m", limit=int(limit))
    df = pd.DataFrame(
        raw,
        columns=[
            "open_time",
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
        raise RuntimeError(f"Empty klines for symbol={symbol}")

    # open_time = candle open time (ms)
    df["timestamp"] = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms")

    for col in ("open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"):
        df[col] = df[col].astype(float)

    # derived (match your historical pipeline)
    eps = 1e-12
    df["buy_volume"] = df["taker_buy_base"]
    df["sell_volume"] = df["volume"] - df["taker_buy_base"]
    df["volume_ratio"] = df["buy_volume"] / (df["sell_volume"] + eps)
    df["long_short_ratio"] = df["taker_buy_base"] / (df["volume"] - df["taker_buy_base"] + eps)

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


def _prefix_columns(df: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    out = df.copy()
    pref = str(symbol)
    rename: Dict[str, str] = {}
    for c in out.columns:
        if c == "timestamp":
            continue
        rename[c] = f"{pref}_{c}"
    return out.rename(columns=rename)


def fetch_latest_multi_symbol_5m(
    *,
    symbols: Iterable[str],
    target_symbol: str,
    limit: int,
    client: Optional[Any] = None,
) -> LatestBarsResult:
    """抓取多幣種 5m 最新 bars，做時間對齊與 prefix。

    Args:
        symbols: e.g. TrainConfig.FEATURE_SYMBOLS
        target_symbol: 用於回傳 latest_closed_price
        limit: 每個 symbol 拉取的 kline 數
        client: 可注入 mock（pytest 使用）
    """
    if client is None:
        client = _get_public_binance_client()

    syms: List[str] = [str(s) for s in symbols]
    if str(target_symbol) not in syms:
        syms = [str(target_symbol)] + syms

    merged: Optional[pd.DataFrame] = None
    per_sym: Dict[str, pd.DataFrame] = {}
    for sym in syms:
        df = _fetch_symbol_klines_5m(client=client, symbol=sym, limit=limit)
        dfp = _prefix_columns(df, symbol=sym)
        per_sym[sym] = dfp
        if merged is None:
            merged = dfp
        else:
            # inner join：確保多幣時間軸完全對齊（與 Env/load_file.py 一致）
            merged = pd.merge(merged, dfp, on="timestamp", how="inner")

    if merged is None or merged.empty:
        raise RuntimeError("Failed to merge multi-symbol 5m data (empty).")

    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # ---- VecEnv/Observer 相容的小技巧：追加一筆 dummy row ----
    # 訓練環境的 MarketData.get_price_seq(step_idx) 會取 [step_idx-window, step_idx)。
    # 若我們希望「最後一根已收盤 bar」能被包含進去，需要 step_idx = last_index+1。
    # 但 step_idx 又會被 TradingObserver 拿來索引 hour_arr[step_idx]，因此我們追加一筆「下一步」row
    # 讓 step_idx 可以安全指向最後一列。
    last_ts = merged["timestamp"].iloc[-1]
    dummy_ts = pd.Timestamp(last_ts) + pd.Timedelta(minutes=5)
    dummy = merged.iloc[[-1]].copy()
    dummy.loc[:, "timestamp"] = dummy_ts
    merged_plus = pd.concat([merged, dummy], ignore_index=True)

    # latest closed bar：使用 merged_plus 的倒數第二列（最後一列是 dummy）
    closed_ts = merged_plus["timestamp"].iloc[-2]
    target_close_col = f"{str(target_symbol)}_close"
    if target_close_col not in merged_plus.columns:
        raise RuntimeError(f"Missing target close column: {target_close_col}")
    last_price = float(merged_plus[target_close_col].iloc[-2])

    return LatestBarsResult(
        df_5m=merged_plus,
        latest_closed_bar_ts=str(closed_ts)[:19],
        latest_closed_price=last_price,
    )


