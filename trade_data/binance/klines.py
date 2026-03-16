"""
Binance 期貨 K 線：取得並轉成與 futures_volume 表對應的 DataFrame。
"""

import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from binance.client import Client

from trade_data.binance.client import get_client


def fetch_futures_klines(
    symbol: str,
    interval: str,
    start_time: datetime,
    end_time: datetime,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    從 Binance 期貨 API 取得 K 線，組成含 volume_ratio、long_short_ratio 的 DataFrame。

    Args:
        symbol: 交易對，如 BTCUSDT。
        interval: K 線間隔，如 Client.KLINE_INTERVAL_5MINUTE。
        start_time: 開始時間。
        end_time: 結束時間。
        max_retries: 未使用，保留供未來重試邏輯。

    Returns:
        欄位：timestamp, open, high, low, close, volume, buy_volume, sell_volume,
        volume_ratio, long_short_ratio, trades, quote_volume。
    """
    client = get_client()
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    all_klines: list[list[Any]] = []
    current_start_ts = start_ts
    request_count = 0
    last_request_time = time.time()

    while current_start_ts < end_ts:
        try:
            current_time = time.time()
            if request_count >= 2400:
                wait_time = 60 - (current_time - last_request_time)
                if wait_time > 0:
                    print(f"\nRate limit reached. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                request_count = 0
                last_request_time = time.time()

            klines = client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=current_start_ts,
                endTime=end_ts,
                limit=1000,
            )
            request_count += 1

            if not klines:
                break

            all_klines.extend(klines)
            current_start_ts = klines[-1][0] + 1
            print(f"\rProcessed data up to {datetime.fromtimestamp(current_start_ts/1000)}", end="")
            time.sleep(0.2)

        except Exception as e:
            if "Too many requests" in str(e):
                print("\nRate limit reached. Waiting 60 seconds...")
                time.sleep(60)
                continue
            print(f"\nError fetching data: {str(e)}")
            break

    df = pd.DataFrame(
        all_klines,
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
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]:
        df[col] = df[col].astype(float)

    df["buy_volume"] = df["taker_buy_base"]
    df["sell_volume"] = df["volume"] - df["taker_buy_base"]
    # 分母為 0 時改為 NaN，讓 CSV 與 Supabase 都用同一份可序列化資料
    ratio_denominator = df["sell_volume"].replace(0, np.nan)
    df["volume_ratio"] = df["buy_volume"] / ratio_denominator
    df["long_short_ratio"] = df["taker_buy_base"] / ratio_denominator

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
    ]
