"""
流程主程式：目前為 K 線 pipeline（fetch → CSV → Supabase）；未來可擴充其他資料流程。
"""

import time
from datetime import datetime, timedelta
from typing import Any

from binance.client import Client

from trade_data.binance.klines import fetch_futures_klines
from trade_data.csv_io import upsert_dataframe_to_csv
from trade_data.supabase_client import upsert_klines


def format_value(value: Any, decimals: int = 2) -> str:
    """將數值格式化為字串，None 或無法轉換時回傳 'N/A'。"""
    if value is None:
        return "N/A"
    try:
        return f"{value:.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def main() -> None:
    """取得期貨 K 線並寫入 CSV 與（若已設定）Supabase。"""
    trading_pairs = [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "DOGEUSDT",
        "1000PEPEUSDT",
    ]
    interval = Client.KLINE_INTERVAL_5MINUTE
    end_time = datetime.now()
    start_time = end_time - timedelta(days=6 * 365)

    print(f"Fetching data from {start_time} to {end_time}")

    for symbol in trading_pairs:
        try:
            print(f"\nFetching {symbol} futures data from {start_time} to {end_time}...")
            df = fetch_futures_klines(symbol, interval, start_time, end_time)

            filename = f"Data/{symbol}_futures_volume_5years_5min.csv"
            df = upsert_dataframe_to_csv(
                df,
                filename,
                key_cols=["timestamp"],
                parse_dates=["timestamp"],
            )
            print(f"Data saved to {filename}")

            print(f"Total records: {len(df)}")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print("Volume Statistics:")
            print(f"  Average Volume: {format_value(df['volume'].mean())}")
            print(f"  Average Buy Volume: {format_value(df['buy_volume'].mean())}")
            print(f"  Average Sell Volume: {format_value(df['sell_volume'].mean())}")
            print(f"  Average Volume Ratio: {format_value(df['volume_ratio'].mean())}")
            print(f"  Average Long-Short Ratio: {format_value(df['long_short_ratio'].mean())}")

            result = upsert_klines(df, symbol)
            if result is None:
                print("Supabase upsert skipped (no config or error).")
            else:
                print("Supabase upsert done.")

            time.sleep(1)

        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            continue
