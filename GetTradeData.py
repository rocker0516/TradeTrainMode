import os
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
import time
import math
from typing import List, Optional

# Binance API Configuration
API_KEY = 'JFyghzzEKteoSzrDlUbNPAYxyCnFwmwylHyNAkxRJhW4xlPdFcN5b9UgYBwU2o0p'
API_SECRET = 'N43ec0htdk2Bp14WuTLKmU0mhFsk4mmk8AMZy6PsPTshccVa5PDaOKu2QJ6Ngth1'

def fetch_futures_data(symbol, interval, start_time, end_time, max_retries=3):
    # Initialize Binance client with API keys
    client = Client(API_KEY, API_SECRET)
    
    # Convert dates to milliseconds
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    
    all_klines = []
    current_start_ts = start_ts
    
    # 添加請求計數器
    request_count = 0
    last_request_time = time.time()
    
    while current_start_ts < end_ts:
        try:
            # 檢查是否需要等待
            current_time = time.time()
            if request_count >= 2400:  # 每分鐘限制
                wait_time = 60 - (current_time - last_request_time)
                if wait_time > 0:
                    print(f"\nRate limit reached. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                request_count = 0
                last_request_time = time.time()
            
            # Fetch klines data
            klines = client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=current_start_ts,
                endTime=end_ts,
                limit=1000
            )
            request_count += 1
            
            if not klines:
                break
            
            all_klines.extend(klines)
            current_start_ts = klines[-1][0] + 1
            
            # 顯示進度
            print(f"\rProcessed data up to {datetime.fromtimestamp(current_start_ts/1000)}", end="")
            
            time.sleep(0.2)
            
        except Exception as e:
            if "Too many requests" in str(e):
                print("\nRate limit reached. Waiting 60 seconds...")
                time.sleep(60)
                continue
            print(f"\nError fetching data: {str(e)}")
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',#時間 開盤價 最高價 最低價 收盤價 成交量
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',#收盤時間 成交量 交易量 買方成交量 買方成交量
        'taker_buy_quote', 'ignore'#賣方成交量 忽略
    ])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Convert string values to float
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 
                'taker_buy_base', 'taker_buy_quote']:
        df[col] = df[col].astype(float)
    
    # Calculate additional volume metrics
    df['buy_volume'] = df['taker_buy_base']  # 買方成交量
    df['sell_volume'] = df['volume'] - df['taker_buy_base']  # 賣方成交量
    df['volume_ratio'] = df['buy_volume'] / df['sell_volume']  # 買賣成交量比
    df['long_short_ratio'] = df['taker_buy_base'] / (df['volume'] - df['taker_buy_base'])  # 多空比
    
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume',
               'buy_volume', 'sell_volume', 'volume_ratio', 'long_short_ratio', 'trades', 'quote_volume']]

def format_value(value, decimals=2):
    if value is None:
        return "N/A"
    try:
        return f"{value:.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"

def upsert_dataframe_to_csv(
    new_df: pd.DataFrame,
    filename: str,
    key_cols: List[str],
    parse_dates: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    將新資料與既有 CSV 合併，重複 key 以新資料覆蓋，其餘直接插入。

    Args:
        new_df: 要寫入的新資料 DataFrame。
        filename: 目標 CSV 檔案路徑。
        key_cols: 判定重複的欄位名稱列表。
        parse_dates: 需要解析為日期的欄位名稱列表。

    Returns:
        合併後的 DataFrame。
    """
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename, parse_dates=parse_dates)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
    else:
        combined = new_df.copy()

    if key_cols:
        combined = combined.sort_values(key_cols).reset_index(drop=True)

    combined.to_csv(filename, index=False)
    return combined

def main():
    # List of trading pairs to fetch
    trading_pairs = [
        'BTCUSDT',
        'ETHUSDT',
        'SOLUSDT', 
        'DOGEUSDT',
        '1000PEPEUSDT'
    ]
    
    # Set parameters
    interval = Client.KLINE_INTERVAL_5MINUTE  # 15-minute intervals
    
    # Calculate date range (5 years)
    end_time = datetime.now()  # 使用昨天的數據
    start_time = end_time - timedelta(days=2*365)
    
    print(f"Fetching data from {start_time} to {end_time}")
    
    # Fetch data for each trading pair
    for symbol in trading_pairs:
        try:
            print(f"\nFetching {symbol} futures data from {start_time} to {end_time}...")
            df = fetch_futures_data(symbol, interval, start_time, end_time)
            
            # Save to CSV
            filename = f"Data/{symbol}_futures_volume_5years_5min.csv"
            df = upsert_dataframe_to_csv(
                df,
                filename,
                key_cols=["timestamp"],
                parse_dates=["timestamp"]
            )
            print(f"Data saved to {filename}")
            
            # Display basic information
            print(f"Total records: {len(df)}")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print("\nVolume Statistics:")
            print(f"Average Volume: {format_value(df['volume'].mean())}")
            print(f"Average Buy Volume: {format_value(df['buy_volume'].mean())}")
            print(f"Average Sell Volume: {format_value(df['sell_volume'].mean())}")
            print(f"Average Volume Ratio: {format_value(df['volume_ratio'].mean())}")
            print(f"Average Long-Short Ratio: {format_value(df['long_short_ratio'].mean())}")
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
