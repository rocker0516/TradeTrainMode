import os
import csv
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Protocol, Iterator
import requests
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoinGlassAPIError(Exception):
    """Base exception for CoinGlass API errors."""
    pass

class CoinGlassRequestError(CoinGlassAPIError):
    """Exception raised for network or request errors."""
    pass

class CoinGlassDataError(CoinGlassAPIError):
    """Exception raised for data parsing or logical errors."""
    pass


class CoinGlassClient:
    """
    Low-level client for CoinGlass API (SRP: Handle HTTP requests).
    """
    BASE_URL = "https://open-api-v4.coinglass.com/api/"

    def __init__(self, api_key: str):
        """
        Initializes the CoinGlass API client.
        
        Args:
            api_key: Your CoinGlass API key.
            
        Raises:
            ValueError: If API key is empty.
        """
        if not api_key:
            raise ValueError("API Key cannot be empty.")
        self._api_key = api_key
        self._session = requests.Session()
        self._session.headers.update({
            "accept": "application/json",
            "CG-API-KEY": self._api_key  # CoinGlass API v4 uses CG-API-KEY header
        })

    def _fetch_history_data(self, endpoint: str, exchange: str = None, symbol: str = None, interval: str = None, limit: int = None, 
                           start_time: Optional[int] = None, end_time: Optional[int] = None, 
                           raise_on_error: bool = True) -> pd.DataFrame:
        """
        Internal method to fetch history data from CoinGlass API.
        
        Args:
            endpoint: API endpoint path (e.g., 'futures/price/history').
            exchange: Exchange name (e.g., 'Binance').
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Time interval (e.g., '5m').
            limit: Maximum number of records to return.
            start_time: Start timestamp in milliseconds (optional).
            end_time: End timestamp in milliseconds (optional).
            raise_on_error: Whether to raise exceptions on API errors (default: True).
            
        Returns:
            DataFrame of history data.
            
        Raises:
            CoinGlassRequestError: If HTTP request fails or API returns error code (when raise_on_error=True).
            CoinGlassDataError: If response data cannot be parsed (when raise_on_error=True).
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if exchange:
            params["exchange"] = exchange
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        url = self.BASE_URL + endpoint

        try:
            logger.debug(f"Requesting: {url} with params: {params}")
            response = self._session.get(url, params=params, timeout=10)
            
            # Check HTTP status code
            if response.status_code == 400:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get("msg", "Bad Request (400)")
                if raise_on_error:
                    raise CoinGlassRequestError(
                        f"API returned 400 Bad Request: {error_msg}. "
                        f"Please check your API key and parameters. Response: {error_data}"
                    )
            
            # Handle rate limit (429 Too Many Requests)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "60")
                error_data = response.json() if response.content else {}
                error_msg = error_data.get("msg", "Rate limit exceeded")
                if raise_on_error:
                    raise CoinGlassRequestError(
                        f"API returned 429 Rate Limit Exceeded: {error_msg}. "
                        f"Please wait {retry_after} seconds before retrying. Response: {error_data}"
                    )
            
            response.raise_for_status()
            data = response.json()
            
            # Check API response code (CoinGlass uses code field in response body)
            api_code = data.get("code")
            if api_code and api_code != "0" and str(api_code) != "200":
                error_msg = data.get("msg", "Unknown error")
                if raise_on_error:
                    raise CoinGlassRequestError(
                        f"API returned error code {api_code}: {error_msg}. "
                        f"Full response: {data}"
                    )
            
            if "data" not in data:
                logger.warning(f"No 'data' field in API response: {data}")
                return pd.DataFrame()
            
            return pd.DataFrame(data["data"])
            
        except requests.exceptions.HTTPError as e:
            if raise_on_error:
                error_detail = ""
                try:
                    error_data = e.response.json() if e.response.content else {}
                    error_detail = f" Response: {error_data}"
                except:
                    error_detail = f" Response text: {e.response.text[:200]}"
                raise CoinGlassRequestError(f"HTTP error during API call: {e}{error_detail}") from e
            return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            if raise_on_error:
                raise CoinGlassRequestError(f"Network error during API call: {e}") from e
            return pd.DataFrame()
        except ValueError as e:
            if raise_on_error:
                raise CoinGlassDataError(f"Invalid JSON response: {e}") from e
            return pd.DataFrame()

    def get_price_history(self, exchange: str, symbol: str, interval: str, limit: int, 
                          start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Calls the price history endpoint.
        
        Args:
            exchange: Exchange name (e.g., 'Binance').
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Time interval (e.g., '5m').
            limit: Maximum number of records to return.
            start_time: Start timestamp in milliseconds (optional).
            end_time: End timestamp in milliseconds (optional).
            
        Returns:
            DataFrame of OHLCV data.
            
        Raises:
            CoinGlassRequestError: If HTTP request fails or API returns error code.
            CoinGlassDataError: If response data cannot be parsed.
        """
        return self._fetch_history_data(
            endpoint="futures/price/history",
            exchange=exchange,
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            raise_on_error=True
        )

    def get_open_interest_history(self, exchange: str, symbol: str, interval: str, limit: int, 
                          start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Calls the open interest history endpoint.
        
        Args:
            exchange: Exchange name (e.g., 'Binance').
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Time interval (e.g., '5m').
            limit: Maximum number of records to return.
            start_time: Start timestamp in milliseconds (optional).
            end_time: End timestamp in milliseconds (optional).
            
        Returns:
            DataFrame of open interest data.
            
        Raises:
            CoinGlassRequestError: If HTTP request fails or API returns error code.
            CoinGlassDataError: If response data cannot be parsed.
        """
        return self._fetch_history_data(
            endpoint="futures/open-interest/history",
            exchange=exchange,
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            raise_on_error=False
        )

    def get_funding_rate_history(self, exchange: str, symbol: str, interval: str, limit: int, 
                          start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Calls the funding rate history endpoint.
        
        Args:
            exchange: Exchange name (e.g., 'Binance').
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Time interval (e.g., '5m').
            limit: Maximum number of records to return.
            start_time: Start timestamp in milliseconds (optional).
            end_time: End timestamp in milliseconds (optional).
            
        Returns:
            DataFrame of funding rate data.
            
        Raises:
            CoinGlassRequestError: If HTTP request fails or API returns error code.
            CoinGlassDataError: If response data cannot be parsed.
        """
        return self._fetch_history_data(
            endpoint="futures/funding-rate/history",
            exchange=exchange,
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            raise_on_error=False
        )
        
    def get_funding_rate_oi_weight_history(self, exchange: str = None, symbol: str = None, interval: str = None, limit: int = None, 
                          start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Calls the funding rate oi weight history endpoint.
        
        Args:
            exchange: Exchange name (e.g., 'Binance').
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Time interval (e.g., '5m').
            limit: Maximum number of records to return.
            start_time: Start timestamp in milliseconds (optional).
            end_time: End timestamp in milliseconds (optional).
            
        Returns:
            DataFrame of funding rate oi weight data.
            
        Raises:
            CoinGlassRequestError: If HTTP request fails or API returns error code.
            CoinGlassDataError: If response data cannot be parsed.
        """
        return self._fetch_history_data(
            endpoint="futures/funding-rate/oi-weight-history",
            symbol=symbol.replace("USDT", ""),
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            raise_on_error=False
        )
    
    def get_funding_rate_vol_weight_history(self, symbol: str, interval: str, limit: int, 
                          start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Calls the funding rate vol weight history endpoint.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Time interval (e.g., '5m').
            limit: Maximum number of records to return.
            start_time: Start timestamp in milliseconds (optional).
            end_time: End timestamp in milliseconds (optional).
            
        Returns:
            DataFrame of funding rate vol weight data.
            
        Raises:
            CoinGlassRequestError: If HTTP request fails or API returns error code.
            CoinGlassDataError: If response data cannot be parsed.
        """
        return self._fetch_history_data(
            endpoint="futures/funding-rate/vol-weight-history",
            symbol=symbol.replace("USDT", ""),
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            raise_on_error=False
        )
    #特定交易所交易對的多空帳戶比率歷史記錄
    def get_global_long_short_account_ratio_history(self, exchange: str, symbol: str, interval: str, limit: int, 
                          start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Calls the global long short account ratio history endpoint.
        
        Args:
            exchange: Exchange name (e.g., 'Binance').
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Time interval (e.g., '5m').
            limit: Maximum number of records to return.
            start_time: Start timestamp in milliseconds (optional).
            end_time: End timestamp in milliseconds (optional).
            
        Returns:
            DataFrame of global long short account ratio data.
            
        Raises:
            CoinGlassRequestError: If HTTP request fails or API returns error code.
            CoinGlassDataError: If response data cannot be parsed.
        """
        return self._fetch_history_data(
            endpoint="futures/global-long-short-account-ratio/history",
            exchange=exchange,
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            raise_on_error=False
        )

        #頂級交易員多空帳戶比率的歷史資料
    def get_top_long_short_account_ratio_history(self, exchange: str, symbol: str, interval: str, limit: int, 
                          start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Calls the top long short account ratio history endpoint.
        
        Args:
            exchange: Exchange name (e.g., 'Binance').
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Time interval (e.g., '5m').
            limit: Maximum number of records to return.
            start_time: Start timestamp in milliseconds (optional).
            end_time: End timestamp in milliseconds (optional).
            
        Returns:
            DataFrame of top long short account ratio data.
            
        Raises:
            CoinGlassRequestError: If HTTP request fails or API returns error code.
            CoinGlassDataError: If response data cannot be parsed.
        """
        return self._fetch_history_data(
            endpoint="futures/top-long-short-account-ratio/history",
            exchange=exchange,
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            raise_on_error=False
        )
         #頂級交易員多空帳戶比率的歷史資料
    def get_top_long_short_position_ratio_history(self, exchange: str, symbol: str, interval: str, limit: int, 
                          start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Calls the top long short position ratio history endpoint.
        
        Args:
            exchange: Exchange name (e.g., 'Binance').
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Time interval (e.g., '5m').
            limit: Maximum number of records to return.
            start_time: Start timestamp in milliseconds (optional).
            end_time: End timestamp in milliseconds (optional).
            
        Returns:
            DataFrame of top long short position ratio data.
            
        Raises:
            CoinGlassRequestError: If HTTP request fails or API returns error code.
            CoinGlassDataError: If response data cannot be parsed.
        """
        return self._fetch_history_data(
            endpoint="futures/top-long-short-position-ratio/history",
            exchange=exchange,
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            raise_on_error=False
        )
def run_coinglass_fetch(
    api_key: str,
    exchange: str,
    symbol: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """
    Main function to orchestrate the process.
    """
    # 1. Setup Dependency Injection
    client = CoinGlassClient(api_key=api_key)

    end_ts = int(end_dt.timestamp() * 1000)
    start_ts = int(start_dt.timestamp() * 1000)
    all_klines: pd.DataFrame = pd.DataFrame()
    before_start_ts = start_ts - 1
    current_start_ts = start_ts

     # 添加請求計數器
    request_count = 0
    last_request_time = time.time()

    while current_start_ts < end_ts and current_start_ts != before_start_ts:
        before_start_ts = current_start_ts
        try:
            # 檢查是否需要等待
            current_time = time.time()
            if request_count >= 300:  # 每分鐘限制
                wait_time = 60 - (current_time - last_request_time)
                if wait_time > 0:
                    print(f"\nRate limit reached. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                request_count = 0
                last_request_time = time.time()
            
            history_data = client.get_price_history(
                exchange=exchange,
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start_ts,
                end_time= 60 * 5 * 1000 * 1000 + current_start_ts
            )
            request_count += 1
            #未平倉合約
            open_interest = client.get_open_interest_history(
                exchange=exchange,
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start_ts,
                end_time= 60 * 5 * 1000 * 1000 + current_start_ts
            )

            if len(open_interest) != 0:
                open_interest = open_interest.rename(columns=lambda c: f"open_interest_{c}" if c != "time" else c)
                history_data = history_data.merge(open_interest, on='time', how='left')
            request_count += 1
            #資金費率
            funding_rate = client.get_funding_rate_history(
                exchange=exchange,
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start_ts,
                end_time= 60 * 5 * 1000 * 1000 + current_start_ts
            )

            if len(funding_rate) != 0:
                funding_rate = funding_rate.rename(columns=lambda c: f"funding_rate_{c}" if c != "time" else c)
                history_data = history_data.merge(funding_rate, on='time', how='left')
            request_count += 1
            
            #資金費率未平倉合約權重
            funding_rate_oi_weight = client.get_funding_rate_oi_weight_history(
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start_ts,
                end_time= 60 * 5 * 1000 * 1000 + current_start_ts
            )
            if len(funding_rate_oi_weight) != 0:
                funding_rate_oi_weight = funding_rate_oi_weight.rename(columns=lambda c: f"funding_rate_oi_weight_{c}" if c != "time" else c)
                history_data = history_data.merge(funding_rate_oi_weight, on='time', how='left')
            request_count += 1
            
            #資金費率成交量權重
            funding_rate_vol_weight = client.get_funding_rate_vol_weight_history(
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start_ts,
                end_time= 60 * 5 * 1000 * 1000 + current_start_ts
            )
            if len(funding_rate_vol_weight) != 0:
                funding_rate_vol_weight = funding_rate_vol_weight.rename(columns=lambda c: f"funding_rate_vol_weight_{c}" if c != "time" else c)
                history_data = history_data.merge(funding_rate_vol_weight, on='time', how='left')
            request_count += 1

            #特定交易所交易對的多空帳戶比率歷史記錄
            global_long_short_account_ratio = client.get_global_long_short_account_ratio_history(
                exchange=exchange,
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start_ts,
                end_time= 60 * 5 * 1000 * 1000 + current_start_ts
            )
            if len(global_long_short_account_ratio) != 0:
                global_long_short_account_ratio = global_long_short_account_ratio.rename(columns=lambda c: f"global_long_short_account_ratio_{c}" if c != "time" else c)
                history_data = history_data.merge(global_long_short_account_ratio, on='time', how='left')
            request_count += 1
            
            #頂級交易員多空帳戶比率歷史資料
            top_long_short_account_ratio = client.get_top_long_short_account_ratio_history(
                exchange=exchange,
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start_ts,
                end_time= 60 * 5 * 1000 * 1000 + current_start_ts
            )
            if len(top_long_short_account_ratio) != 0:
                top_long_short_account_ratio = top_long_short_account_ratio.rename(columns=lambda c: f"top_long_short_account_ratio_{c}" if c != "time" else c)
                history_data = history_data.merge(top_long_short_account_ratio, on='time', how='left')
            request_count += 1
            
            #頂級交易員多空帳戶比率歷史資料
            top_long_short_position_ratio = client.get_top_long_short_position_ratio_history(
                exchange=exchange,
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start_ts,
                end_time= 60 * 5 * 1000 * 1000 + current_start_ts
            )
            if len(top_long_short_position_ratio) != 0:
                top_long_short_position_ratio = top_long_short_position_ratio.rename(columns=lambda c: f"top_long_short_position_ratio_{c}" if c != "time" else c)
                history_data = history_data.merge(top_long_short_position_ratio, on='time', how='left')
            request_count += 1

            all_klines = pd.concat([all_klines, history_data]) # 合併 dataframes，不會有重複的 timestamp
            last_timestamp = all_klines['time'].max()
            first_timestamp = all_klines['time'].min()
            
            current_start_ts = int(last_timestamp) + 1
            
            # 顯示進度
            print(f"\rProcessed data up to {datetime.fromtimestamp(current_start_ts/1000)}", end="")
            
            time.sleep(0.2)
            
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            print(f"Failed to fetch data: {e}")
            break
    
    return all_klines

if __name__ == "__main__":
    # Configuration via parameters/env
    API_KEY = os.getenv("COINGLASS_API_KEY", "1e41abd6360a4d1486b770e83982e33a")
     # List of trading pairs to fetch
    trading_pairs = [
        'BTCUSDT'
    ]
    EXCHANGE = "Binance"
    INTERVAL = "5m"
        # 2. Define Time Range (1 year)
    end_dt = datetime.now() - timedelta(minutes=5)
    start_dt = end_dt - timedelta(days=365 * 2)

    if API_KEY == "YOUR_API_KEY_HERE":
        print("Please set COINGLASS_API_KEY environment variable or edit the script.")
    else:
        # Fetch data for each trading pair
        for symbol in trading_pairs:
            try:
                print(f"\nFetching {symbol} futures data from {start_dt} to {end_dt}...")
                all_klines = run_coinglass_fetch(API_KEY, EXCHANGE, symbol, INTERVAL, start_dt, end_dt)
                
                # 將字典列表轉換為 DataFrame
                if all_klines.empty:
                    print(f"No data fetched for {symbol}")
                    continue
                
                # 按時間戳排序並去重
                if not all_klines.empty:
                    all_klines = all_klines.sort_values('time').drop_duplicates(subset=['time'], keep='first')
                    all_klines = all_klines.reset_index(drop=True)

                all_klines['time'] = pd.to_datetime(all_klines['time'], unit='ms')
                
                # Save to CSV
                filename = f"Data/{symbol}_futures_volume_coinglass_5years_5min.csv"
                all_klines.to_csv(filename, index=False)
                print(f"Data saved to {filename}")
                
                # Display basic information
                print(f"Total records: {len(all_klines)}")
                print(f"Date range: {all_klines['time'].min()} to {all_klines['time'].max()}")
                
                # Add a small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                continue
        

