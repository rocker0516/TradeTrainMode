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
from Env.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        self._max_retries = int(os.getenv("COINGLASS_MAX_RETRIES", "6"))
        self._min_request_interval_seconds = float(os.getenv("COINGLASS_MIN_REQUEST_INTERVAL_SECONDS", "1.0"))
        self._last_request_time = 0.0

    @staticmethod
    def _safe_json(response: requests.Response) -> Dict[str, Any]:
        """Safely parse JSON response."""
        try:
            return response.json() if response.content else {}
        except ValueError:
            return {}

    def _request_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        raise_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Request API data with built-in waiting/retry when rate limited (429).
        """
        for attempt in range(self._max_retries + 1):
            try:
                elapsed = time.time() - self._last_request_time
                if elapsed < self._min_request_interval_seconds:
                    time.sleep(self._min_request_interval_seconds - elapsed)

                response = self._session.get(url, params=params, timeout=10)
                self._last_request_time = time.time()
                data = self._safe_json(response)

                api_code = str(data.get("code", ""))
                api_msg = str(data.get("msg", "")).lower()
                is_rate_limited = (
                    response.status_code == 429
                    or api_code == "429"
                    or "too many requests" in api_msg
                )

                if is_rate_limited:
                    retry_after = response.headers.get("Retry-After")
                    retry_after_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 0
                    wait_seconds = max(retry_after_seconds, min(60, 2 ** attempt))
                    if attempt < self._max_retries:
                        logger.warning(
                            "Too Many Requests. Waiting %ss (retry %s/%s)...",
                            wait_seconds,
                            attempt + 1,
                            self._max_retries
                        )
                        time.sleep(wait_seconds)
                        continue
                    if raise_on_error:
                        raise CoinGlassRequestError(
                            f"API returned error code 429: Too Many Requests. Full response: {data}"
                        )
                    return {}

                response.raise_for_status()
                return data
            except requests.exceptions.RequestException as e:
                if attempt < self._max_retries:
                    wait_seconds = min(60, 2 ** attempt)
                    logger.warning(
                        "Request failed. Waiting %ss before retry %s/%s. Error: %s",
                        wait_seconds,
                        attempt + 1,
                        self._max_retries,
                        e
                    )
                    time.sleep(wait_seconds)
                    continue
                if raise_on_error:
                    raise CoinGlassRequestError(f"Network error during API call: {e}") from e
                return {}

        if raise_on_error:
            raise CoinGlassRequestError("Request failed after retries.")
        return {}

    def _fetch_fear_greed_history(self, raise_on_error: bool = True) -> pd.DataFrame:
        """
        Internal method to fetch history data from CoinGlass API.
        
        Args:
            raise_on_error: Whether to raise exceptions on API errors (default: True).
            
        Returns:
            DataFrame of fear-greed data with columns: time, fear_greed_index, price.
            
        Raises:
            CoinGlassRequestError: If HTTP request fails or API returns error code (when raise_on_error=True).
            CoinGlassDataError: If response data cannot be parsed (when raise_on_error=True).
        """
       
        url = self.BASE_URL + "index/fear-greed-history"

        try:
            logger.debug(f"Requesting: {url} ")
            data = self._request_json(url=url, raise_on_error=raise_on_error)
            if not data:
                return pd.DataFrame()
            
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
            
            # Parse the nested structure: data is a list containing one object with three lists
            data_array = data["data"]
            if not data_array or len(data_array) == 0:
                logger.warning(f"Empty 'data' array in API response")
                return pd.DataFrame()
            
            # Validate required fields
            if not all(key in data_array for key in ["data_list", "price_list", "time_list"]):
                if raise_on_error:
                    raise CoinGlassDataError(
                        f"Missing required fields in response data. "
                        f"Expected 'data_list', 'price_list', 'time_list'. Got: {list(data_array.keys())}"
                    )
                logger.warning(f"Missing required fields in response data: {list(data_array.keys())}")
                return pd.DataFrame()
            
            data_list = data_array["data_list"]
            price_list = data_array["price_list"]
            time_list = data_array["time_list"]
            
            # Validate that all lists have the same length
            if not (len(data_list) == len(price_list) == len(time_list)):
                if raise_on_error:
                    raise CoinGlassDataError(
                        f"List length mismatch: data_list={len(data_list)}, "
                        f"price_list={len(price_list)}, time_list={len(time_list)}"
                    )
                logger.warning(
                    f"List length mismatch: data_list={len(data_list)}, "
                    f"price_list={len(price_list)}, time_list={len(time_list)}"
                )
                return pd.DataFrame()
            
            # Create DataFrame from the three lists
            df = pd.DataFrame({
                "time": time_list,
                "fear_greed_index": data_list,
                "price": price_list
            })
            
            return df
            
        except CoinGlassRequestError:
            if raise_on_error:
                raise
            return pd.DataFrame()
        except ValueError as e:
            if raise_on_error:
                raise CoinGlassDataError(f"Invalid JSON response: {e}") from e
            return pd.DataFrame()

    def _fetch_index_history(self, endpoint: str, raise_on_error: bool = True) -> pd.DataFrame:
        """
        Internal method to fetch index history data from CoinGlass API.
        
        Args:
            endpoint: API endpoint path (e.g., 'index/altcoin-season').
            raise_on_error: Whether to raise exceptions on API errors (default: True).
            
        Returns:
            DataFrame of index data.
            
        Raises:
            CoinGlassRequestError: If HTTP request fails or API returns error code (when raise_on_error=True).
            CoinGlassDataError: If response data cannot be parsed (when raise_on_error=True).
        """
       
        url = self.BASE_URL + endpoint

        try:
            logger.debug(f"Requesting: {url} ")
            data = self._request_json(url=url, raise_on_error=raise_on_error)
            if not data:
                return pd.DataFrame()
            
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
            
            # Parse the nested structure: data is a list containing one object with three lists
            data_array = data["data"]
            if not data_array or len(data_array) == 0:
                logger.warning(f"Empty 'data' array in API response")
                return pd.DataFrame()
            
            
            # Create DataFrame from the data array
            df = pd.DataFrame(data_array)
            
            return df
            
        except CoinGlassRequestError:
            if raise_on_error:
                raise
            return pd.DataFrame()
        except ValueError as e:
            if raise_on_error:
                raise CoinGlassDataError(f"Invalid JSON response: {e}") from e
            return pd.DataFrame()

    def _fetch_history_data(self, endpoint: str, exchange: str = None, exchange_list: str = None, symbol: str = None, interval: str = None, limit: int = None, 
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
        if exchange_list:
            params["exchange_list"] = exchange_list
        if exchange:
            params["exchange"] = exchange
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        url = self.BASE_URL + endpoint

        try:
            logger.debug(f"Requesting: {url} with params: {params}")
            data = self._request_json(url=url, params=params, raise_on_error=raise_on_error)
            if not data:
                return pd.DataFrame()
            
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
            
        except CoinGlassRequestError:
            if raise_on_error:
                raise
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

           #吃單買賣量歷史資料
    def get_taker_buy_sell_volume_history(self, exchange: str, symbol: str, interval: str, limit: int, 
                          start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Calls the taker buy sell volume history endpoint.
        
        Args:
            exchange: Exchange name (e.g., 'Binance').
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Time interval (e.g., '5m').
            limit: Maximum number of records to return.
            start_time: Start timestamp in milliseconds (optional).
            end_time: End timestamp in milliseconds (optional).
            
        Returns:
            DataFrame of taker buy sell volume data.
            
        Raises:
            CoinGlassRequestError: If HTTP request fails or API returns error code.
            CoinGlassDataError: If response data cannot be parsed.
        """
        return self._fetch_history_data(
            endpoint="futures/v2/taker-buy-sell-volume/history",
            exchange=exchange,
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            raise_on_error=False
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
         #交易對多空平倉的歷史資料
    def get_liquidation_history(self, exchange: str, symbol: str, interval: str, limit: int, 
                          start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Calls the liquidation history endpoint.
        
        Args:
            exchange: Exchange name (e.g., 'Binance').
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Time interval (e.g., '5m').
            limit: Maximum number of records to return.
            start_time: Start timestamp in milliseconds (optional).
            end_time: End timestamp in milliseconds (optional).
            
        Returns:
            DataFrame of liquidation data.
            
        Raises:
            CoinGlassRequestError: If HTTP request fails or API returns error code.
            CoinGlassDataError: If response data cannot be parsed.
        """
        return self._fetch_history_data(
            endpoint="futures/liquidation/history",
            exchange=exchange,
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            raise_on_error=False
        )
         #多空清算總計歷史資料
    def get_liquidation_aggregated_history(self, exchange_list: str, symbol: str, interval: str, limit: int, 
                          start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Calls the liquidation aggregated history endpoint.
        
        Args:
            exchange: Exchange name (e.g., 'Binance').
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Time interval (e.g., '5m').
            limit: Maximum number of records to return.
            start_time: Start timestamp in milliseconds (optional).
            end_time: End timestamp in milliseconds (optional).
            
        Returns:
            DataFrame of liquidation aggregated data.
            
        Raises:
            CoinGlassRequestError: If HTTP request fails or API returns error code.
            CoinGlassDataError: If response data cannot be parsed.
        """
        return self._fetch_history_data(
            endpoint="futures/liquidation/aggregated-history",
            exchange_list=exchange_list,
            symbol=symbol.replace("USDT", ""),
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            raise_on_error=False
        )
            #提供期貨交易訂單簿的歷史數據，包括特定價格範圍內的總買賣量
    def get_ask_bids_history(self, exchange: str, symbol: str, interval: str, limit: int, 
                          start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Calls the ask-bids history endpoint.
        
        Args:
            exchange: Exchange name (e.g., 'Binance').
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Time interval (e.g., '5m').
            limit: Maximum number of records to return.
            start_time: Start timestamp in milliseconds (optional).
            end_time: End timestamp in milliseconds (optional).
            
        Returns:
            DataFrame of ask-bids data.
            
        Raises:
            CoinGlassRequestError: If HTTP request fails or API returns error code.
            CoinGlassDataError: If response data cannot be parsed.
        """
        return self._fetch_history_data(
            endpoint="futures/orderbook/ask-bids-history",
            exchange=exchange,
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            raise_on_error=False
        )
          #期貨交易總表的歷史數據，包括特定價格範圍內的總買賣價差
    def get_aggregated_ask_bids_history(self, exchange_list: str, symbol: str, interval: str, limit: int, 
                          start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Calls the aggregated ask-bids history endpoint.
        
        Args:
            exchange_list: Exchange list (e.g., 'Binance,Bitget').
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Time interval (e.g., '5m').
            limit: Maximum number of records to return.
            start_time: Start timestamp in milliseconds (optional).
            end_time: End timestamp in milliseconds (optional).
            
        Returns:
            DataFrame of aggregated ask-bids data.
            
        Raises:
            CoinGlassRequestError: If HTTP request fails or API returns error code.
            CoinGlassDataError: If response data cannot be parsed.
        """
        return self._fetch_history_data(
            endpoint="futures/orderbook/aggregated-ask-bids-history",
            exchange_list=exchange_list,
            symbol=symbol.replace("USDT", ""),
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            raise_on_error=False
        )
         #期貨交易總表的歷史數據，包括特定價格範圍內的總買賣價差
    def get_large_limit_order_history(self, exchange: str, symbol: str, interval: str, limit: int, 
                          start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Calls the large-limit-order history endpoint.
        
        Args:
            exchange: Exchange name (e.g., 'Binance').
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Time interval (e.g., '5m').
            limit: Maximum number of records to return.
            start_time: Start timestamp in milliseconds (optional).
            end_time: End timestamp in milliseconds (optional).
            
        Returns:
            DataFrame of large-limit-order data.
            
        Raises:
            CoinGlassRequestError: If HTTP request fails or API returns error code.
            CoinGlassDataError: If response data cannot be parsed.
        """
        return self._fetch_history_data(
            endpoint="futures/orderbook/large-limit-order-history",
            exchange=exchange,
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            raise_on_error=False
        )
        #提供歷史鯨魚指數資料。
    def get_whale_index_history(self, exchange: str, symbol: str, interval: str, limit: int, 
                          start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Calls the whale-index history endpoint.
        
        Args:
            exchange: Exchange name (e.g., 'Binance').
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Time interval (e.g., '5m').
            limit: Maximum number of records to return.
            start_time: Start timestamp in milliseconds (optional).
            end_time: End timestamp in milliseconds (optional).
            
        Returns:
            DataFrame of whale-index data.
            
        Raises:
            CoinGlassRequestError: If HTTP request fails or API returns error code.
            CoinGlassDataError: If response data cannot be parsed.
        """
        return self._fetch_history_data(
            endpoint="futures/whale-index/history",
            exchange=exchange,
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            raise_on_error=False
        )
         #提供加密貨幣的每日借貸利率。
    def get_borrow_interest_rate_history(self, exchange: str, symbol: str, interval: str, limit: int, 
                          start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        return self._fetch_history_data(
            endpoint="borrow-interest-rate/history",
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
    diff_ms: int
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
                end_time= diff_ms * 1000 + current_start_ts
            )
            request_count += 1
            #未平倉合約
            open_interest = client.get_open_interest_history(
                exchange=exchange,
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start_ts,
                end_time= diff_ms * 1000 + current_start_ts
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
                end_time= diff_ms * 1000 + current_start_ts
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
                end_time= diff_ms * 1000 + current_start_ts
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
                end_time= diff_ms * 1000 + current_start_ts
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
                end_time= diff_ms * 1000 + current_start_ts
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
                end_time= diff_ms * 1000 + current_start_ts
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
                end_time= diff_ms * 1000 + current_start_ts
            )
            if len(top_long_short_position_ratio) != 0:
                top_long_short_position_ratio = top_long_short_position_ratio.rename(columns=lambda c: f"top_long_short_position_ratio_{c}" if c != "time" else c)
                history_data = history_data.merge(top_long_short_position_ratio, on='time', how='left')
            request_count += 1

            #交易對多空平倉的歷史資料
            liquidation = client.get_liquidation_history(
                exchange=exchange,
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start_ts,
                end_time= diff_ms * 1000 + current_start_ts
            )
            if len(liquidation) != 0:
                liquidation = liquidation.rename(columns=lambda c: f"liquidation_{c}" if c != "time" else c)
                history_data = history_data.merge(liquidation, on='time', how='left')
            request_count += 1
            
            #多空清算總計歷史資料
            liquidation_aggregated = client.get_liquidation_aggregated_history(
                exchange_list=exchange,
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start_ts,
                end_time= diff_ms * 1000 + current_start_ts
            )
            if len(liquidation_aggregated) != 0:
                liquidation_aggregated = liquidation_aggregated.rename(columns=lambda c: f"liquidation_aggregated_{c}" if c != "time" else c)
                history_data = history_data.merge(liquidation_aggregated, on='time', how='left')
            request_count += 1

            #提供期貨交易訂單簿的歷史數據，包括特定價格範圍內的總買賣量
            ask_bids = client.get_ask_bids_history(
                exchange=exchange,
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start_ts,
                end_time= diff_ms * 1000 + current_start_ts
            )
            if len(ask_bids) != 0:
                ask_bids = ask_bids.rename(columns=lambda c: f"ask_bids_{c}" if c != "time" else c)
                history_data = history_data.merge(ask_bids, on='time', how='left')
            request_count += 1

            #期貨交易總表的歷史數據，包括特定價格範圍內的總買賣價差
            aggregated_ask_bids = client.get_aggregated_ask_bids_history(
                exchange_list=exchange,
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start_ts,                
                end_time= diff_ms * 1000 + current_start_ts

            )
            if len(aggregated_ask_bids) != 0:
                aggregated_ask_bids = aggregated_ask_bids.rename(columns=lambda c: f"aggregated_ask_bids_{c}" if c != "time" else c)
                history_data = history_data.merge(aggregated_ask_bids, on='time', how='left')
            request_count += 1

            #吃單買賣量歷史資料
            taker_buy_sell_volume = client.get_taker_buy_sell_volume_history(
                exchange=exchange,
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start_ts,                
                end_time= diff_ms * 1000 + current_start_ts
            )
            if len(taker_buy_sell_volume) != 0:
                taker_buy_sell_volume = taker_buy_sell_volume.rename(columns=lambda c: f"taker_buy_sell_volume_{c}" if c != "time" else c)
                history_data = history_data.merge(taker_buy_sell_volume, on='time', how='left')
            request_count += 1

            #期貨交易總表的歷史數據，包括特定價格範圍內的總買賣價差
            #large_limit_order = client.get_large_limit_order_history(
            #    exchange=exchange,
            #    symbol=symbol,
            #    interval=interval,
            #    limit=1000,
            #    start_time=current_start_ts,
            #    end_time= diff_ms * 1000 + current_start_ts
            #)
            #if len(large_limit_order) != 0:
            #    large_limit_order = large_limit_order.rename(columns=lambda c: f"large_limit_order_{c}" if c != "time" else c)
            #    history_data = history_data.merge(large_limit_order, on='time', how='left')
            #request_count += 1


            all_klines = pd.concat([all_klines, history_data]) # 合併 dataframes，不會有重複的 timestamp
            last_timestamp = all_klines['time'].max()
            first_timestamp = all_klines['time'].min()
            
            current_start_ts = int(last_timestamp) + 1
            
            # 顯示進度
            print(f"\rProcessed data up to {datetime.fromtimestamp(current_start_ts/1000)}", end="")
            
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            print(f"Failed to fetch data: {e}")
            break
    
    return all_klines

def run_once() -> None:
    """
    依據 Config 進行一次 CoinGlass 補資料。
    """
    api_key = os.getenv("COINGLASS_API_KEY", "1e41abd6360a4d1486b770e83982e33a")
    if not api_key:
        raise ValueError("Please set COINGLASS_API_KEY environment variable before running.")

    client = CoinGlassClient(api_key=api_key)
    trading_pairs = list(getattr(Config, "COINGLASS_FETCH_TRADING_PAIRS", ("BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT")))
    exchange = str(getattr(Config, "COINGLASS_EXCHANGE", "Binance"))
    interval = str(getattr(Config, "COINGLASS_FETCH_INTERVAL", "1d"))
    lookback_days = int(getattr(Config, "COINGLASS_FETCH_LOOKBACK_DAYS", 365 * 6))

    diff = timedelta(days=1)
    diff_ms = int(diff.total_seconds() * 1000)
    end_dt = datetime.now() - diff
    start_dt = end_dt - timedelta(days=lookback_days)

    for symbol in trading_pairs:
        try:
            print(f"\nFetching {symbol} futures data from {start_dt} to {end_dt}...")
            all_klines = run_coinglass_fetch(api_key, exchange, symbol, interval, start_dt, end_dt, diff_ms)

            if all_klines.empty:
                print(f"No data fetched for {symbol}")
                continue

            all_klines = all_klines.sort_values("time").drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)
            all_klines["time"] = pd.to_datetime(all_klines["time"], unit="ms")

            filename = f"Data/{symbol}_futures_volume_coinglass_5years_{interval}.csv"
            clean_klines = all_klines.dropna()
            upsert_dataframe_to_csv(
                clean_klines,
                filename,
                key_cols=["time"],
                parse_dates=["time"]
            )
            print(f"Data saved to {filename}")
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            continue

    try:
        fear_greed = client._fetch_fear_greed_history()
        if not fear_greed.empty and "time" in fear_greed.columns:
            fear_greed["time"] = pd.to_datetime(fear_greed["time"], unit="ms")
            filename = "Data/fear_greed_index_history_1d.csv"
            upsert_dataframe_to_csv(fear_greed, filename, key_cols=["time"], parse_dates=["time"])
            print(f"Fear & Greed data saved to {filename}")

        altcoin_season = client._fetch_index_history(endpoint="index/altcoin-season")
        if not altcoin_season.empty and "timestamp" in altcoin_season.columns:
            altcoin_season["timestamp"] = pd.to_datetime(altcoin_season["timestamp"], unit="ms")
            filename = "Data/altcoin_season_index_history_1d.csv"
            upsert_dataframe_to_csv(altcoin_season, filename, key_cols=["timestamp"], parse_dates=["timestamp"])
            print(f"Altcoin Season data saved to {filename}")

        bitcoin_sth_sopr = client._fetch_index_history(endpoint="index/bitcoin-sth-sopr")
        if not bitcoin_sth_sopr.empty and "timestamp" in bitcoin_sth_sopr.columns:
            bitcoin_sth_sopr["timestamp"] = pd.to_datetime(bitcoin_sth_sopr["timestamp"], unit="ms")
            filename = "Data/bitcoin_sth_sopr_index_history_1d.csv"
            upsert_dataframe_to_csv(bitcoin_sth_sopr, filename, key_cols=["timestamp"], parse_dates=["timestamp"])
            print(f"Bitcoin STH SOPR data saved to {filename}")

        bitcoin_lth_sopr = client._fetch_index_history(endpoint="index/bitcoin-lth-sopr")
        if not bitcoin_lth_sopr.empty and "timestamp" in bitcoin_lth_sopr.columns:
            bitcoin_lth_sopr["timestamp"] = pd.to_datetime(bitcoin_lth_sopr["timestamp"], unit="ms")
            filename = "Data/bitcoin_sth_sopr_index_history_1d.csv"
            upsert_dataframe_to_csv(bitcoin_lth_sopr, filename, key_cols=["timestamp"], parse_dates=["timestamp"])
            print(f"Bitcoin LTH SOPR data saved to {filename}")

        bitcoin_macro_oscillator = client._fetch_index_history(endpoint="index/bitcoin-macro-oscillator")
        if not bitcoin_macro_oscillator.empty and "timestamp" in bitcoin_macro_oscillator.columns:
            bitcoin_macro_oscillator["timestamp"] = pd.to_datetime(bitcoin_macro_oscillator["timestamp"], unit="ms")
            filename = "Data/bitcoin_macro_oscillator_index_history_1d.csv"
            upsert_dataframe_to_csv(
                bitcoin_macro_oscillator,
                filename,
                key_cols=["timestamp"],
                parse_dates=["timestamp"]
            )
            print(f"Bitcoin Macro Oscillator data saved to {filename}")
    except Exception as e:
        print(f"Error fetching Fear & Greed data: {str(e)}")


def run_periodic_service() -> None:
    """
    依據 Config 設定定時補資料。
    """
    interval_seconds = int(getattr(Config, "COINGLASS_FETCH_INTERVAL_SECONDS", 3600))
    run_on_startup = bool(getattr(Config, "DATA_FETCH_RUN_ON_STARTUP", True))
    max_cycles = int(getattr(Config, "DATA_FETCH_MAX_CYCLES", 0))

    cycle = 0
    if run_on_startup:
        cycle += 1
        print(f"\n[CoinGlass Fetch Service] Cycle {cycle} started.")
        run_once()
        if max_cycles > 0 and cycle >= max_cycles:
            print("[CoinGlass Fetch Service] Reached max cycles, service stopped.")
            return

    while True:
        print(f"[CoinGlass Fetch Service] Sleeping {interval_seconds} seconds...")
        time.sleep(interval_seconds)
        cycle += 1
        print(f"\n[CoinGlass Fetch Service] Cycle {cycle} started.")
        try:
            run_once()
        except Exception as e:
            print(f"[CoinGlass Fetch Service] Cycle failed: {e}")

        if max_cycles > 0 and cycle >= max_cycles:
            print("[CoinGlass Fetch Service] Reached max cycles, service stopped.")
            return


if __name__ == "__main__":
    service_enabled = bool(getattr(Config, "DATA_FETCH_SERVICE_ENABLED", True))
    try:
        if service_enabled:
            run_periodic_service()
        else:
            run_once()
    except KeyboardInterrupt:
        print("\n[CoinGlass Fetch Service] Stopped by user.")
