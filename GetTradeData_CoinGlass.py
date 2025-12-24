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
            response = self._session.get(url, timeout=10)
            
            time.sleep(0.2)
            
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
            response = self._session.get(url, timeout=10)
            
            time.sleep(0.2)
            
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
            
            # Parse the nested structure: data is a list containing one object with three lists
            data_array = data["data"]
            if not data_array or len(data_array) == 0:
                logger.warning(f"Empty 'data' array in API response")
                return pd.DataFrame()
            
            
            # Create DataFrame from the data array
            df = pd.DataFrame(data_array)
            
            return df
            
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
            response = self._session.get(url, params=params, timeout=10)
            
            time.sleep(0.2)
            
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

if __name__ == "__main__":
    # Configuration via parameters/env
    API_KEY = os.getenv("COINGLASS_API_KEY", "1e41abd6360a4d1486b770e83982e33a")
    client = CoinGlassClient(api_key=API_KEY)
     # List of trading pairs to fetch
    trading_pairs = [
        'BTCUSDT',
        'ETHUSDT',
        'SOLUSDT',
        'DOGEUSDT',
        '1000PEPEUSDT'
    ]
    EXCHANGE = "Binance"
    INTERVAL = "1h"
        # 2. Define Time Range (1 year)
    diff = timedelta(hours=1)
    diff_ms = diff.total_seconds() * 1000
    end_dt = datetime.now() - diff
    start_dt = end_dt - timedelta(days=365 * 5)

    if API_KEY == "YOUR_API_KEY_HERE":
        print("Please set COINGLASS_API_KEY environment variable or edit the script.")
    else:
        # Fetch data for each trading pair
        for symbol in trading_pairs:
            try:
                print(f"\nFetching {symbol} futures data from {start_dt} to {end_dt}...")
                all_klines = run_coinglass_fetch(API_KEY, EXCHANGE, symbol, INTERVAL, start_dt, end_dt, int(diff_ms))
                
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
                filename = f"Data/{symbol}_futures_volume_coinglass_5years_{INTERVAL}.csv"
                all_klines.to_csv(filename, index=False)
                print(f"Data saved to {filename}")

                # Add a small delay to avoid rate limiting
                time.sleep(1)
            
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                continue
        try:
            
            fear_greed = client._fetch_fear_greed_history()
                
            if not fear_greed.empty:
                # Convert timestamp to datetime if time column exists
                if 'time' in fear_greed.columns:
                        fear_greed['time'] = pd.to_datetime(fear_greed['time'], unit='ms')
                    
                filename = f"Data/fear_greed_index_history.csv"
                fear_greed.to_csv(filename, index=False)
                print(f"Fear & Greed data saved to {filename}")
                print(f"Fear & Greed records: {len(fear_greed)}")
            else:
                print("No Fear & Greed data fetched")
                
            # Display basic information
            print(f"Total records: {len(fear_greed)}")
            print(f"Date range: {fear_greed['time'].min()} to {fear_greed['time'].max()}")

            altcoin_season = client._fetch_index_history(endpoint="index/altcoin-season")
            if not altcoin_season.empty:
                # Convert timestamp to datetime if time column exists
                if 'timestamp' in altcoin_season.columns:
                    altcoin_season['timestamp'] = pd.to_datetime(altcoin_season['timestamp'], unit='ms')
                    
                    filename = f"Data/altcoin_season_index_history.csv"
                    altcoin_season.to_csv(filename, index=False)
                    print(f"Altcoin Season data saved to {filename}")
                    print(f"Altcoin Season records: {len(altcoin_season)}")
                else:
                    print("No Altcoin Season data fetched")
                
                # Display basic information
                print(f"Total records: {len(altcoin_season)}")
                print(f"Date range: {altcoin_season['timestamp'].min()} to {altcoin_season['timestamp'].max()}")

            bitcoin_sth_sopr = client._fetch_index_history(endpoint="index/bitcoin-sth-sopr")
            if not bitcoin_sth_sopr.empty:
                # Convert timestamp to datetime if time column exists
                if 'timestamp' in bitcoin_sth_sopr.columns:
                    bitcoin_sth_sopr['timestamp'] = pd.to_datetime(bitcoin_sth_sopr['timestamp'], unit='ms')
                    
                    filename = f"Data/bitcoin_sth_sopr_index_history.csv"
                    bitcoin_sth_sopr.to_csv(filename, index=False)
                    print(f"Bitcoin STH SOPR data saved to {filename}")
                    print(f"Bitcoin STH SOPR records: {len(bitcoin_sth_sopr)}")

            
            bitcoin_lth_sopr = client._fetch_index_history(endpoint="index/bitcoin-lth-sopr")
            if not bitcoin_lth_sopr.empty:
                # Convert timestamp to datetime if time column exists
                if 'timestamp' in bitcoin_lth_sopr.columns:
                    bitcoin_lth_sopr['timestamp'] = pd.to_datetime(bitcoin_lth_sopr['timestamp'], unit='ms')
                    
                    filename = f"Data/bitcoin_sth_sopr_index_history.csv"
                    bitcoin_lth_sopr.to_csv(filename, index=False)
                    print(f"Bitcoin LTH SOPR data saved to {filename}")
                    print(f"Bitcoin LTH SOPR records: {len(bitcoin_lth_sopr)}")
            
            bitcoin_macro_oscillator = client._fetch_index_history(endpoint="index/bitcoin-macro-oscillator")
            if not bitcoin_macro_oscillator.empty:
                # Convert timestamp to datetime if time column exists
                if 'timestamp' in bitcoin_macro_oscillator.columns:
                    bitcoin_macro_oscillator['timestamp'] = pd.to_datetime(bitcoin_macro_oscillator['timestamp'], unit='ms')
                    
                    filename = f"Data/bitcoin_macro_oscillator_index_history.csv"
                    bitcoin_macro_oscillator.to_csv(filename, index=False)
                    print(f"Bitcoin Macro Oscillator data saved to {filename}")
                    print(f"Bitcoin Macro Oscillator records: {len(bitcoin_macro_oscillator)}")
                
        except Exception as e:
            print(f"Error fetching Fear & Greed data: {str(e)}")
            

