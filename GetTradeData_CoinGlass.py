import os
import csv
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Protocol, Iterator
import requests

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

@dataclass
class OHLCVData:
    """
    Data model representing a single candle (OHLCV).
    """
    timestamp: int  # Milliseconds
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume_usd: float

    def to_dict(self) -> Dict[str, Any]:
        """Converts the data to a dictionary."""
        return {
            "time": self.timestamp,
            "open": self.open_price,
            "high": self.high_price,
            "low": self.low_price,
            "close": self.close_price,
            "volume_usd": self.volume_usd
        }

    @classmethod
    def from_api_dict(cls, data: Dict[str, Any]) -> 'OHLCVData':
        """Creates an instance from API response dictionary."""
        try:
            return cls(
                timestamp=int(data['t']),
                open_price=float(data['o']),
                high_price=float(data['h']),
                low_price=float(data['l']),
                close_price=float(data['c']),
                volume_usd=float(data['v'])
            )
        except KeyError as e:
            # Fallback for alternative field names if API differs
            try:
                return cls(
                    timestamp=int(data['time']),
                    open_price=float(data['open']),
                    high_price=float(data['high']),
                    low_price=float(data['low']),
                    close_price=float(data['close']),
                    volume_usd=float(data['volume_usd'])
                )
            except KeyError as inner_e:
                raise CoinGlassDataError(f"Missing field in API response: {inner_e}") from e
        except (ValueError, TypeError) as e:
            raise CoinGlassDataError(f"Invalid data type in API response: {e}") from e

class IMarketDataProvider(ABC):
    """
    Abstract Base Class for market data providers (DIP).
    """
    @abstractmethod
    def fetch_history(self, exchange: str, symbol: str, interval: str, 
                      start_time: int, end_time: int) -> Iterator[OHLCVData]:
        """
        Fetches historical OHLCV data.

        Args:
            exchange: The exchange name (e.g., 'Binance').
            symbol: The trading pair symbol (e.g., 'BTCUSDT').
            interval: The time interval (e.g., '5m').
            start_time: Start timestamp in milliseconds.
            end_time: End timestamp in milliseconds.

        Returns:
            Iterator of OHLCVData objects.
        """
        pass

class CoinGlassClient:
    """
    Low-level client for CoinGlass API (SRP: Handle HTTP requests).
    """
    BASE_URL = "https://open-api-v4.coinglass.com/api/futures/price/history"

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

    def get_price_history(self, exchange: str, symbol: str, interval: str, limit: int, 
                          start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[Dict[str, Any]]:
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
            List of OHLCV data dictionaries.
            
        Raises:
            CoinGlassRequestError: If HTTP request fails or API returns error code.
            CoinGlassDataError: If response data cannot be parsed.
        """
        params = {
            "exchange": exchange,
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        try:
            logger.debug(f"Requesting: {self.BASE_URL} with params: {params}")
            response = self._session.get(self.BASE_URL, params=params, timeout=10)
            
            # Check HTTP status code
            if response.status_code == 400:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get("msg", "Bad Request (400)")
                raise CoinGlassRequestError(
                    f"API returned 400 Bad Request: {error_msg}. "
                    f"Please check your API key and parameters. Response: {error_data}"
                )
            
            # Handle rate limit (429 Too Many Requests)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "60")
                error_data = response.json() if response.content else {}
                error_msg = error_data.get("msg", "Rate limit exceeded")
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
                raise CoinGlassRequestError(
                    f"API returned error code {api_code}: {error_msg}. "
                    f"Full response: {data}"
                )
            
            if "data" not in data:
                logger.warning(f"No 'data' field in API response: {data}")
                return []
            
            return data["data"], response.headers
            
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_data = e.response.json() if e.response.content else {}
                error_detail = f" Response: {error_data}"
            except:
                error_detail = f" Response text: {e.response.text[:200]}"
            raise CoinGlassRequestError(f"HTTP error during API call: {e}{error_detail}") from e
        except requests.exceptions.RequestException as e:
            raise CoinGlassRequestError(f"Network error during API call: {e}") from e
        except ValueError as e:
            raise CoinGlassDataError(f"Invalid JSON response: {e}") from e

class CoinGlassHistoryFetcher(IMarketDataProvider):
    """
    High-level fetcher for CoinGlass data (SRP: Orchestrate fetching logic).
    """
    def __init__(self, client: CoinGlassClient):
        self._client = client

    def fetch_history(self, exchange: str, symbol: str, interval: str, 
                      start_time: int, end_time: int) -> Iterator[OHLCVData]:
        """
        Fetches historical data handling pagination with loop-based batching.
        
        This method automatically handles pagination by making multiple API calls
        in a loop until all data within the time range is fetched.
        Similar to GetTradeData.py's fetch_futures_data loop structure.
        """
        current_start = start_time
        limit = 1000  # API maximum limit per request
        
        # Rate limit tracking (similar to GetTradeData.py)
        request_count = 0
        last_request_time = time.time()
        max_requests_per_minute = 60  # Conservative default, adjust based on API limits
        
        logger.info(
            f"Starting fetch for {symbol} from {datetime.fromtimestamp(start_time/1000)} "
            f"to {datetime.fromtimestamp(end_time/1000)}"
        )

        while current_start < end_time:
            try:
                # Check rate limit (similar to GetTradeData.py)
                current_time = time.time()
                if request_count >= max_requests_per_minute:
                    wait_time = 60 - (current_time - last_request_time)
                    if wait_time > 0:
                        logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                    request_count = 0
                    last_request_time = time.time()
                
                # Fetch batch
                raw_data, response_headers = self._client.get_price_history(
                    exchange=exchange,
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    start_time=current_start,
                    end_time=end_time
                )
                
                request_count += 1
                
                if not raw_data:
                    logger.info("No more data returned from API. Fetch complete.")
                    break
                
                # Yield all records in this batch
                for item in raw_data:
                    ohlcv = OHLCVData.from_api_dict(item)
                    if ohlcv.timestamp > end_time:
                        continue
                    yield ohlcv
                
                # Update start time for next batch using last record's timestamp
                # Similar to GetTradeData.py: current_start_ts = klines[-1][0] + 1
                last_time = raw_data[-1].get('time') or raw_data[-1].get('t')
                if not last_time:
                    logger.warning("Could not determine last timestamp, stopping.")
                    break
                
                current_start = int(last_time) + 1  # Avoid duplicate by adding 1ms
                
                # Display progress (similar to GetTradeData.py)
                current_dt = datetime.fromtimestamp(current_start / 1000)
                print(f"\rProcessed data up to {current_dt.strftime('%Y-%m-%d %H:%M:%S')}", end="")
                
                # Small delay between requests (similar to GetTradeData.py)
                time.sleep(0.2)
                
            except CoinGlassRequestError as e:
                error_msg = str(e)
                # Handle rate limit errors (similar to GetTradeData.py)
                if "429" in error_msg or "Rate Limit" in error_msg or "rate limit" in error_msg.lower() or "Too many requests" in error_msg:
                    logger.warning("Rate limit reached. Waiting 60 seconds...")
                    time.sleep(60)
                    continue
                logger.error(f"Error fetching data: {e}")
                break
                
            except CoinGlassAPIError as e:
                logger.error(f"Error fetching data: {e}")
                break

class IDataStorage(ABC):
    """
    Abstract Base Class for data storage (DIP).
    """
    @abstractmethod
    def save(self, data: Iterator[OHLCVData], destination: str) -> None:
        """Saves data to the destination."""
        pass

class CSVDataStorage(IDataStorage):
    """
    Saves data to CSV (SRP).
    """
    def save(self, data: Iterator[OHLCVData], destination: str) -> None:
        """
        Saves the iterator of OHLCVData to a CSV file.
        """
        try:
            # We need to consume the iterator. 
            # Note: For very large datasets, we might want to write in chunks.
            # Here we will write row by row.
            
            file_exists = os.path.isfile(destination)
            
            with open(destination, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["time", "open", "high", "low", "close", "volume_usd"])
                writer.writeheader()
                
                count = 0
                for item in data:
                    writer.writerow(item.to_dict())
                    count += 1
                    
            logger.info(f"Successfully saved {count} records to {destination}")
            
        except IOError as e:
            raise IOError(f"Failed to write to CSV file {destination}: {e}") from e

def run_coinglass_fetch(api_key: str, exchange: str, symbol: str, interval: str, output_file: str):
    """
    Main function to orchestrate the process.
    """
    # 1. Setup Dependency Injection
    client = CoinGlassClient(api_key=api_key)
    fetcher = CoinGlassHistoryFetcher(client=client)
    storage = CSVDataStorage()

    # 2. Define Time Range (1 year)
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=365 * 5)
    
    end_ts = int(end_dt.timestamp() * 1000)
    start_ts = int(start_dt.timestamp() * 1000)

    # 3. Fetch and Save
    try:
        data_iterator = fetcher.fetch_history(
            exchange=exchange,
            symbol=symbol,
            interval=interval,
            start_time=start_ts,
            end_time=end_ts
        )
        
        storage.save(data_iterator, output_file)
        print(f"Data saved to {output_file}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"Failed to fetch data: {e}")

if __name__ == "__main__":
    # Configuration via parameters/env
    API_KEY = os.getenv("COINGLASS_API_KEY", "1e41abd6360a4d1486b770e83982e33a")
    EXCHANGE = "Binance"
    SYMBOL = "BTCUSDT"
    INTERVAL = "5m"
    OUTPUT_FILE = "Data/BTCUSDT_futures_volume_5years_5min.csv"

    if API_KEY == "YOUR_API_KEY_HERE":
        print("Please set COINGLASS_API_KEY environment variable or edit the script.")
    else:
        run_coinglass_fetch(API_KEY, EXCHANGE, SYMBOL, INTERVAL, OUTPUT_FILE)

