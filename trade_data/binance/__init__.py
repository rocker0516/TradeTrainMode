"""
Binance 資料來源（可擴充：K 線、funding rate、open interest 等）。
"""

from trade_data.binance.client import get_client
from trade_data.binance.klines import fetch_futures_klines

__all__ = ["get_client", "fetch_futures_klines"]
