"""
共用 Binance Client，從環境變數讀取 API 金鑰，供各 fetcher（klines、未來 funding rate 等）使用。
"""

import os
from typing import Optional

from binance.client import Client


def get_client() -> Client:
    """
    從環境變數建立 Binance Client。
    使用 BINANCE_API_KEY、BINANCE_API_SECRET；未設定時拋錯。

    Returns:
        Binance Client 實例。

    Raises:
        ValueError: 當環境變數未設定時。
    """
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise ValueError(
            "BINANCE_API_KEY and BINANCE_API_SECRET must be set in environment"
        )
    return Client(api_key, api_secret)
