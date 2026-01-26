"""CoinGlass API client (v4).

只負責 HTTP 與資料解析（SRP）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
import requests


class CoinGlassAPIError(Exception):
    """Base exception for CoinGlass API errors."""


class CoinGlassRequestError(CoinGlassAPIError):
    """Network/HTTP/API code errors."""


class CoinGlassDataError(CoinGlassAPIError):
    """Response parsing/data errors."""


@dataclass(frozen=True)
class CoinGlassClientOptions:
    timeout_s: float = 10.0
    rate_limit_sleep_s: float = 0.2


class CoinGlassClient:
    """Low-level client for CoinGlass API."""

    BASE_URL = "https://open-api-v4.coinglass.com/api/"

    def __init__(self, *, api_key: str, options: Optional[CoinGlassClientOptions] = None) -> None:
        if not api_key:
            raise ValueError("CoinGlass api_key is required (use env COINGLASS_API_KEY).")
        self._api_key = api_key
        self._options = options or CoinGlassClientOptions()
        self._session = requests.Session()
        self._session.headers.update({"accept": "application/json", "CG-API-KEY": self._api_key})

    def fetch_json(self, *, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self.BASE_URL + endpoint.lstrip("/")
        try:
            resp = self._session.get(url, params=params, timeout=self._options.timeout_s)
            time.sleep(self._options.rate_limit_sleep_s)

            # basic HTTP errors
            if resp.status_code == 400:
                data = resp.json() if resp.content else {}
                raise CoinGlassRequestError(f"CoinGlass 400 Bad Request: {data}")
            if resp.status_code == 429:
                data = resp.json() if resp.content else {}
                retry_after = resp.headers.get("Retry-After")
                raise CoinGlassRequestError(f"CoinGlass 429 Rate limited (Retry-After={retry_after}): {data}")

            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            raise CoinGlassRequestError(f"CoinGlass request failed: {e}") from e
        except ValueError as e:
            raise CoinGlassDataError(f"CoinGlass invalid JSON: {e}") from e

        # CoinGlass uses `code` in body; common success: "0" or "200"
        code = data.get("code")
        if code is not None and str(code) not in ("0", "200"):
            raise CoinGlassRequestError(f"CoinGlass API code={code} msg={data.get('msg')!r}")
        return data

    def fetch_dataframe(self, *, endpoint: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        data = self.fetch_json(endpoint=endpoint, params=params)
        if "data" not in data or data["data"] is None:
            return pd.DataFrame()
        return pd.DataFrame(data["data"])

    def fetch_fear_greed_history(self) -> pd.DataFrame:
        """index/fear-greed-history -> columns: time, fear_greed_index, price (time is ms)."""
        data = self.fetch_json(endpoint="index/fear-greed-history", params=None)
        raw = data.get("data")
        if raw is None:
            return pd.DataFrame()

        # API 可能回傳 dict 或 [dict]
        if isinstance(raw, list):
            raw_obj = raw[0] if raw else {}
        else:
            raw_obj = raw

        if not isinstance(raw_obj, dict):
            raise CoinGlassDataError(f"Unexpected fear-greed data type: {type(raw_obj)}")

        for k in ("data_list", "price_list", "time_list"):
            if k not in raw_obj:
                raise CoinGlassDataError(f"Missing key {k!r} in fear-greed response")

        data_list = raw_obj["data_list"]
        price_list = raw_obj["price_list"]
        time_list = raw_obj["time_list"]
        if not (len(data_list) == len(price_list) == len(time_list)):
            raise CoinGlassDataError("fear-greed list length mismatch")

        return pd.DataFrame({"time": time_list, "fear_greed_index": data_list, "price": price_list})


