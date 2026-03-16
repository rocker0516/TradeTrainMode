"""
單元測試：trade_data.supabase_client.upsert_klines。
使用 mock Supabase client，不連真實 DB。
"""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trade_data.supabase_client import get_client, upsert_klines


def test_upsert_klines_adds_symbol_and_calls_upsert_with_conflict() -> None:
    """傳入的 DataFrame 會被加上 symbol，並以 chunk 呼叫 upsert，on_conflict 為 symbol,timestamp。"""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:05:00"]),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 1100.0],
            "buy_volume": [600.0, 650.0],
            "sell_volume": [400.0, 450.0],
            "volume_ratio": [1.5, 1.44],
            "long_short_ratio": [1.5, 1.44],
            "trades": [100, 110],
            "quote_volume": [100000.0, 111000.0],
        }
    )
    mock_table = MagicMock()
    mock_client = MagicMock()
    mock_client.table.return_value = mock_table

    with patch("trade_data.supabase_client._get_supabase_client", return_value=mock_client):
        result = upsert_klines(df, "BTCUSDT", table="futures_volume")

    assert result == "ok"
    assert mock_table.upsert.called
    # upsert(rows, on_conflict="symbol,timestamp")
    pos = mock_table.upsert.call_args[0]
    kwargs = mock_table.upsert.call_args[1]
    assert kwargs.get("on_conflict") == "symbol,timestamp"
    rows = pos[0] if pos else kwargs.get("rows", [])
    assert isinstance(rows, list) and len(rows) == 2
    assert rows[0].get("symbol") == "BTCUSDT"
    assert "timestamp" in rows[0]


def test_get_client_returns_none_when_env_not_set() -> None:
    """環境變數未設定時 get_client 回傳 None（不拋錯）。"""
    from trade_data.supabase_client import _get_supabase_client

    with patch.dict(
        os.environ,
        {"SUPABASE_URL": "", "SUPABASE_SERVICE_ROLE_KEY": "", "SUPABASE_ANON_KEY": ""},
        clear=False,
    ):
        client = _get_supabase_client()
    assert client is None


def test_upsert_klines_returns_none_when_client_is_none() -> None:
    """當 Supabase client 為 None（環境未設定）時，upsert_klines 回傳 None，不呼叫 Supabase。"""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 00:00:00"]),
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
            "volume": [1000.0],
            "buy_volume": [600.0],
            "sell_volume": [400.0],
            "volume_ratio": [1.5],
            "long_short_ratio": [1.5],
            "trades": [100],
            "quote_volume": [100000.0],
        }
    )
    with patch("trade_data.supabase_client._get_supabase_client", return_value=None):
        result = upsert_klines(df, "BTCUSDT")
    assert result is None


def test_upsert_klines_normalizes_inf_to_none() -> None:
    """遇到 inf/-inf 時要轉成 None，避免 JSON 序列化錯誤。"""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 00:00:00"]),
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
            "volume": [1000.0],
            "buy_volume": [600.0],
            "sell_volume": [0.0],
            "volume_ratio": [float("inf")],
            "long_short_ratio": [float("-inf")],
            "trades": [100],
            "quote_volume": [100000.0],
        }
    )
    mock_table = MagicMock()
    mock_client = MagicMock()
    mock_client.table.return_value = mock_table

    with patch("trade_data.supabase_client._get_supabase_client", return_value=mock_client):
        result = upsert_klines(df, "BTCUSDT", table="futures_volume")

    assert result == "ok"
    rows = mock_table.upsert.call_args[0][0]
    assert rows[0]["volume_ratio"] is None
    assert rows[0]["long_short_ratio"] is None
