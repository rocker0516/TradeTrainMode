"""
Supabase 連線與寫入：upsert_klines 等，依環境變數決定是否可用。
"""

import os
import math
from typing import Any, Optional, Tuple

import pandas as pd


def _build_supabase_client() -> Tuple[Optional[Any], Optional[str]]:
    """
    建立 Supabase client，並在失敗時回傳可讀原因。

    Returns:
        (client, error_message)
    """
    try:
        from supabase import create_client
    except ImportError:
        return None, "未安裝 supabase 套件（請先 pip install supabase）"

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get(
        "SUPABASE_ANON_KEY"
    )
    if not url:
        return None, "缺少 SUPABASE_URL"
    if not key:
        return None, "缺少 SUPABASE_SERVICE_ROLE_KEY 或 SUPABASE_ANON_KEY"

    try:
        return create_client(url, key), None
    except Exception as exc:
        return None, f"create_client 失敗: {exc}"


def _get_supabase_client() -> Optional[Any]:
    """
    取得 Supabase client，失敗時回傳 None。
    """
    client, _ = _build_supabase_client()
    return client


def get_client() -> Optional[Any]:
    """
    從環境變數建立 Supabase client。
    需要 SUPABASE_URL 與 SUPABASE_SERVICE_ROLE_KEY（或 SUPABASE_ANON_KEY）。
    未設定或未安裝 supabase 時回傳 None，不拋錯。
    """
    return _get_supabase_client()


def verify_supabase(table: str = "futures_volume") -> bool:
    """
    驗證 Supabase 連線與表存取：檢查環境變數、連線及對指定表的讀取權限。

    Args:
        table: 要驗證的表名，預設 futures_volume。

    Returns:
        驗證成功 True，否則 False。
    """
    client, error = _build_supabase_client()
    if client is None:
        print("Supabase 驗證失敗：無法建立 client。")
        if error:
            print(f"  原因: {error}")
        return False
    try:
        # 讀取一筆以確認連線與表存在、有權限
        client.table(table).select("symbol,timestamp").limit(1).execute()
        print(f"Supabase 驗證成功：已連線，表 '{table}' 可存取。")
        return True
    except Exception as e:
        print(f"Supabase 驗證失敗：{e}")
        return False


# 單次 upsert 建議筆數，避免 payload 過大
_UPSERT_CHUNK_SIZE = 1000


def _normalize_json_value(value: Any) -> Any:
    """
    將資料清洗為 JSON 可序列化的值：
    - NaN/NaT -> None
    - inf/-inf -> None
    """
    if pd.isna(value):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def upsert_klines(
    df: pd.DataFrame,
    symbol: str,
    table: str = "futures_volume",
) -> Optional[str]:
    """
    將 K 線 DataFrame 以 upsert 寫入 Supabase 表。
    會加上 symbol、將 timestamp 轉成 ISO 字串，並分 chunk 寫入。

    Args:
        df: K 線 DataFrame（欄位與 futures_volume 表一致，不含 symbol）。
        symbol: 交易對，如 BTCUSDT。
        table: 目標表名，預設 futures_volume。

    Returns:
        成功時回傳 "ok" 或筆數說明；環境未設定或失敗時回傳 None。
    """
    client = _get_supabase_client()
    if client is None:
        return None

    df = df.copy()
    df["symbol"] = symbol
    # 供 Supabase timestamptz 使用：ISO 字串
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).astype(str)
    # trades 對應 bigint
    if "trades" in df.columns:
        df["trades"] = df["trades"].astype("Int64")

    columns = [
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "buy_volume",
        "sell_volume",
        "volume_ratio",
        "long_short_ratio",
        "trades",
        "quote_volume",
    ]
    df = df[columns]

    try:
        total = 0
        for start in range(0, len(df), _UPSERT_CHUNK_SIZE):
            chunk = df.iloc[start : start + _UPSERT_CHUNK_SIZE]
            rows = chunk.to_dict(orient="records")
            # 將 NaN / inf / -inf 轉成 None，避免 JSON 序列化錯誤
            for row in rows:
                for k, v in row.items():
                    row[k] = _normalize_json_value(v)
            client.table(table).upsert(
                rows,
                on_conflict="symbol,timestamp",
            ).execute()
            total += len(rows)
        return "ok" if total else None
    except Exception as e:
        print(f"Supabase upsert error: {e}")
        return None
