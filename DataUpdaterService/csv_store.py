"""CSV persistence helpers.

重點：
- 提供「找最後 timestamp」功能，讓 updater 做增量抓取
- 提供原子寫入：先寫 temp，再 os.replace，避免 service 中途掛掉留下半套 CSV
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd


class CsvStoreError(RuntimeError):
    """CSV 讀寫錯誤。"""


def read_last_timestamp(
    file_path: Path,
    *,
    time_col: str,
) -> Optional[pd.Timestamp]:
    """讀取 CSV 內指定時間欄位的最大值。

    Args:
        file_path: CSV path
        time_col: 欄位名稱（例：'timestamp' 或 'time'）

    Returns:
        最大 timestamp（若檔案不存在或解析不到，回傳 None）
    """
    if not file_path.exists():
        return None
    try:
        # 只讀時間欄位，避免每次 update 都把整個大 CSV 讀進來
        df = pd.read_csv(str(file_path), usecols=[time_col])
    except ValueError:
        # 缺欄位：視為無法判斷最後時間，讓上層改用 backfill window
        return None
    except (OSError, pd.errors.EmptyDataError) as e:
        raise CsvStoreError(f"Failed to read CSV: {file_path}") from e

    ts = pd.to_datetime(df[time_col], errors="coerce").max()
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def upsert_dataframe_to_csv_atomic(
    new_df: pd.DataFrame,
    *,
    filename: Path,
    key_cols: List[str],
    parse_dates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """將新資料與既有 CSV 合併（重複 key 以新資料覆蓋），並以原子方式寫回。

    注意：此函式會讀取既有 CSV 全內容，因此對極大檔案可能較慢；
    但可避免 service 寫到一半留下破檔。
    """
    filename.parent.mkdir(parents=True, exist_ok=True)

    # 重要：若 existing_df 會 parse_dates，new_df 也要同步轉型，避免 sort/merge 時混用 str/Timestamp
    incoming = new_df.copy()
    if parse_dates:
        for c in parse_dates:
            if c in incoming.columns:
                incoming[c] = pd.to_datetime(incoming[c], errors="coerce")
    if filename.exists():
        existing_df = pd.read_csv(str(filename), parse_dates=parse_dates)
        combined = pd.concat([existing_df, incoming], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
    else:
        combined = incoming.copy()

    if key_cols:
        combined = combined.sort_values(key_cols).reset_index(drop=True)

    tmp_path = filename.with_suffix(filename.suffix + ".tmp")
    combined.to_csv(str(tmp_path), index=False)
    os.replace(str(tmp_path), str(filename))
    return combined


