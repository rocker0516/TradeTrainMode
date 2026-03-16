"""
CSV 讀寫：將新資料與既有 CSV 依 key 合併（upsert）後寫回。
"""

import os
from typing import List, Optional

import pandas as pd


def upsert_dataframe_to_csv(
    new_df: pd.DataFrame,
    filename: str,
    key_cols: List[str],
    parse_dates: Optional[List[str]] = None,
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
