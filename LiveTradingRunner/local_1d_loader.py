"""Load local 1d CSVs from Data/ for live runner.

與 `Env/load_file.py` 的 1d 合併規則對齊：
- 讀取 `Data/*_1d.csv`
- 推導 prefix（symbol 或 macro 名稱）
- 欄位 rename 成 `{prefix}_{col}`（timestamp 不加 prefix）
- 以 timestamp 做 outer join（避免單一來源缺時間導致整張表變空）
"""

from __future__ import annotations

import glob
import os
from typing import List

import pandas as pd


def _infer_prefix(basename: str) -> str:
    stem, _ext = os.path.splitext(str(basename))
    head = stem.split("_")[0]
    if "USDT" in head:
        return head
    if stem.endswith("_index_history_1d"):
        return stem.replace("_index_history_1d", "")
    return stem


def load_local_1d_data(*, data_dir: str | None = None) -> pd.DataFrame:
    """載入 Data 目錄下所有 *_1d.csv，合併成 df_1d。"""
    if data_dir is None:
        # 以專案結構為基準（避免使用者從任意 cwd 執行時找不到 Data）
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))
    files_1d: List[str] = sorted(glob.glob(os.path.join(str(data_dir), "*_1d.csv")))
    if not files_1d:
        raise ValueError(f"No *_1d.csv files found in {data_dir!r}")

    df_all = pd.DataFrame()
    for f in files_1d:
        df = pd.read_csv(f)
        prefix = _infer_prefix(os.path.basename(f))

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        elif "time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["time"])
        else:
            raise ValueError(f"CSV missing timestamp/time: {os.path.basename(f)}")

        df = df.rename(columns=lambda c: f"{prefix}_{c}" if c != "timestamp" else c)
        if df_all.empty:
            df_all = df
        else:
            df_all = pd.merge(df_all, df, on="timestamp", how="outer")

    if df_all.empty:
        raise ValueError("Failed to load any 1d data (merged empty).")

    df_all = (
        df_all.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )
    return df_all


