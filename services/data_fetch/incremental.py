"""
Utilities for incremental range resolution.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd


def interval_to_timedelta(interval: str, default_step: timedelta) -> timedelta:
    """Convert interval token to timedelta."""
    mapping = {
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "1d": timedelta(days=1),
    }
    return mapping.get(interval, default_step)


def resolve_start_time(
    filename: str,
    time_col: str,
    fallback_start_time: datetime,
    step: timedelta,
) -> datetime:
    """
    Resolve incremental start timestamp from existing CSV.
    """
    if not os.path.exists(filename):
        return fallback_start_time
    try:
        existing = pd.read_csv(filename, usecols=[time_col])
        if existing.empty:
            return fallback_start_time
        existing[time_col] = pd.to_datetime(existing[time_col], errors="coerce")
        max_ts = existing[time_col].max()
        if pd.isna(max_ts):
            return fallback_start_time
        next_start = max_ts.to_pydatetime() + step
        return max(fallback_start_time, next_start)
    except Exception:
        return fallback_start_time

