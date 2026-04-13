"""LiveRefreshRenderer：不連續 timestamp 時 x 軸應錨定最末段，避免全域時間跨距過大。"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import pandas as pd

from LiveTradingRunner.live_render import LiveRefreshRenderer


def test_shared_xlim_uses_trailing_window_not_global_span() -> None:
    """中間有大空段的 OHLC 仍應以尾端約 n×5m 寬度設 xlim。"""
    sym = "TESTUSDT"
    data_dir = os.path.join(os.path.dirname(__file__), "..", "Data")
    r = LiveRefreshRenderer(symbol=sym, max_visible_bars=50, data_dir=data_dir)
    t_old = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    t_mid = pd.Timestamp("2024-06-01 00:00:00", tz="UTC")
    t_new = t_mid + pd.Timedelta(minutes=5)
    rows = [
        {"timestamp": t_old, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05},
        {"timestamp": t_mid, "open": 2.0, "high": 2.1, "low": 1.9, "close": 2.05},
        {"timestamp": t_new, "open": 2.1, "high": 2.2, "low": 2.0, "close": 2.15},
    ]
    df = pd.DataFrame(rows)
    r._apply_shared_xlim_from_ohlc(df)
    lo, hi = r._ax_price.get_xlim()
    span_days = float(hi - lo)
    # 3 根 K → 錨定寬度約 2×5m + 邊距，遠小於資料列首末跨越多個月
    assert span_days < 2.0 / 24.0
    wide_span = float(mdates.date2num(t_new) - mdates.date2num(t_old))
    assert wide_span > span_days * 10.0
