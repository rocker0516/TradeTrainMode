"""LiveRefreshRenderer：GATE regime + conviction 序列與 equity 對齊時可正常重繪。"""

from __future__ import annotations

import os

# 必須在 import matplotlib / LiveRefreshRenderer 之前指定，避免互動後端在 CI 失敗
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from LiveTradingRunner.live_render import LiveRefreshRenderer


@pytest.mark.parametrize("n", [3, 8])
def test_refresh_gate_regime_and_conviction_aligned(n: int) -> None:
    """傳入與 OHLC 根數一致的歷史時，refresh 不應拋錯。"""
    sym = "TESTUSDT"
    data_dir = os.path.join(os.path.dirname(__file__), "..", "Data")
    ts0 = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    rows = []
    for i in range(n):
        t = ts0 + pd.Timedelta(minutes=5 * i)
        p = 100.0 + float(i) * 0.1
        rows.append(
            {
                "timestamp": t,
                "open": p,
                "high": p + 0.2,
                "low": p - 0.2,
                "close": p + 0.05,
            }
        )
    df_price = pd.DataFrame(rows)
    eq = [1000.0 + float(i) for i in range(n)]
    pos = [0.0] * n
    tm: list[str | None] = [None] * n
    gl: list[list[str]] = [[] for _ in range(n)]
    gr = [1.0 if i % 3 == 0 else (-1.0 if i % 3 == 1 else 0.0) for i in range(n)]
    cv = [abs(np.tanh(0.1 * float(i))) for i in range(n)]
    tt = [float(np.tanh(0.1 * float(i) * (-1 if i % 2 else 1))) for i in range(n)]
    gb = [1.0 if i % 4 == 0 else 0.0 for i in range(n)]

    r = LiveRefreshRenderer(symbol=sym, max_visible_bars=max(50, n + 10), data_dir=data_dir)
    r.refresh(
        closed_bar_ts=str(df_price["timestamp"].iloc[-1]),
        paper_equity_usdt=float(eq[-1]),
        paper_profit_usdt=float(eq[-1] - 1000.0),
        fee_rate_pct=0.04,
        final_pos_pct=0.0,
        equity_history=eq,
        ts_history=[str(x) for x in df_price["timestamp"].tolist()],
        df_price=df_price,
        position_history=pos,
        trade_marker_history=tm,
        gate_labels_rows_history=gl,
        max_position_pct=0.8,
        gate_regime_history=gr,
        conviction_strength_history=cv,
        trend_tanh_signed_history=tt,
        gate_b_liquidity_history=gb,
    )
