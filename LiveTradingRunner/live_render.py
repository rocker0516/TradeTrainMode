"""Simple refresh renderer for live loop.

單一視窗刷新顯示（三子圖）：
- 上：session/API 同源 OHLC（蠟燭優先；無資料時可 fallback 本機 CSV）+ B/S + GATE A/C 變化標籤
- 中：倉位曲線 ``final_pos_pct``
- 下：紙上資金曲線
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class LiveRefreshRenderer:
    """Single-window renderer refreshed on each new bar."""

    symbol: str
    max_visible_bars: int
    data_dir: str

    def __post_init__(self) -> None:
        self._max_visible = int(max(10, self.max_visible_bars))
        self._csv_path = os.path.join(self.data_dir, f"{self.symbol}_futures_volume_5years_5min.csv")
        plt.ion()
        self._fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(13, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1, 1]},
        )
        self._ax_price, self._ax_pos, self._ax_equity = axes
        self._fig.tight_layout(pad=1.2)

    def _load_recent_klines_csv_fallback(self) -> pd.DataFrame:
        """當 session OHLC 緩衝為空時，以本機 CSV 作 offline fallback。"""
        if not os.path.exists(self._csv_path):
            return pd.DataFrame()
        df = pd.read_csv(self._csv_path)
        if df.empty or "timestamp" not in df.columns:
            return pd.DataFrame()
        needed = ["timestamp", "open", "high", "low", "close"]
        for col in needed:
            if col not in df.columns:
                return pd.DataFrame()
        out = df[needed].copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
        tail_n = int(max(self._max_visible, 288))
        return out.tail(tail_n)

    @staticmethod
    def _bar_scale_fontsize(_ax: plt.Axes, _fig: plt.Figure, n_bars: int) -> float:
        """依可視 K 根數估算字級（避免依賴尚未完成的 canvas renderer）。"""
        _ = _ax
        _ = _fig
        return float(np.clip(900.0 / max(n_bars, 1), 6.0, 16.0))

    def _overlay_bs_and_gate_history(
        self,
        df_price: pd.DataFrame,
        trade_marker_history: Sequence[Optional[str]],
        gate_labels_rows_history: Sequence[Sequence[str]],
        take: int,
    ) -> None:
        """在與決策歷史對齊的每一根可視 K 上疊加 B/S 與 GATE A/C（標籤會一直留在圖上直到滑出視窗）。"""
        if df_price.empty or take <= 0:
            return
        n = len(df_price)
        h = len(trade_marker_history)
        hg = len(gate_labels_rows_history)
        if h < take or hg < take:
            return
        n_bars = max(n, 1)
        fs = self._bar_scale_fontsize(self._ax_price, self._fig, n_bars)

        for k in range(take):
            row = df_price.iloc[n - take + k]
            hi = h - take + k
            m = trade_marker_history[hi]
            gls: List[str] = list(gate_labels_rows_history[hi]) if hi < hg else []

            ts = pd.Timestamp(row["timestamp"])
            lx = float(mdates.date2num(ts))
            bar_low = float(row["low"])
            bar_high = float(row["high"])
            rng = max(bar_high - bar_low, abs(bar_high) * 1e-6)
            y_step = 0.12 * rng

            for i, gtext in enumerate(gls[:6]):
                y = bar_high + y_step * (1.4 + float(i) * 0.85)
                self._ax_price.text(
                    lx,
                    y,
                    str(gtext),
                    ha="center",
                    va="bottom",
                    fontsize=max(fs * 0.85, 5.0),
                    color="darkviolet",
                )

            if m in ("B", "S"):
                color = "green" if m == "B" else "red"
                y_mark = bar_low - 0.2 * rng
                self._ax_price.text(
                    lx,
                    y_mark,
                    str(m),
                    ha="center",
                    va="top",
                    fontsize=max(fs * 1.1, 7.0),
                    fontweight="bold",
                    color=color,
                )

    def _draw_price(
        self,
        df_price: pd.DataFrame,
        trade_marker_history: Sequence[Optional[str]],
        gate_labels_rows_history: Sequence[Sequence[str]],
        take: int,
    ) -> None:
        self._ax_price.clear()
        if df_price.empty:
            self._ax_price.set_title(f"{self.symbol} - waiting for 5m data")
            return
        try:
            import mplfinance as mpf

            plot_df = df_price.set_index("timestamp").rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                }
            )
            mpf.plot(
                plot_df,
                type="candle",
                ax=self._ax_price,
                volume=False,
                style="charles",
                show_nontrading=True,
                warn_too_much_data=max(2000, int(len(plot_df) + 10)),
            )
        except Exception:
            self._ax_price.plot(df_price["timestamp"], df_price["close"], color="tab:blue", linewidth=1.0)
        self._ax_price.set_ylabel("Price")
        self._ax_price.xaxis_date()
        self._overlay_bs_and_gate_history(
            df_price,
            trade_marker_history,
            gate_labels_rows_history,
            take,
        )

    def _apply_datetime_xaxis(self) -> None:
        """三圖共用實際日期時間刻度（與資料 timestamp 一致，通常為交易所 UTC）。"""
        for ax in (self._ax_price, self._ax_pos, self._ax_equity):
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=16))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

    def refresh(
        self,
        *,
        closed_bar_ts: str,
        paper_equity_usdt: float,
        paper_profit_usdt: float,
        fee_rate_pct: float,
        final_pos_pct: float,
        equity_history: Sequence[float],
        ts_history: Sequence[str],
        df_price: pd.DataFrame,
        position_history: Sequence[float],
        trade_marker_history: Sequence[Optional[str]],
        gate_labels_rows_history: Sequence[Sequence[str]],
        max_position_pct: float,
    ) -> None:
        """Refresh one window with latest chart and PnL state.

        Args:
            closed_bar_ts: 本輪決策對應之已收盤 bar 時間（標題用）。
            paper_equity_usdt: 紙上權益。
            paper_profit_usdt: 紙上損益（相對初始資金）。
            fee_rate_pct: 手續費率（百分比）。
            final_pos_pct: 本步 effective 倉位比例。
            equity_history: 紙上權益序列（與決策步對齊）。
            ts_history: 每步 bar 時間字串（保留供未來擴充）。
            df_price: session OHLC（timestamp/open/high/low/close）；可為空表以觸發 CSV fallback。
            position_history: 每步 ``final_pos_pct``。
            trade_marker_history: 每步 ``B`` / ``S`` / ``None``，與 equity 等長。
            gate_labels_rows_history: 每步 GATE A/C 變化字串列表（可空），與 equity 等長。
            max_position_pct: 中圖 Y 軸對稱範圍參考。
        """
        _ = ts_history
        df_use = df_price.copy()
        if df_use.empty:
            df_use = self._load_recent_klines_csv_fallback()
        if not df_use.empty:
            df_use = df_use.sort_values("timestamp").tail(self._max_visible)

        n_price = len(df_use)
        h_eq = len(equity_history)
        h_m = len(trade_marker_history)
        h_g = len(gate_labels_rows_history)
        h_align = int(min(h_eq, h_m, h_g))
        take = int(min(h_align, n_price)) if (n_price > 0 and h_align > 0) else 0

        self._draw_price(df_use, trade_marker_history, gate_labels_rows_history, take)

        self._ax_pos.clear()

        if take > 0:
            ts_tail = pd.to_datetime(df_use["timestamp"].iloc[-take:])
            xp = mdates.date2num(ts_tail)
            yp = np.asarray(list(position_history)[-take:], dtype=float)
            self._ax_pos.plot(xp, yp, color="tab:orange", linewidth=1.2)
        m = float(max(0.01, abs(max_position_pct)))
        self._ax_pos.set_ylim(-1.05 * m, 1.05 * m)
        self._ax_pos.set_ylabel("Pos %")

        self._ax_equity.clear()
        if take > 0:
            xe = xp
            ye = np.asarray(list(equity_history)[-take:], dtype=float)
            self._ax_equity.plot(xe, ye, color="tab:green", linewidth=1.2)
        self._ax_equity.set_ylabel("Equity")
        self._ax_equity.set_xlabel("時間（YYYY-MM-DD HH:MM，資料 timestamp）")

        title = (
            f"{self.symbol} | bar={closed_bar_ts} | equity={paper_equity_usdt:.2f} "
            f"| profit={paper_profit_usdt:.2f} | fee={fee_rate_pct:.4f}% | pos={final_pos_pct:.3f}"
        )
        self._fig.suptitle(title, fontsize=11)
        self._apply_datetime_xaxis()
        self._fig.autofmt_xdate(rotation=18)
        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    def idle(self) -> None:
        """Keep window responsive without full redraw."""
        plt.pause(0.001)
