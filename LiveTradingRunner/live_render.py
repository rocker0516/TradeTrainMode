"""Simple refresh renderer for live loop.

單一視窗刷新顯示（四子圖）：
- 上：OHLC + B/S + GATE A/C 變化標籤（僅在變化時顯示文字）
- 次：日線 A/C regime（階梯、跨日才變）+ reward 用 |tanh(scale×trend)| + 可選 5m signed tanh、Gate B 流動性帶
- 中：倉位曲線 ``final_pos_pct``
- 下：紙上資金曲線
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

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
    _ax_conv_r: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._max_visible = int(max(10, self.max_visible_bars))
        self._csv_path = os.path.join(self.data_dir, f"{self.symbol}_futures_volume_5years_5min.csv")
        plt.ion()
        # 不在此處 tight_layout：refresh 會動態加 twinx / legend / suptitle，初次 layout 會與後續衝突而跑版。
        # 勿用 sharex：mplfinance 在部分設定下會用「棒號」座標，與 mdates.date2num 混用會整張圖跑版。
        self._fig, axes = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(13, 11),
            sharex=False,
            gridspec_kw={"height_ratios": [3, 1, 1, 1]},
        )
        self._ax_price, self._ax_gate_conv, self._ax_pos, self._ax_equity = axes

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

    def _draw_price_matplotlib_ohlc(self, df_price: pd.DataFrame) -> None:
        """
        以 matplotlib 手動畫 K 線，x 一律為 mdates.date2num(timestamp)。

        避免 mplfinance 在 show_nontrading=False 等情況使用「棒號」座標，與下方子圖的
        date2num 不一致而導致 sharex 時整張圖壓縮到左側。
        """
        ts_raw = pd.to_datetime(df_price["timestamp"], errors="coerce")
        df = df_price.loc[ts_raw.notna()].copy().reset_index(drop=True)
        if df.empty:
            return
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if not bool(ts.notna().all()):
            return
        x = np.asarray(mdates.date2num(ts), dtype=float)
        o = df["open"].astype(float).to_numpy()
        h = df["high"].astype(float).to_numpy()
        l = df["low"].astype(float).to_numpy()
        c = df["close"].astype(float).to_numpy()
        n = int(len(x))
        if n >= 2:
            width = float(np.median(np.diff(np.sort(x)))) * 0.68
        else:
            width = 3.0 / (24.0 * 60.0)
        width = max(width, 1e-9)
        for i in range(n):
            self._ax_price.plot(
                [x[i], x[i]],
                [l[i], h[i]],
                color="black",
                linewidth=0.85,
                solid_capstyle="round",
                zorder=1,
            )
            up = bool(c[i] >= o[i])
            col = "#26a69a" if up else "#ef5350"
            body_bot = float(min(o[i], c[i]))
            body_h = max(abs(float(c[i] - o[i])), max(float(h[i] - l[i]), 1e-12) * 1e-5)
            self._ax_price.bar(
                x[i],
                body_h,
                width=width,
                bottom=body_bot,
                color=col,
                edgecolor=col,
                linewidth=0.35,
                align="center",
                zorder=2,
            )
        self._ax_price.set_ylabel("Price")
        self._ax_price.xaxis_date()

    def _draw_price(
        self,
        df_price: pd.DataFrame,
        trade_marker_history: Sequence[Optional[str]],
        gate_labels_rows_history: Sequence[Sequence[str]],
        take: int,
        sideway_mask_tail: Optional[np.ndarray] = None,
    ) -> None:
        self._ax_price.clear()
        if df_price.empty:
            self._ax_price.set_title(f"{self.symbol} - waiting for 5m data")
            return
        self._draw_price_matplotlib_ohlc(df_price)
        if sideway_mask_tail is not None and take > 0 and len(df_price) >= take:
            ts_tail = pd.to_datetime(df_price["timestamp"].iloc[-take:])
            side = np.asarray(sideway_mask_tail, dtype=bool).reshape(-1)
            if side.size == take:
                high_tail = np.asarray(df_price["high"].iloc[-take:], dtype=float)
                low_tail = np.asarray(df_price["low"].iloc[-take:], dtype=float)
                y_min = float(np.nanmin(low_tail))
                y_max = float(np.nanmax(high_tail))
                if np.isfinite(y_min) and np.isfinite(y_max) and (y_max > y_min):
                    self._ax_price.fill_between(
                        ts_tail,
                        y_min,
                        y_max,
                        where=side,
                        step="post",
                        color="khaki",
                        alpha=0.16,
                        label="Sideway region",
                    )
                    self._ax_price.legend(loc="upper left", fontsize=7)
        self._overlay_bs_and_gate_history(
            df_price,
            trade_marker_history,
            gate_labels_rows_history,
            take,
        )

    def _draw_gate_and_conviction(
        self,
        xp: np.ndarray,
        regime_tail: np.ndarray,
        strength_tail: np.ndarray,
        trend_signed_tail: Optional[np.ndarray] = None,
        gate_b_tail: Optional[np.ndarray] = None,
    ) -> None:
        """日線 A/C regime（左）+ |tanh|（右）；可選 5m signed tanh、Gate B（每根 5m 更新）。"""
        self._ax_gate_conv.clear()
        if self._ax_conv_r is not None:
            try:
                self._ax_conv_r.remove()
            except Exception:
                pass
            self._ax_conv_r = None
        if xp.size == 0:
            self._ax_gate_conv.set_ylabel("GATE / conv")
            return
        if gate_b_tail is not None and len(gate_b_tail) == len(xp):
            gb = np.clip(np.asarray(gate_b_tail, dtype=float), 0.0, 1.0)
            self._ax_gate_conv.fill_between(
                xp,
                -1.14,
                -1.14 + 0.13 * gb,
                step="post",
                color="seagreen",
                alpha=0.32,
                label="Gate B liq (5m)",
            )
        self._ax_gate_conv.step(
            xp,
            regime_tail,
            where="post",
            color="navy",
            linewidth=1.4,
            label="1d A/C regime (+1=A / -1=C / 0=N)",
        )
        if trend_signed_tail is not None and len(trend_signed_tail) == len(xp):
            self._ax_gate_conv.plot(
                xp,
                np.asarray(trend_signed_tail, dtype=float),
                color="darkcyan",
                linestyle=":",
                linewidth=1.15,
                alpha=0.9,
                label="tanh(scale×trend) 5m signed",
            )
        self._ax_gate_conv.axhline(0.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
        self._ax_gate_conv.set_ylim(-1.15, 1.15)
        self._ax_gate_conv.set_ylabel("Regime / 5m trend")
        # 標題過長會佔滿子圖上方、擠壓曲線；改短標題 + 圖例放邊外避免與資料重疊
        self._ax_gate_conv.set_title(
            "1d A/C (step) | Gate B band | 5m tanh (dotted) | R: |tanh| strength",
            fontsize=8,
        )
        self._ax_gate_conv.legend(loc="upper left", fontsize=6, framealpha=0.92)

        ax_r = self._ax_gate_conv.twinx()
        self._ax_conv_r = ax_r
        ax_r.plot(
            xp,
            strength_tail,
            color="darkorange",
            linewidth=1.1,
            alpha=0.95,
            label="|tanh(scale*trend)|",
        )
        ax_r.set_ylim(0.0, 1.05)
        ax_r.set_ylabel("Conv strength")
        ax_r.legend(loc="upper right", fontsize=6, framealpha=0.92)

    def _apply_datetime_xaxis(self) -> None:
        """四圖共用實際日期時間刻度（與資料 timestamp 一致，通常為交易所 UTC）。"""
        for ax in (self._ax_price, self._ax_gate_conv, self._ax_pos, self._ax_equity):
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=16))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

    def _apply_shared_xlim_from_ohlc(self, df_price: pd.DataFrame) -> None:
        """
        強制四子圖 x 軸與 OHLC 可視區間一致（與手動 OHLC / date2num 同一座標系）。

        以「最後一根 K」為錨點、寬度約 ``min(n_rows, max_visible) × 5m``，避免 session
        內 timestamp 不連續時用全域 min/max 把中間空段撐滿視窗。
        """
        if df_price.empty or "timestamp" not in df_price.columns:
            return
        ts = pd.to_datetime(df_price["timestamp"], errors="coerce")
        ts = ts.dropna()
        if len(ts) == 0:
            return
        n_bars = int(min(len(ts), self._max_visible))
        t1 = pd.Timestamp(ts.iloc[-1])
        t0 = t1 - pd.Timedelta(minutes=5 * max(0, n_bars - 1))
        try:
            x0 = float(mdates.date2num(t0))
            x1 = float(mdates.date2num(t1))
        except (TypeError, ValueError, OverflowError):
            return
        if not (np.isfinite(x0) and np.isfinite(x1)):
            return
        span = max(float(x1 - x0), 1.0 / (24.0 * 60.0))
        pad = float(span * 0.03)
        lo, hi = x0 - pad, x1 + pad
        for ax in (self._ax_price, self._ax_gate_conv, self._ax_pos, self._ax_equity):
            ax.set_xlim(lo, hi)
            # 避免後續事件 / autoscale 再次把 x 軸收成單點
            ax.set_autoscalex_on(False)

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
        gate_regime_history: Optional[Sequence[float]] = None,
        conviction_strength_history: Optional[Sequence[float]] = None,
        trend_tanh_signed_history: Optional[Sequence[float]] = None,
        gate_b_liquidity_history: Optional[Sequence[float]] = None,
        sideway_mask_history: Optional[Sequence[bool]] = None,
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
            gate_regime_history: 每步 regime -1/0/1（與 reward gate 語意一致）；缺省則以 0 填滿。
            conviction_strength_history: 每步 |tanh(scale*trend_score)|；缺省則以 0 填滿。
            trend_tanh_signed_history: 每步 tanh(scale×trend) 帶符號（5m）；缺省不畫點線。
            gate_b_liquidity_history: 每步 Gate B 0/1（5m 流動性）；缺省不畫綠帶。
            sideway_mask_history: 每步是否盤整（True=盤整），缺省不畫盤整區域。
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

        gr = list(gate_regime_history) if gate_regime_history is not None else []
        cv = list(conviction_strength_history) if conviction_strength_history is not None else []
        if len(gr) < h_eq:
            gr = [0.0] * (h_eq - len(gr)) + gr
        if len(cv) < h_eq:
            cv = [0.0] * (h_eq - len(cv)) + cv
        if len(gr) > h_eq:
            gr = gr[-h_eq:]
        if len(cv) > h_eq:
            cv = cv[-h_eq:]

        tt: List[float] = []
        if trend_tanh_signed_history is not None:
            tt = list(trend_tanh_signed_history)
            if len(tt) < h_eq:
                tt = [0.0] * (h_eq - len(tt)) + tt
            if len(tt) > h_eq:
                tt = tt[-h_eq:]
        gb: List[float] = []
        if gate_b_liquidity_history is not None:
            gb = list(gate_b_liquidity_history)
            if len(gb) < h_eq:
                gb = [0.0] * (h_eq - len(gb)) + gb
            if len(gb) > h_eq:
                gb = gb[-h_eq:]
        sw: List[bool] = []
        if sideway_mask_history is not None:
            sw = [bool(v) for v in list(sideway_mask_history)]
            if len(sw) < h_eq:
                sw = [False] * (h_eq - len(sw)) + sw
            if len(sw) > h_eq:
                sw = sw[-h_eq:]

        align_parts = [h_eq, h_m, h_g, n_price, len(gr), len(cv)]
        if trend_tanh_signed_history is not None:
            align_parts.append(len(tt))
        if gate_b_liquidity_history is not None:
            align_parts.append(len(gb))
        if sideway_mask_history is not None:
            align_parts.append(len(sw))
        h_align = int(min(align_parts)) if (n_price > 0 and h_eq > 0) else 0
        take = h_align

        y_sw = np.asarray(sw[-take:], dtype=bool) if (sideway_mask_history is not None and take > 0) else None
        self._draw_price(df_use, trade_marker_history, gate_labels_rows_history, take, y_sw)

        if take > 0:
            ts_tail = pd.to_datetime(df_use["timestamp"].iloc[-take:])
            xp = mdates.date2num(ts_tail)
            y_reg = np.asarray(gr[-take:], dtype=float)
            y_cv = np.asarray(cv[-take:], dtype=float)
            y_tt = (
                np.asarray(tt[-take:], dtype=float) if trend_tanh_signed_history is not None else None
            )
            y_gb = np.asarray(gb[-take:], dtype=float) if gate_b_liquidity_history is not None else None
            self._draw_gate_and_conviction(xp, y_reg, y_cv, y_tt, y_gb)
        else:
            self._draw_gate_and_conviction(np.array([]), np.array([]), np.array([]))

        self._ax_pos.clear()

        if take > 0:
            ts_tail = pd.to_datetime(df_use["timestamp"].iloc[-take:])
            xp = mdates.date2num(ts_tail)
            yp = np.asarray(list(position_history)[-take:], dtype=float)
            if take <= 3:
                self._ax_pos.plot(
                    xp,
                    yp,
                    color="tab:orange",
                    linewidth=1.2,
                    marker="o",
                    markersize=7,
                )
            else:
                self._ax_pos.plot(xp, yp, color="tab:orange", linewidth=1.2)
        m = float(max(0.01, abs(max_position_pct)))
        self._ax_pos.set_ylim(-1.05 * m, 1.05 * m)
        self._ax_pos.set_ylabel("Pos %")

        self._ax_equity.clear()
        if take > 0:
            ts_tail = pd.to_datetime(df_use["timestamp"].iloc[-take:])
            xp = mdates.date2num(ts_tail)
            ye = np.asarray(list(equity_history)[-take:], dtype=float)
            if take <= 3:
                self._ax_equity.plot(
                    xp,
                    ye,
                    color="tab:green",
                    linewidth=1.2,
                    marker="o",
                    markersize=7,
                )
            else:
                self._ax_equity.plot(xp, ye, color="tab:green", linewidth=1.2)
        self._ax_equity.set_ylabel("Equity")
        self._ax_equity.set_xlabel("Time (YYYY-MM-DD HH:MM, bar timestamps, typically UTC)")

        title = (
            f"{self.symbol} | bar={closed_bar_ts} | equity={paper_equity_usdt:.2f} "
            f"| profit={paper_profit_usdt:.2f} | fee={fee_rate_pct:.4f}% | pos={final_pos_pct:.3f}"
        )
        self._fig.suptitle(title, fontsize=11, y=0.995)
        self._apply_datetime_xaxis()
        # autofmt_xdate 會自行 subplots_adjust，常與 twinx / 多子圖疊加後跑版；改手動旋轉 + 統一 tight_layout
        for ax in (self._ax_price, self._ax_gate_conv, self._ax_pos, self._ax_equity):
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(18)
                lbl.set_ha("right")
        try:
            self._fig.tight_layout(
                rect=(0.02, 0.02, 0.98, 0.92),
                pad=0.6,
                h_pad=0.85,
                w_pad=0.4,
            )
        except Exception:
            # 極端情況（例如某軸無資料）tight_layout 可能失敗，仍嘗試畫出
            self._fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.08, hspace=0.28)
        # 必須在 layout 之後：修正 sharex +「決策序列比 K 線根數短」時的 x 軸被壓扁
        if not df_use.empty:
            self._apply_shared_xlim_from_ohlc(df_use)
        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    def idle(self) -> None:
        """Keep window responsive without full redraw."""
        plt.pause(0.001)
