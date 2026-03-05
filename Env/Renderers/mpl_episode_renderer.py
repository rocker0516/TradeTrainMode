from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from Env.Renderers.base_renderer import BaseEpisodeRenderer


def _add_stats_block(
    fig: Any,
    ax: Any,
    env: Any,
    info: Optional[dict],
    init_bal: float,
    start: int,
    end_excl: int,
) -> None:
    """在圖上固定區塊顯示交易統計（普遍市場交易統計）。"""
    if info is None:
        return
    lines = []
    try:
        final_bal = float(info.get("final_balance", 0.0) or 0.0)
        profit = float(info.get("profit", 0.0)) if "profit" in info else (final_bal - init_bal)
        return_pct = (profit / init_bal * 100.0) if init_bal > 0 else 0.0
        max_dd = float(info.get("episode_max_dd", 0.0) or 0.0)
        fees = float(info.get("total_fees", 0.0) or 0.0)
        trades = int(info.get("episode_trade_count", 0) or 0)
        slc = int(info.get("episode_stop_loss_count", 0) or 0)
        liqc = int(info.get("episode_liq_count", 0) or 0)
        term = str(info.get("termination_reason", "") or "")
        ep_steps = int(info.get("episode_steps", 0) or 0)
        max_steps = int(getattr(env, "max_episode_steps", 0) or getattr(env, "episode_max_steps", 0) or 0)
        fee_to_pnl = (fees / max(abs(profit), 1e-12)) if abs(profit) >= 1e-12 else float("nan")

        lines.append(f"Return %: {return_pct:+.2f}")
        lines.append(f"PnL: {profit:+.2f}")
        lines.append(f"Max DD: {max_dd:.3f}")
        lines.append(f"Fees: {fees:.2f}")
        lines.append(f"Trades: {trades}")
        lines.append(f"SL/LIQ: {slc}/{liqc}")
        lines.append(f"Steps: {ep_steps}/{max_steps}" if max_steps > 0 else f"Steps: {ep_steps}")
        lines.append(f"Fee/PnL: {fee_to_pnl:.2%}" if not np.isnan(fee_to_pnl) else "Fee/PnL: —")
        if len(term) > 22:
            term = term[:19] + "..."
        lines.append(f"Term: {term}")

        # 可選：未實現盈虧、開倉均價、風險指標
        try:
            ex = getattr(env, "executor", None)
            md = getattr(env, "market_data", None)
            if ex is not None and md is not None and hasattr(md, "close_arr") and end_excl > 0:
                idx = min(end_excl - 1, len(md.close_arr) - 1)
                mark = float(md.close_arr[idx])
                if hasattr(ex, "unrealized_pnl") and mark > 0:
                    u = ex.unrealized_pnl(mark)
                    lines.append(f"Unrealized PnL: {u:+.2f}")
                if hasattr(ex, "position") and getattr(ex.position, "entry_price", 0.0):
                    entry = float(ex.position.entry_price)
                    if entry > 0:
                        lines.append(f"Entry: {entry:.2f}")
        except Exception:
            pass
        for key in ("used_margin", "margin_ratio", "liq_distance"):
            if key in info and info[key] is not None:
                try:
                    lines.append(f"{key}: {info[key]}")
                except Exception:
                    pass
    except Exception:
        return
    if not lines:
        return
    text = "\n".join(lines)
    try:
        # 使用 figure 座標、右側留邊距，避免統計區塊貼邊被裁切
        fig.text(0.97, 0.48, text, transform=fig.transFigure, fontsize=7, verticalalignment="top", horizontalalignment="right", family="monospace", bbox=dict(boxstyle="round,pad=0.35", facecolor="wheat", alpha=0.9))
    except Exception:
        pass


def _export_episode_events(env: Any, png_path: Optional[str], start: int, end_excl: int) -> None:
    """可選：將 _episode_events 匯出為 JSON，檔名與圖檔對應。"""
    if not png_path:
        return
    events = list(getattr(env, "_episode_events", []) or [])
    # 簡化：只保留可 JSON 序列化的欄位
    out = []
    for ev in events:
        step_idx = ev.get("step_idx")
        if not (start <= step_idx < end_excl):
            continue
        out.append({k: (str(v) if hasattr(v, "isoformat") else v) for k, v in ev.items()})
    base = os.path.splitext(png_path)[0]
    json_path = base + ".json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=0)
    except OSError:
        pass


@dataclass(frozen=True)
class RenderOutput:
    """Render 的輸出設定。"""

    save_dir: str
    save: bool = True
    show: bool = True
    dpi: int = 140
    figsize: tuple[float, float] = (14.0, 9.0)
    export_events: bool = False  # 是否匯出 _episode_events 為 JSON


class MplfinanceEpisodeRenderer(BaseEpisodeRenderer):
    """以 mplfinance 繪製 episode 圖：K 線 + 交易標記 + 曲線。"""

    def __init__(self, *, output: RenderOutput) -> None:
        self._output = output

    @staticmethod
    def _format_delta_qty(delta_qty: float) -> str:
        """格式化 Δqty（資產單位）字串。"""
        x = float(delta_qty)
        ax = abs(x)
        if ax < 1:
            return f"{x:+.6f}"
        if ax < 10:
            return f"{x:+.4f}"
        return f"{x:+.2f}"

    @staticmethod
    def _safe_mkdir(path: str) -> None:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            # do not fail training due to render path
            pass

    @staticmethod
    def _can_show_gui() -> bool:
        """盡力判斷是否可 show GUI（策略 A：可顯示就顯示，否則只存檔）。"""
        try:
            import matplotlib

            backend = str(matplotlib.get_backend() or "").lower()
            # Agg / pdf / svg 等一般不可互動顯示
            if "agg" in backend or "pdf" in backend or "svg" in backend or "cairo" in backend:
                return False
            return True
        except Exception:
            return False

    def render_episode(self, *, env: Any, info: Optional[dict] = None) -> Optional[str]:
        """
        Render 當前 episode 圖並（可選）存檔/顯示。

        重要（Windows 常見崩潰修正）：
        - 你遇到的 `Tcl_AsyncDelete: async handler deleted by the wrong thread` 幾乎都是
          Matplotlib 使用 Tk 後端（TkAgg）時，在非主執行緒/非 Tk 事件迴圈情境呼叫 close/cleanup 造成。
        - 我們在 `show=False`（訓練/CI/Headless 最常見）時，強制切到非互動式 `Agg` 後端，
          讓 savefig/close 不再依賴 Tk，避免訓練中途被 render 崩潰。
        """

        # Lazy import: avoid heavy import during training unless render is called.
        # 必須在 import mplfinance/matplotlib.pyplot 之前設定 backend，否則可能已經鎖定 TkAgg。
        try:
            import matplotlib

            if not bool(getattr(self._output, "show", True)):
                # force=True：即使 matplotlib 已初始化，也盡力切換到 Agg（若 pyplot 已 import 則可能無效）
                matplotlib.use("Agg", force=True)

            import matplotlib.pyplot as plt
            import mplfinance as mpf
            # 圖表內標籤用英文，避免無 CJK 字型時 Glyph missing 警告（mplfinance 會覆寫 rcParams）
            _l_pos = "Position (green=long / red=short)"
            _l_ret = "Return (vs initial balance)"
            _l_eq = "Equity / Wallet"
            _l_vol = "Volume"
        except Exception:
            return None

        md = getattr(env, "market_data", None)
        tr = getattr(env, "tracker", None)
        if md is None or tr is None:
            return None

        data_len = int(len(getattr(md, "df_5m", [])))
        episode_start_step = int(getattr(env, "episode_start_step", 0) or 0)
        # 只顯示「以最後一步為結尾、長度 = window_size」的區間，圖表聚焦可讀
        last_step = getattr(env, "_last_executed_step_idx", None)
        if last_step is not None:
            last_step = int(last_step)
        else:
            episode_max_steps = int(getattr(env, "episode_max_steps", 0) or 0)
            last_step = min(episode_start_step + max(0, episode_max_steps - 1), data_len - 1)
        window_size = int(getattr(env, "window_size", 288) or 288)
        end_excl = min(last_step + 1, data_len)
        start = max(0, end_excl - window_size)
        end_excl = min(end_excl, data_len)
        start = int(np.clip(start, 0, max(0, end_excl - 1)))
        if end_excl - start < 5:
            return None

        # Time index
        try:
            ts = md.df_5m["timestamp"].iloc[start:end_excl]
            if not pd.api.types.is_datetime64_any_dtype(ts):
                ts = pd.to_datetime(ts)
        except Exception:
            return None

        # OHLC
        try:
            o = md.open_arr[start:end_excl]
            h = md.high_arr[start:end_excl]
            l = md.low_arr[start:end_excl]
            c = md.close_arr[start:end_excl]
        except Exception:
            return None

        df = pd.DataFrame(
            {"Open": o, "High": h, "Low": l, "Close": c},
            index=pd.DatetimeIndex(ts),
        )

        # Curves (raw series; NaN will naturally blank after done)
        acct = tr.account_series
        equity_raw = np.array(acct.get("equity_raw", np.full(data_len, np.nan)), dtype=float)[start:end_excl]
        wallet_raw = np.array(acct.get("wallet_raw", np.full(data_len, np.nan)), dtype=float)[start:end_excl]
        pos_size = np.array(acct.get("position_size", np.full(data_len, np.nan)), dtype=float)[start:end_excl]

        init_bal = float(getattr(env, "initial_balance", 0.0) or 0.0)
        ret = (equity_raw / init_bal - 1.0) if init_bal > 0 else np.full_like(equity_raw, np.nan)
        ret_zero = np.zeros_like(ret)

        # 持倉曲線正負分色：多單一色、空單一色
        pos_long = np.where(pos_size > 0, pos_size, np.nan)
        pos_short = np.where(pos_size < 0, pos_size, np.nan)

        # Volume（若有）
        has_volume = "volume" in getattr(md, "df_5m", pd.DataFrame()).columns
        vol_series = None
        if has_volume:
            try:
                vol_series = md.df_5m["volume"].iloc[start:end_excl].values
                vol_series = pd.Series(vol_series, index=df.index)
            except Exception:
                has_volume = False

        # Build addplots: panel 1=pos, 2=ret, 3=equity/wallet; 可選 4=volume
        addplots = []
        try:
            addplots.append(mpf.make_addplot(pd.Series(pos_long, index=df.index), panel=1, color="tab:green", ylabel="pos_long"))
            addplots.append(mpf.make_addplot(pd.Series(pos_short, index=df.index), panel=1, color="tab:red", ylabel="pos_short"))
            addplots.append(mpf.make_addplot(pd.Series(ret, index=df.index), panel=2, color="tab:green", ylabel="ret"))
            addplots.append(mpf.make_addplot(pd.Series(ret_zero, index=df.index), panel=2, color="gray", linestyle="--", ylabel=""))
            addplots.append(mpf.make_addplot(pd.Series(equity_raw, index=df.index), panel=3, color="tab:blue", ylabel="equity"))
            addplots.append(mpf.make_addplot(pd.Series(wallet_raw, index=df.index), panel=3, color="tab:gray"))
            if has_volume and vol_series is not None:
                addplots.append(mpf.make_addplot(vol_series, panel=4, type="bar", color="tab:gray", ylabel="vol"))
        except Exception:
            addplots = []

        # 標題：僅保留簡短識別（symbol + env_id + 起迄），詳細統計已在右側區塊，避免單行過長被切
        symbol = getattr(env, "target_symbol", None) or "Symbol"
        env_id = int(getattr(env, "env_id", 0))
        start_ts = getattr(env, "episode_start_timestamp", None) or str(start)
        last_step = getattr(env, "_last_executed_step_idx", None)
        if last_step is not None and hasattr(md, "df_5m") and len(md.df_5m) > 0:
            try:
                end_ts = md.df_5m["timestamp"].iloc[min(int(last_step), len(md.df_5m) - 1)]
                if hasattr(end_ts, "strftime"):
                    end_ts = end_ts.strftime("%Y-%m-%d %H:%M")
                else:
                    end_ts = str(end_ts)[:16]
            except Exception:
                end_ts = str(last_step)
        else:
            end_ts = str(end_excl - 1)
        title = f"{symbol}  env{env_id}  {start_ts} → {end_ts}"

        # panel_ratios 必須與實際 panel 數一致（1 main + len(unique addplot panels)）
        n_panels = 1 + (4 if (has_volume and addplots) else (3 if addplots else 0))
        panel_ratios = (3, 1, 1, 1, 1) if n_panels == 5 else ((3, 1, 1, 1) if n_panels == 4 else (1,))
        # Plot
        # 刻意畫到 max_episode_steps，資料量可能很大；提高門檻避免 mplfinance 警告
        n_bars = len(df)
        fig, axes = mpf.plot(
            df,
            type="candle",
            style="yahoo",
            addplot=addplots if addplots else [],
            volume=False,
            title=title,
            returnfig=True,
            figsize=self._output.figsize,
            panel_ratios=panel_ratios,
            xrotation=15,
            datetime_format="%Y-%m-%d %H:%M",
            tight_layout=True,
            warn_too_much_data=max(50000, n_bars + 10000),
        )
        if not isinstance(axes, (list, tuple)):
            axes = [axes]
        # 縮小標題字體，避免過大與裁切
        try:
            if axes:
                axes[0].set_title(axes[0].get_title(), fontsize=10)
        except Exception:
            pass

        # 最下方兩圖表（及持倉、成交量）補上說明（有 CJK 字型用中文，否則英文）
        try:
            n_ax = len(axes)
            if n_ax >= 2:
                axes[1].text(0.02, 0.96, _l_pos, transform=axes[1].transAxes, fontsize=7, va="top", color="gray")
            if n_ax >= 3:
                axes[2].set_ylabel(_l_ret, fontsize=8)
                axes[2].text(0.02, 0.96, _l_ret, transform=axes[2].transAxes, fontsize=7, va="top", color="gray")
            if n_ax >= 4:
                axes[3].set_ylabel(_l_eq, fontsize=8)
                axes[3].text(0.02, 0.96, _l_eq, transform=axes[3].transAxes, fontsize=7, va="top", color="gray")
            if n_ax >= 5:
                axes[4].set_ylabel(_l_vol, fontsize=8)
                axes[4].text(0.02, 0.96, _l_vol, transform=axes[4].transAxes, fontsize=7, va="top", color="gray")
        except Exception:
            pass

        # Annotations: trade events & SL/LIQ & termination line
        try:
            ax_price = axes[0] if axes else None
        except Exception:
            ax_price = None

        if ax_price is not None:
            # mplfinance 的 x 軸實際為列索引 (0..n-1)，標籤才顯示 datetime；事件與終止線需用整數位置對齊
            n_bars_plot = len(df)
            def _bar_x(step_idx: int):
                """step_idx 對應在圖上的 x 位置（列索引，與 K 線對齊）。"""
                loc = step_idx - start
                if loc < 0 or loc >= n_bars_plot:
                    return None
                return loc

            # termination vertical line
            done_step_idx = getattr(env, "_last_executed_step_idx", None)
            if done_step_idx is not None:
                try:
                    done_step_idx = int(done_step_idx)
                    x_done = _bar_x(done_step_idx)
                    if x_done is not None:
                        ax_price.axvline(x_done, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
                except Exception:
                    pass

            # event markers：只畫箭頭/符號，不畫文字標籤；同 bar 內自動避開重疊
            MAX_MARKERS_PER_BAR = 8
            MIN_MARKER_GAP_FRAC = 0.08  # 同 bar 內標記最小間距（佔 K 線 span 比例）
            _priority = {"liq": 0, "liquidation": 0, "sl": 1, "stop_loss": 1, "close": 2, "entry": 3, "reduce": 4, "flip": 5}
            events = list(getattr(env, "_episode_events", []) or [])
            by_bar: dict[int, list[dict]] = {}
            for ev in events:
                try:
                    step_idx = int(ev.get("step_idx"))
                    if _bar_x(step_idx) is None:
                        continue
                    by_bar.setdefault(step_idx, []).append(ev)
                except Exception:
                    continue

            # 第一輪：計算每個標記的 (x, y, marker, color, s)，並按 bar 收集
            bar_markers: dict[int, list[dict]] = {}  # step_idx -> list of {x, y, marker, color, s, is_last, overflow}
            for step_idx, bar_evs in by_bar.items():
                loc = _bar_x(step_idx)
                if loc is None:
                    continue
                bar_evs.sort(key=lambda e: _priority.get(str(e.get("type", "")).lower(), 99))
                to_draw = bar_evs[:MAX_MARKERS_PER_BAR]
                hi, lo = float(h[loc]), float(l[loc])
                span = max(1e-8, hi - lo)
                per_bar_count: dict[tuple[str, bool], int] = {}
                row: list[dict] = []
                for i, ev in enumerate(to_draw):
                    try:
                        ev_type = str(ev.get("type", "")).lower()
                        delta = float(ev.get("delta_qty", 0.0))
                        prev_size = float(ev.get("prev_size", 0.0))
                        new_size = float(ev.get("new_size", prev_size + delta))
                        is_buy = delta > 0
                        k = (ev_type, is_buy)
                        n = per_bar_count.get(k, 0)
                        per_bar_count[k] = n + 1
                        off = (0.15 + 0.12 * n) * span
                        if is_buy:
                            y = lo - off
                            marker = "^"
                        else:
                            y = hi + off
                            marker = "v"
                        if ev_type in {"sl", "stop_loss"}:
                            marker = "X"
                            color = "purple"
                            y = (lo - off) if prev_size > 0.0 else (hi + off)
                        elif ev_type in {"liq", "liquidation"}:
                            marker = "X"
                            color = "black"
                            y = (lo - off) if prev_size > 0.0 else (hi + off)
                        elif ev_type == "entry":
                            color = "green" if new_size > 0 else "red"
                        elif ev_type == "reduce":
                            color = "red" if prev_size > 0 else "green"
                        elif ev_type == "close":
                            color = "red" if prev_size > 0 else "green"
                        elif ev_type == "flip":
                            color = "green" if new_size > 0 else "red"
                        else:
                            color = "green" if is_buy else "red"
                        n_on_bar = len(to_draw)
                        s = 12 if n_on_bar > 4 else 18  # 與 K 線粗細相當，避免箭頭過大
                        is_last = ev == to_draw[-1]
                        overflow = len(bar_evs) - MAX_MARKERS_PER_BAR if (is_last and len(bar_evs) > MAX_MARKERS_PER_BAR) else 0
                        row.append({"x": loc, "y": y, "marker": marker, "color": color, "s": s, "is_last": is_last, "overflow": overflow})
                    except Exception:
                        continue
                if row:
                    bar_markers[step_idx] = row

            # 同 bar 內避開重疊：依 y 排序後，相鄰兩點間距至少 min_gap
            for step_idx, row in bar_markers.items():
                if len(row) <= 1:
                    continue
                loc = _bar_x(step_idx)
                if loc is None:
                    continue
                hi, lo = float(h[loc]), float(l[loc])
                span = max(1e-8, hi - lo)
                min_gap = span * MIN_MARKER_GAP_FRAC
                # 依 y 升序，從上往下擠開
                row_sorted = sorted(row, key=lambda m: m["y"])
                for i in range(1, len(row_sorted)):
                    if row_sorted[i]["y"] - row_sorted[i - 1]["y"] < min_gap:
                        row_sorted[i]["y"] = row_sorted[i - 1]["y"] + min_gap

            # 第二輪：只畫箭頭（scatter），不畫文字；僅 overflow 時畫一個 "+N"
            for step_idx, row in bar_markers.items():
                for m in row:
                    ax_price.scatter([m["x"]], [m["y"]], marker=m["marker"], s=m["s"], color=m["color"], zorder=5)
                overflow = next((m["overflow"] for m in row if m.get("overflow", 0) > 0), 0)
                if overflow > 0 and row:
                    last = row[-1]
                    ax_price.annotate(
                        f"+{overflow}",
                        (last["x"], last["y"]),
                        textcoords="offset points",
                        xytext=(4, -4),
                        fontsize=6,
                        color="gray",
                        alpha=0.8,
                    )

            # 事件圖例（Entry / Reduce / Close / Flip / SL / LIQ）
            try:
                from matplotlib.lines import Line2D

                legend_handles = [
                    Line2D([0], [0], marker="^", color="w", markerfacecolor="green", markersize=8, label="Entry"),
                    Line2D([0], [0], marker="v", color="w", markerfacecolor="red", markersize=8, label="Reduce"),
                    Line2D([0], [0], marker="v", color="w", markerfacecolor="darkred", markersize=8, label="Close"),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, label="Flip"),
                    Line2D([0], [0], marker="X", color="w", markerfacecolor="purple", markersize=8, label="SL"),
                    Line2D([0], [0], marker="X", color="w", markerfacecolor="black", markersize=8, label="LIQ"),
                ]
                ax_price.legend(handles=legend_handles, loc="upper left", fontsize=7, ncol=2)
            except Exception:
                pass

            # 交易統計區塊（普遍市場交易統計）
            try:
                _add_stats_block(fig, axes[0], env, info, init_bal, start, end_excl)
            except Exception:
                pass

        # 報酬曲線 0 軸線已由 addplot ret_zero 繪製

        # 時間軸可讀性：長 episode 時限制 x 軸 tick 數量
        try:
            n_bars = end_excl - start
            if n_bars > 100:
                for ax in axes:
                    if hasattr(ax, "xaxis") and hasattr(ax.xaxis, "set_major_locator"):
                        import matplotlib.ticker as mtick
                        ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=12, integer=False))
        except Exception:
            pass

        # Save / show (strategy A)
        out_path: Optional[str] = None
        if self._output.save:
            self._safe_mkdir(self._output.save_dir)
            # filename includes symbol/env_id/start_step/done_step
            safe_symbol = str(getattr(env, "target_symbol", "SYMBOL")).replace("/", "_")
            done_idx = getattr(env, "_last_executed_step_idx", None)
            done_idx = int(done_idx) if done_idx is not None else int(getattr(env, "current_step", 0))
            fname = f"{safe_symbol}_env{int(getattr(env, 'env_id', 0)):03d}_s{start}_d{done_idx}.png"
            out_path = os.path.join(self._output.save_dir, fname)
            try:
                fig.savefig(out_path, dpi=int(self._output.dpi))
            except Exception:
                out_path = None
            # 單筆交易匯出（可選），在存檔後執行
            if getattr(self._output, "export_events", False) and out_path:
                _export_episode_events(env, out_path, start, end_excl)

        can_show = bool(self._output.show and self._can_show_gui())
        if can_show:
            try:
                plt.show(block=False)
                plt.pause(0.001)
            except Exception:
                pass
        # 若真的有 GUI 且要求 show，避免立刻 close 導致視窗瞬間消失。
        # 在 headless / show=False 的情況，才關閉 fig 釋放記憶體。
        if not can_show:
            try:
                plt.close(fig)
            except Exception:
                pass

        return out_path

