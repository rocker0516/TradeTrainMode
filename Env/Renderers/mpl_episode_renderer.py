from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from Env.Renderers.base_renderer import BaseEpisodeRenderer


@dataclass(frozen=True)
class RenderOutput:
    """Render 的輸出設定。"""

    save_dir: str
    save: bool = True
    show: bool = True
    dpi: int = 140
    figsize: tuple[float, float] = (14.0, 9.0)


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
        except Exception:
            return None

        md = getattr(env, "market_data", None)
        tr = getattr(env, "tracker", None)
        if md is None or tr is None:
            return None

        start = int(getattr(env, "episode_start_step", 0) or 0)
        max_steps = int(getattr(env, "episode_max_steps", 0) or 0)
        if max_steps <= 0:
            return None

        # 顯示範圍：固定用 episode_max_steps 長度
        # 以 bar 數量計：取 [start, start+max_steps) 共 max_steps 根 K
        end_excl = int(start + max_steps)
        data_len = int(len(getattr(md, "df_5m", [])))
        end_excl = int(np.clip(end_excl, 0, data_len))
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

        # Build addplots
        addplots = []
        try:
            addplots.append(mpf.make_addplot(pd.Series(pos_size, index=df.index), panel=1, color="tab:orange", ylabel="pos_qty"))
            addplots.append(mpf.make_addplot(pd.Series(ret, index=df.index), panel=2, color="tab:green", ylabel="ret"))
            addplots.append(mpf.make_addplot(pd.Series(equity_raw, index=df.index), panel=3, color="tab:blue", ylabel="equity"))
            addplots.append(mpf.make_addplot(pd.Series(wallet_raw, index=df.index), panel=3, color="tab:gray"))
        except Exception:
            addplots = []

        # Title / subtitle
        title_parts: list[str] = []
        symbol = getattr(env, "target_symbol", None)
        if symbol:
            title_parts.append(str(symbol))
        title_parts.append(f"env_id={int(getattr(env, 'env_id', 0))}")
        title_parts.append(f"start={getattr(env, 'episode_start_timestamp', None) or start}")
        if info:
            try:
                final_eq = float(info.get("final_balance", np.nan))
                profit = float(info.get("profit", np.nan))
                max_dd = float(info.get("episode_max_dd", np.nan))
                fees = float(info.get("total_fees", np.nan))
                trades = int(info.get("episode_trade_count", 0) or 0)
                slc = int(info.get("episode_stop_loss_count", 0) or 0)
                liqc = int(info.get("episode_liq_count", 0) or 0)
                term = info.get("termination_reason", None)
                title_parts.append(
                    f"final_eq={final_eq:.2f} profit={profit:.2f} max_dd={max_dd:.3f} fees={fees:.2f} trades={trades} sl={slc} liq={liqc} term={term}"
                )
            except Exception:
                pass
        title = " | ".join(title_parts)

        # Plot
        fig, axes = mpf.plot(
            df,
            type="candle",
            style="yahoo",
            addplot=addplots if addplots else None,
            volume=False,
            title=title,
            returnfig=True,
            figsize=self._output.figsize,
            panel_ratios=(3, 1, 1, 1),
            xrotation=15,
            datetime_format="%Y-%m-%d %H:%M",
            tight_layout=True,
        )

        # Annotations: trade events & SL/LIQ & termination line
        try:
            ax_price = axes[0]
        except Exception:
            ax_price = None

        if ax_price is not None:
            # termination vertical line
            done_step_idx = getattr(env, "_last_executed_step_idx", None)
            if done_step_idx is not None:
                try:
                    done_step_idx = int(done_step_idx)
                    if start <= done_step_idx < end_excl:
                        done_ts = df.index[done_step_idx - start]
                        ax_price.axvline(done_ts, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
                except Exception:
                    pass

            # event markers
            events = list(getattr(env, "_episode_events", []) or [])
            # group counts to avoid overlap on same candle
            per_bar_count: dict[tuple[int, str], int] = {}

            for ev in events:
                try:
                    step_idx = int(ev.get("step_idx"))
                    if not (start <= step_idx < end_excl):
                        continue
                    ev_type = str(ev.get("type", "")).lower()
                    delta = float(ev.get("delta_qty", 0.0))
                    # x
                    x = df.index[step_idx - start]
                    # y base from candle
                    hi = float(h[step_idx - start])
                    lo = float(l[step_idx - start])

                    # direction: buy (up) if delta>0 else sell (down)
                    is_buy = delta > 0
                    dir_key = "up" if is_buy else "down"
                    k = (step_idx, dir_key)
                    n = per_bar_count.get(k, 0)
                    per_bar_count[k] = n + 1

                    # offset stacking
                    span = max(1e-8, hi - lo)
                    off = (0.15 + 0.12 * n) * span
                    if is_buy:
                        y = lo - off
                        marker = "^"
                        color = "green"
                    else:
                        y = hi + off
                        marker = "v"
                        color = "red"

                    # special styles
                    label_prefix = ev_type
                    if ev_type in {"sl", "stop_loss"}:
                        marker = "X"
                        color = "purple"
                        label_prefix = "SL"
                        # place at extreme to highlight
                        y = (lo - off) if (float(ev.get("prev_size", 0.0)) > 0.0) else (hi + off)
                    elif ev_type in {"liq", "liquidation"}:
                        marker = "X"
                        color = "black"
                        label_prefix = "LIQ"
                        y = (lo - off) if (float(ev.get("prev_size", 0.0)) > 0.0) else (hi + off)
                    elif ev_type == "entry":
                        label_prefix = "entry"
                    elif ev_type == "close":
                        label_prefix = "close"
                    elif ev_type == "reduce":
                        label_prefix = "reduce"

                    ax_price.scatter([x], [y], marker=marker, s=70, color=color, zorder=5)

                    txt = f"{label_prefix} {self._format_delta_qty(delta)}"
                    ax_price.annotate(
                        txt,
                        (x, y),
                        textcoords="offset points",
                        xytext=(6, 6 if is_buy else -10),
                        fontsize=8,
                        color=color,
                        alpha=0.9,
                    )
                except Exception:
                    continue

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

