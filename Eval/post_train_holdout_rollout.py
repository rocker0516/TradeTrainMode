"""
訓練結束後：在 holdout 設定的 eval 切片上跑單一 episode，並以 Live 風格圖表顯示摘要。
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Callable, Deque, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from Env.config import Config
from Env.trading_env import TradingEnvironment
from LiveTradingRunner.live_render import LiveRefreshRenderer
from LiveTradingRunner.render_history_utils import (
    conviction_strength_from_trend_score,
    gate_ac_change_labels,
    gate_flags_to_regime_indicator,
    trade_bs_from_delta,
    trend_tanh_signed_from_trend_score,
)


def post_train_max_episode_steps_cap(*, data_split: bool, holdout_months: int) -> int:
    """
    決定傳入環境的 ``max_episode_steps`` 上限。

    - 有資料切分時：給極大值，讓 reset 時以「資料剩餘長度」夾成真正 episode 上限。
    - 無切分時：以「約當 holdout 月數」的 5m 棒數上限，避免全歷史過長。

    Args:
        data_split: 是否啟用 train/eval 時間切分。
        holdout_months: 與 CLI holdout 月數一致（僅 no-data-split 時用於步數上限）。

    Returns:
        應寫入 ``TradingEnvironment`` kwargs 的 ``max_episode_steps``。
    """
    if data_split:
        return 1_000_000_000
    return max(1, int(288 * 30 * max(1, int(holdout_months))))


def unwrap_to_trading_env(env: gym.Env) -> TradingEnvironment:
    """
    自 Phase AB wrapper 鏈取出底層 ``TradingEnvironment``（含 ``TradingEnvPhaseA`` 子類）。

    Args:
        env: 最外層 gym Env。

    Returns:
        底層交易環境。

    Raises:
        TypeError: 無法 unwrap 到 ``TradingEnvironment``。
    """
    cur: Any = env
    while not isinstance(cur, TradingEnvironment):
        if not hasattr(cur, "env"):
            raise TypeError(
                f"Cannot unwrap to TradingEnvironment: stuck at {type(cur).__name__}"
            )
        cur = cur.env
    return cur


def _ohlc_slice_for_live_render(
    df_5m: pd.DataFrame,
    *,
    target_symbol: str,
    i0: int,
    i1: int,
) -> Optional[pd.DataFrame]:
    """
    自合併後 ``df_5m``（多為 ``{SYMBOL}_open`` 前綴欄位）取出區間，並輸出
    ``LiveRefreshRenderer`` 需要的 ``timestamp/open/high/low/close``。

    Args:
        df_5m: ``MarketData`` 與環境共用的 5m 表。
        target_symbol: 主交易對（與 ``TradingEnvironment.target_symbol`` 一致）。
        i0: 起始列索引（含）。
        i1: 結束列索引（含）。

    Returns:
        標準 OHLC 表；無法對應欄位時為 ``None``。
    """
    if df_5m.empty or "timestamp" not in df_5m.columns:
        return None
    sym = str(target_symbol).strip().upper()
    p_open = f"{sym}_open"
    p_high = f"{sym}_high"
    p_low = f"{sym}_low"
    p_close = f"{sym}_close"
    sl = df_5m.loc[i0:i1]
    if p_close in df_5m.columns and p_high in df_5m.columns and p_low in df_5m.columns:
        open_series = df_5m[p_open] if p_open in df_5m.columns else df_5m[p_close]
        out = pd.DataFrame(
            {
                "timestamp": sl["timestamp"].values,
                "open": open_series.loc[sl.index].values,
                "high": sl[p_high].values,
                "low": sl[p_low].values,
                "close": sl[p_close].values,
            }
        )
        return out
    if "close" in df_5m.columns and "high" in df_5m.columns and "low" in df_5m.columns:
        o_vals = sl["open"].values if "open" in df_5m.columns else sl["close"].values
        out = pd.DataFrame(
            {
                "timestamp": sl["timestamp"].values,
                "open": o_vals,
                "high": sl["high"].values,
                "low": sl["low"].values,
                "close": sl["close"].values,
            }
        )
        return out
    return None


def _gate_tuple_from_obs(obs: Any) -> Optional[Tuple[float, float, float]]:
    if not isinstance(obs, dict) or "gate_flags" not in obs:
        return None
    gf = obs["gate_flags"]
    arr = np.asarray(gf, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return None
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def run_holdout_rollout_and_show_live_render(
    *,
    model: SAC,
    env_thunk: Callable[[], gym.Env],
    data_split: bool,
    holdout_months: int,
    deterministic: bool,
    seed: Optional[int],
    project_root: str,
    render_pos_delta_eps: float = 0.02,
    max_visible_bars: int = 2000,
) -> None:
    """
    建立單一 env（非 VecEnv）、跑滿一個 episode，最後開啟與 live loop 相同版面的圖表。

    Args:
        model: 已訓練的 SAC（記憶體內，無需重新 load）。
        env_thunk: ``make_env`` 回傳的 thunk；kwargs 應已含 eval 切片與正確 ``max_episode_steps``／random_start。
        data_split: 是否做過 train/eval 切分（僅用於 log 說明）。
        holdout_months: holdout 月數（僅 log）。
        deterministic: 推論是否 deterministic。
        seed: ``reset(seed=...)``；可為 None。
        project_root: 專案根（``Data/`` 目錄）。
        render_pos_delta_eps: B/S 標記門檻（與 live 一致）。
        max_visible_bars: 價格圖最多顯示棒數。
    """
    env = env_thunk()
    try:
        if seed is not None:
            obs, _ = env.reset(seed=int(seed))
        else:
            obs, _ = env.reset()

        equity_history: List[float] = []
        position_history: List[float] = []
        trade_marker_history: List[Optional[str]] = []
        gate_labels_rows_history: List[List[str]] = []
        gate_regime_history: List[float] = []
        conviction_strength_history: List[float] = []
        trend_tanh_signed_history: List[float] = []
        gate_b_liquidity_history: List[float] = []
        prev_gate: Optional[Tuple[float, float, float]] = None
        prev_pos: float = 0.0
        obs_curr: Any = obs

        done = False
        while not done:
            curr_gate = _gate_tuple_from_obs(obs_curr)
            gate_labels_rows_history.append(gate_ac_change_labels(prev_gate, curr_gate))
            if curr_gate is not None:
                prev_gate = curr_gate

            action, _ = model.predict(obs_curr, deterministic=bool(deterministic))
            obs_next, _reward, terminated, truncated, info = env.step(action)
            info_d = info if isinstance(info, dict) else {}
            base = unwrap_to_trading_env(env)

            _df5 = getattr(base.market_data, "df_5m", None)
            _md_len = int(len(_df5)) if _df5 is not None else 0
            _step_idx = int(getattr(base, "_last_executed_step_idx", 0))
            if _md_len > 0:
                _step_idx = max(0, min(_step_idx, _md_len - 1))
                _met = base.market_data.get_market_metrics(_step_idx)
                _ts = float(_met.get("trend_score", 0.0))
                _gf = base.market_data.get_gate_flags(_step_idx)
                _scale = float(
                    getattr(
                        getattr(base, "reward_calculator", None),
                        "conviction_trend_score_scale",
                        getattr(Config, "CONVICTION_TREND_SCORE_SCALE", 10.0),
                    )
                )
                gate_regime_history.append(gate_flags_to_regime_indicator(_gf))
                conviction_strength_history.append(
                    conviction_strength_from_trend_score(_ts, scale=_scale)
                )
                trend_tanh_signed_history.append(
                    trend_tanh_signed_from_trend_score(_ts, scale=_scale)
                )
                _gfa = np.asarray(_gf, dtype=np.float64).reshape(-1)
                gate_b_liquidity_history.append(
                    float(_gfa[1]) if _gfa.size >= 2 else 0.0
                )
            else:
                gate_regime_history.append(0.0)
                conviction_strength_history.append(0.0)
                trend_tanh_signed_history.append(0.0)
                gate_b_liquidity_history.append(0.0)

            eq = float(info_d.get("equity", float("nan")))
            if np.isfinite(eq):
                equity_history.append(eq)
            else:
                equity_history.append(float(getattr(base, "initial_balance", 0.0)))

            pos = float(info_d.get("last_final_pos_pct", 0.0))
            position_history.append(pos)

            trade_marker_history.append(
                trade_bs_from_delta(
                    prev_final_pos_pct=prev_pos,
                    new_final_pos_pct=pos,
                    eps=float(render_pos_delta_eps),
                )
            )
            prev_pos = pos

            done = bool(terminated or truncated)
            obs_curr = obs_next

        if not equity_history:
            print("[post_train_holdout_render] no steps recorded; skip render.", flush=True)
            return

        df_5m = base.market_data.df_5m
        target_sym = str(getattr(base, "target_symbol", "BTCUSDT")).strip().upper()

        ep_start = int(getattr(base, "episode_start_step", 0))
        last_bar_idx = int(getattr(base, "_last_executed_step_idx", ep_start))
        i0 = max(0, ep_start)
        i1 = min(len(df_5m) - 1, last_bar_idx)
        if i1 < i0:
            i1 = i0
        df_price = _ohlc_slice_for_live_render(
            df_5m, target_symbol=target_sym, i0=i0, i1=i1
        )
        if df_price is None or df_price.empty:
            print(
                "[post_train_holdout_render] cannot map OHLC columns "
                f"(target_symbol={target_sym!r}); skip render.",
                flush=True,
            )
            return

        n_decisions = len(equity_history)
        split_mode = "eval_slice" if data_split else "full_data_capped"
        print(
            f"[post_train_holdout_render] split={split_mode} holdout_months={holdout_months} "
            f"decision_steps={n_decisions} bars_ohlc=[{i0},{i1}] — opening chart…",
            flush=True,
        )

        try:
            from Eval.train_config import TrainConfig

            max_pos = float(getattr(TrainConfig, "MAX_POSITION_PCT", 0.8))
        except Exception:
            max_pos = 0.8

        fee_pct = float(getattr(base, "transaction_fee", 0.01))
        last_eq = float(equity_history[-1]) if equity_history else float(base.initial_balance)
        init_bal = float(base.initial_balance)
        last_ts = df_price["timestamp"].iloc[-1] if not df_price.empty else ""

        mv = int(max(200, min(max_visible_bars, max(n_decisions + 64, 500))))
        renderer = LiveRefreshRenderer(
            symbol=target_sym, max_visible_bars=mv, data_dir=os.path.join(project_root, "Data")
        )

        n_eq = len(equity_history)
        if n_eq > 0 and ep_start + n_eq <= len(df_5m):
            ts_hist = [str(x) for x in df_5m["timestamp"].iloc[ep_start : ep_start + n_eq].tolist()]
        else:
            ts_hist = [str(last_ts)] * max(1, n_eq)

        eq_deque: Deque[float] = deque(equity_history)
        pos_deque: Deque[float] = deque(position_history)
        tm_deque: Deque[Optional[str]] = deque(trade_marker_history)
        gl_deque: Deque[List[str]] = deque(gate_labels_rows_history)

        renderer.refresh(
            closed_bar_ts=str(last_ts),
            paper_equity_usdt=last_eq,
            paper_profit_usdt=float(last_eq - init_bal),
            fee_rate_pct=fee_pct,
            final_pos_pct=float(position_history[-1]) if position_history else 0.0,
            equity_history=list(eq_deque),
            ts_history=ts_hist,
            df_price=df_price,
            position_history=list(pos_deque),
            trade_marker_history=list(tm_deque),
            gate_labels_rows_history=list(gl_deque),
            max_position_pct=max_pos,
            gate_regime_history=gate_regime_history,
            conviction_strength_history=conviction_strength_history,
            trend_tanh_signed_history=trend_tanh_signed_history,
            gate_b_liquidity_history=gate_b_liquidity_history,
        )

        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show(block=True)
    except Exception as exc:
        print(f"[post_train_holdout_render] skipped: {exc}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
