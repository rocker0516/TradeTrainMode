"""Live trading loop (every 5 minutes) using SB3 best_model.zip.

設計重點：
- 交易 API 安全開關：預設不呼叫交易 API（不下單）；必須顯式 `--enable_trade_api`。
- 市場資料：使用 Binance 公開端點抓取 5m K 線（不需金鑰）。
- 1d 特徵：使用本機 `Data/*_1d.csv`。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# 確保可用 `python -m LiveTradingRunner.live_trading_loop` 直接執行
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Eval.train_config import TrainConfig
from LiveTradingRunner.live_runner_env_config import LiveRunnerEnvConfig
from LiveTradingRunner.fee_provider import BinanceFeeRateProvider
from LiveTradingRunner.live_render import LiveRefreshRenderer
from LiveTradingRunner.render_history_utils import (
    atr_ratio_from_ohlc,
    build_sideway_mask_from_series,
    gate_ac_change_labels,
    trade_bs_from_delta,
)


@dataclass
class LiveRunnerConfig:
    """Live runner 設定。"""

    symbol: str
    model_path: str
    leverage: int
    max_position_pct: float
    poll_interval_sec: int
    deterministic: bool
    enable_trade_api: bool
    dry_run: bool
    default_min_notional_usdt: float

    # data/feature
    feature_symbols: Tuple[str, ...]
    window_size_5m: int
    window_size_1d: int

    # ---- align with Train/Eval env wrappers & action processing ----
    # 用於模擬 Train/Eval 的 ActionRepeatWrapper（降低決策頻率）
    action_repeat: int = 1
    # 用於模擬 TradingEnvironment 的 ActionProcessor（單步加倉上限）
    max_step_pos_change_pct: float = 0.0
    # 用於模擬 TradeExecutor 的 deadband（避免微小調倉刷手續費）
    min_position_change: float = 0.0
    # 用於模擬 ActionProcessor 的 no-trade hysteresis（更穩定回到空倉）
    no_trade_entry_threshold: float = 0.0
    no_trade_exit_threshold: float = 0.0
    # 用於模擬 risk_base 的更新頻率（Train/Eval 一般用 WINDOW_SIZE_5M）
    risk_base_update_steps: int = 288
    # debug prints（放在最後，避免 dataclass default/non-default 順序限制）
    print_account_info: bool = False
    print_open_positions: bool = False
    print_obs_account_context: bool = False
    # ---- live matplotlib render ----
    render_pos_delta_eps: float = 0.02
    render_max_visible_kline_bars: int = 500
    render_max_kline_buffer_rows: int = 2000


@dataclass
class LiveRunnerState:
    """跨 tick 保存狀態（避免同一根 K 線重複下單）。"""

    last_processed_closed_ts: Optional[np.datetime64] = None
    # 用於模擬 ActionRepeatWrapper：以「已處理過的新收盤 bar 次數」做為 step 計數
    decision_step: int = 0
    # 用於模擬 risk_base（單步加倉上限的基準資金）
    risk_base_usdt: Optional[float] = None
    steps_since_risk_base_update: int = 0
    # 用於 action_repeat：非決策 step 時沿用上一個 final_pos_pct
    last_final_pos_pct: float = 0.0
    # ---- paper trading state (market_state 以外，供 CSV / render) ----
    paper_equity_usdt: float = 1000.0
    last_mark_price: Optional[float] = None
    cumulative_fee_usdt: float = 0.0
    # ---- live render：K 線 session 緩衝（與 API merged df 同源）----
    kline_session_rows: List[Dict[str, Union[str, float]]] = field(default_factory=list)
    # 上一輪 obs 的 gate_flags（A/B/C），供偵測 A/C 變化；None 表示尚未初始化
    last_gate_flags: Optional[Tuple[float, float, float]] = None


@dataclass(frozen=True)
class TickDecision:
    """單次 tick 的決策輸出（不一定會下單）。"""

    closed_bar_ts: str
    last_price: float
    action_raw: float
    action_clipped: float
    target_pos_pct: float
    final_pos_pct: float
    equity_usdt: Optional[float]
    current_position_qty: Optional[float]
    target_position_qty: Optional[float]
    delta_qty: Optional[float]
    # debug: named account/context obs snapshot（對齊 Env/Components/observer.py 的欄位順序）
    obs_account_context_named: Optional[Dict[str, Dict[str, float]]] = None
    gate_flags: Optional[Tuple[float, float, float]] = None
    # 本步實際用於 obs／紙上損益／圖表標題的手續費率（百分比）；與幣安 API 或 fallback 一致
    fee_rate_pct: Optional[float] = None
    # 與訓練 reward 對齊：regime ∈ {-1,0,1}、conviction_strength = |tanh(scale*trend_score)|
    regime_indicator: float = 0.0
    conviction_strength: float = 0.0
    # 5m：tanh(scale×trend) 帶符號；圖表用（reward 只用絕對值當 strength）
    trend_tanh_signed: float = 0.0


def _state_to_json(state: LiveRunnerState) -> Dict[str, Any]:
    """將 LiveRunnerState 轉為可 JSON 序列化 dict。"""
    ts = state.last_processed_closed_ts
    return {
        "last_processed_closed_ts": str(ts) if ts is not None else None,
        "decision_step": int(getattr(state, "decision_step", 0)),
        "risk_base_usdt": float(state.risk_base_usdt) if state.risk_base_usdt is not None else None,
        "steps_since_risk_base_update": int(getattr(state, "steps_since_risk_base_update", 0)),
        "last_final_pos_pct": float(getattr(state, "last_final_pos_pct", 0.0)),
        "paper_equity_usdt": float(getattr(state, "paper_equity_usdt", 1000.0)),
        "last_mark_price": float(state.last_mark_price) if state.last_mark_price is not None else None,
        "cumulative_fee_usdt": float(getattr(state, "cumulative_fee_usdt", 0.0)),
        "kline_session_rows": list(getattr(state, "kline_session_rows", []) or []),
        "last_gate_flags": (
            None
            if getattr(state, "last_gate_flags", None) is None
            else [float(x) for x in getattr(state, "last_gate_flags", ())]
        ),
    }


def _state_from_json(payload: Dict[str, Any]) -> LiveRunnerState:
    """由 JSON dict 還原 LiveRunnerState。"""
    raw = payload.get("last_processed_closed_ts")
    decision_step = int(payload.get("decision_step", 0) or 0)
    risk_base_usdt = payload.get("risk_base_usdt", None)
    try:
        risk_base_usdt_f = None if risk_base_usdt is None else float(risk_base_usdt)
    except (TypeError, ValueError):
        risk_base_usdt_f = None
    steps_since_risk = int(payload.get("steps_since_risk_base_update", 0) or 0)
    try:
        last_final_pos_pct = float(payload.get("last_final_pos_pct", 0.0) or 0.0)
    except (TypeError, ValueError):
        last_final_pos_pct = 0.0
    try:
        paper_equity_usdt = float(payload.get("paper_equity_usdt", 1000.0) or 1000.0)
    except (TypeError, ValueError):
        paper_equity_usdt = 1000.0
    last_mark_raw = payload.get("last_mark_price", None)
    try:
        last_mark_price = None if last_mark_raw is None else float(last_mark_raw)
    except (TypeError, ValueError):
        last_mark_price = None
    try:
        cumulative_fee_usdt = float(payload.get("cumulative_fee_usdt", 0.0) or 0.0)
    except (TypeError, ValueError):
        cumulative_fee_usdt = 0.0

    klines_raw = payload.get("kline_session_rows", [])
    kline_session_rows: List[Dict[str, Union[str, float]]] = []
    if isinstance(klines_raw, list):
        for item in klines_raw:
            if isinstance(item, dict):
                kline_session_rows.append(dict(item))
    lgf_raw = payload.get("last_gate_flags", None)
    last_gate_flags: Optional[Tuple[float, float, float]] = None
    if isinstance(lgf_raw, (list, tuple)) and len(lgf_raw) >= 3:
        try:
            last_gate_flags = (float(lgf_raw[0]), float(lgf_raw[1]), float(lgf_raw[2]))
        except (TypeError, ValueError):
            last_gate_flags = None

    if raw is None:
        return LiveRunnerState(
            last_processed_closed_ts=None,
            decision_step=decision_step,
            risk_base_usdt=risk_base_usdt_f,
            steps_since_risk_base_update=steps_since_risk,
            last_final_pos_pct=last_final_pos_pct,
            paper_equity_usdt=paper_equity_usdt,
            last_mark_price=last_mark_price,
            cumulative_fee_usdt=cumulative_fee_usdt,
            kline_session_rows=kline_session_rows,
            last_gate_flags=last_gate_flags,
        )
    try:
        return LiveRunnerState(
            last_processed_closed_ts=np.datetime64(str(raw)),
            decision_step=decision_step,
            risk_base_usdt=risk_base_usdt_f,
            steps_since_risk_base_update=steps_since_risk,
            last_final_pos_pct=last_final_pos_pct,
            paper_equity_usdt=paper_equity_usdt,
            last_mark_price=last_mark_price,
            cumulative_fee_usdt=cumulative_fee_usdt,
            kline_session_rows=kline_session_rows,
            last_gate_flags=last_gate_flags,
        )
    except (TypeError, ValueError):
        return LiveRunnerState(
            last_processed_closed_ts=None,
            decision_step=decision_step,
            risk_base_usdt=risk_base_usdt_f,
            steps_since_risk_base_update=steps_since_risk,
            last_final_pos_pct=last_final_pos_pct,
            paper_equity_usdt=paper_equity_usdt,
            last_mark_price=last_mark_price,
            cumulative_fee_usdt=cumulative_fee_usdt,
            kline_session_rows=kline_session_rows,
            last_gate_flags=last_gate_flags,
        )


def _append_state_csv(
    *,
    csv_path: str,
    decision: TickDecision,
    state: LiveRunnerState,
    fee_rate_pct: float,
) -> None:
    """Append market_state 以外狀態紀錄到 CSV。"""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    fieldnames = [
        "timestamp",
        "closed_bar_ts",
        "last_price",
        "paper_equity_usdt",
        "paper_profit_usdt",
        "cumulative_fee_usdt",
        "fee_rate_pct",
        "action_raw",
        "action_clipped",
        "target_pos_pct",
        "final_pos_pct",
        "decision_step",
    ]
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "closed_bar_ts": str(decision.closed_bar_ts),
        "last_price": float(decision.last_price),
        "paper_equity_usdt": float(state.paper_equity_usdt),
        "paper_profit_usdt": float(state.paper_equity_usdt - 1000.0),
        "cumulative_fee_usdt": float(state.cumulative_fee_usdt),
        "fee_rate_pct": float(fee_rate_pct),
        "action_raw": float(decision.action_raw),
        "action_clipped": float(decision.action_clipped),
        "target_pos_pct": float(decision.target_pos_pct),
        "final_pos_pct": float(decision.final_pos_pct),
        "decision_step": int(getattr(state, "decision_step", 0)),
    }
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _tick_decision_json_payload(decision: TickDecision) -> Dict[str, Any]:
    """將 ``TickDecision`` 轉成可 ``json.dumps`` 的 dict（tuple → list）。"""
    d: Dict[str, Any] = dict(decision.__dict__)
    gf = d.get("gate_flags")
    if isinstance(gf, tuple):
        d["gate_flags"] = [float(x) for x in gf]
    return d


def _build_sideway_mask_from_df_price(df_plot: pd.DataFrame) -> List[bool]:
    """由 session OHLC 估算 sideway mask（對齊 E2 build_sideway_labels 公式）。"""
    if df_plot is None or df_plot.empty:
        return []
    required = {"high", "low", "close"}
    if not required.issubset(set(df_plot.columns)):
        return []
    close_vals = pd.to_numeric(df_plot["close"], errors="coerce").ffill().fillna(0.0).tolist()
    high_vals = pd.to_numeric(df_plot["high"], errors="coerce").ffill().fillna(0.0).tolist()
    low_vals = pd.to_numeric(df_plot["low"], errors="coerce").ffill().fillna(0.0).tolist()
    atr_ratio_vals = atr_ratio_from_ohlc(high_vals, low_vals, close_vals)
    return build_sideway_mask_from_series(close_vals, atr_ratio_vals)


def _update_paper_equity(
    *,
    state: LiveRunnerState,
    last_price: float,
    prev_final_pos_pct: float,
    new_final_pos_pct: float,
    fee_rate_pct: float,
) -> None:
    """更新紙上資金：先計算持倉收益，再扣除調倉手續費。"""
    prev_price = state.last_mark_price
    equity = max(1e-8, float(getattr(state, "paper_equity_usdt", 1000.0)))
    if prev_price is not None and prev_price > 0:
        price_ret = (float(last_price) - float(prev_price)) / float(prev_price)
        equity = equity * (1.0 + float(prev_final_pos_pct) * float(price_ret))
    turnover = abs(float(new_final_pos_pct) - float(prev_final_pos_pct))
    fee_paid = max(0.0, equity * turnover * (float(fee_rate_pct) / 100.0))
    equity = max(1e-8, equity - fee_paid)
    state.paper_equity_usdt = float(equity)
    state.cumulative_fee_usdt = float(getattr(state, "cumulative_fee_usdt", 0.0) + fee_paid)
    state.last_mark_price = float(last_price)


def _sleep_with_idle(*, total_seconds: float, renderer: Optional[LiveRefreshRenderer]) -> None:
    """Sleep while keeping GUI event loop responsive."""
    remain = float(max(0.0, total_seconds))
    slice_sec = 0.1
    while remain > 0.0:
        if renderer is not None:
            renderer.idle()
        step = slice_sec if remain > slice_sec else remain
        time.sleep(step)
        remain -= step


def _load_state(state_path: str) -> LiveRunnerState:
    """從檔案載入 state；若不存在或損毀則回傳新 state。"""
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return _state_from_json(payload)
    except FileNotFoundError:
        return LiveRunnerState()
    except (json.JSONDecodeError, OSError):
        return LiveRunnerState()
    return LiveRunnerState()


def _save_state(state_path: str, state: LiveRunnerState) -> None:
    """將 state 寫回檔案（原子寫入，避免中斷造成檔案半寫）。"""
    tmp_path = f"{state_path}.tmp"
    os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_state_to_json(state), f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, state_path)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live trading loop using SB3 model (5m)")
    parser.add_argument("--symbol", type=str, default=str(getattr(TrainConfig, "SYMBOL", "BTCUSDT")))
    # 有 default 時不要設 required=True，否則即使有 default 也會被 argparse 視為必填
    parser.add_argument(
        "--model",
        type=str,
        default=str(getattr(TrainConfig, "MODEL_PATH", "models/best_model.zip")),
        help="Path to SB3 model .zip",
    )
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument("--max_position_pct", type=float, default=float(getattr(TrainConfig, "MAX_POSITION_PCT", 0.8)))
    parser.add_argument("--poll_interval_sec", type=int, default=10) # 10 seconds poll interval
    parser.add_argument("--deterministic", action="store_true")

    # ---- safety switches ----
    parser.add_argument(
        "--enable_trade_api",
        default=bool(getattr(TrainConfig, "ENABLE_TRADE_API", False)),
        action=argparse.BooleanOptionalAction,
        help="Enable real trading API calls (account/positions/orders). Default OFF for safety.",
    )
    parser.add_argument(
        "--dry_run",
        default=bool(getattr(TrainConfig, "DRY_RUN", True)),
        action=argparse.BooleanOptionalAction,
        help="If set, do not place real orders even when --enable_trade_api is ON.",
    )
    parser.add_argument(
        "--print_account_info",
        default=bool(getattr(TrainConfig, "PRINT_ACCOUNT_INFO", False)),
        action=argparse.BooleanOptionalAction,
        help="Debug: print account summary each tick (only when --enable_trade_api).",
    )
    parser.add_argument(
        "--print_open_positions",
        default=bool(getattr(TrainConfig, "PRINT_OPEN_POSITIONS", False)),
        action=argparse.BooleanOptionalAction,
        help="Debug: print open positions each tick (only when --enable_trade_api).",
    )
    parser.add_argument(
        "--print_obs_account_context",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Debug: print obs values for account_state/cost_state each tick.",
    )
    parser.add_argument("--default_min_notional_usdt", type=float, default=10.0)
    parser.add_argument(
        "--feature_symbols_csv",
        type=str,
        default="",
        help="Comma-separated feature symbols override. Empty = use TrainConfig.FEATURE_SYMBOLS.",
    )
    parser.add_argument(
        "--window_size_5m",
        type=int,
        default=0,
        help="Override live obs 5m window size. 0 = use default source.",
    )
    parser.add_argument(
        "--window_size_1d",
        type=int,
        default=0,
        help="Override live obs 1d window size. 0 = use default source.",
    )
    parser.add_argument(
        "--state_path",
        type=str,
        default=os.path.join(_PROJECT_ROOT, ".live_runner_state.json"),
        help="Path to persisted state file (prevents duplicate processing after restart).",
    )
    parser.add_argument(
        "--record_dir",
        type=str,
        default="",
        help="If set, write tick outputs as JSONL into this directory (safe to stop/restart).",
    )
    parser.add_argument(
        "--state_csv_path",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "logs", "live_state_history.csv"),
        help="Append-only CSV path for non-market_state live states.",
    )
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open one refresh window for live chart/PNL.",
    )
    parser.add_argument(
        "--render_max_visible_bars",
        type=int,
        default=500,
        help="Max closed 5m bars shown on live price chart (tail of session buffer).",
    )
    parser.add_argument(
        "--render_max_kline_buffer_rows",
        type=int,
        default=2000,
        help="Max rows persisted in session K-line buffer (state JSON).",
    )
    parser.add_argument(
        "--render_pos_delta_eps",
        type=float,
        default=0.02,
        help="Min |Δfinal_pos_pct| to annotate B/S on the decision bar.",
    )

    # ---- mode ----
    #mode = parser.add_mutually_exclusive_group(required=True)
    #mode.add_argument("--once", action="store_true", help="Run a single tick and exit.")
    #mode.add_argument("--loop", action="store_true", help="Loop forever, processing each new 5m closed bar once.")
    parser.add_argument("--once", action="store_true", default=bool(getattr(TrainConfig, "ONCE", True)), help="Run a single tick and exit.")
    parser.add_argument("--loop", action="store_true", default=bool(getattr(TrainConfig, "LOOP", False)), help="Loop forever, processing each new 5m closed bar once.")
    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace) -> LiveRunnerConfig:
    # 支援使用者直接 `python LiveTradingRunner/live_trading_loop.py` 從任意 cwd 執行：
    # - model_path 若為相對路徑，改以專案根為基準
    model_path = str(args.model)
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(os.path.join(_PROJECT_ROOT, model_path))
    if str(getattr(args, "feature_symbols_csv", "")).strip():
        feature_symbols = tuple(
            part.strip().upper()
            for part in str(args.feature_symbols_csv).split(",")
            if part.strip()
        )
    else:
        feature_symbols = tuple(getattr(TrainConfig, "FEATURE_SYMBOLS", (str(args.symbol),)))

    default_w5m = int(getattr(TrainConfig, "WINDOW_SIZE_5M", LiveRunnerEnvConfig.WINDOW_SIZE_5M_DEFAULT))
    default_w1d = int(getattr(TrainConfig, "WINDOW_SIZE_1D", LiveRunnerEnvConfig.WINDOW_SIZE_1D_DEFAULT))
    w5m = int(getattr(args, "window_size_5m", 0) or 0)
    w1d = int(getattr(args, "window_size_1d", 0) or 0)

    return LiveRunnerConfig(
        symbol=str(args.symbol),
        model_path=str(model_path),
        leverage=int(args.leverage),
        max_position_pct=float(args.max_position_pct),
        poll_interval_sec=int(args.poll_interval_sec),
        deterministic=bool(args.deterministic),
        enable_trade_api=bool(args.enable_trade_api),
        dry_run=bool(args.dry_run),
        print_account_info=bool(getattr(args, "print_account_info", True)),
        print_open_positions=bool(getattr(args, "print_open_positions", True)),
        print_obs_account_context=bool(getattr(args, "print_obs_account_context", False)),
        default_min_notional_usdt=float(args.default_min_notional_usdt),
        feature_symbols=feature_symbols,
        window_size_5m=int(w5m if w5m > 0 else default_w5m),
        window_size_1d=int(w1d if w1d > 0 else default_w1d),
        # align with Train/run_sac_lag.py wrappers & env_kwargs
        action_repeat=int(getattr(TrainConfig, "ACTION_REPEAT", LiveRunnerEnvConfig.ACTION_REPEAT)),
        max_step_pos_change_pct=float(
            getattr(TrainConfig, "MAX_STEP_POS_CHANGE_PCT", LiveRunnerEnvConfig.MAX_STEP_POS_CHANGE_PCT)
        ),
        min_position_change=float(
            getattr(TrainConfig, "MIN_POSITION_CHANGE", LiveRunnerEnvConfig.MIN_POSITION_CHANGE)
        ),
        no_trade_entry_threshold=float(
            getattr(TrainConfig, "NO_TRADE_ENTRY_THRESHOLD", LiveRunnerEnvConfig.NO_TRADE_ENTRY_THRESHOLD)
        ),
        no_trade_exit_threshold=float(
            getattr(TrainConfig, "NO_TRADE_EXIT_THRESHOLD", LiveRunnerEnvConfig.NO_TRADE_EXIT_THRESHOLD)
        ),
        risk_base_update_steps=int(
            getattr(TrainConfig, "WINDOW_SIZE_5M", LiveRunnerEnvConfig.RISK_BASE_UPDATE_STEPS)
        ),
        render_pos_delta_eps=float(getattr(args, "render_pos_delta_eps", 0.02)),
        render_max_visible_kline_bars=int(max(10, int(getattr(args, "render_max_visible_bars", 500)))),
        render_max_kline_buffer_rows=int(max(50, int(getattr(args, "render_max_kline_buffer_rows", 2000)))),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI 入口。"""

    args = _parse_args(argv)
    cfg = _build_config(args)

    # 延遲匯入：避免在單元測試中強制載入重型依賴
    from LiveTradingRunner.runner_core import LiveRunner

    fee_provider = BinanceFeeRateProvider(default_fee_pct=float(LiveRunnerEnvConfig.DEFAULT_TRANSACTION_FEE_PCT))
    runner = LiveRunner(cfg, fee_provider=fee_provider)
    state_path = str(getattr(args, "state_path", os.path.join(_PROJECT_ROOT, ".live_runner_state.json")))
    state = _load_state(state_path)

    record_dir = str(getattr(args, "record_dir", "")).strip()
    state_csv_path = str(getattr(args, "state_csv_path", os.path.join(_PROJECT_ROOT, "logs", "live_state_history.csv")))
    render_enabled = bool(getattr(args, "render", True))
    hist_max = int(max(100, int(cfg.render_max_visible_kline_bars) + 64))
    renderer = LiveRefreshRenderer(
        symbol=str(cfg.symbol),
        max_visible_bars=int(cfg.render_max_visible_kline_bars),
        data_dir=os.path.join(_PROJECT_ROOT, "Data"),
    ) if render_enabled else None
    equity_history: deque[float] = deque(maxlen=hist_max)
    ts_history: deque[str] = deque(maxlen=hist_max)
    position_history: deque[float] = deque(maxlen=hist_max)
    # 與 equity 同長：每步一筆，供價格圖在每根對應 K 上持續顯示 B/S 與 GATE A/C
    trade_marker_history: deque[Optional[str]] = deque(maxlen=hist_max)
    gate_labels_rows_history: deque[List[str]] = deque(maxlen=hist_max)
    gate_regime_history: deque[float] = deque(maxlen=hist_max)
    conviction_strength_history: deque[float] = deque(maxlen=hist_max)
    trend_tanh_signed_history: deque[float] = deque(maxlen=hist_max)
    gate_b_liquidity_history: deque[float] = deque(maxlen=hist_max)
    record_fp = None
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
        record_fp = open(os.path.join(record_dir, "ticks.jsonl"), "a", encoding="utf-8")

    run_once_mode = bool(args.once) and (not bool(args.loop))
    if run_once_mode:
        # 預設 argparse：--once 預設 True、--loop 預設 False → 多數人未加參數時只跑 1 次就結束（常被誤以為當掉）
        print(
            "[live_loop] 單次模式：本次跑完即結束。若要持續每根新 5m 收盤都跑，請加上參數 `--loop`。",
            flush=True,
        )
        try:
            prev_pos_pct = float(getattr(state, "last_final_pos_pct", 0.0))
            decision = runner.run_once(state=state)
            if decision is not None:
                fee_rate_pct = float(decision.fee_rate_pct) if decision.fee_rate_pct is not None else float(
                    fee_provider.get_fee_rate_percent(symbol=str(cfg.symbol))
                )
                _update_paper_equity(
                    state=state,
                    last_price=float(decision.last_price),
                    prev_final_pos_pct=prev_pos_pct,
                    new_final_pos_pct=float(decision.final_pos_pct),
                    fee_rate_pct=fee_rate_pct,
                )
                _append_state_csv(csv_path=state_csv_path, decision=decision, state=state, fee_rate_pct=fee_rate_pct)
                ts_history.append(str(decision.closed_bar_ts))
                equity_history.append(float(state.paper_equity_usdt))
                position_history.append(float(decision.final_pos_pct))
                df_plot = pd.DataFrame(state.kline_session_rows)
                if not df_plot.empty:
                    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"], errors="coerce")
                    df_plot = df_plot.dropna(subset=["timestamp"])
                sideway_mask_history = _build_sideway_mask_from_df_price(df_plot)
                gate_labels = gate_ac_change_labels(state.last_gate_flags, decision.gate_flags)
                trade_marker = trade_bs_from_delta(
                    prev_final_pos_pct=prev_pos_pct,
                    new_final_pos_pct=float(decision.final_pos_pct),
                    eps=float(cfg.render_pos_delta_eps),
                )
                trade_marker_history.append(trade_marker)
                gate_labels_rows_history.append(gate_labels)
                gate_regime_history.append(float(decision.regime_indicator))
                conviction_strength_history.append(float(decision.conviction_strength))
                trend_tanh_signed_history.append(float(decision.trend_tanh_signed))
                _gb = (
                    float(decision.gate_flags[1])
                    if decision.gate_flags is not None and len(decision.gate_flags) >= 2
                    else 0.0
                )
                gate_b_liquidity_history.append(_gb)
                if renderer is not None:
                    renderer.refresh(
                        closed_bar_ts=str(decision.closed_bar_ts),
                        paper_equity_usdt=float(state.paper_equity_usdt),
                        paper_profit_usdt=float(state.paper_equity_usdt - 1000.0),
                        fee_rate_pct=fee_rate_pct,
                        final_pos_pct=float(decision.final_pos_pct),
                        equity_history=list(equity_history),
                        ts_history=list(ts_history),
                        df_price=df_plot,
                        position_history=list(position_history),
                        trade_marker_history=list(trade_marker_history),
                        gate_labels_rows_history=list(gate_labels_rows_history),
                        max_position_pct=float(cfg.max_position_pct),
                        gate_regime_history=list(gate_regime_history),
                        conviction_strength_history=list(conviction_strength_history),
                        trend_tanh_signed_history=list(trend_tanh_signed_history),
                        gate_b_liquidity_history=list(gate_b_liquidity_history),
                        sideway_mask_history=sideway_mask_history,
                    )
                if decision.gate_flags is not None:
                    state.last_gate_flags = decision.gate_flags
                print(decision.__dict__)
                try:
                    _save_state(state_path, state)
                except OSError:
                    pass
                if record_fp is not None:
                    record_fp.write(json.dumps(_tick_decision_json_payload(decision), ensure_ascii=False) + "\n")
                    record_fp.flush()
                print("[live_loop] 單次模式：tick 完成，正常結束。", flush=True)
            else:
                print(
                    "[live_loop] 單次模式：無新收盤 5m bar（與 state 中 last_processed 相同），"
                    "未產生 decision；正常結束。",
                    flush=True,
                )
        except Exception as exc:
            print(f"[live_loop] 單次模式錯誤（請看以下訊息或 traceback）: {exc}", flush=True)
            raise
        finally:
            if record_fp is not None:
                record_fp.close()
        return

    print("[live_loop] 循環模式：持續輪詢（Ctrl+C 結束）。", flush=True)
    startup_force = True
    while True:
        try:
            prev_pos_pct = float(getattr(state, "last_final_pos_pct", 0.0))
            decision = runner.run_once(state=state, force=startup_force)
            startup_force = False
            if decision is not None:
                fee_rate_pct = float(decision.fee_rate_pct) if decision.fee_rate_pct is not None else float(
                    fee_provider.get_fee_rate_percent(symbol=str(cfg.symbol))
                )
                _update_paper_equity(
                    state=state,
                    last_price=float(decision.last_price),
                    prev_final_pos_pct=prev_pos_pct,
                    new_final_pos_pct=float(decision.final_pos_pct),
                    fee_rate_pct=fee_rate_pct,
                )
                _append_state_csv(csv_path=state_csv_path, decision=decision, state=state, fee_rate_pct=fee_rate_pct)
                ts_history.append(str(decision.closed_bar_ts))
                equity_history.append(float(state.paper_equity_usdt))
                position_history.append(float(decision.final_pos_pct))
                df_plot = pd.DataFrame(state.kline_session_rows)
                if not df_plot.empty:
                    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"], errors="coerce")
                    df_plot = df_plot.dropna(subset=["timestamp"])
                sideway_mask_history = _build_sideway_mask_from_df_price(df_plot)
                gate_labels = gate_ac_change_labels(state.last_gate_flags, decision.gate_flags)
                trade_marker = trade_bs_from_delta(
                    prev_final_pos_pct=prev_pos_pct,
                    new_final_pos_pct=float(decision.final_pos_pct),
                    eps=float(cfg.render_pos_delta_eps),
                )
                trade_marker_history.append(trade_marker)
                gate_labels_rows_history.append(gate_labels)
                gate_regime_history.append(float(decision.regime_indicator))
                conviction_strength_history.append(float(decision.conviction_strength))
                trend_tanh_signed_history.append(float(decision.trend_tanh_signed))
                _gb = (
                    float(decision.gate_flags[1])
                    if decision.gate_flags is not None and len(decision.gate_flags) >= 2
                    else 0.0
                )
                gate_b_liquidity_history.append(_gb)
                if renderer is not None:
                    renderer.refresh(
                        closed_bar_ts=str(decision.closed_bar_ts),
                        paper_equity_usdt=float(state.paper_equity_usdt),
                        paper_profit_usdt=float(state.paper_equity_usdt - 1000.0),
                        fee_rate_pct=fee_rate_pct,
                        final_pos_pct=float(decision.final_pos_pct),
                        equity_history=list(equity_history),
                        ts_history=list(ts_history),
                        df_price=df_plot,
                        position_history=list(position_history),
                        trade_marker_history=list(trade_marker_history),
                        gate_labels_rows_history=list(gate_labels_rows_history),
                        max_position_pct=float(cfg.max_position_pct),
                        gate_regime_history=list(gate_regime_history),
                        conviction_strength_history=list(conviction_strength_history),
                        trend_tanh_signed_history=list(trend_tanh_signed_history),
                        gate_b_liquidity_history=list(gate_b_liquidity_history),
                        sideway_mask_history=sideway_mask_history,
                    )
                if decision.gate_flags is not None:
                    state.last_gate_flags = decision.gate_flags
                print(decision.__dict__)
                try:
                    _save_state(state_path, state)
                except OSError:
                    pass
                if record_fp is not None:
                    record_fp.write(json.dumps(_tick_decision_json_payload(decision), ensure_ascii=False) + "\n")
                    record_fp.flush()
            else:
                # 無新 bar：只維持視窗事件循環，不重畫圖，避免閃爍
                if renderer is not None:
                    renderer.idle()
            _sleep_with_idle(total_seconds=float(cfg.poll_interval_sec), renderer=renderer)
        except KeyboardInterrupt:
            print("使用者中斷，結束。")
            if record_fp is not None:
                record_fp.close()
            return
        except Exception as exc:
            # 網路/DNS 等暫時性錯誤不應讓 loop 終止；等待後自動重試
            print(f"[live_loop] transient error: {exc}", flush=True)
            if renderer is not None:
                renderer.idle()
            _sleep_with_idle(total_seconds=float(cfg.poll_interval_sec), renderer=renderer)


if __name__ == "__main__":
    main()


