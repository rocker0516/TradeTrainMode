"""Live trading loop (every 5 minutes) using SB3 best_model.zip.

設計重點：
- 交易 API 安全開關：預設不呼叫交易 API（不下單）；必須顯式 `--enable_trade_api`。
- 市場資料：使用 Binance 公開端點抓取 5m K 線（不需金鑰）。
- 1d 特徵：使用本機 `Data/*_1d.csv`。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

# 確保可用 `python -m LiveTradingRunner.live_trading_loop` 直接執行
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Env.config import Config
from Train.train_config import TrainConfig


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


def _state_to_json(state: LiveRunnerState) -> Dict[str, Any]:
    """將 LiveRunnerState 轉為可 JSON 序列化 dict。"""
    ts = state.last_processed_closed_ts
    return {
        "last_processed_closed_ts": str(ts) if ts is not None else None,
        "decision_step": int(getattr(state, "decision_step", 0)),
        "risk_base_usdt": float(state.risk_base_usdt) if state.risk_base_usdt is not None else None,
        "steps_since_risk_base_update": int(getattr(state, "steps_since_risk_base_update", 0)),
        "last_final_pos_pct": float(getattr(state, "last_final_pos_pct", 0.0)),
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

    if raw is None:
        return LiveRunnerState(
            last_processed_closed_ts=None,
            decision_step=decision_step,
            risk_base_usdt=risk_base_usdt_f,
            steps_since_risk_base_update=steps_since_risk,
            last_final_pos_pct=last_final_pos_pct,
        )
    try:
        return LiveRunnerState(
            last_processed_closed_ts=np.datetime64(str(raw)),
            decision_step=decision_step,
            risk_base_usdt=risk_base_usdt_f,
            steps_since_risk_base_update=steps_since_risk,
            last_final_pos_pct=last_final_pos_pct,
        )
    except (TypeError, ValueError):
        return LiveRunnerState(
            last_processed_closed_ts=None,
            decision_step=decision_step,
            risk_base_usdt=risk_base_usdt_f,
            steps_since_risk_base_update=steps_since_risk,
            last_final_pos_pct=last_final_pos_pct,
        )


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
        default=str(getattr(TrainConfig, "MODEL_PATH", "models/sac_lag_BTCUSDT/best_model/best_model.zip")),
        help="Path to SB3 model .zip",
    )
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument("--max_position_pct", type=float, default=float(getattr(TrainConfig, "MAX_POSITION_PCT", 0.7)))
    parser.add_argument("--poll_interval_sec", type=int, default=10) # 10 seconds poll interval
    parser.add_argument("--deterministic", action="store_true")

    # ---- safety switches ----
    parser.add_argument(
        "--enable_trade_api",
        default=bool(getattr(TrainConfig, "ENABLE_TRADE_API", True)),
        action="store_true",
        help="Enable real trading API calls (account/positions/orders). Default OFF for safety.",
    )
    parser.add_argument(
        "--dry_run",
        default=bool(getattr(TrainConfig, "DRY_RUN", True)),
        action="store_true",
        help="If set, do not place real orders even when --enable_trade_api is ON.",
    )
    parser.add_argument(
        "--print_account_info",
        default=bool(getattr(TrainConfig, "PRINT_ACCOUNT_INFO", True)),
        action="store_true",
        help="Debug: print account summary each tick (only when --enable_trade_api).",
    )
    parser.add_argument(
        "--print_open_positions",
        default=bool(getattr(TrainConfig, "PRINT_OPEN_POSITIONS", True)),
        action="store_true",
        help="Debug: print open positions each tick (only when --enable_trade_api).",
    )
    parser.add_argument(
        "--print_obs_account_context",
        default=True,
        action="store_true",
        help="Debug: print obs values for account_state/cost_state each tick.",
    )
    parser.add_argument("--default_min_notional_usdt", type=float, default=10.0)
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
        feature_symbols=tuple(getattr(TrainConfig, "FEATURE_SYMBOLS", (str(args.symbol),))),
        window_size_5m=int(getattr(TrainConfig, "WINDOW_SIZE_5M", 288)),
        window_size_1d=int(getattr(TrainConfig, "WINDOW_SIZE_1D", 14)),
        # align with Train/run_sac_lag.py wrappers & env_kwargs
        action_repeat=int(getattr(TrainConfig, "ACTION_REPEAT", 1)),
        max_step_pos_change_pct=float(getattr(Config, "MAX_STEP_POS_CHANGE_PCT", 0.0)),
        min_position_change=float(getattr(TrainConfig, "MIN_POSITION_CHANGE", getattr(Config, "MIN_POSITION_CHANGE", 0.0))),
        no_trade_entry_threshold=float(getattr(TrainConfig, "NO_TRADE_ENTRY_THRESHOLD", getattr(Config, "NO_TRADE_ENTRY_THRESHOLD", 0.0))),
        no_trade_exit_threshold=float(getattr(TrainConfig, "NO_TRADE_EXIT_THRESHOLD", getattr(Config, "NO_TRADE_EXIT_THRESHOLD", 0.0))),
        risk_base_update_steps=int(getattr(TrainConfig, "WINDOW_SIZE_5M", getattr(Config, "RISK_BASE_UPDATE_STEPS", 288))),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI 入口。"""

    args = _parse_args(argv)
    cfg = _build_config(args)

    # 延遲匯入：避免在單元測試中強制載入重型依賴
    from LiveTradingRunner.runner_core import LiveRunner

    runner = LiveRunner(cfg)
    state_path = str(getattr(args, "state_path", os.path.join(_PROJECT_ROOT, ".live_runner_state.json")))
    state = _load_state(state_path)

    record_dir = str(getattr(args, "record_dir", "")).strip()
    record_fp = None
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
        record_fp = open(os.path.join(record_dir, "ticks.jsonl"), "a", encoding="utf-8")

    if bool(args.once):
        decision = runner.run_once(state=state)
        if decision is not None:
            print(decision.__dict__)
            try:
                _save_state(state_path, state)
            except OSError:
                pass
            if record_fp is not None:
                record_fp.write(json.dumps(decision.__dict__, ensure_ascii=False) + "\n")
                record_fp.flush()
        if record_fp is not None:
            record_fp.close()
        return

    while True:
        try:
            decision = runner.run_once(state=state)
            if decision is not None:
                print(decision.__dict__)
                try:
                    _save_state(state_path, state)
                except OSError:
                    pass
                if record_fp is not None:
                    record_fp.write(json.dumps(decision.__dict__, ensure_ascii=False) + "\n")
                    record_fp.flush()
            time.sleep(cfg.poll_interval_sec)
        except KeyboardInterrupt:
            print("使用者中斷，結束。")
            if record_fp is not None:
                record_fp.close()
            return


if __name__ == "__main__":
    main()


