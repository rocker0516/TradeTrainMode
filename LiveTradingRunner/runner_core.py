"""Core implementation for LiveTradingRunner.

此模組刻意把「可測試的核心流程」與 argparse/CLI 分離，讓 pytest 可以用 mock
驗證交易 API 的呼叫序列與參數。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from pprint import pformat

from ApiTrading import ITradingClient, build_trading_client
from Env.Components.action_processor import ActionProcessor
from Env.Executors.trade_executor import TradeExecutor
from Env.config import Config
from LiveTradingRunner.binance_market_data import LatestBarsResult, fetch_latest_multi_symbol_5m
from LiveTradingRunner.live_trading_loop import LiveRunnerConfig, LiveRunnerState, TickDecision
from LiveTradingRunner.local_1d_loader import load_local_1d_data
from LiveTradingRunner.obs_builder import LiveObsBuilder
from LiveTradingRunner.position_sizer import compute_delta_position_qty


def _build_account_and_context_obs_named(obs: dict) -> Dict[str, Dict[str, float]]:
    """把 account/context 向量轉成「欄位名稱 -> 數值」。

    說明：
    - 欄位順序完全對齊 `Env/Components/observer.py`：
      - `TradingObserver._get_account_obs()` 的 `account_state = np.array([...])`
      - `TradingObserver._get_context_obs()` 的 cost index 定義
    """

    account_state_names = [
        "position_side",              # 0
        "position_size_norm",         # 1
        "actual_pos_pct",             # 2  執行後真實倉位比例 [-1, 1]
        "equity_ratio",               # 3
        "realized_pnl_ratio",         # 4
        "unrealized_pnl_atr",         # 5
        "drawdown",                   # 6
        "liq_distance_atr",           # 7
        "stop_loss_distance_atr",     # 8
        "margin_usage_ratio",         # 9
        "cooldown_remaining_norm",    # 10
        "fee_rate",                   # 11
        "rolling_fee_ratio",          # 12
        "trade_count_log",            # 13
        "stop_loss_count_log",        # 14
        "holding_time_log",           # 15
        "buffer_to_min_balance_ratio", # 16
        "steps_since_trade_norm",     # 17
        "trade_freq_remaining_ratio", # 18
        "trade_freq_blocked_last",    # 19
        "entry_price_ratio",          # 20
        "stop_loss_price_ratio",      # 21
        "recent_flat_ratio",          # 22（與 cost_flat 同口徑，實盤可算）
    ]
    cost_state_names = [
        "placeholder",  # 佔位欄位（cost_state 已清空）
    ]

    def _named_values(x: Any, names: list[str]) -> dict[str, float]:
        arr = np.asarray(x).reshape(-1)
        if len(arr) != len(names):
            return {
                "__error__": float("nan"),
                "__shape__": float(len(arr)),
            }
        out: dict[str, float] = {}
        for i, n in enumerate(names):
            try:
                out[str(n)] = float(arr[i])
            except (TypeError, ValueError):
                out[str(n)] = 0.0
        return out

    return {
        "account_state": _named_values(obs.get("account_state"), account_state_names),
        "cost_state": _named_values(obs.get("cost_state"), cost_state_names),
    }


def _print_account_and_context_obs(obs: dict) -> Dict[str, Dict[str, float]]:
    """印出傳進 policy 的 account/context（同時回傳 named dict，方便存檔）。"""
    named = _build_account_and_context_obs_named(obs)
    payload = {"obs_account_context_named": named}
    print(pformat(payload, width=120, sort_dicts=False), flush=True)
    return named


@dataclass
class LiveRunner:
    """Live runner 核心類別（後續 todo 會補齊實作）。"""

    config: LiveRunnerConfig

    @staticmethod
    def _spaces_from_obs(obs: dict) -> tuple[gym.Space, gym.Space]:
        """由一次實際 obs 推導 observation_space/action_space，避免讀取 zip 內的 space pickle。"""
        spaces: dict[str, gym.Space] = {}
        for k, v in obs.items():
            arr = np.asarray(v)
            spaces[str(k)] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=arr.shape,
                dtype=arr.dtype,
            )
        obs_space = gym.spaces.Dict(spaces)
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        return obs_space, act_space

    def run_once(self, *, state: LiveRunnerState) -> Optional[TickDecision]:
        """執行單次 tick。

        Returns:
            - 若本次沒有新 5m 收盤 bar（避免重複處理），回傳 None。
            - 否則回傳 TickDecision（不代表一定有下單；由 enable_trade_api/dry_run 決定）。
        """
        cfg = self.config

        # --- lazy init (avoid heavy imports in module import time) ---
        if not hasattr(self, "_df_1d"):
            # 依你規格：1d 來自本機 Data；先 cache（你之後若要 daily reload 再擴充）
            self._df_1d = load_local_1d_data()
        if not hasattr(self, "_trade_client"):
            self._trade_client = None
            self._leverage_set = False

        # 5m：抓最新資料（含 dummy row）
        lookback = int(max(288, cfg.window_size_5m))
        limit = int(min(1000, lookback + 50))
        latest: LatestBarsResult = fetch_latest_multi_symbol_5m(
            symbols=cfg.feature_symbols,
            target_symbol=cfg.symbol,
            limit=limit,
            client=None,
        )

        closed_ts = pd.Timestamp(latest.latest_closed_bar_ts)
        closed_ts64 = np.datetime64(closed_ts.to_datetime64())
        if state.last_processed_closed_ts is not None and closed_ts64 == state.last_processed_closed_ts:
            return None

        equity = None
        current_pos_qty = None
        filters = None

        if bool(cfg.enable_trade_api):
            if self._trade_client is None:
                self._trade_client = build_trading_client(testnet=False)
            tc: ITradingClient = self._trade_client
            if not bool(getattr(self, "_leverage_set", False)):
                tc.set_leverage(symbol=cfg.symbol, leverage=int(cfg.leverage))
                self._leverage_set = True

            # Debug prints（避免在未啟用 trade_api 時誤觸）
            if bool(getattr(cfg, "print_account_info", False)):
                try:
                    summary = tc.get_account_summary()
                except Exception as exc:
                    summary = {"error": str(exc)}
                print({"account_summary": summary}, flush=True)
            if bool(getattr(cfg, "print_open_positions", False)):
                try:
                    positions = tc.get_open_positions()
                except Exception as exc:
                    positions = [{"error": str(exc)}]
                print({"open_positions": positions}, flush=True)

            equity = float(tc.get_equity_usdt())
            current_pos_qty = float(tc.get_current_position_size(cfg.symbol))
            filters = tc.get_symbol_filters(symbol=cfg.symbol, default_min_notional=float(cfg.default_min_notional_usdt))

        # ---- build a lightweight executor for action processing alignment (same as Train/Eval env) ----
        # - 不負責真的下單（真下單走 ITradingClient）
        # - 只用來複用 ActionProcessor 的 max_step_pos_change / hysteresis 邏輯
        init_balance = float(equity) if equity is not None else float(getattr(Config, "INITIAL_BALANCE", 1000.0))
        exec_for_action = TradeExecutor(
            initial_balance=float(init_balance),
            fee_rate=float(getattr(Config, "TRANSACTION_FEE", 0.01)),
            leverage=float(cfg.leverage),
            min_trade_qty=0.0,  # live 端以交易所 filters 控制最小可成交單位
            maintenance_margin_rate=float(getattr(Config, "MAINTENANCE_MARGIN_RATE", 0.005)),
            margin_mode="isolated",
            stop_loss_atr=float(getattr(Config, "STOP_LOSS_ATR", 2.0)),
            stop_loss_liq_buffer_pct=float(getattr(Config, "STOP_LOSS_LIQ_BUFFER_PCT", 0.0)),
            min_position_change=float(getattr(cfg, "min_position_change", 0.0)),
        )
        exec_for_action.wallet_balance = float(init_balance)
        if current_pos_qty is not None:
            exec_for_action.position.size = float(current_pos_qty)

        # obs
        obs_builder = LiveObsBuilder(
            target_symbol=cfg.symbol,
            feature_symbols=cfg.feature_symbols,
            window_size_5m=cfg.window_size_5m,
            window_size_1d=cfg.window_size_1d,
            leverage=float(cfg.leverage),
            obs_dtype="float32",
        )
        obs_result = obs_builder.build(
            df_5m=latest.df_5m,
            df_1d=self._df_1d,
            equity_usdt=equity,
            current_position_qty=current_pos_qty,
        )
        obs_named: Optional[Dict[str, Dict[str, float]]] = None
        if bool(getattr(cfg, "print_obs_account_context", False)):
            obs_named = _print_account_and_context_obs(obs_result.obs)

        # model load：延後到拿到一次真實 obs 後，並用 obs 推導出的 space 覆蓋 zip 內的 pickle（避免 numpy 版本差異）
        if not hasattr(self, "_model"):
            obs_space, act_space = self._spaces_from_obs(obs_result.obs)
            self._model = self._load_model(cfg.model_path, observation_space=obs_space, action_space=act_space)

        # predict
        action_repeat = int(max(1, int(getattr(cfg, "action_repeat", 1))))
        should_predict = (int(getattr(state, "decision_step", 0)) % action_repeat) == 0

        action_raw = 0.0
        if should_predict:
            action, _ = self._model.predict(obs_result.obs, deterministic=bool(cfg.deterministic))
            try:
                action_raw = float(np.asarray(action).reshape(-1)[0])
            except Exception:
                action_raw = 0.0
        else:
            # 非決策 step：沿用上一個 action（近似 ActionRepeatWrapper）
            action_raw = float(getattr(state, "last_final_pos_pct", 0.0))

        # 1) ActionClipWrapper（Train/Eval 同款）：限制在 [-max_position_pct, +max_position_pct]
        action_clipped = float(np.clip(float(action_raw), -float(cfg.max_position_pct), float(cfg.max_position_pct)))

        # 2) ActionProcessor（Train/Eval 同款）：hysteresis + 單步加倉上限
        ap = ActionProcessor(
            leverage=float(cfg.leverage),
            max_step_pos_change_pct=float(getattr(cfg, "max_step_pos_change_pct", 0.0)),
            min_position_change=float(getattr(cfg, "min_position_change", 0.0)),
            no_trade_entry_threshold=float(getattr(cfg, "no_trade_entry_threshold", 0.0)),
            no_trade_exit_threshold=float(getattr(cfg, "no_trade_exit_threshold", 0.0)),
        )

        # risk_base：近似 TradingEnvironment 的 daily_risk_base（每 risk_base_update_steps 更新一次）
        rb_steps = int(max(1, int(getattr(cfg, "risk_base_update_steps", 288))))
        if getattr(state, "risk_base_usdt", None) is None:
            state.risk_base_usdt = float(init_balance)
            state.steps_since_risk_base_update = 0
        if int(getattr(state, "steps_since_risk_base_update", 0)) >= rb_steps:
            state.risk_base_usdt = float(init_balance)
            state.steps_since_risk_base_update = 0

        # env 的 process_action 介面吃 ndarray shape(1,)
        action_used = np.array([action_clipped], dtype=np.float32)
        target_pos_pct, _is_flip = ap.process_action(action_used, exec_for_action, float(latest.latest_closed_price))
        final_pos_pct = ap.calculate_effective_action(
            float(target_pos_pct),
            exec_for_action,
            float(latest.latest_closed_price),
            float(state.risk_base_usdt) if state.risk_base_usdt is not None else float(init_balance),
        )

        # 更新 state：供 action_repeat / risk_base 用
        state.last_final_pos_pct = float(final_pos_pct)
        state.decision_step = int(getattr(state, "decision_step", 0) + 1)
        state.steps_since_risk_base_update = int(getattr(state, "steps_since_risk_base_update", 0) + 1)

        target_qty = None
        delta_qty = None

        # 下單（只在 enable_trade_api 時）
        if bool(cfg.enable_trade_api) and equity is not None and current_pos_qty is not None and filters is not None:
            sizing = compute_delta_position_qty(
                equity_usdt=float(equity),
                last_price=float(latest.latest_closed_price),
                leverage=float(cfg.leverage),
                target_pos_pct=float(final_pos_pct),
                current_position_qty=float(current_pos_qty),
            )
            target_qty = float(sizing.target_position_qty)
            delta_qty = float(sizing.delta_qty)

            # 若 enable_trade_api 但 dry_run：仍走 place_delta_order，但帶 dry_run=True（不送單）
            tc2: ITradingClient = self._trade_client
            tc2.place_delta_order(
                symbol=cfg.symbol,
                delta=float(delta_qty),
                step_size=float(filters.step_size),
                min_notional=float(filters.min_notional),
                last_price=float(latest.latest_closed_price),
                dry_run=bool(cfg.dry_run),
            )

        state.last_processed_closed_ts = closed_ts64

        return TickDecision(
            closed_bar_ts=str(latest.latest_closed_bar_ts),
            last_price=float(latest.latest_closed_price),
            action_raw=float(action_raw),
            action_clipped=float(action_clipped),
            target_pos_pct=float(target_pos_pct),
            final_pos_pct=float(final_pos_pct),
            equity_usdt=float(equity) if equity is not None else None,
            current_position_qty=float(current_pos_qty) if current_pos_qty is not None else None,
            target_position_qty=float(target_qty) if target_qty is not None else None,
            delta_qty=float(delta_qty) if delta_qty is not None else None,
            obs_account_context_named=obs_named,
        )

    @staticmethod
    def _load_model(model_path: str, *, observation_space: gym.Space, action_space: gym.Space) -> Any:
        """載入 SB3 SAC 模型（lazy import）。"""
        # ---- numpy pickle compatibility shim ----
        # 你的 best_model.zip 可能是在不同 numpy 版本下儲存，cloudpickle 反序列化時會引用 numpy 內部模組路徑。
        # 常見情況：模型儲存於 numpy>=2（引用 numpy._core.*），但執行環境是 numpy<2（只有 numpy.core.*）。
        # 這裡在載入前補 sys.modules alias，避免 ModuleNotFoundError。
        import sys
        import types

        try:
            import numpy._core.numeric  # type: ignore  # noqa: F401
        except ModuleNotFoundError:
            try:
                import numpy.core.numeric as _numeric  # type: ignore

                # ensure package module exists
                if "numpy._core" not in sys.modules:
                    sys.modules["numpy._core"] = types.ModuleType("numpy._core")
                sys.modules["numpy._core.numeric"] = _numeric
            except ModuleNotFoundError:
                # 若連 numpy.core.numeric 都不存在，讓後續 SAC.load 正常拋錯（不在這裡吞掉）
                pass

        from stable_baselines3 import SAC

        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"

        # 只做推論時，不需要 zip 內那些「訓練續跑狀態」（_last_obs/rng/buffers）。
        # 直接用 custom_objects 覆蓋，避免 numpy/gym 的 pickle 版本不相容。
        custom_objects = {
            "_last_obs": None,
            "_last_original_obs": None,
            "_last_episode_starts": None,
            "ep_info_buffer": None,
            "ep_success_buffer": None,
            # 若 lr_schedule 反序列化失敗，推論也用不到；給一個最簡單可呼叫函式
            "lr_schedule": (lambda _progress: 0.0),
            # 用我們從 obs 推導出的 spaces 覆蓋（避免 gymnasium space 內含 np_random pickle）
            "observation_space": observation_space,
            "action_space": action_space,
        }
        return SAC.load(str(model_path), device=device, custom_objects=custom_objects)


