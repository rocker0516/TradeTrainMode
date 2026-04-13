"""Build Env-compatible Dict observation for live inference.

本模組的責任（SRP）：
- 把 live 資料（df_5m/df_1d + 交易帳戶狀態）轉成與訓練環境一致的 observation dict。

重要：
- observation key/shape/dtype 必須與訓練時完全一致（SB3 MultiInputPolicy）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from Env.config import Config
from Env.Components.market_data import MarketData
from Env.Components.observer import TradingObserver
from Env.Executors.trade_executor import TradeExecutor

from LiveTradingRunner.live_obs_align import default_last_action_effects
from LiveTradingRunner.render_history_utils import (
    conviction_strength_from_trend_score,
    gate_flags_to_regime_indicator,
    trend_tanh_signed_from_trend_score,
)


@dataclass(frozen=True)
class LiveObsBuildResult:
    """Observation 與 meta。"""

    obs: Dict[str, np.ndarray]
    step_idx: int
    current_price: float
    # 與訓練 reward 對齊，供 Live 圖表 GATE/conviction 子圖使用
    regime_indicator: float = 0.0
    conviction_strength: float = 0.0
    trend_score: float = 0.0
    # 與 conviction 同源，保留符號供圖表顯示 5m 細微波動（reward 仍用 abs）
    trend_tanh_signed: float = 0.0


class LiveObsBuilder:
    """Live observation builder."""

    def __init__(
        self,
        *,
        target_symbol: str,
        feature_symbols: Tuple[str, ...],
        window_size_5m: int,
        window_size_1d: int,
        leverage: float,
        obs_dtype: str = "float32",
        fee_rate_percent: Optional[float] = None,
    ) -> None:
        self.target_symbol = str(target_symbol)
        self.feature_symbols = tuple(feature_symbols)
        self.window_size_5m = int(window_size_5m)
        self.window_size_1d = int(window_size_1d)
        self.leverage = float(leverage)
        self.obs_dtype = str(obs_dtype)
        # 與 Config.TRANSACTION_FEE 同單位（百分比，0.04 = 0.04%）；None 時沿用 Config（訓練對齊）
        self._fee_rate_percent = None if fee_rate_percent is None else float(fee_rate_percent)

    def build(
        self,
        *,
        df_5m: pd.DataFrame,
        df_1d: pd.DataFrame,
        equity_usdt: Optional[float],
        current_position_qty: Optional[float],
        last_action_effects: Optional[Dict[str, float]] = None,
        account_metrics: Optional[Dict[str, Any]] = None,
    ) -> LiveObsBuildResult:
        """建立與 Env 相容的 Dict observation。

        Args:
            df_5m: 已經是 multi-symbol wide table（含 `{symbol}_close` 等欄位）且最後一列為 dummy row。
            df_1d: 本機 1d wide table（至少含 timestamp）。
            equity_usdt: 若 trading API 開啟，建議填入真實帳戶 equity；否則可為 None。
            current_position_qty: 若 trading API 開啟，填入目前持倉數量（>0 long, <0 short）；否則 None。
            last_action_effects: 上一決策步的執行摘要（對齊 env 的 `_last_action_effects`）；None 則全零重置。
            account_metrics: 與 `TradingEnvironment._get_observation` 相同 key；None 則使用向後相容的簡化預設。
        """
        if df_5m.empty:
            raise ValueError("df_5m is empty")
        if df_1d.empty:
            raise ValueError("df_1d is empty")

        # step_idx 指向最後一列（dummy row），讓 price_seq 能包含最後一根已收盤 bar
        step_idx = int(len(df_5m) - 1)

        market_data = MarketData(
            df_5m=df_5m,
            df_1d=df_1d,
            window_size=self.window_size_5m,
            window_size_1d=self.window_size_1d,
            target_symbol=self.target_symbol,
            feature_symbols=list(self.feature_symbols),
        )
        observer = TradingObserver(
            self.window_size_5m,
            self.window_size_1d,
            market_data,
            obs_dtype=self.obs_dtype,
        )

        # executor：用 Env 內同款邏輯生成 account/cost/risk 特徵
        init_balance = float(equity_usdt) if equity_usdt is not None else float(Config.INITIAL_BALANCE)
        fee_for_obs = (
            float(self._fee_rate_percent)
            if self._fee_rate_percent is not None
            else float(getattr(Config, "TRANSACTION_FEE", 0.01))
        )
        executor = TradeExecutor(
            initial_balance=init_balance,
            fee_rate=fee_for_obs,
            leverage=float(self.leverage),
            min_trade_qty=0.001,
            maintenance_margin_rate=float(getattr(Config, "MAINTENANCE_MARGIN_RATE", 0.005)),
            margin_mode="isolated",
            stop_loss_atr=float(getattr(Config, "STOP_LOSS_ATR", 2.0)),
            stop_loss_liq_buffer_pct=float(getattr(Config, "STOP_LOSS_LIQ_BUFFER_PCT", 0.0)),
            min_position_change=0.0,
        )

        # 將 live 帳戶狀態灌進 executor（最小可用版本）
        if equity_usdt is not None:
            executor.wallet_balance = float(equity_usdt)
        if current_position_qty is not None:
            executor.position.size = float(current_position_qty)

        # 取當下價格（注意：MarketData 會 clip index，因此 step_idx 對 dummy row 也安全）
        metrics = market_data.get_market_metrics(step_idx)
        current_price = float(metrics["close"])
        if current_price <= 0.0:
            raise ValueError("current_price must be > 0")

        # entry_price 用 current_price 近似（TODO：若要更準確可從交易所拿 entryPrice）
        if abs(float(executor.position.size)) > 1e-12 and executor.position.entry_price <= 0.0:
            executor.position.entry_price = float(current_price)
        if abs(float(executor.position.size)) > 1e-12 and float(executor.position.entry_price) > 0.0:
            executor.used_margin = float(
                max(0.0, abs(float(executor.position.size)) * float(executor.position.entry_price) / float(self.leverage))
            )

        atr_ratio = float(metrics.get("atr_ratio", 0.0))
        atr_est = float(atr_ratio) * float(current_price)

        risk_signals = observer.compute_risk_signals(
            executor=executor,
            current_price=current_price,
            atr_est=atr_est,
            step_idx=step_idx,
            total_steps=len(df_5m),
        )

        if account_metrics is not None:
            am = dict(account_metrics)
        else:
            am = {
                "initial_balance": init_balance,
                "max_equity_so_far": init_balance,
                "episode_stop_loss_count": 0,
                "episode_liq_count": 0,
                "risk_budget": 1.0,
                "steps_since_trade": float(self.window_size_5m),
                "holding_steps": 0.0,
                "last_step_fee": 0.0,
                "rolling_fee_sum": 0.0,
                "cooldown_remaining": 0.0,
                "min_balance": float(getattr(Config, "MIN_BALANCE", init_balance * 0.6)),
                "episode_steps": 0,
                "episode_max_steps": int(getattr(Config, "MAX_EPISODE_STEPS", 1)),
                "trade_freq_remaining_ratio": 1.0,
                "trade_freq_blocked_last": 0.0,
                "recent_flat_ratio": 0.5,
            }

        lae = default_last_action_effects() if last_action_effects is None else dict(last_action_effects)

        obs = observer.get_observation(
            step_idx=step_idx,
            executor=executor,
            market_data=market_data,
            account_metrics=am,
            risk_signals=risk_signals,
            last_action_effects=lae,
        )

        trend_score_live = float(metrics.get("trend_score", 0.0))
        gf_live = market_data.get_gate_flags(step_idx)
        regime_ind = float(gate_flags_to_regime_indicator(gf_live))
        scale_live = float(getattr(Config, "CONVICTION_TREND_SCORE_SCALE", 10.0))
        conv_strength = float(conviction_strength_from_trend_score(trend_score_live, scale=scale_live))
        tanh_signed = float(trend_tanh_signed_from_trend_score(trend_score_live, scale=scale_live))

        return LiveObsBuildResult(
            obs=obs,
            step_idx=step_idx,
            current_price=current_price,
            regime_indicator=regime_ind,
            conviction_strength=conv_strength,
            trend_score=trend_score_live,
            trend_tanh_signed=tanh_signed,
        )


