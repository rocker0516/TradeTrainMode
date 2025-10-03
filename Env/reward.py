from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RewardNormalizationConfig:
    """獎勵正規化設定：將原始獎勵組件縮放至穩定範圍。

    參數說明：
        pnl_scale: 對 PnL 比率進行 tanh 前的縮放倍數。
        entry_scale: 對進場質量分數進行裁剪前的縮放倍數。
        stop_scale: 對止盈/止損分數進行裁剪前的縮放倍數。
        holding_scale: 對持倉質量分數進行裁剪前的縮放倍數。
        penalty_scale: 對交易懲罰分數進行裁剪前的縮放倍數。
        clip_range: 最終合成獎勵的裁剪範圍上限（對稱）。
    """
    pnl_scale: float = 20.0
    entry_scale: float = 100.0
    stop_scale: float = 500.0
    holding_scale: float = 10.0
    penalty_scale: float = 1000.0
    clip_range: float = 1.0


@dataclass(frozen=True)
class RewardWeights:
    """各獎勵組件的加權係數。"""
    pnl: float = 0.6
    entry: float = 0.06
    stop: float = 0.06
    holding: float = 0.04
    penalty: float = 0.04
    risk_management: float = 0.2


@dataclass
class RewardSignals:
    """原始獎勵訊號容器（正規化前）。

    所有欄位應由環境在當前步驟的狀態與前一個動作下計算得到。

    欄位說明：
        pnl_ratio: 平倉時計算之實現損益占初始資金的比例。
        entry_score: 進場時機質量分數。
        stop_score: 止盈/止損執行分數。
        holding_score: 持倉管理質量分數。
        penalty_score: 交易懲罰/頻率分數；負值為懲罰，正值為小額獎勵。
        risk_score: 風險管理分數；此分數已為最終尺度（不再正規化）。
    """
    pnl_ratio: float = 0.0
    entry_score: float = 0.0
    stop_score: float = 0.0
    holding_score: float = 0.0
    penalty_score: float = 0.0
    risk_score: float = 0.0


class RewardNormalizer:
    """將原始獎勵訊號以確定性規則正規化至穩定範圍。"""

    def __init__(self, config: RewardNormalizationConfig | None = None) -> None:
        self._config = config or RewardNormalizationConfig()

    def normalize_pnl(self, pnl_ratio: float) -> float:
        scaled = pnl_ratio * self._config.pnl_scale
        return float(np.tanh(scaled))

    def normalize_entry(self, entry_score: float) -> float:
        scaled = entry_score * self._config.entry_scale
        return float(np.clip(scaled, -0.5, 0.5))

    def normalize_stop(self, stop_score: float) -> float:
        scaled = stop_score * self._config.stop_scale
        return float(np.clip(scaled, 0.0, 1.0))

    def normalize_holding(self, holding_score: float) -> float:
        scaled = holding_score * self._config.holding_scale
        return float(np.clip(scaled, -0.3, 0.3))

    def normalize_penalty(self, penalty_score: float) -> float:
        scaled = penalty_score * self._config.penalty_scale
        return float(np.clip(scaled, -0.5, 0.0))

    @property
    def clip_range(self) -> float:
        return self._config.clip_range


class BaseRewardCalculator:
    """從各獎勵組件訊號組合出最終的標量獎勵。

    將正規化、加權與最終裁剪的邏輯封裝於此。環境負責計算原始訊號，並委派至本類進行合成。
    """

    def __init__(
        self,
        weights: RewardWeights | None = None,
        normalizer: RewardNormalizer | None = None,
    ) -> None:
        self._weights = weights or RewardWeights()
        self._normalizer = normalizer or RewardNormalizer()

    def calculate(self, signals: RewardSignals) -> float:
        """根據原始獎勵訊號計算最終獎勵。

        參數：
            signals: 當前步驟的原始獎勵組件訊號。

        回傳：
            依設定範圍裁剪後之最終標量獎勵值。
        """
        pnl_term = self._normalizer.normalize_pnl(signals.pnl_ratio)
        entry_term = self._normalizer.normalize_entry(signals.entry_score)
        stop_term = self._normalizer.normalize_stop(signals.stop_score)
        holding_term = self._normalizer.normalize_holding(signals.holding_score)

        # 懲罰項維持原始「非對稱」特性：
        #  - 負值：縮放後再裁剪（懲罰下限）
        #  - 正值：給予輕微獎勵且上限封頂
        if signals.penalty_score < 0.0:
            penalty_term = self._normalizer.normalize_penalty(signals.penalty_score)
        else:
            penalty_term = float(np.clip(signals.penalty_score * 50.0, 0.0, 0.1))

        risk_term = float(signals.risk_score)

        total = 0.0
        total += pnl_term * self._weights.pnl
        total += entry_term * self._weights.entry
        total += stop_term * self._weights.stop
        total += holding_term * self._weights.holding
        total += penalty_term * self._weights.penalty
        total += risk_term * self._weights.risk_management

        total = float(np.clip(total, -self._normalizer.clip_range, self._normalizer.clip_range))
        return total


@dataclass
class RewardContext:
    """計算獎勵所需的環境上下文（由環境在當步提供）。

    欄位說明：
        df: 歷史資料（需至少包含 'close' 與 'volume' 欄位）。
        current_step: 當前步索引。
        initial_balance: 初始資金。
        transaction_fee: 交易手續費比例。
        leverage: 槓桿倍數。
        min_balance: 終止門檻資金。
        total_value: 當前總資產。
        balance: 可用資金（未含未實現損益）。
        btc_held: 當前持倉數量，正多負空。
        avg_entry_price: 持倉平均進場價格。
        position_holding_time: 目前持倉持有時間步數。
        last_position: 上一步持倉數量。
        stop_triggered_this_step: 若本步觸發止盈或止損，則為 'take_profit' 或 'stop_loss'，否則 None。
        closed_trades: 已完成交易紀錄（需含 'pnl' 欄位）。
    """

    df: pd.DataFrame
    current_step: int
    initial_balance: float
    transaction_fee: float
    leverage: float
    min_balance: float
    total_value: float
    balance: float
    btc_held: float
    avg_entry_price: float
    position_holding_time: int
    last_position: float
    stop_triggered_this_step: Optional[str]
    closed_trades: List[Dict]


class RewardCalculator(BaseRewardCalculator):  # type: ignore[misc]
    """擴充：提供從上下文直接計算最終獎勵的介面。"""

    # ---- 以下為環境內原有的輔助邏輯轉移至此處 ----
    @staticmethod
    def _just_opened_position(ctx: RewardContext) -> bool:
        return ctx.last_position == 0 and ctx.btc_held != 0

    @staticmethod
    def _position_closed_this_step(ctx: RewardContext) -> bool:
        return ctx.last_position != 0 and ctx.btc_held == 0

    @staticmethod
    def _calculate_closed_trade_pnl(ctx: RewardContext) -> float:
        if not ctx.closed_trades:
            return 0.0
        return float(ctx.closed_trades[-1].get('pnl', 0.0))

    @staticmethod
    def _calculate_short_term_momentum(ctx: RewardContext, periods: int = 5) -> float:
        if ctx.current_step < periods:
            return 0.0
        recent = ctx.df.iloc[ctx.current_step - periods:ctx.current_step]['close']
        if len(recent) < 2:
            return 0.0
        return float((recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0])

    @staticmethod
    def _check_volume_confirmation(ctx: RewardContext, multiplier: float = 1.2) -> bool:
        if ctx.current_step < 10:
            return False
        current_volume = ctx.df.iloc[ctx.current_step]['volume']
        avg_volume = ctx.df.iloc[ctx.current_step - 10:ctx.current_step]['volume'].mean()
        return bool(current_volume > avg_volume * multiplier)

    @classmethod
    def _evaluate_entry_timing(cls, ctx: RewardContext) -> float:
        if not cls._just_opened_position(ctx):
            return 0.0
        entry_score = 0.0
        momentum = cls._calculate_short_term_momentum(ctx)
        volume_confirm = cls._check_volume_confirmation(ctx)
        if (ctx.btc_held > 0 and momentum > 0) or (ctx.btc_held < 0 and momentum < 0):
            entry_score += 0.001
        if volume_confirm:
            entry_score += 0.0005
        if (ctx.btc_held > 0 and momentum < -0.005) or (ctx.btc_held < 0 and momentum > 0.005):
            entry_score -= 0.0005
        return float(entry_score)

    @staticmethod
    def _evaluate_stop_execution(ctx: RewardContext) -> float:
        if ctx.stop_triggered_this_step is None:
            return 0.0
        if ctx.stop_triggered_this_step == 'take_profit':
            return 0.002
        if ctx.stop_triggered_this_step == 'stop_loss':
            return 0.001
        return 0.0

    @staticmethod
    def _evaluate_holding_decision(ctx: RewardContext) -> float:
        if ctx.btc_held == 0:
            return 0.0
        current_price = float(ctx.df.iloc[ctx.current_step]['close'])
        if ctx.avg_entry_price > 0:
            if ctx.btc_held > 0:
                unrealized_pnl = (current_price - ctx.avg_entry_price) * abs(ctx.btc_held)
            else:
                unrealized_pnl = (ctx.avg_entry_price - current_price) * abs(ctx.btc_held)
            holding_reward = unrealized_pnl / ctx.initial_balance * 0.1
        else:
            holding_reward = 0.0
        if ctx.position_holding_time > 100:
            holding_reward -= 0.0001 * (ctx.position_holding_time - 100)
        return float(holding_reward)

    @staticmethod
    def _calculate_risk_management_reward(ctx: RewardContext, action: np.ndarray) -> float:
        risk_reward = 0.0
        current_price = float(ctx.df.iloc[ctx.current_step]['close'])
        balance_ratio = ctx.total_value / ctx.initial_balance if ctx.initial_balance > 0 else 0.0

        if balance_ratio > 1.0:
            risk_reward += 0.05 * (balance_ratio - 1.0)
        elif balance_ratio > 0.8:
            risk_reward += 0.01
        elif balance_ratio < 0.5:
            risk_reward -= 0.2 * (0.5 - balance_ratio)
        elif balance_ratio < 0.2:
            risk_reward -= 0.5

        if ctx.total_value <= ctx.min_balance:
            risk_reward -= 1.0

        if ctx.btc_held != 0:
            position_value = abs(ctx.btc_held * current_price)
            leverage_used = position_value / ctx.total_value if ctx.total_value > 0 else 10.0
            if leverage_used > 0.8:
                risk_reward -= 0.1 * (leverage_used - 0.8)
            if leverage_used > 0.95:
                risk_reward -= 0.5

        position_percent = float(abs(action[0]))
        if position_percent < 0.3:
            risk_reward += 0.01
        elif position_percent > 0.8:
            risk_reward -= 0.05

        stop_loss_percent = float(action[2])
        if stop_loss_percent > 1.0:
            risk_reward += 0.005
        elif stop_loss_percent < 0.5:
            risk_reward -= 0.002

        if ctx.total_value > ctx.initial_balance * 0.8:
            risk_reward += 0.001

        return float(risk_reward)

    # ---- 將上下文轉為 RewardSignals，並組合最終獎勵 ----
    def compute_signals(self, ctx: RewardContext, action: np.ndarray) -> RewardSignals:
        signals = RewardSignals()
        if self._position_closed_this_step(ctx):
            signals.pnl_ratio = self._calculate_closed_trade_pnl(ctx) / ctx.initial_balance if ctx.initial_balance > 0 else 0.0
        if self._just_opened_position(ctx):
            signals.entry_score = self._evaluate_entry_timing(ctx)
        if ctx.stop_triggered_this_step is not None:
            signals.stop_score = self._evaluate_stop_execution(ctx)
        if ctx.btc_held != 0:
            signals.holding_score = self._evaluate_holding_decision(ctx)

        if abs(float(action[0])) > 0.05:
            if self._position_closed_this_step(ctx) and self._calculate_closed_trade_pnl(ctx) > 0:
                signals.penalty_score = 0.001
            elif self._just_opened_position(ctx):
                signals.penalty_score = 0.0005
            else:
                signals.penalty_score = -ctx.transaction_fee * 0.2

        signals.risk_score = self._calculate_risk_management_reward(ctx, action)
        return signals

    def compute_reward(self, ctx: RewardContext, action: np.ndarray) -> float:
        """從上下文直接計算最終獎勵。"""
        signals = self.compute_signals(ctx, action)
        return self.calculate(signals)


