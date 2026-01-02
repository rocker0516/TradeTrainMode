from __future__ import annotations

"""
成本線（Cost / Constraint）計算器

用途：
- 將環境內部可觀測的「成本/風險接近程度」轉成可被 Lagrangian-SAC 使用的 cost 訊號。
- cost 會以 `info["cost"]`（總成本）與 `info["cost_breakdown"]`（分項）回傳，方便訓練與解析。

設計原則：
- SRP：只負責 cost 計算，不耦合 Reward/TradeExecutor/MarketData 的細節。
- 可解析：每個分項都有明確語意（手續費、接近爆倉、保證金壓力、回撤、止損缺失...）。
"""

from dataclasses import dataclass
from typing import Dict, Mapping


def _relu(x: float) -> float:
    return float(x) if x > 0.0 else 0.0


def _clip(x: float, low: float, high: float) -> float:
    if x < low:
        return float(low)
    if x > high:
        return float(high)
    return float(x)


@dataclass(frozen=True)
class CostWeights:
    """成本權重（可由 Config 覆寫）。"""

    w_fee: float = 1.0
    w_liq_proximity: float = 1.0
    w_margin_proximity: float = 0.5
    w_dd: float = 0.2
    w_stop_missing: float = 0.5
    w_stop_proximity: float = 0.2
    w_liq_event: float = 5.0
    w_stop_event: float = 1.0


class CostCalculator:
    """
    成本計算器（單一責任）。

    你可以把它視為「約束訊號產生器」：
    - 連續成本（proximity）：越接近爆倉/保證金不足/止損越近，cost 越高
    - 事件成本（event）：真的爆倉/真的止損時，給較高 cost
    """

    def __init__(
        self,
        *,
        liq_warn_pct: float,
        stop_warn_pct: float,
        maintenance_margin_rate: float,
        weights: CostWeights | None = None,
    ) -> None:
        self.liq_warn_pct = float(max(1e-8, liq_warn_pct))
        self.stop_warn_pct = float(max(1e-8, stop_warn_pct))
        self.maintenance_margin_rate = float(max(0.0, maintenance_margin_rate))
        self.weights = weights or CostWeights()

    def compute(
        self,
        *,
        step_fee_ratio: float,
        current_dd: float,
        risk_signals: Mapping[str, object],
        liq_triggered: bool,
        stop_loss_triggered: bool,
    ) -> Dict[str, object]:
        """
        計算 cost（總成本 + 分項）。

        Args:
            step_fee_ratio: 本 step 手續費 / initial_balance（建議已做過穩定化/clip）
            current_dd: 當前回撤（0~1）
            risk_signals: TradingObserver.compute_risk_signals 的輸出
            liq_triggered: 是否本 step 觸發爆倉
            stop_loss_triggered: 是否本 step 觸發止損

        Returns:
            dict：
            - cost: float（總成本）
            - cost_breakdown: dict（分項，供解析）
        """
        w = self.weights

        # --- 基礎成本：交易摩擦 ---
        fee_cost = max(0.0, float(step_fee_ratio))

        # --- 接近爆倉成本：距離爆倉越近 cost 越高（0~1）---
        abs_gap_pct = float(risk_signals.get("abs_gap_pct", 0.0))
        liq_prox = _relu((self.liq_warn_pct - abs_gap_pct) / self.liq_warn_pct)
        liq_prox = _clip(liq_prox, 0.0, 1.0)

        # --- 保證金壓力成本：margin_ratio 低於 mmr 越多 cost 越高（0~1）---
        margin_ratio = float(risk_signals.get("margin_ratio", 0.0))
        mmr = self.maintenance_margin_rate
        margin_prox = 0.0
        if mmr > 0.0:
            margin_prox = _relu((mmr - margin_ratio) / mmr)
            margin_prox = _clip(margin_prox, 0.0, 1.0)

        # --- 回撤成本 ---
        dd_cost = _clip(max(0.0, float(current_dd)), 0.0, 1.0)

        # --- 止損相關成本 ---
        stop_missing = float(risk_signals.get("stop_loss_missing", 0.0))
        stop_missing = 1.0 if stop_missing >= 0.5 else 0.0

        sl_gap_pct = float(risk_signals.get("sl_gap_pct", 0.0))
        # sl_gap_pct 可能為負；我們取 abs 距離
        stop_prox = _relu((self.stop_warn_pct - abs(sl_gap_pct)) / self.stop_warn_pct)
        stop_prox = _clip(stop_prox, 0.0, 1.0)

        # --- 事件成本（真的觸發時給大懲罰，便於 constraint learning）---
        liq_event_cost = 1.0 if bool(liq_triggered) else 0.0
        stop_event_cost = 1.0 if bool(stop_loss_triggered) else 0.0

        total = (
            w.w_fee * fee_cost
            + w.w_liq_proximity * liq_prox
            + w.w_margin_proximity * margin_prox
            + w.w_dd * dd_cost
            + w.w_stop_missing * stop_missing
            + w.w_stop_proximity * stop_prox
            + w.w_liq_event * liq_event_cost
            + w.w_stop_event * stop_event_cost
        )

        return {
            "cost": float(total),
            "cost_breakdown": {
                "fee_cost": float(fee_cost),
                "liq_proximity_cost": float(liq_prox),
                "margin_proximity_cost": float(margin_prox),
                "dd_cost": float(dd_cost),
                "stop_missing_cost": float(stop_missing),
                "stop_proximity_cost": float(stop_prox),
                "liq_event_cost": float(liq_event_cost),
                "stop_event_cost": float(stop_event_cost),
                # raw values（方便 debug/解析）
                "abs_gap_pct": float(abs_gap_pct),
                "margin_ratio": float(margin_ratio),
                "sl_gap_pct": float(sl_gap_pct),
            },
        }


