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
    # Friction / frequency (path costs)
    w_fee_equity: float = 1.0
    w_turnover: float = 0.5
    w_trade_event: float = 0.05

    w_liq_proximity: float = 1.0
    w_margin_proximity: float = 0.5
    w_dd: float = 0.2
    w_balance_proximity: float = 2.0
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
        balance_warn_up_ratio: float = 0.5,
        prox_curve_power: float = 2.0,
        weights: CostWeights | None = None,
    ) -> None:
        self.liq_warn_pct = float(max(1e-8, liq_warn_pct))
        self.stop_warn_pct = float(max(1e-8, stop_warn_pct))
        self.maintenance_margin_rate = float(max(0.0, maintenance_margin_rate))
        self.balance_warn_up_ratio = float(max(1e-8, balance_warn_up_ratio))
        self.prox_curve_power = float(max(1.0, prox_curve_power))
        self.weights = weights or CostWeights()

    def _curve(self, x01: float) -> float:
        """把 0~1 的 proximity 變成更末端敏感的曲線（用於 survival）。"""
        x = _clip(float(x01), 0.0, 1.0)
        p = float(self.prox_curve_power)
        return float(x**p)

    def compute(
        self,
        *,
        step_fee_ratio: float,
        step_fee: float = 0.0,
        equity: float = 0.0,
        min_balance: float = 0.0,
        turnover_ratio: float = 0.0,
        traded: bool = False,
        current_dd: float,
        risk_signals: Mapping[str, object],
        liq_triggered: bool,
        stop_loss_triggered: bool,
    ) -> Dict[str, object]:
        """
        計算 cost（總成本 + 分項）。

        Args:
            step_fee_ratio: 本 step 手續費 / initial_balance（建議已做過穩定化/clip）
            step_fee: 本 step 手續費（絕對值，供 equity-normalized cost 使用）
            equity: 本 step 權益（供 equity-normalized cost / balance proximity 使用）
            min_balance: 最低允許權益（供 balance proximity 使用）
            turnover_ratio: 本 step 換手比例（turnover_notional / (equity*leverage)）
            traded: 是否本 step 有發生實際交易（事件型 frequency cost）
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

        # -------------------------
        # Friction / frequency costs (path)
        # -------------------------
        # (1) 舊版：以 initial_balance 正規化的 step fee（保留相容性）
        fee_cost = max(0.0, float(step_fee_ratio))

        # (2) 新版：以 equity 正規化的 step fee（更能反映「接近死亡時費用更致命」）
        safe_equity = max(1e-8, float(equity))
        fee_equity_cost = max(0.0, float(step_fee)) / safe_equity
        # clip：避免極端值把 cost 打爆（這個比例本來就應該非常小）
        fee_equity_cost = _clip(fee_equity_cost, 0.0, 1.0)

        # (3) 換手成本：turnover_ratio 原本就是正規化值，但仍做 clip
        turnover_cost = _clip(max(0.0, float(turnover_ratio)), 0.0, 1.0)

        # (4) 交易事件成本：只要 traded 就扣固定成本（抑制抖動刷單）
        trade_event_cost = 1.0 if bool(traded) else 0.0

        # --- 接近爆倉成本：距離爆倉越近 cost 越高（0~1）---
        abs_gap_pct = float(risk_signals.get("abs_gap_pct", 0.0))
        liq_prox = _relu((self.liq_warn_pct - abs_gap_pct) / self.liq_warn_pct)
        liq_prox = self._curve(_clip(liq_prox, 0.0, 1.0))

        # --- 保證金壓力成本：margin_ratio 低於 mmr 越多 cost 越高（0~1）---
        margin_ratio = float(risk_signals.get("margin_ratio", 0.0))
        mmr = self.maintenance_margin_rate
        margin_prox = 0.0
        if mmr > 0.0:
            margin_prox = _relu((mmr - margin_ratio) / mmr)
            margin_prox = self._curve(_clip(margin_prox, 0.0, 1.0))

        # --- 接近最低權益成本：equity 越接近 min_balance cost 越高（0~1）---
        balance_prox = 0.0
        mb = float(min_balance)
        if mb > 0.0:
            eq_over_min = float(safe_equity / mb)
            # 當 equity <= min_balance -> cost=1
            # 當 equity >= (1+warn)*min_balance -> cost=0
            warn = float(self.balance_warn_up_ratio)
            if eq_over_min <= 1.0:
                balance_prox = 1.0
            else:
                upper = 1.0 + warn
                balance_prox = _relu((upper - eq_over_min) / max(1e-8, warn))
                balance_prox = self._curve(_clip(balance_prox, 0.0, 1.0))

        # --- 回撤成本 (Refactored: rely on Main Reward) ---
        dd_cost = _clip(max(0.0, float(current_dd)), 0.0, 1.0) if w.w_dd > 0 else 0.0

        # --- 止損相關成本 (Refactored: only penalize MISSING stop loss) ---
        # 1. Stop Loss Missing (Violation)
        stop_missing = float(risk_signals.get("stop_loss_missing", 0.0))
        stop_missing = 1.0 if stop_missing >= 0.5 else 0.0

        # 2. Stop Loss Proximity / Event (Trading outcome, NOT violation)
        sl_gap_pct = float(risk_signals.get("sl_gap_pct", 0.0))
        stop_prox = 0.0
        if w.w_stop_proximity > 0:
             stop_prox = _relu((self.stop_warn_pct - abs(sl_gap_pct)) / self.stop_warn_pct)
             stop_prox = self._curve(_clip(stop_prox, 0.0, 1.0))

        stop_event_cost = 1.0 if (bool(stop_loss_triggered) and w.w_stop_event > 0) else 0.0

        # --- 事件成本 ---
        liq_event_cost = 1.0 if bool(liq_triggered) else 0.0

        # Split costs (方便雙 λ)
        cost_fric = (
            w.w_fee * fee_cost
            + w.w_fee_equity * fee_equity_cost
            + w.w_turnover * turnover_cost
            + w.w_trade_event * trade_event_cost
        )

        cost_risk = (
            w.w_liq_proximity * liq_prox
            + w.w_margin_proximity * margin_prox
            + w.w_balance_proximity * balance_prox
            + w.w_dd * dd_cost
            + w.w_stop_missing * stop_missing
            + w.w_stop_proximity * stop_prox
            + w.w_liq_event * liq_event_cost
            + w.w_stop_event * stop_event_cost
        )

        # Cap total per-step cost to reasonable bounds (e.g. 1.0) unless catastrophe
        # This helps gradient stability.
        # But we allow >1.0 if multiple violations occur or event triggers.
        # Ideally, we want the "normal operational cost" to be small.
        
        total = float(cost_risk + cost_fric)

        return {
            "cost": float(total),
            "cost_risk": float(cost_risk),
            "cost_fric": float(cost_fric),
            "cost_breakdown": {
                # friction
                "fee_cost": float(fee_cost),
                "fee_equity_cost": float(fee_equity_cost),
                "turnover_cost": float(turnover_cost),
                "trade_event_cost": float(trade_event_cost),
                # risk
                "liq_proximity_cost": float(liq_prox),
                "margin_proximity_cost": float(margin_prox),
                "balance_proximity_cost": float(balance_prox),
                "dd_cost": float(dd_cost),
                "stop_missing_cost": float(stop_missing),
                "stop_proximity_cost": float(stop_prox),
                "liq_event_cost": float(liq_event_cost),
                "stop_event_cost": float(stop_event_cost),
                # raw values（方便 debug/解析）
                "abs_gap_pct": float(abs_gap_pct),
                "margin_ratio": float(margin_ratio),
                "sl_gap_pct": float(sl_gap_pct),
                "equity": float(safe_equity),
                "min_balance": float(mb),
            },
        }


