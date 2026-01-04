from __future__ import annotations

"""
成本線（Cost / Constraint）計算器

用途：
- 將環境內部的「死亡事件」（實際爆倉、資金耗盡）轉成可被 Lagrangian-SAC 使用的 cost 訊號。
- 依照需求，只保留死亡懲罰，移除所有摩擦成本與過程風險成本。
"""

from dataclasses import dataclass
from typing import Dict, Mapping


@dataclass(frozen=True)
class CostWeights:
    """成本權重（只保留死亡懲罰）。"""
    w_liq_event: float = 5.0
    w_bankrupt_event: float = 5.0


class CostCalculator:
    """
    成本計算器（簡化版：只懲罰死亡）。
    
    Cost 只在以下情況產生：
    1. liq_triggered = True (爆倉)
    2. equity <= min_balance (資金耗盡/破產)
    """

    def __init__(
        self,
        weights: CostWeights | None = None,
    ) -> None:
        self.weights = weights or CostWeights()

    def compute(
        self,
        *,
        liq_triggered: bool,
        equity: float,
        min_balance: float,
        **kwargs  # 忽略其他不再使用的參數 (step_fee, turnover 等)
    ) -> Dict[str, object]:
        """
        計算 cost（總成本 + 分項）。

        Args:
            liq_triggered: 是否本 step 觸發爆倉
            equity: 本 step 權益
            min_balance: 最低允許權益

        Returns:
            dict：
            - cost: float（總成本）
            - cost_breakdown: dict（分項，供解析）
        """
        w = self.weights

        # 1. 爆倉事件成本
        liq_event_cost = float(w.w_liq_event) if bool(liq_triggered) else 0.0

        # 2. 資金耗盡成本 (Bankruptcy)
        # 當權益低於最小餘額時，視為死亡
        bankrupt_event_cost = 0.0
        if equity <= min_balance:
             bankrupt_event_cost = float(w.w_bankrupt_event)

        total = liq_event_cost + bankrupt_event_cost

        return {
            "cost": float(total),
            # 兼容舊接口的 key (risk/fric)，這裡全部歸類為 risk
            "cost_risk": float(total), 
            "cost_fric": 0.0,
            "cost_breakdown": {
                "liq_event_cost": float(liq_event_cost),
                "bankrupt_event_cost": float(bankrupt_event_cost),
            },
        }
