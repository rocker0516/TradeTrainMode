from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class CostWeights:
    """
    成本權重/設定 (新架構)
    雖然公式已標準化，但保留此類別以備未來擴充 (例如是否啟用某條線的開關)。
    目前主要用於佔位，參數皆預設為 1.0 或由 Config 控制。
    """
    pass


class CostCalculator:
    """
    正規化成本計算器 (Normalized Cost Calculator)

    設計原則：
    所有成本皆正規化為「佔當前權益的比例」 (Cost / Equity)，
    確保約束條件在不同資金規模下具有尺度不變性 (Scale Invariance)。

    公式：
    1. Death Cost: 1.0 (若發生爆倉/破產，視為損失 100% 權益)
    2. Fric Cost : StepFee / Equity (本步手續費佔權益的比例)
    """

    def __init__(self, weights: CostWeights | None = None) -> None:
        self.weights = weights or CostWeights()

    def compute(
        self,
        *,
        liq_triggered: bool,
        equity: float,
        min_balance: float,
        step_fee: float,
        **kwargs
    ) -> Dict[str, float]:
        """
        計算正規化成本。

        Args:
            liq_triggered: 是否觸發爆倉
            equity: 當前權益 (E_t)
            min_balance: 最低資金門檻
            step_fee: 本步產生的手續費 (絕對金額)

        Returns:
            Dict:
            - cost: 總正規化成本 (供單一 Lambda 使用)
            - cost_risk: 死亡成本 (1.0 or 0.0)
            - cost_fric: 摩擦成本 (Fee / Equity)
            - cost_breakdown: 詳細分項
        """
        # 防除以零保護：使用 min_balance 或極小值做為分母下限
        # 若 equity 已經低於 0，則保護值為 1e-4，避免負值或除零炸裂
        safe_equity = max(equity, 1e-4)

        # 1. 死亡/風險成本 (c_risk)
        # 定義：發生死亡事件 = 100% 權益損失風險實現 -> Cost = 1.0
        is_dead = liq_triggered or (equity <= min_balance)
        c_death = 1.0 if is_dead else 0.0

        # 2. 摩擦/換手成本 (c_fric)
        # 定義：手續費佔當前權益的比例
        # c_fric = Fee_t / E_t
        c_fric = step_fee / safe_equity

        # 總成本 (若訓練端只支援單一 cost channel，則相加)
        # 通常死亡成本 (1.0) 會遠大於摩擦成本 (e.g. 0.001)，
        # 所以直接相加在數學上是合理的 (死亡是主導項)。
        total_cost = c_death + c_fric

        return {
            "cost": float(total_cost),       # 總和 (供 Env.info['cost'] 使用)
            "cost_risk": float(c_death),     # 獨立通道 (供多 Lambda 使用)
            "cost_fric": float(c_fric),      # 獨立通道 (供多 Lambda 使用)
            "cost_breakdown": {
                "death_cost": float(c_death),
                "fric_cost": float(c_fric),
            },
        }
