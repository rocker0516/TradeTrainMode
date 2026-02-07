from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

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
            step_fee: 本步產生的手續費 (絕對金額；包含加倉/減倉/平倉)
            step_fee_add_only: （可選，從 kwargs 傳入）僅計入「加碼/加曝險」的手續費，用於排除減倉/平倉的摩擦成本線

        Returns:
            Dict:
            - cost: 總正規化成本 (供單一 Lambda 使用)
            - cost_risk: 死亡成本 (1.0 or 0.0)
            - cost_fric: 摩擦成本 (目前固定為 0.0)
            - cost_breakdown: 詳細分項
        """
        # 防除以零保護：使用 min_balance 或極小值做為分母下限
        # 若 equity 已經低於 0，則保護值為 1e-4，避免負值或除零炸裂
        safe_equity = max(equity, 1e-4)

        # 1. 死亡/風險成本 (c_risk)
        # 定義：發生死亡事件 = 100% 權益損失風險實現 -> Cost = 1.0
        is_dead = liq_triggered or (equity <= min_balance)
        c_death = 1.0 if is_dead else 0.0

        # 風險通道：只代表死亡事件
        c_risk = float(c_death)
        
        # 2. 摩擦成本 (c_fric)
        # 目前固定為 0.0，未來可擴充實現
        c_fric = 0.0
        
        # 總成本 (目前只有 cost_risk，因為 cost_fric 為 0)
        total_cost = c_death
        
        return {
            "cost": float(total_cost),       # 總和 (供 Env.info['cost'] 使用)
            "cost_risk": float(c_risk),      # 獨立通道 (供多 Lambda 使用)
            "cost_fric": float(c_fric),      # 摩擦成本通道 (目前固定為 0)
            "cost_breakdown": {
                "death_cost": float(c_death),
                "fric_cost": float(c_fric),
            },
        }
