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
    fric_fee_weight: float = 1.0
    fric_turnover_weight: float = 0.0
    fric_trade_activity_weight: float = 0.0
    fric_extreme_weight: float = 0.0
    fric_extreme_threshold: float = 0.0


class CostCalculator:
    """
    正規化成本計算器 (Normalized Cost Calculator)

    設計原則：
    所有成本皆正規化為「佔當前權益的比例」 (Cost / Equity)，
    確保約束條件在不同資金規模下具有尺度不變性 (Scale Invariance)。

    公式：
    1. Death Cost: 死亡時 = 1.0 + (剩餘步數/總步數)，未死亡 = 0；越早死懲罰越大（最多 2.0）
    2. Fric Cost : StepFee / Equity (本步手續費佔權益的比例)
    3. Dense Buffer Cost (cost_risk_dense): 每步 (1 - buffer_to_min_balance_ratio)^2，僅在接近死亡線時變大（方案 B 獨立通道）
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
            episode_steps: （可選）本回合已執行步數（不含本步）；與 episode_max_steps 同時提供時，死亡成本隨剩餘步數加權
            episode_max_steps: （可選）本回合最大步數
            step_fee_add_only: （可選）僅計入「加碼/加曝險」的手續費
            initial_balance: （可選）初始資金；與 min_balance 同時提供時，計算 buffer_to_min_balance_ratio 以輸出 cost_risk_dense
            cost_fric_scale: （可選）cost_fric 放大係數，預設 1.0；設為 100~1000 可讓 lambda_fee 懲罰與 reward 同數量級

        Returns:
            Dict:
            - cost: 總正規化成本（不含 cost_risk_dense，僅 death + fric）
            - cost_risk: 死亡成本 (0 或 [1.0, 2.0]，剩餘步數越多越大)
            - cost_risk_dense: 每步 dense 懲罰 (1 - buffer_ratio)^2，僅在接近死亡線時變大
            - cost_fric: 摩擦成本（fee + activity + extreme 三分量加總）
            - cost_fric_fee_component: fee 分量（scaled）
            - cost_fric_activity_component: activity 分量（scaled）
            - cost_fric_extreme_component: extreme 分量（scaled）
            - cost_breakdown: 詳細分項
        """
        # 防除以零保護：使用 min_balance 或極小值做為分母下限
        safe_equity = max(equity, 1e-4)

        # 1. 死亡/風險成本 (c_risk)
        # 定義：死亡時 = 1.0 + (剩餘步數/總步數)，區間 [1.0, 2.0]；越早死（剩餘步數多）懲罰越大
        is_dead = liq_triggered or (equity <= min_balance)
        if is_dead:
            ep_steps = kwargs.get("episode_steps")
            ep_max = kwargs.get("episode_max_steps")
            if ep_max is not None and ep_max >= 1 and ep_steps is not None:
                # 本步為第 (episode_steps+1) 步，剩餘步數 = max(0, max - steps - 1)
                remaining = max(0, int(ep_max) - int(ep_steps) - 1)
                ratio = float(remaining) / float(max(1, int(ep_max)))
                c_death = 1.0 + ratio  # [1.0, 2.0]
            else:
                c_death = 1.0
        else:
            c_death = 0.0

        c_risk = float(c_death)

        # 2. Dense buffer 成本 (cost_risk_dense)：每步 (1 - buffer_to_min_balance_ratio)^2，只在很危險時才變大
        initial_balance = kwargs.get("initial_balance")
        if initial_balance is not None and float(initial_balance) > 0:
            buffer_to_min = (float(equity) - float(min_balance)) / float(initial_balance)
            buffer_ratio = max(0.0, min(1.0, buffer_to_min))
            c_dense = (1.0 - buffer_ratio) ** 2
        else:
            c_dense = 0.0
        c_risk_dense = float(c_dense)
        
        # 3. 摩擦成本 (c_fric)
        # 預設使用本步手續費；若有提供 step_fee_add_only（僅加碼/加曝險）則優先使用
        # 正規化為「手續費佔當前權益比例」，保持尺度不變性
        # cost_fric_scale：放大係數，使 cost_fric 與 reward 同數量級（預設 1.0；Phase B 可設 100~1000）
        step_fee_add_only = kwargs.get("step_fee_add_only")
        fee_source = step_fee_add_only if step_fee_add_only is not None else step_fee
        fee_ratio = max(0.0, float(fee_source)) / safe_equity

        turnover_ratio = max(0.0, float(kwargs.get("turnover_ratio", 0.0)))
        trade_activity = max(0.0, float(kwargs.get("trade_activity", 0.0)))

        fric_fee_weight = max(0.0, float(kwargs.get("cost_fric_fee_weight", self.weights.fric_fee_weight)))
        fric_turnover_weight = max(0.0, float(kwargs.get("cost_fric_turnover_weight", self.weights.fric_turnover_weight)))
        fric_trade_activity_weight = max(
            0.0,
            float(kwargs.get("cost_fric_trade_activity_weight", self.weights.fric_trade_activity_weight)),
        )
        fric_extreme_weight = max(0.0, float(kwargs.get("cost_fric_extreme_weight", self.weights.fric_extreme_weight)))
        fric_extreme_threshold = max(
            0.0,
            float(kwargs.get("cost_fric_extreme_threshold", self.weights.fric_extreme_threshold)),
        )

        fee_component_raw = fric_fee_weight * fee_ratio
        activity_component_raw = (
            fric_turnover_weight * turnover_ratio
            + fric_trade_activity_weight * trade_activity
        )
        extreme_signal = max(turnover_ratio, trade_activity)
        extreme_component_raw = fric_extreme_weight * max(0.0, extreme_signal - fric_extreme_threshold) ** 2

        fric_scale = float(kwargs.get("cost_fric_scale", 1.0))
        fric_scale = max(1e-12, fric_scale)
        fee_component = fee_component_raw * fric_scale
        activity_component = activity_component_raw * fric_scale
        extreme_component = extreme_component_raw * fric_scale
        c_fric = fee_component + activity_component + extreme_component
        
        # 總成本 (供 Env.info['cost'] 使用；不含 cost_risk_dense，dense 由獨立 lambda 處理)
        total_cost = c_death + c_fric
        
        return {
            "cost": float(total_cost),             # 總和 (death + fric)
            "cost_risk": float(c_risk),            # 死亡成本通道
            "cost_risk_dense": float(c_risk_dense),  # dense 緩衝懲罰通道 (方案 B 獨立)
            "cost_fric": float(c_fric),            # 摩擦成本通道
            "cost_fric_fee_component": float(fee_component),
            "cost_fric_activity_component": float(activity_component),
            "cost_fric_extreme_component": float(extreme_component),
            "cost_fric_turnover_signal": float(turnover_ratio),
            "cost_fric_trade_activity_signal": float(trade_activity),
            "cost_breakdown": {
                "death_cost": float(c_death),
                "dense_buffer_cost": float(c_risk_dense),
                "fric_cost": float(c_fric),
            },
        }
