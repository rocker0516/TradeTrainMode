from __future__ import annotations

from typing import Any, Dict


class CostCalculator:
    """
    正規化成本計算器 (Normalized Cost Calculator)

    設計原則：
    所有成本皆正規化為「佔當前權益的比例」 (Cost / Equity)，
    確保約束條件在不同資金規模下具有尺度不變性 (Scale Invariance)。

    公式：
    1. Death Cost: 死亡時 = 1.0 + (剩餘步數/總步數)，未死亡 = 0；越早死懲罰越大（最多 2.0）
    2. Dense Buffer Cost (cost_risk_dense): 每步 (1 - buffer_to_min_balance_ratio)^2，僅在接近死亡線時變大（方案 B 獨立通道）
    3. Turnover Cost (cost_turnover): 以正規化 turnover ratio 作為獨立成本通道；
       採「線性 + 超過門檻後二次加重」的非線性形式，專門提高高換手行為的成本

    註：摩擦／手續費已從成本通道移除；交易摩擦僅透過環境 PnL（transaction_fee）反映，不再輸出 cost_fric。
    """

    def __init__(
        self,
        *,
        turnover_quadratic_coef: float = 10.0,
        turnover_quadratic_threshold: float = 0.02,
    ) -> None:
        """
        Args:
            turnover_quadratic_coef: turnover 超過門檻後的二次懲罰係數
            turnover_quadratic_threshold: 啟動二次懲罰的門檻
        """
        self.turnover_quadratic_coef = max(0.0, float(turnover_quadratic_coef))
        self.turnover_quadratic_threshold = max(0.0, float(turnover_quadratic_threshold))

    def compute(
        self,
        *,
        liq_triggered: bool,
        equity: float,
        min_balance: float,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        計算正規化成本。

        Args:
            liq_triggered: 是否觸發爆倉
            equity: 當前權益 (E_t)
            min_balance: 最低資金門檻
            episode_steps: （可選）本回合已執行步數（不含本步）；與 episode_max_steps 同時提供時，死亡成本隨剩餘步數加權
            episode_max_steps: （可選）本回合最大步數
            initial_balance: （可選）初始資金；與 min_balance 同時提供時，計算 buffer_to_min_balance_ratio 以輸出 cost_risk_dense
            turnover_ratio: （可選）只含加曝險的正規化 turnover ratio
            turnover_ratio_full: （可選）完整倉位變動的正規化 turnover ratio
            penalize_turnover_reduction: （可選）若為 True，減碼/平倉也計入 turnover 成本

        Returns:
            Dict:
            - cost: 總正規化成本（僅死亡事件；不含 cost_risk_dense，dense 由獨立 lambda 處理）
            - cost_risk: 死亡成本 (0 或 [1.0, 2.0]，剩餘步數越多越大)
            - cost_risk_dense: 每步 dense 懲罰 (1 - buffer_ratio)^2，僅在接近死亡線時變大
            - cost_turnover: 每步 turnover 成本（線性 + 高換手二次加重）
            - cost_breakdown: 詳細分項（death、dense_buffer、turnover）
        """
        # 防除以零保護：使用 min_balance 或極小值做為分母下限
        _ = max(equity, 1e-4)

        # 1. 死亡/風險成本 (c_risk)
        is_dead = liq_triggered or (equity <= min_balance)
        if is_dead:
            ep_steps = kwargs.get("episode_steps")
            ep_max = kwargs.get("episode_max_steps")
            if ep_max is not None and ep_max >= 1 and ep_steps is not None:
                remaining = max(0, int(ep_max) - int(ep_steps) - 1)
                ratio = float(remaining) / float(max(1, int(ep_max)))
                c_death = 1.0 + ratio  # [1.0, 2.0]
            else:
                c_death = 1.0
        else:
            c_death = 0.0

        c_risk = float(c_death)

        # 2. Dense buffer 成本 (cost_risk_dense)
        initial_balance = kwargs.get("initial_balance")
        if initial_balance is not None and float(initial_balance) > 0:
            buffer_to_min = (float(equity) - float(min_balance)) / float(initial_balance)
            buffer_ratio = max(0.0, min(1.0, buffer_to_min))
            c_dense = (1.0 - buffer_ratio) ** 2
        else:
            c_dense = 0.0
        c_risk_dense = float(c_dense)

        # 3. Turnover 成本（獨立通道，不併入 total_cost）
        penalize_turnover_reduction = bool(kwargs.get("penalize_turnover_reduction", False))
        turnover_ratio = float(kwargs.get("turnover_ratio", 0.0) or 0.0)
        turnover_ratio_full = float(kwargs.get("turnover_ratio_full", turnover_ratio) or 0.0)
        turnover_base = max(0.0, float(turnover_ratio_full if penalize_turnover_reduction else turnover_ratio))
        turnover_excess = max(0.0, turnover_base - self.turnover_quadratic_threshold)
        turnover_linear_cost = float(turnover_base)
        turnover_quadratic_cost = float(self.turnover_quadratic_coef * (turnover_excess**2))
        c_turnover = float(turnover_linear_cost + turnover_quadratic_cost)

        total_cost = c_death

        return {
            "cost": float(total_cost),
            "cost_risk": float(c_risk),
            "cost_risk_dense": float(c_risk_dense),
            "cost_turnover": float(c_turnover),
            "cost_breakdown": {
                "death_cost": float(c_death),
                "dense_buffer_cost": float(c_risk_dense),
                "turnover_cost": float(c_turnover),
                "turnover_linear_cost": float(turnover_linear_cost),
                "turnover_quadratic_cost": float(turnover_quadratic_cost),
            },
        }
