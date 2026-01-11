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
            - cost_fric: 摩擦成本 (Fee / Equity)
            - cost_sl_buf: 止損安全緩衝成本（密集、0~1、ATR 無量綱）
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
        #
        # 重要：若外部提供 step_fee_add_only，則 cost_fric 會「排除減倉/平倉」，
        # 只在加碼/加曝險時才計入摩擦成本。
        fee_for_fric = kwargs.get("step_fee_add_only", step_fee)
        try:
            fee_for_fric = float(fee_for_fric)
        except (TypeError, ValueError):
            fee_for_fric = float(step_fee)
        c_fric = fee_for_fric / safe_equity

        # 3. Stop-Buffer Cost（止損成本線）
        # 定義距離（以 ATR 正規化）：d_t = |P_t - SL_t| / ATR_t
        # 成本：c_sl_buf = clip( max(0, d_min - d_t) / d_scale, 0, 1 )
        #
        # 語義：不是罰虧損，而是罰「你把倉位放在快撞止損的地方還不撤」。
        # 若持倉但缺少 SL 或 ATR 無法估計，視為不安全 -> stop_missing_cost = 1.0。
        has_position = bool(kwargs.get("has_position", False))
        current_price = float(kwargs.get("current_price", 0.0) or 0.0)
        stop_loss_price = float(kwargs.get("stop_loss_price", 0.0) or 0.0)
        atr = float(kwargs.get("atr", 0.0) or 0.0)
        d_min = float(kwargs.get("stop_buffer_d_min", 0.3))
        d_scale = float(kwargs.get("stop_buffer_d_scale", max(d_min, 1e-12)))
        d_scale_safe = max(d_scale, 1e-12)

        sl_buf_cost = 0.0
        stop_missing_cost = 0.0
        if has_position:
            # stop_loss_price==0 表示未設定；atr<=0 表示無法估計（資料不足或極端情況）
            if stop_loss_price <= 0.0 or atr <= 1e-12:
                stop_missing_cost = 1.0
            else:
                d_t = abs(current_price - stop_loss_price) / atr
                raw = max(0.0, d_min - d_t) / d_scale_safe
                sl_buf_cost = min(1.0, max(0.0, raw))

        # channel 值：若缺 SL，直接視為最大不安全；否則使用 buffer 公式
        c_sl_buf = max(sl_buf_cost, stop_missing_cost)

        # 總成本 (若訓練端只支援單一 cost channel，則相加)
        # 通常死亡成本 (1.0) 會遠大於摩擦成本 (e.g. 0.001)，
        # 所以直接相加在數學上是合理的 (死亡是主導項)。
        total_cost = c_death + c_fric + c_sl_buf

        return {
            "cost": float(total_cost),       # 總和 (供 Env.info['cost'] 使用)
            "cost_risk": float(c_death),     # 獨立通道 (供多 Lambda 使用)
            "cost_fric": float(c_fric),      # 獨立通道 (供多 Lambda 使用)
            "cost_sl_buf": float(c_sl_buf),  # 獨立通道 (供多 Lambda 使用)
            "cost_breakdown": {
                "death_cost": float(c_death),
                "fric_cost": float(c_fric),
                "sl_buf_cost": float(sl_buf_cost),
                "stop_missing_cost": float(stop_missing_cost),
            },
        }
