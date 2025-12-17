
import numpy as np
from typing import Dict, List, Optional, TypedDict
from abc import ABC, abstractmethod
from .config import Config


class CostSignal(TypedDict, total=False):
    """
    標準化成本訊號。
    - turnover_notional_change：當前步倉位名義變化量（正值）
    - turnover_notional_scale：規模基準；若缺省則使用 equity*leverage
    - done/termination_reason：用於判斷是否「爆倉死亡」
    - maintenance_margin: 維持保證金金額
    - sl_gap_pct: 當前價格距離止損的比例（可正可負；靠近 0 表示接近止損）
    - stop_loss_missing: 有倉位但未設止損時為 1，否則 0
    - leverage_ratio: pos_notional / equity（有效槓桿 proxy）
    - maint_margin_ratio: maintenance_margin / equity（保證金壓力 proxy）
    """
    equity: float
    turnover_notional_change: float
    turnover_notional_scale: float
    done: bool
    termination_reason: Optional[str]
    maintenance_margin: float
    sl_gap_pct: float
    stop_loss_missing: float
    leverage_ratio: float
    maint_margin_ratio: float
    maintenance_margin_rate: float


class BaseCostCalculator(ABC):
    """成本計算抽象介面。"""

    @abstractmethod
    def calculate_cost(self, signal: CostSignal) -> float:
        ...

    def reset(self, initial_balance: float):
        """預留給需狀態的成本；此處為無狀態實作。"""
        return None


class TurnoverCost(BaseCostCalculator):
    """
    成本線1：換手率成本
    c_t = |Δposition_notional| / notional_scale
    - notional_scale 預設採用 equity * leverage，亦可由外部傳入。
    - 本版本預設不裁剪（clip=None），避免「大跳倉」與「中跳倉」被視為同等成本而產生錯誤誘因。
    """

    def __init__(self, default_scale: Optional[float] = None, clip: Optional[float] = None):
        self.default_scale = default_scale
        self.clip = clip

    def calculate_cost(self, signal: CostSignal) -> float:
        change = abs(float(signal.get("turnover_notional_change", 0.0)))
        scale = signal.get("turnover_notional_scale") or self.default_scale

        if scale is None or scale <= 0:
            equity = float(signal.get("equity", 0.0))
            scale = max(equity * getattr(Config, "LEVERAGE", 1.0), 1e-8)

        cost = change / max(scale, 1e-8)
        if self.clip is not None:
            cost = min(self.clip, cost)
        return float(cost)


class DeathCost(BaseCostCalculator):
    """
    成本線2：爆倉/強平死亡成本
    - 若 episode 因「爆倉/強平/資金不足」提前結束：c_t = 1
    - 自然結束（data_exhausted）或尚未結束：c_t = 0
    - 採「終局給 1」的實作，簡化回填。
    """

    def __init__(self, death_reasons: Optional[List[str]] = None):
        # fee_limit 不再列入死亡原因，避免將手續費上限當作硬性成本線
        self.death_reasons = set(
            death_reasons
            or ["liq_triggered", "balance_insufficient"]
        )

    def calculate_cost(self, signal: CostSignal) -> float:
        done = bool(signal.get("done", False))
        if not done:
            return 0.0

        reason = signal.get("termination_reason")
        if reason in (None, "data_exhausted"):
            return 0.0

        return 1.0 if reason in self.death_reasons else 0.0


class MarginCost(BaseCostCalculator):
    """
    成本線3：Margin Safety (保證金安全邊際)
    - 監控 m_t = Equity / MaintenanceMargin
    - 目標：m_t >= m_target
    - 成本：max(0, m_target - m_t)
    """

    def __init__(self, target_margin_ratio: Optional[float] = None):
        self.target = target_margin_ratio or getattr(Config, "COST_MARGIN_TARGET", 1.5)

    def calculate_cost(self, signal: CostSignal) -> float:
        mm = float(signal.get("maintenance_margin", 0.0))
        equity = float(signal.get("equity", 0.0))

        # 若無倉位 (mm=0) 或權益極高，視為無限安全
        if mm <= 1e-8:
            return 0.0
        
        m_t = equity / mm
        
        # Hinge Loss: 若 m_t < target，產生成本
        violation = self.target - m_t
        cost = max(0.0, violation)
        
        # 可選：正規化或裁剪 (此處簡單保持原始 Hinge，讓 Lambda 自動縮放)
        # 若需要限制範圍，可考慮 min(cost, self.target)
        return float(cost)

class StopLossProximityCost(BaseCostCalculator):
    """
    成本線4（新）：Stop-loss Proximity (止損接近度 + 無止損動態懲罰)
    - 路徑型成本：當 |sl_gap_pct| < d_safe_sl 時產生成本
        c_slprox = max(0, (d_safe_sl - |sl_gap_pct|) / d_safe_sl)
    - 無止損（stop_loss_missing=1）時，不使用固定常數，而是用風險程度動態縮放：
        c_missing = stop_loss_missing * clip(alpha*maint_margin_ratio + (1-alpha)*(leverage_ratio/Lcap), 0, 1)
    - 最終：C6 = clip(c_slprox + c_missing, 0, clip_max)
    """

    def __init__(self):
        self.d_safe = float(getattr(Config, "COST_SL_SAFE_BAND", 0.01))
        self.alpha = float(getattr(Config, "COST_SL_MISSING_ALPHA", 0.8))
        self.lcap = float(getattr(Config, "COST_SL_MISSING_LCAP", 10.0))
        self.clip_max = float(getattr(Config, "COST_SL_CLIP", 1.0))

    def calculate_cost(self, signal: CostSignal) -> float:
        mm = float(signal.get("maintenance_margin", 0.0))
        equity = float(signal.get("equity", 0.0))
        if mm <= 1e-8 or equity <= 1e-12:
            # 無倉位或權益無效 -> 不計止損接近度成本
            return 0.0

        sl_gap_pct = float(signal.get("sl_gap_pct", 0.0))
        stop_loss_missing = float(signal.get("stop_loss_missing", 0.0))

        # -- 1) sl proximity --
        c_slprox = 0.0
        if self.d_safe > 1e-12 and stop_loss_missing < 0.5:
            c_slprox = max(0.0, (self.d_safe - abs(sl_gap_pct)) / self.d_safe)

        # -- 2) missing stop dynamic penalty (NOT constant) --
        maint_margin_ratio = float(signal.get("maint_margin_ratio", 0.0))
        # 若沒提供則以 maintenance_margin / equity 推導
        if maint_margin_ratio <= 0.0 and equity > 0:
            maint_margin_ratio = mm / equity
        maint_margin_ratio = float(np.clip(maint_margin_ratio, 0.0, 2.0))

        leverage_ratio = float(signal.get("leverage_ratio", 0.0))
        # 若沒提供且有 mmr 可推導：pos_notional = maintenance_margin / mmr
        if leverage_ratio <= 0.0:
            mmr = float(signal.get("maintenance_margin_rate", 0.0))
            if mmr > 1e-12:
                leverage_ratio = (maint_margin_ratio / mmr)
        leverage_ratio = float(np.clip(leverage_ratio, 0.0, 10.0))

        lcap = max(self.lcap, 1e-8)
        risk_mix = self.alpha * maint_margin_ratio + (1.0 - self.alpha) * (leverage_ratio / lcap)
        risk_mix = float(np.clip(risk_mix, 0.0, 1.0))
        c_missing = (1.0 if stop_loss_missing > 0.5 else 0.0) * risk_mix

        c_total = c_slprox + c_missing
        c_total = float(np.clip(c_total, 0.0, self.clip_max))
        return c_total


class CombinedCostCalculator:
    """
    成本計算入口：
    - C1: turnover (換手率)
    - C2: death (爆倉/強平提前終局)
    - C3: margin (保證金安全)
    - C4: stop_loss_proximity (止損接近度/無止損動態懲罰)
    """

    def __init__(
        self,
        num_envs: int = 1,
        turnover_scale: Optional[float] = None,
        death_reasons: Optional[List[str]] = None,
    ):
        self.turnover_cost = TurnoverCost(default_scale=turnover_scale)
        self.death_cost = DeathCost(death_reasons=death_reasons)
        self.margin_cost = MarginCost()
        self.stop_loss_cost = StopLossProximityCost()
        self.num_envs = num_envs

    def reset(self, env_indices: List[int], initial_balances: List[float]):
        # 本版成本為無狀態，預留接口以便未來擴充。
        return None

    def calculate_costs(self, infos: List[Dict]) -> np.ndarray:
        costs: List[List[float]] = []

        for info in infos:
            equity = float(info.get("equity", 0.0))
            risk = info.get("risk_signals") or {}
            signal: CostSignal = {
                "equity": equity,
                "turnover_notional_change": float(info.get("turnover_notional_change", 0.0)),
                "turnover_notional_scale": float(info.get("turnover_notional_scale", 0.0)),
                "done": bool(info.get("done", False)),
                "termination_reason": info.get("termination_reason"),
                "maintenance_margin": float(info.get("maintenance_margin", 0.0)),
                # stop-loss proximity inputs (from env risk_signals)
                "sl_gap_pct": float(risk.get("sl_gap_pct", 0.0)),
                "stop_loss_missing": float(risk.get("stop_loss_missing", 0.0)),
                # optional helpers (if env provides; otherwise derived in StopLossProximityCost)
                "leverage_ratio": float(info.get("leverage_ratio", 0.0)),
                "maint_margin_ratio": float(info.get("maint_margin_ratio", 0.0)),
                "maintenance_margin_rate": float(info.get("maintenance_margin_rate", 0.0)),
            }

            c_turnover = self.turnover_cost.calculate_cost(signal)
            c_death = self.death_cost.calculate_cost(signal)
            c_margin = self.margin_cost.calculate_cost(signal)
            c_sl = self.stop_loss_cost.calculate_cost(signal)

            costs.append([c_turnover, c_death, c_margin, c_sl])

        return np.array(costs, dtype=np.float32)
