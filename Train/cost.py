
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
    """
    equity: float
    turnover_notional_change: float
    turnover_notional_scale: float
    done: bool
    termination_reason: Optional[str]


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
    - 上限裁剪至 1.0，避免 Qc 發散。
    """

    def __init__(self, default_scale: Optional[float] = None, clip: float = 1.0):
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
    - 若 episode 因「爆倉/強平/資金不足/費用上限」提前結束：c_t = 1
    - 自然結束（data_exhausted）或尚未結束：c_t = 0
    - 採「終局給 1」的實作，簡化回填。
    """

    def __init__(self, death_reasons: Optional[List[str]] = None):
        self.death_reasons = set(
            death_reasons
            or ["liq_triggered", "balance_insufficient", "fee_limit"]
        )

    def calculate_cost(self, signal: CostSignal) -> float:
        done = bool(signal.get("done", False))
        if not done:
            return 0.0

        reason = signal.get("termination_reason")
        if reason in (None, "data_exhausted"):
            return 0.0

        return 1.0 if reason in self.death_reasons else 0.0


class CombinedCostCalculator:
    """
    成本計算入口：
    - C1: turnover (換手率)
    - C2: death (爆倉/強平提前終局)
    """

    def __init__(
        self,
        num_envs: int = 1,
        turnover_scale: Optional[float] = None,
        death_reasons: Optional[List[str]] = None,
    ):
        self.turnover_cost = TurnoverCost(default_scale=turnover_scale)
        self.death_cost = DeathCost(death_reasons=death_reasons)
        self.num_envs = num_envs

    def reset(self, env_indices: List[int], initial_balances: List[float]):
        # 本版成本為無狀態，預留接口以便未來擴充。
        return None

    def calculate_costs(self, infos: List[Dict]) -> np.ndarray:
        costs: List[List[float]] = []

        for info in infos:
            equity = float(info.get("equity", 0.0))
            signal: CostSignal = {
                "equity": equity,
                "turnover_notional_change": float(info.get("turnover_notional_change", 0.0)),
                "turnover_notional_scale": float(info.get("turnover_notional_scale", 0.0)),
                "done": bool(info.get("done", False)),
                "termination_reason": info.get("termination_reason"),
            }

            c_turnover = self.turnover_cost.calculate_cost(signal)
            c_death = self.death_cost.calculate_cost(signal)

            costs.append([c_turnover, c_death])

        return np.array(costs, dtype=np.float32)
