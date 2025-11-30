import numpy as np
from typing import Dict, List
from abc import ABC, abstractmethod
from .config import Config

class BaseCostCalculator(ABC):
    """
    Abstract base class for cost calculation in SAC-Lagrangian.
    """
    @abstractmethod
    def calculate_cost(self, info: Dict) -> float:
        pass

class MarginRiskCost(BaseCostCalculator):
    """
    Cost 1: Margin Risk / Liquidation Risk
    
    Formula:
        MR = Equity / MaintenanceMargin
        If MR >= M_safe: cost = 0
        If 1 <= MR < M_safe: cost = (M_safe - MR) / (M_safe - 1)
        If MR < 1 (Liq): cost = 1 + C_liq
    """
    def __init__(self, m_safe: float = 1.5, c_liq: float = 5.0):
        self.m_safe = m_safe
        self.c_liq = c_liq

    def calculate_cost(self, info: Dict) -> float:
        # Check for liquidation signal first
        if info.get('liq_triggered', False):
            return 1.0 + self.c_liq
        
        equity = info.get('equity', 0.0)
        mm = info.get('maintenance_margin', 0.0)
        
        if mm <= 1e-9:
            # No position or negligible margin -> Safe
            return 0.0
            
        mr = equity / mm
        
        if mr >= self.m_safe:
            return 0.0
        elif mr >= 1.0:
            # Linear scaling
            return (self.m_safe - mr) / (self.m_safe - 1.0)
        else:
            # Should be covered by liq_triggered, but just in case
            return 1.0 + self.c_liq

class DrawdownCost(BaseCostCalculator):
    """
    Cost 2: Drawdown Risk (Segmented)
    
    Segments:
    1. DD <= Warn: Cost = 0
    2. Warn < DD <= Crit: Linear increase from 0 to 0.5
    3. DD > Crit: Linear increase from 0.5 to 1.0
    """
    def __init__(self, warn: float = 0.1, crit: float = 0.2, terminal_penalty: float = 5.0):
        self.warn = warn
        self.crit = crit
        self.terminal_penalty = terminal_penalty
        self.max_equity = 0.0
        self.initial_balance = 1.0 # Placeholder, updated on reset
        
    def reset(self, initial_balance: float):
        self.max_equity = initial_balance
        self.initial_balance = initial_balance
        
    def calculate_cost(self, info: Dict) -> float:
        equity = info.get('equity', 0.0)
        
        # Update running max
        if equity > self.max_equity:
            self.max_equity = equity
            
        if self.max_equity <= 0:
            return 0.0
            
        dd = 1.0 - (equity / self.max_equity)
        
        cost = 0.0
        if dd <= self.warn:
            cost = 0.0
        elif dd <= self.crit:
            # Segment 1: Warn to Crit -> Cost 0.0 to 0.5
            # (DD - Warn) / (Crit - Warn) * 0.5
            if self.crit > self.warn:
                cost = ((dd - self.warn) / (self.crit - self.warn)) * 0.5
            else:
                cost = 0.5 # Edge case
        else:
            # Segment 2: > Crit -> Cost 0.5 to 1.0
            # 0.5 + (DD - Crit) / (1 - Crit) * 0.5
            cost = 0.5 + ((dd - self.crit) / (1.0 - self.crit)) * 0.5
            
        # Terminal penalty check
        if info.get('termination_reason') == 'liq_triggered':
             # Extra penalty
             if self.initial_balance > 0:
                 penalty = self.terminal_penalty * (self.max_equity - equity) / self.initial_balance
                 cost += penalty
                 
        return max(0.0, cost)

class FeeRiskCost(BaseCostCalculator):
    """
    Cost 3: Fee Risk (Step Fee / Initial Balance)
    Directly penalizes incurring fees (trading volume).
    """
    def calculate_cost(self, info: Dict) -> float:
        # Env calculates step_fee_ratio = step_fee / initial_balance
        return info.get('step_fee_ratio', 0.0)

class CombinedCostCalculator:
    def __init__(self, num_envs: int = 1):
        self.margin_cost = MarginRiskCost(m_safe=Config.MARGIN_SAFE, c_liq=Config.COST_LIQ_PENALTY)
        # Drawdown cost needs state (max_equity) per environment
        self.dd_costs = [DrawdownCost(warn=Config.DD_WARN, crit=Config.DD_CRIT, terminal_penalty=Config.DD_MAX_PENALTY) for _ in range(num_envs)]
        self.fee_cost = FeeRiskCost()
        self.num_envs = num_envs

    def reset(self, env_indices: List[int], initial_balances: List[float]):
        for idx, balance in zip(env_indices, initial_balances):
            self.dd_costs[idx].reset(balance)

    def calculate_costs(self, infos: List[Dict]) -> np.ndarray:
        """
        Returns shape (num_envs, num_constraints)
        """
        costs = []
        for i, info in enumerate(infos):
            c1 = self.margin_cost.calculate_cost(info)
            c2 = self.dd_costs[i].calculate_cost(info)
            c3 = self.fee_cost.calculate_cost(info)
            costs.append([c1, c2, c3])
        return np.array(costs, dtype=np.float32)
