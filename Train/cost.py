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
    Cost 2: Drawdown Risk
    
    Formula:
        DD = 1 - E_t / E_max_t
        If DD < DD_soft: cost = 0
        Else: cost = (DD - DD_soft) / (1 - DD_soft)
        
        Terminal Penalty (if defined): alpha * (E_max - E_t) / E_0
        (We handle terminal penalty in the loop or here if 'done' flag provided, 
         but calculating cost usually happens step-wise. The prompt implies 
         c_T += penalty. We need 'done' and 'initial_balance' context.)
    """
    def __init__(self, dd_soft: float = 0.2, terminal_penalty: float = 5.0):
        self.dd_soft = dd_soft
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
        if dd > self.dd_soft:
            cost = (dd - self.dd_soft) / (1.0 - self.dd_soft)
            
        # Terminal penalty check
        # Info usually doesn't contain 'done' unless we pass it or infer from termination_reason
        # But the environment returns 'done'. The calculator is called after step.
        # Let's check if we can access termination reason.
        if info.get('termination_reason') == 'liq_triggered':
             # Extra penalty
             # c_T += alpha * (E_max - E_T) / E_0
             # Note: if liq, E_T might be small.
             if self.initial_balance > 0:
                 penalty = self.terminal_penalty * (self.max_equity - equity) / self.initial_balance
                 cost += penalty
                 
        return max(0.0, cost)

class CombinedCostCalculator:
    def __init__(self, num_envs: int = 1):
        self.margin_cost = MarginRiskCost(m_safe=Config.MARGIN_SAFE, c_liq=Config.COST_LIQ_PENALTY)
        # Drawdown cost needs state (max_equity) per environment
        self.dd_costs = [DrawdownCost(dd_soft=Config.DD_SOFT, terminal_penalty=Config.DD_MAX_PENALTY) for _ in range(num_envs)]
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
            costs.append([c1, c2])
        return np.array(costs, dtype=np.float32)
