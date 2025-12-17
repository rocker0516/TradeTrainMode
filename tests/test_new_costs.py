
import pytest
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Train.cost import CombinedCostCalculator
from Train.config import Config


class TestNewCosts:
    @pytest.fixture
    def calculator(self):
        Config.TURNOVER_NOTIONAL_SCALE = 10_000.0  # 固定尺度方便測試
        calc = CombinedCostCalculator(num_envs=1, turnover_scale=Config.TURNOVER_NOTIONAL_SCALE)
        calc.reset([0], [10000.0])
        return calc

    def test_turnover_cost_basic(self, calculator):
        info = {
            "equity": 10000.0,
            "turnover_notional_change": 1000.0,  # 10% 變化
            "turnover_notional_scale": 10000.0,
            "done": False,
            "termination_reason": None,
        }
        costs = calculator.calculate_costs([info])
        assert np.isclose(costs[0][0], 0.1)
        assert costs[0][1] == 0.0

    def test_turnover_cost_clipped(self, calculator):
        info = {
            "equity": 10000.0,
            "turnover_notional_change": 20_000.0,  # > scale
            "turnover_notional_scale": 10_000.0,
            "done": False,
            "termination_reason": None,
        }
        costs = calculator.calculate_costs([info])
        # clip 已移除：成本應為 200%
        assert np.isclose(costs[0][0], 2.0)

    def test_death_cost_not_done(self, calculator):
        info = {
            "equity": 10000.0,
            "turnover_notional_change": 0.0,
            "turnover_notional_scale": 10_000.0,
            "done": False,
            "termination_reason": None,
        }
        costs = calculator.calculate_costs([info])
        assert costs[0][1] == 0.0

    def test_death_cost_liquidation(self, calculator):
        info = {
            "equity": 5000.0,
            "turnover_notional_change": 0.0,
            "turnover_notional_scale": 10_000.0,
            "done": True,
            "termination_reason": "liq_triggered",
        }
        costs = calculator.calculate_costs([info])
        assert costs[0][1] == 1.0

    def test_death_cost_natural_end(self, calculator):
        info = {
            "equity": 12000.0,
            "turnover_notional_change": 0.0,
            "turnover_notional_scale": 10_000.0,
            "done": True,
            "termination_reason": "data_exhausted",
        }
        costs = calculator.calculate_costs([info])
        assert costs[0][1] == 0.0
