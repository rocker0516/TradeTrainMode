"""
Cost guidance tests for C1 and C4 improvements.
"""

from __future__ import annotations

import numpy as np

from Train.cost import TurnoverCost, StopLossProximityCost
from Train.config import Config


def test_c1_turnover_penalizes_flip_more_and_relaxes_risk_reducing(monkeypatch) -> None:
    # Configure multipliers for deterministic behavior
    monkeypatch.setattr(Config, "COST_TURNOVER_FLIP_MULT", 1.0, raising=False)          # x2 on flip
    monkeypatch.setattr(Config, "COST_TURNOVER_RISK_REDUCING_MULT", 0.25, raising=False)  # x0.25 when de-risking

    c1 = TurnoverCost(default_scale=100.0)
    base_signal = {
        "equity": 10_000.0,
        "turnover_notional_change": 10.0,
        "turnover_notional_scale": 100.0,
    }

    base = c1.calculate_cost({**base_signal, "is_flip": False, "is_risk_reducing": False})
    flip = c1.calculate_cost({**base_signal, "is_flip": True, "is_risk_reducing": False})
    derisk = c1.calculate_cost({**base_signal, "is_flip": False, "is_risk_reducing": True})

    assert np.isclose(base, 0.1, atol=1e-9)
    assert np.isclose(flip, 0.2, atol=1e-9)      # doubled
    assert np.isclose(derisk, 0.025, atol=1e-9)  # quartered


def test_c4_directional_cost_higher_when_misaligned(monkeypatch) -> None:
    # Enable directional guidance
    monkeypatch.setattr(Config, "COST_SL_TREND_WEIGHT", 0.10, raising=False)
    monkeypatch.setattr(Config, "COST_SL_CLIP", 10.0, raising=False)  # avoid clipping masking differences

    c4 = StopLossProximityCost()

    # Keep other terms zero: far from stop, has stop, and set mm/equity so method runs.
    common = {
        "maintenance_margin": 50.0,
        "equity": 10_000.0,
        "sl_gap_pct": 1.0,            # far from stop band => c_slprox ~ 0
        "stop_loss_missing": 0.0,     # no missing-stop penalty
        "maint_margin_ratio": 0.005,
        "maintenance_margin_rate": 0.005,
        "leverage_ratio": 10.0,       # max risk_scale ~ 1
    }

    # Uptrend proxy: trend_score > 0
    aligned_long = c4.calculate_cost({**common, "trend_score": 3.0, "position_pct": 0.8})
    misaligned_short = c4.calculate_cost({**common, "trend_score": 3.0, "position_pct": -0.8})

    assert misaligned_short > aligned_long


