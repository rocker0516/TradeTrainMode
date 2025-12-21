"""
Dashboard alignment tests.

Ensures Avg Profit and Avg Balance are computed from consistent episode subsets
when 'survived_*' stats are present.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from Train.train import format_dashboard


def test_dashboard_uses_survived_balances_when_survived_profits_present() -> None:
    stats = {
        "profits": deque([20.0, -50.0], maxlen=100),
        "survived_profits": deque([20.0], maxlen=100),
        "balances": deque([12_000.0, 5_000.0], maxlen=100),
        "survived_balances": deque([12_000.0], maxlen=100),
        "lengths": deque([10, 10], maxlen=100),
        "reasons": deque(["data_exhausted", "liq_triggered"], maxlen=100),
        "fees": deque([0.0, 0.0], maxlen=100),
        "trades": deque([0, 0], maxlen=100),
        "longs": deque([0, 0], maxlen=100),
        "shorts": deque([0, 0], maxlen=100),
        "sl_counts": deque([0, 0], maxlen=100),
        "max_trade_losses": deque([0.0, -50.0], maxlen=100),
        "missing_sl_rates": deque([0.0, 0.0], maxlen=100),
        "abs_sl_gap_means": deque([0.0, 0.0], maxlen=100),
        "pos_pct_means": deque([0.0, 0.0], maxlen=100),
        "abs_pos_pct_means": deque([0.0, 0.0], maxlen=100),
        "pos_time_rates": deque([0.0, 0.0], maxlen=100),
    }
    metrics = {}
    costs = np.zeros((1, 4), dtype=np.float32)
    cost_limits = np.ones(4, dtype=np.float32)

    s = format_dashboard(
        global_step=0,
        total_episodes=2,
        fps=0.0,
        stats=stats,
        metrics=metrics,
        costs=costs,
        cost_limits=cost_limits,
        current_fee_rate=0.0,
    )

    # Should show survived avg balance 12,000 and also show all avg balance 8,500
    assert "Avg Balance" in s
    assert "12,000.00" in s
    assert "All: 8,500.00" in s


