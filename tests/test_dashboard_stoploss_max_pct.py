"""
Dashboard stop-loss metrics tests.

Ensures Max StopLoss% is printed as the maximum stop-loss distance (entry->stop) percent.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from Train.train import format_dashboard


def test_dashboard_prints_max_stoploss_percent() -> None:
    stats = {
        "profits": deque([0.0, 0.0], maxlen=100),
        "survived_profits": deque([], maxlen=100),
        "balances": deque([10_000.0, 10_000.0], maxlen=100),
        "survived_balances": deque([], maxlen=100),
        "lengths": deque([10, 10], maxlen=100),
        "reasons": deque(["data_exhausted", "data_exhausted"], maxlen=100),
        "fees": deque([0.0, 0.0], maxlen=100),
        "trades": deque([10, 20], maxlen=100),
        "longs": deque([5, 10], maxlen=100),
        "shorts": deque([5, 10], maxlen=100),
        "sl_counts": deque([1, 5], maxlen=100),
        # Max stop-loss distance (entry->stop) is 25% here, so Max StopLoss% should print 25.0%
        "max_stop_loss_dists": deque([0.10, 0.25], maxlen=100),
        # Keep hit-rate metric present to satisfy dashboard formatting
        "sl_rates": deque([0.10, 0.10], maxlen=100),
        "max_trade_losses": deque([0.0, 0.0], maxlen=100),
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

    assert "Max StopLoss%" in s
    assert "25.0%" in s


