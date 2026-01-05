from __future__ import annotations

import pytest

from Train.lagrangian import compute_trade_stats


def test_compute_trade_stats_empty() -> None:
    stats = compute_trade_stats([])
    assert stats["avg_fee"] == 0.0
    assert stats["avg_liq"] == 0.0
    assert stats["avg_dd"] == 0.0
    assert stats["avg_long_entries"] == 0.0
    assert stats["avg_short_entries"] == 0.0
    assert stats["avg_stop_loss"] == 0.0


def test_compute_trade_stats_basic() -> None:
    ep_infos = [
        {
            "total_fees": 10.0,
            "episode_liq_count": 1,
            "episode_max_dd": 0.1,
            "long_entry_count": 2,
            "short_entry_count": 0,
            "episode_stop_loss_count": 3,
        },
        {
            "total_fees": 30.0,
            "episode_liq_count": 0,
            "episode_max_dd": 0.3,
            "long_entry_count": 0,
            "short_entry_count": 4,
            "episode_stop_loss_count": 1,
        },
    ]

    stats = compute_trade_stats(ep_infos)
    assert stats["avg_fee"] == pytest.approx(20.0)
    assert stats["avg_liq"] == pytest.approx(0.5)
    assert stats["avg_dd"] == pytest.approx(0.2)
    assert stats["avg_long_entries"] == pytest.approx(1.0)
    assert stats["avg_short_entries"] == pytest.approx(2.0)
    assert stats["avg_stop_loss"] == pytest.approx(2.0)


