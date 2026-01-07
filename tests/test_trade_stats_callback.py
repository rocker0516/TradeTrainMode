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
    assert stats["avg_long_closes"] == 0.0
    assert stats["avg_short_closes"] == 0.0
    assert stats["avg_stop_loss"] == 0.0
    assert stats["avg_active_exits"] == 0.0
    assert stats["stop_loss_rate_pct"] == 0.0
    assert stats["active_exit_rate_pct"] == 0.0
    assert stats["exit_coverage_rate_pct"] == 0.0


def test_compute_trade_stats_basic() -> None:
    ep_infos = [
        {
            "total_fees": 10.0,
            "episode_liq_count": 1,
            "episode_max_dd": 0.1,
            "long_entry_count": 2,
            "short_entry_count": 0,
            "long_close_count": 1,
            "short_close_count": 0,
            "episode_stop_loss_count": 1,
            "episode_active_exit_count": 1,
        },
        {
            "total_fees": 30.0,
            "episode_liq_count": 0,
            "episode_max_dd": 0.3,
            "long_entry_count": 0,
            "short_entry_count": 4,
            "long_close_count": 0,
            "short_close_count": 2,
            "episode_stop_loss_count": 1,
            "episode_active_exit_count": 0,
        },
    ]

    stats = compute_trade_stats(ep_infos)
    assert stats["avg_fee"] == pytest.approx(20.0)
    assert stats["avg_liq"] == pytest.approx(0.5)
    assert stats["avg_dd"] == pytest.approx(0.2)
    assert stats["avg_long_entries"] == pytest.approx(1.0)
    assert stats["avg_short_entries"] == pytest.approx(2.0)
    assert stats["avg_long_closes"] == pytest.approx(0.5)
    assert stats["avg_short_closes"] == pytest.approx(1.0)
    assert stats["avg_stop_loss"] == pytest.approx(1.0)
    assert stats["avg_active_exits"] == pytest.approx(0.5)
    # total_exits = (stop_loss=2) + (active_exit=1) = 3
    assert stats["stop_loss_rate_pct"] == pytest.approx((2 / 3) * 100.0)
    assert stats["active_exit_rate_pct"] == pytest.approx((1 / 3) * 100.0)
    # total_closes = (1+0) + (0+2) = 3 -> coverage = 100%
    assert stats["exit_coverage_rate_pct"] == pytest.approx(100.0)


