from __future__ import annotations

import pytest

from Train.lagrangian import compute_end_result_stats


def test_compute_end_result_stats_empty() -> None:
    stats = compute_end_result_stats([])
    assert stats["n"] == 0
    assert stats["terminated_count"] == 0
    assert stats["truncated_count"] == 0
    assert stats["terminated_rate"] == 0.0
    assert stats["truncated_rate"] == 0.0
    assert stats["reason_counts"] == {}
    assert stats["reason_rates"] == {}
    assert stats["avg_episode_len"] == 0.0
    assert stats["avg_final_balance"] == 0.0


def test_compute_end_result_stats_basic_counts_and_rates() -> None:
    ep_infos = [
        {
            "terminated": True,
            "truncated": False,
            "termination_reason": "liq_triggered",
            "final_balance": 900.0,
            "episode": {"l": 10, "r": 1.0, "t": 0.1},
        },
        {
            "terminated": False,
            "truncated": True,
            "termination_reason": "max_steps_reached",
            "final_balance": 1100.0,
            "episode": {"l": 20, "r": 2.0, "t": 0.2},
        },
        {
            "terminated": True,
            "truncated": False,
            "termination_reason": "balance_insufficient",
            "final_balance": 800.0,
            "episode": {"l": 30, "r": 3.0, "t": 0.3},
        },
        # 缺 reason -> unknown
        {
            "terminated": False,
            "truncated": True,
            "final_balance": 1000.0,
            "episode": {"l": 40, "r": 4.0, "t": 0.4},
        },
    ]

    stats = compute_end_result_stats(ep_infos)
    assert stats["n"] == 4
    assert stats["terminated_count"] == 2
    assert stats["truncated_count"] == 2
    assert stats["terminated_rate"] == pytest.approx(50.0)
    assert stats["truncated_rate"] == pytest.approx(50.0)

    assert stats["reason_counts"]["liq_triggered"] == 1
    assert stats["reason_counts"]["max_steps_reached"] == 1
    assert stats["reason_counts"]["balance_insufficient"] == 1
    assert stats["reason_counts"]["unknown"] == 1

    assert stats["reason_rates"]["unknown"] == pytest.approx(25.0)
    assert stats["avg_episode_len"] == pytest.approx(25.0)
    assert stats["avg_final_balance"] == pytest.approx(950.0)


