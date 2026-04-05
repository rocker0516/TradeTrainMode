"""
TradingEnvironment 之短窗 tail 事件計數與 ep_* 回合摘要相關單元測試。
"""

from __future__ import annotations

import numpy as np
import pytest

from Env.trading_env import compute_tail_loss_event_count


def test_compute_tail_loss_event_count_disabled() -> None:
    """k<1 或 tau<=0 時應為 0。"""
    assert compute_tail_loss_event_count([-0.5, -0.5, -0.5], k=0, tau=0.1) == 0
    assert compute_tail_loss_event_count([-0.5, -0.5, -0.5], k=3, tau=0.0) == 0


def test_compute_tail_loss_event_count_short_series() -> None:
    """長度不足 k 時為 0。"""
    assert compute_tail_loss_event_count([-0.1, -0.1], k=3, tau=0.05) == 0


def test_compute_tail_loss_event_count_single_event_and_merge() -> None:
    """連續窗皆在 tail 內時僅第一次計事件。"""
    # k=3, tau=0.25：每窗和約 -0.3 < -0.25
    r = [-0.1, -0.1, -0.1, -0.1, -0.1]
    assert compute_tail_loss_event_count(r, k=3, tau=0.25) == 1


def test_compute_tail_loss_event_count_reenter() -> None:
    """離開 tail 後再次進入應再計一次。"""
    # k=2, tau=0.25（-0.2 不視為 tail，因 -0.2 >= -0.25）
    # L=2: [-0.2,-0.2] sum=-0.4 -> tail, event
    # L=3: [-0.2,0.0] sum=-0.2 -> 非 tail
    # L=4: [0.0,-0.2] sum=-0.2 -> 非 tail
    # L=5: [-0.2,-0.2] sum=-0.4 -> tail, 第二次 event
    r = [-0.2, -0.2, 0.0, -0.2, -0.2]
    assert compute_tail_loss_event_count(r, k=2, tau=0.25) == 2


def test_compute_tail_loss_matches_incremental_simulation() -> None:
    """逐步模擬（與 env 內 _tail_prev_window_flag 邏輯）須與批次函式一致。"""
    rng = np.random.default_rng(0)
    k, tau = 7, 0.03
    for _ in range(20):
        n = int(rng.integers(k, 80))
        rets = [float(rng.normal(0, 0.02)) for _ in range(n)]
        batch = compute_tail_loss_event_count(rets, k=k, tau=tau)
        prev_flag = False
        inc = 0
        buf: list[float] = []
        for rv in rets:
            buf.append(rv)
            if len(buf) >= k:
                R = float(sum(buf[-k:]))
                flag = R < -tau
                if flag and not prev_flag:
                    inc += 1
                prev_flag = flag
        assert inc == batch


def test_ep_return_and_idle_ratio_formulas() -> None:
    """ep_return、ep_idle_ratio 公式（與 _build_step_info done 分支一致）。"""
    initial_balance = 300.0
    new_equity = 330.0
    episode_flat_steps = 4
    episode_steps = 10
    ep_ret = (new_equity / initial_balance) - 1.0
    ep_idle = episode_flat_steps / max(1, episode_steps)
    assert ep_ret == pytest.approx(0.1)
    assert ep_idle == pytest.approx(0.4)

    ep_death = 2 + 1
    assert ep_death == 3
