from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym

from Train.lagrangian import SharedLagrangianController, LagrangianRewardWrapper


class _MockEnvWithBreakdown(gym.Env):
    """可控的 mock env：提供 cost 與 cost_breakdown，且第 2 步結束回合。"""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self._i = 0

    def reset(self, **kwargs):
        self._i = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self._i += 1
        # 固定兩步：
        # step1: reward=1.0, cost=0.5 (a=0.2, b=0.3)
        # step2: reward=2.0, cost=1.0 (a=0.4, b=0.6) -> done
        if self._i == 1:
            reward = 1.0
            info = {"cost": 0.5, "cost_breakdown": {"a": 0.2, "b": 0.3}}
            return np.array([0.0], dtype=np.float32), reward, False, False, info
        reward = 2.0
        info = {"cost": 1.0, "cost_breakdown": {"a": 0.4, "b": 0.6}}
        return np.array([0.0], dtype=np.float32), reward, True, False, info


def test_wrapper_episode_metrics_reward_decomposition() -> None:
    # 固定 lambda=2.0
    ctrl = SharedLagrangianController(cost_limit=0.1, lambda_init=2.0)
    env = _MockEnvWithBreakdown()
    wrapped = LagrangianRewardWrapper(env, ctrl, reward_scale=10.0)

    wrapped.reset()
    _, r1, _, _, _ = wrapped.step([0])
    _, r2, terminated, _, info = wrapped.step([0])

    assert terminated is True
    # modified_reward = raw*scale - lam*cost
    assert r1 == pytest.approx(1.0 * 10.0 - 2.0 * 0.5)
    assert r2 == pytest.approx(2.0 * 10.0 - 2.0 * 1.0)

    metrics = info["episode_metrics"]
    assert metrics["return_orig"] == pytest.approx(3.0)
    assert metrics["return_orig_scaled"] == pytest.approx(30.0)
    assert metrics["return_cost"] == pytest.approx(1.5)
    assert metrics["cost_breakdown"]["a"] == pytest.approx(0.6)
    assert metrics["cost_breakdown"]["b"] == pytest.approx(0.9)

    # penalty_total = lam * cost（正數大小）
    assert metrics["cost_penalty_total"] == pytest.approx(2.0 * 1.5)
    assert metrics["cost_penalty_breakdown"]["a"] == pytest.approx(2.0 * 0.6)
    assert metrics["cost_penalty_breakdown"]["b"] == pytest.approx(2.0 * 0.9)

    # return_total = sum(modified_reward)
    expected_total = (1.0 * 10.0 - 2.0 * 0.5) + (2.0 * 10.0 - 2.0 * 1.0)
    assert metrics["return_total"] == pytest.approx(expected_total)


