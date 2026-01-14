from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym

from Train.optimized_dict_replay_buffer import OptimizedDictReplayBuffer


def test_optimized_dict_replay_buffer_supports_optimize_memory_usage_true() -> None:
    obs_space = gym.spaces.Dict(
        {
            "a": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "b": gym.spaces.Box(low=-1.0, high=1.0, shape=(3, 4), dtype=np.float32),
        }
    )
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    buf = OptimizedDictReplayBuffer(
        buffer_size=16,
        observation_space=obs_space,
        action_space=act_space,
        n_envs=1,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    obs = {"a": np.array([0.1, -0.2], dtype=np.float32), "b": np.zeros((3, 4), dtype=np.float32)}
    next_obs = {"a": np.array([0.2, -0.1], dtype=np.float32), "b": np.ones((3, 4), dtype=np.float32)}
    action = np.array([0.0], dtype=np.float32)
    reward = np.array([1.0], dtype=np.float32)
    done = np.array([0.0], dtype=np.float32)

    buf.add(obs=obs, next_obs=next_obs, action=action, reward=reward, done=done, infos=[{}])
    samples = buf.sample(batch_size=1)

    assert set(samples.observations.keys()) == {"a", "b"}
    assert set(samples.next_observations.keys()) == {"a", "b"}
    assert samples.observations["a"].shape[-1] == 2
    assert samples.next_observations["b"].shape[-2:] == (3, 4)
    assert samples.actions.shape[-1] == 1
