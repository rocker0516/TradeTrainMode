from __future__ import annotations

"""
Optimized Dict Replay Buffer for SB3 (SAC/TD3).

背景：
- SB3 2.7.0 內建的 DictReplayBuffer 直接禁止 optimize_memory_usage=True：
  `assert not optimize_memory_usage, "DictReplayBuffer does not support optimize_memory_usage"`
- 但對本專案這種大尺寸 Dict observation（price_seq, price_seq_1d...）而言，
  replay buffer 會成為 RAM 爆炸的主因。

做法：
- 參考 SB3 ReplayBuffer 的 memory efficient 版本：
  - 不另外存 next_observations
  - 將 next_obs 寫入 observations[(pos+1) % buffer_size]
  - sample 時避免取到 index=self.pos（那格是下一筆寫入位置，transition 無效）

注意：
- 這個技巧依賴「每個 global step 同時寫入 n_envs 的 obs/next_obs」，
  與 SB3 ReplayBuffer 的 vectorized storage 模式一致。
- optimize_memory_usage 與 handle_timeout_termination 同時開啟在 SB3 已知會有 bug（#934），
  這裡跟 ReplayBuffer 一樣直接禁止該組合。
"""

from typing import Any, Optional, Union

import numpy as np
from gymnasium import spaces
import torch as th

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class DictReplayBufferSamplesWithAux:
    """
    具備與 DictReplayBufferSamples 相同欄位，外加 step_log_returns、has_positions，
    供 auxiliary loss（預測 step 報酬符號）使用。SAC 仍可依 batch.observations 等正常使用。
    """
    __slots__ = ("observations", "next_observations", "actions", "dones", "rewards", "discounts", "step_log_returns", "has_positions")

    def __init__(
        self,
        observations,
        next_observations,
        actions,
        dones,
        rewards,
        step_log_returns: th.Tensor,
        has_positions: th.Tensor,
        discounts=None,
    ):
        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.dones = dones
        self.rewards = rewards
        self.discounts = discounts
        self.step_log_returns = step_log_returns
        self.has_positions = has_positions


class OptimizedDictReplayBuffer(ReplayBuffer):
    """
    Dict Replay buffer with optimize_memory_usage support (for SB3 >= 2.0).
    """

    observation_space: spaces.Dict
    obs_shape: dict[str, tuple[int, ...]]  # type: ignore[assignment]
    observations: dict[str, np.ndarray]  # type: ignore[assignment]
    next_observations: dict[str, np.ndarray]  # only created when optimize_memory_usage=False

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        # Bypass ReplayBuffer.__init__ allocations; reuse its utilities (_normalize_obs/_normalize_reward/to_torch/_maybe_cast_dtype).
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "OptimizedDictReplayBuffer must be used with Dict obs space only"

        # Same behavior as SB3 buffers: real storage is per-env
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Keep same safety constraint as SB3 ReplayBuffer
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "OptimizedDictReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = bool(optimize_memory_usage)

        # observations storage
        self.observations = {
            key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }

        if not self.optimize_memory_usage:
            self.next_observations = {
                key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
                for key, _obs_shape in self.obs_shape.items()
            }

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # 供 auxiliary loss（預測 step 報酬符號）使用
        self.step_log_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.has_positions = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = np.array(obs[key])

        if self.optimize_memory_usage:
            # Store next obs in-place in the observations ring buffer
            next_pos = (self.pos + 1) % self.buffer_size
            for key in self.observations.keys():
                if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                    next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
                self.observations[key][next_pos] = np.array(next_obs[key])
        else:
            for key in self.next_observations.keys():
                if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                    next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
                self.next_observations[key][self.pos] = np.array(next_obs[key])

        # Reshape to handle multi-dim and discrete action spaces
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        self.step_log_returns[self.pos] = np.array([float(info.get("step_log_return", 0.0)) for info in infos])
        self.has_positions[self.pos] = np.array([1.0 if info.get("has_position", False) else 0.0 for info in infos])

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):  # type: ignore[override]
        # Match SB3 ReplayBuffer behavior when optimize_memory_usage=True:
        # Do not sample index `self.pos` because transition is invalid (next_obs overlaps).
        if not self.optimize_memory_usage:
            return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)

        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}, env)
        assert isinstance(obs_, dict)

        if self.optimize_memory_usage:
            next_obs_ = self._normalize_obs(
                {key: obs[(batch_inds + 1) % self.buffer_size, env_indices, :] for key, obs in self.observations.items()},
                env,
            )
        else:
            next_obs_ = self._normalize_obs(
                {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()},
                env,
            )
        assert isinstance(next_obs_, dict)

        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        step_log_returns = self.to_torch(self.step_log_returns[batch_inds, env_indices].reshape(-1, 1))
        has_positions = self.to_torch(self.has_positions[batch_inds, env_indices].reshape(-1, 1))

        return DictReplayBufferSamplesWithAux(
            observations=observations,
            next_observations=next_observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
            step_log_returns=step_log_returns,
            has_positions=has_positions,
            discounts=None,
        )

