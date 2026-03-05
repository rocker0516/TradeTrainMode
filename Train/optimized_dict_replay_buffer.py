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
        # Copy to avoid modification by reference；用 copyto 寫入預分配 slot，避免 np.array 產生臨時大陣列（it/s 優化）
        for key in self.observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            np.copyto(self.observations[key][self.pos], obs[key])

        if self.optimize_memory_usage:
            next_pos = (self.pos + 1) % self.buffer_size
            for key in self.observations.keys():
                if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                    next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
                np.copyto(self.observations[key][next_pos], next_obs[key])
        else:
            for key in self.next_observations.keys():
                if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                    next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
                np.copyto(self.next_observations[key][self.pos], next_obs[key])

        # Reshape to handle multi-dim and discrete action spaces
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _to_torch_cuda_friendly(self, arr: np.ndarray):
        """搬到 device，cuda 時使用 non_blocking=True 讓傳輸與後續計算有機會重疊。"""
        if not isinstance(arr, np.ndarray):
            return self.to_torch(arr)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        t = th.from_numpy(arr)
        dev = self.device
        if str(dev).startswith("cuda"):
            return t.to(dev, non_blocking=True)
        return t.to(dev)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:  # type: ignore[override]
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

        # 大體積 obs 用 non_blocking 搬到 GPU，減少「卡在傳輸」的瓶頸（僅 cuda 時）
        observations = {key: self._to_torch_cuda_friendly(obs) for key, obs in obs_.items()}
        next_observations = {key: self._to_torch_cuda_friendly(obs) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
        )

