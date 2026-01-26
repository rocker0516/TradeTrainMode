from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

# Windows/pytest 直跑時，sys.path 可能未包含專案根，導致 `import Train` 失敗
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Train.eval_callback import ConstraintEvalCallback, EvalConfig, EvalConstraints


class _TwoStepEpisodeEnv(gym.Env):
    """
    測試用最小環境：
    - 兩步結束一個 episode
    - 每步都提供 info["cost"]
    - episode 結束時提供你評估所需的 info keys
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        initial_balance: float = 10_000.0,
        final_balance: float = 11_000.0,
        episode_max_dd: float = 0.1,
        mean_step_cost: float = 0.0005,
        termination_reason: Optional[str] = None,
        episode_liq_count: int = 0,
        episode_stop_loss_count: int = 0,
        episode_trade_count: int = 2,
        long_entry_count: int = 1,
        short_entry_count: int = 1,
        episode_holding_steps: int = 1,
    ) -> None:
        super().__init__()
        self.initial_balance = float(initial_balance)
        self._final_balance = float(final_balance)
        self._episode_max_dd = float(episode_max_dd)
        self._mean_step_cost = float(mean_step_cost)
        self._termination_reason = termination_reason
        self._episode_liq_count = int(episode_liq_count)
        self._episode_stop_loss_count = int(episode_stop_loss_count)
        self._episode_trade_count = int(episode_trade_count)
        self._long_entry_count = int(long_entry_count)
        self._short_entry_count = int(short_entry_count)
        self._episode_holding_steps = int(episode_holding_steps)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._t = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._t = 0
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._t += 1
        done = self._t >= 2

        info: Dict[str, Any] = {"cost": float(self._mean_step_cost)}
        if done:
            info.update(
                {
                    "termination_reason": self._termination_reason,
                    "final_balance": float(self._final_balance),
                    "episode_max_dd": float(self._episode_max_dd),
                    "episode_stop_loss_count": int(self._episode_stop_loss_count),
                    "episode_liq_count": int(self._episode_liq_count),
                    "episode_trade_count": int(self._episode_trade_count),
                    "long_entry_count": int(self._long_entry_count),
                    "short_entry_count": int(self._short_entry_count),
                    "episode_holding_steps": int(self._episode_holding_steps),
                }
            )

        obs = np.zeros((1,), dtype=np.float32)
        reward = 0.0
        terminated = bool(done)
        truncated = False
        return obs, float(reward), terminated, truncated, info


@dataclass
class _DummyModel:
    """提供 callback 所需的最小 model 介面。"""

    num_timesteps: int
    logger: Any
    saved_paths: list[str]

    def predict(self, observation: Any, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        # DummyVecEnv 需要 shape=(n_envs, action_dim)
        return np.zeros((1, 1), dtype=np.float32), None

    def save(self, path: str) -> None:
        self.saved_paths.append(str(path))

    def get_env(self) -> None:
        # 測試不使用 training_env
        return None


def test_eval_not_triggered_before_freq(tmp_path: Path) -> None:
    eval_env = DummyVecEnv([lambda: _TwoStepEpisodeEnv()])
    model = _DummyModel(num_timesteps=0, logger=configure(folder=None, format_strings=[]), saved_paths=[])
    best_dir = tmp_path / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    cb = ConstraintEvalCallback(
        eval_env=eval_env,
        eval_config=EvalConfig(
            enabled=True,
            eval_every_timesteps=200,
            n_eval_episodes=1,
            deterministic=True,
            reject_if_death_event=True,
            constraints=EvalConstraints(max_dd_limit=0.4, mean_cost_limit=0.001),
            save_best_model=True,
            best_model_path=str(best_dir / "best_model"),
            print_each_episode=False,
        ),
    )
    cb.init_callback(model)

    model.num_timesteps = 199
    cb.on_step()
    assert model.saved_paths == []


def test_eval_rejects_when_death_event_occurs(tmp_path: Path) -> None:
    # 只要有死亡事件（強平/資金耗盡）就整次淘汰，不保存 best
    eval_env = DummyVecEnv(
        [lambda: _TwoStepEpisodeEnv(termination_reason="liq_triggered", episode_liq_count=1, final_balance=50.0)]
    )
    model = _DummyModel(num_timesteps=0, logger=configure(folder=None, format_strings=[]), saved_paths=[])
    best_dir = tmp_path / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    cb = ConstraintEvalCallback(
        eval_env=eval_env,
        eval_config=EvalConfig(
            enabled=True,
            eval_every_timesteps=1,
            n_eval_episodes=1,
            deterministic=True,
            reject_if_death_event=True,
            constraints=EvalConstraints(max_dd_limit=0.4, mean_cost_limit=0.001),
            save_best_model=True,
            best_model_path=str(best_dir / "best_model"),
            print_each_episode=False,
        ),
    )
    cb.init_callback(model)

    model.num_timesteps = 1
    cb.on_step()
    assert model.saved_paths == []


def test_eval_saves_best_when_constraints_pass_and_no_death(tmp_path: Path) -> None:
    eval_env = DummyVecEnv([lambda: _TwoStepEpisodeEnv(final_balance=12_000.0, episode_max_dd=0.2, mean_step_cost=0.0005)])
    model = _DummyModel(num_timesteps=0, logger=configure(folder=None, format_strings=[]), saved_paths=[])
    best_dir = tmp_path / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    cb = ConstraintEvalCallback(
        eval_env=eval_env,
        eval_config=EvalConfig(
            enabled=True,
            eval_every_timesteps=1,
            n_eval_episodes=1,
            deterministic=True,
            reject_if_death_event=True,
            constraints=EvalConstraints(max_dd_limit=0.4, mean_cost_limit=0.001),
            save_best_model=True,
            best_model_path=str(best_dir / "best_model"),
            print_each_episode=False,
        ),
    )
    cb.init_callback(model)

    model.num_timesteps = 1
    cb.on_step()
    assert len(model.saved_paths) == 1

