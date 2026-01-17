from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym

from Env.wrappers import ActionRepeatWrapper
from Env.trading_env import TradingEnvironment


def test_action_repeat_wrapper_breaks_on_stop_loss(patch_env_load_data, make_synth_market, monkeypatch) -> None:
    """repeat 期間若觸發止損，wrapper 必須提早 break，避免持續重複下單。"""
    market = make_synth_market(n_5m=80, n_1d=40, start_price=100.0, step_5m=0.0, spread_5m=1.0)

    # 避免一開始 ATR 被 prev_close=0 污染（前 14 根會偏大），讓環境從較後面的 step 開始。
    # 讓 step=(window_size+1)=21 觸發止損：entry≈100, atr≈2 => stop≈96；設 low 明顯低於 stop
    market.df_5m.loc[21, "low"] = 50.0

    patch_env_load_data(market=market)

    # 強制 cooldown=0，純測 wrapper 行為
    from Env import config as env_config

    monkeypatch.setattr(env_config.Config, "STOP_LOSS_COOLDOWN_STEPS", 0, raising=True)

    env = TradingEnvironment(
        env_id=0,
        random_start=False,
        window_size=20,
        window_size_1d=7,
        min_episode_steps=30,
        max_episode_steps=30,
        stop_loss_atr=2.0,
    )
    wrapped = ActionRepeatWrapper(env, repeat=5)

    wrapped.reset(seed=1)
    start_step = env.current_step

    # 以滿倉多重複 5 次，但第二次內部 step 就會止損 -> 應提早 break
    action = np.array([1.0], dtype=np.float32)
    _, _, done, truncated, info = wrapped.step(action)

    assert info.get("stop_loss_triggered", False) is True or done or truncated
    assert env.current_step - start_step < 5


def test_action_repeat_wrapper_accumulates_cost_channels_and_breakdown() -> None:
    """repeat>1 時，wrapper 必須累積 cost_* 與 cost_breakdown，避免 λ 更新視窗看到 0。"""

    class DummyEnv(gym.Env):
        def __init__(self) -> None:
            super().__init__()
            self.i = 0
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        def reset(self, *args, **kwargs):
            self.i = 0
            return np.zeros((1,), dtype=np.float32), {}

        def step(self, action):
            self.i += 1
            # 每步固定回傳一樣的成本，方便檢查是否累加
            info = {
                "cost": 1.0,
                "cost_risk": 0.1,
                "cost_fric": 0.2,
                "cost_sl_buf": 0.3,
                "cost_sl_event": 0.4,
                "cost_breakdown": {"death_cost": 0.1, "fric_cost": 0.2, "sl_buf_cost": 0.3},
            }
            terminated = False
            truncated = False
            reward = 0.0
            obs = np.zeros((1,), dtype=np.float32)
            return obs, reward, terminated, truncated, info

    wrapped = ActionRepeatWrapper(DummyEnv(), repeat=3)
    wrapped.reset()
    _, _, _, _, info = wrapped.step(np.array([0.0], dtype=np.float32))

    assert info["cost"] == pytest.approx(3.0)
    assert info["cost_risk"] == pytest.approx(0.3)
    assert info["cost_fric"] == pytest.approx(0.6)
    assert info["cost_sl_buf"] == pytest.approx(0.9)
    assert info["cost_sl_event"] == pytest.approx(1.2)
    assert isinstance(info.get("cost_breakdown"), dict)
    assert info["cost_breakdown"]["death_cost"] == pytest.approx(0.3)
    assert info["cost_breakdown"]["fric_cost"] == pytest.approx(0.6)
    assert info["cost_breakdown"]["sl_buf_cost"] == pytest.approx(0.9)


