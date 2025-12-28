from __future__ import annotations

import numpy as np

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


