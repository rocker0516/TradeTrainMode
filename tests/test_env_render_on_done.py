from __future__ import annotations

import os

import numpy as np


def test_env_render_on_done_injects_render_path(tmp_path) -> None:
    """
    Regression test for VecEnv-style auto-reset:

    - When render_on_done=True, TradingEnvironment should render on the terminal step
      and inject the saved image path into info["render_path"].
    - This allows callbacks to access render output even if a VecEnv resets immediately.
    """
    try:
        from Env.trading_env import TradingEnvironment
    except Exception as e:
        raise AssertionError(f"Failed to import TradingEnvironment: {e}")

    out_dir = tmp_path / "renders"
    env = TradingEnvironment(
        env_id=0,
        random_start=False,
        data_split_enabled=False,
        # keep episode short for test speed
        max_episode_steps=30,
        min_episode_steps=10,
        render_enabled=True,
        render_dir=str(out_dir),
        render_save=True,
        render_show=False,
        render_on_done=True,
    )

    _obs, _info = env.reset(seed=123)

    done = False
    last_info: dict | None = None
    for _ in range(60):
        if done:
            break
        action = np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)
        _obs, _reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        last_info = info if isinstance(info, dict) else None

    assert done is True
    assert last_info is not None
    assert "render_path" in last_info, "terminal info should include render_path when render_on_done=True"
    out_path = str(last_info["render_path"])
    assert os.path.exists(out_path), f"render output missing: {out_path}"
    assert os.path.getsize(out_path) > 0

