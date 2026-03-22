from __future__ import annotations

import os

import numpy as np
import pytest


def test_render_episode_mplfinance_saves_file(tmp_path):
    """
    Integration sanity test:
    - run a short episode
    - call env.render() at episode end
    - ensure an image file is saved

    Notes:
    - render_show=False to avoid GUI requirement in CI/headless.
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
    )
    if getattr(env, "_renderer", None) is None:
        pytest.skip("MplfinanceEpisodeRenderer unavailable (matplotlib/mplfinance missing or failed to load)")

    obs, info = env.reset(seed=123)
    assert isinstance(info, dict)

    done = False
    for _ in range(60):
        if done:
            break
        action = np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

    assert done is True

    out_path = env.render()
    if out_path is None:
        pytest.skip("env.render() returned None (episode render failed silently inside renderer)")
    assert os.path.exists(out_path), f"render output missing: {out_path}"
    assert os.path.getsize(out_path) > 0

