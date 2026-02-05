"""
整合測試：合成資料 + 多 episode 隨機步 + render_on_done。

- 不依賴 Data/*.csv，使用 patch_env_load_data + make_synth_market。
- 驗證每 episode 結束時 info["render_path"] 存在且檔案有效。
- 可選：檢查 obs / reward / info 基本型別與 key，確保整條 step 管線正常。
"""
from __future__ import annotations

import os

import numpy as np
import pytest


def test_env_random_steps_with_render_integration(
    tmp_path,
    patch_env_load_data,
    make_synth_market,
) -> None:
    """
    多 episode 隨機步 + render_on_done：每 episode 結束應產出 render 圖檔。
    """
    try:
        from Env.trading_env import TradingEnvironment
    except Exception as e:
        raise AssertionError(f"Failed to import TradingEnvironment: {e}")

    market = make_synth_market(n_5m=150, n_1d=40)
    patch_env_load_data(market=market)

    out_dir = tmp_path / "renders"
    env = TradingEnvironment(
        env_id=0,
        random_start=False,
        data_split_enabled=False,
        window_size=10,
        window_size_1d=5,
        max_episode_steps=50,
        min_episode_steps=10,
        render_enabled=True,
        render_dir=str(out_dir),
        render_save=True,
        render_show=False,
        render_on_done=True,
    )

    render_paths: list[str] = []
    num_episodes = 3

    for ep in range(num_episodes):
        obs, info = env.reset(seed=42 + ep)
        assert isinstance(obs, dict), "obs should be dict"
        assert isinstance(info, dict), "reset info should be dict"

        done = False
        last_info: dict | None = None
        step_count = 0
        max_steps_per_ep = 80

        while not done and step_count < max_steps_per_ep:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            last_info = info if isinstance(info, dict) else None
            step_count += 1
            assert isinstance(reward, (float, np.floating)), "reward should be float"
            if step_count == 1 and ep == 0:
                for k, v in obs.items():
                    assert hasattr(v, "shape"), f"obs['{k}'] should have shape"

        assert done is True, f"episode {ep} should finish done"
        assert last_info is not None, "terminal step should have info"
        assert "render_path" in last_info, (
            "terminal info should include render_path when render_on_done=True"
        )
        out_path = str(last_info["render_path"])
        assert os.path.exists(out_path), f"render output missing: {out_path}"
        assert os.path.getsize(out_path) > 0, f"render file empty: {out_path}"
        render_paths.append(out_path)

        if "final_balance" in last_info:
            assert isinstance(last_info["final_balance"], (int, float))
        if "episode_max_dd" in last_info:
            assert isinstance(last_info["episode_max_dd"], (int, float))

    assert len(render_paths) == num_episodes, "should have one render path per episode"
    env.close()
