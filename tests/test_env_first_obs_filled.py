"""驗證訓練/評估時第一步 obs 是否在 warmup 之後正確填滿。

起點應至少為 max(window_size, window_size_1d * 288) 步，以保證：
- 5m 序列：有足夠 lookback
- 1d 序列：有足夠日線（288 步/天）
"""
from __future__ import annotations

import numpy as np
import pytest

from Env.trading_env import TradingEnvironment


def _warmup_steps(window_size: int, window_size_1d: int, ensure_filled_obs: bool) -> int:
    """與 TradingEnvironment.reset() 一致的 warmup 計算。"""
    if ensure_filled_obs:
        return max(1, int(max(window_size, window_size_1d * 288)))
    return max(1, window_size)


@pytest.fixture()
def patch_load_data_and_mk_env(monkeypatch, make_synth_market):
    """Patch load_data 並回傳建立 env 的工廠（可指定 n_5m）。"""
    def _factory(
        *,
        n_5m: int = 800,
        n_1d: int = 60,
        window_size: int = 10,
        window_size_1d: int = 2,
        random_start: bool = True,
        ensure_filled_obs: bool = True,
        **kwargs,
    ):
        market = make_synth_market(n_5m=n_5m, n_1d=n_1d)
        def _fake_load():
            return market.df_5m.copy(), market.df_1d.copy()
        monkeypatch.setattr("Env.trading_env.load_data", _fake_load, raising=True)
        return TradingEnvironment(
            env_id=0,
            df_5m=market.df_5m,
            df_1d=market.df_1d,
            random_start=random_start,
            window_size=window_size,
            window_size_1d=window_size_1d,
            ensure_filled_obs=ensure_filled_obs,
            min_episode_steps=10,
            max_episode_steps=500,
            **kwargs,
        )
    return _factory


def test_first_obs_warmup_when_ensure_filled_obs_true(patch_load_data_and_mk_env):
    """ensure_filled_obs=True 時，reset 後 episode_start_step >= max(ws, ws_1d*288)。"""
    window_size = 10
    window_size_1d = 2
    warmup = _warmup_steps(window_size, window_size_1d, ensure_filled_obs=True)
    assert warmup == 2 * 288, "warmup 應為 576"

    env = patch_load_data_and_mk_env(
        n_5m=700,
        window_size=window_size,
        window_size_1d=window_size_1d,
        random_start=True,
        ensure_filled_obs=True,
    )
    obs, info = env.reset(seed=42)
    start_step = info["episode_start_step"]
    assert start_step >= warmup, (
        f"ensure_filled_obs=True 時起點應 >= {warmup} (max(window_size, window_size_1d*288))，得到 {start_step}"
    )
    # 第一步 obs 應有有效 shape，且 1d 序列不應全為 0（有足夠歷史時會填滿）
    assert "price_seq_1d_target" in obs
    assert obs["price_seq_1d_target"].shape[0] == window_size_1d
    # 至少應有部分非零（表示有日線資料填入）
    non_zero = np.count_nonzero(obs["price_seq_1d_target"])
    assert non_zero > 0, "第一步 1d target 序列應有至少部分非零（已過 warmup）"


def test_first_obs_warmup_when_ensure_filled_obs_false(patch_load_data_and_mk_env):
    """ensure_filled_obs=False 時，起點僅保證 >= window_size。"""
    window_size = 10
    window_size_1d = 2
    warmup_min = _warmup_steps(window_size, window_size_1d, ensure_filled_obs=False)
    assert warmup_min == 10

    env = patch_load_data_and_mk_env(
        n_5m=500,
        window_size=window_size,
        window_size_1d=window_size_1d,
        random_start=True,
        ensure_filled_obs=False,
    )
    obs, info = env.reset(seed=123)
    start_step = info["episode_start_step"]
    assert start_step >= warmup_min, f"ensure_filled_obs=False 時起點應 >= {warmup_min}，得到 {start_step}"


def test_eval_style_config_uses_filled_obs(monkeypatch, make_synth_market):
    """模擬 eval 使用的 to_eval_dict：data_mode=eval 且 ensure_filled_obs=True。"""
    # 需足夠 5m 筆數才能讓起點 >= warmup（36 vs 21*288=6048）
    n_5m = 6500
    market = make_synth_market(n_5m=n_5m, n_1d=60)
    def _fake_load():
        return market.df_5m.copy(), market.df_1d.copy()
    monkeypatch.setattr("Env.trading_env.load_data", _fake_load, raising=True)

    env = TradingEnvironment(
        env_id=0,
        df_5m=market.df_5m,
        df_1d=market.df_1d,
        random_start=True,
        window_size=36,
        window_size_1d=21,
        data_mode="eval",
        ensure_filled_obs=True,
        min_episode_steps=100,
        max_episode_steps=5000,
    )
    warmup = max(36, 21 * 288)
    obs, info = env.reset(seed=1)
    start_step = info["episode_start_step"]
    assert start_step >= warmup, (
        f"Eval 環境起點應 >= {warmup} (max(36, 21*288))，得到 {start_step}"
    )
