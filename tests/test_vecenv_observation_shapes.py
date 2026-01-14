from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stable_baselines3.common.vec_env import DummyVecEnv

from Env.trading_env import TradingEnvironment


def _make_market_with_1d_start_later(*, n_5m: int = 200, n_1d: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    建立一組「5m 比 1d 更早開始」的資料，重現真實資料 inner join 後可能出現的狀況：
    - df_5m 起點早於 df_1d 起點
    - 若 map_5m_to_1d 未做下界保護，price_seq_1d padding 可能產生非固定長度
    """
    ts_5m = pd.date_range("2020-01-01", periods=n_5m, freq="5min")
    ts_1d = pd.date_range("2020-03-01", periods=n_1d, freq="1d")  # 故意晚兩個月

    df_5m = pd.DataFrame(
        {
            "timestamp": ts_5m,
            "open": np.full(n_5m, 100.0),
            "high": np.full(n_5m, 101.0),
            "low": np.full(n_5m, 99.0),
            "close": np.full(n_5m, 100.0),
            "volume": np.full(n_5m, 100.0),
            "buy_volume": np.full(n_5m, 55.0),
            "sell_volume": np.full(n_5m, 45.0),
            "trades": np.full(n_5m, 500.0),
            "quote_volume": np.full(n_5m, 10000.0),
            "volume_ratio": np.full(n_5m, 1.0),
            "long_short_ratio": np.full(n_5m, 1.0),
        }
    )
    df_1d = pd.DataFrame(
        {
            "timestamp": ts_1d,
            "open": np.full(n_1d, 100.0),
            "high": np.full(n_1d, 102.0),
            "low": np.full(n_1d, 98.0),
            "close": np.full(n_1d, 100.0),
            "volume": np.full(n_1d, 1000.0),
            # 這裡不提供 coinglass/macro 欄位，FeatureTransformer 會補 0，重點是 shape 必須固定
        }
    )
    return df_5m, df_1d


def test_vecenv_can_stack_dict_observations_without_shape_mismatch(monkeypatch) -> None:
    """
    目的：避免你遇到的錯誤：
      ValueError: all input arrays must have the same shape

    這個測試用 DummyVecEnv（同進程）快速重現「多環境 stack observation」的行為，
    並確保 price_seq_1d 永遠是固定 shape。
    """
    df_5m, df_1d = _make_market_with_1d_start_later()

    def _fake_load_data():
        return df_5m.copy(), df_1d.copy()

    # TradingEnvironment 是從 Env.trading_env import load_data
    monkeypatch.setattr("Env.trading_env.load_data", _fake_load_data, raising=True)

    def make_one(_rank: int):
        return TradingEnvironment(env_id=_rank, random_start=True, window_size=32, window_size_1d=30, target_symbol="BTCUSDT")

    venv = DummyVecEnv([lambda r=i: make_one(r) for i in range(4)])
    obs = venv.reset()

    # reset 後 obs 已經被 stack 成 batch，驗證關鍵 key shape
    # 不要硬編碼特徵維度：feature set 可能擴充（例如加入結構化趨勢 + 跨市場摘要）。
    seq_5m_dim = int(venv.observation_space["price_seq"].shape[1])
    seq_1d_dim = int(venv.observation_space["price_seq_1d"].shape[1])
    assert obs["price_seq"].shape == (4, 32, seq_5m_dim)
    assert obs["price_seq_1d"].shape == (4, 30, seq_1d_dim)

    # step 一次也要能 stack
    actions = np.zeros((4, 1), dtype=np.float32)
    obs2, rewards, dones, infos = venv.step(actions)
    assert obs2["price_seq"].shape == (4, 32, seq_5m_dim)
    assert obs2["price_seq_1d"].shape == (4, 30, seq_1d_dim)


