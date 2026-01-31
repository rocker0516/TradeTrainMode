from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Env.obs_quality import ObsQualityAuditor
from Env.trading_env import TradingEnvironment


def _make_minimal_df(*, n_5m: int = 500, n_1d: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """建立一組最小可用資料，用於快速跑 obs audit。"""
    ts_5m = pd.date_range("2020-01-01", periods=int(n_5m), freq="5min")
    ts_1d = pd.date_range("2019-12-01", periods=int(n_1d), freq="1d")

    df_5m = pd.DataFrame(
        {
            "timestamp": ts_5m,
            "open": np.linspace(100.0, 110.0, num=int(n_5m)),
            "high": np.linspace(101.0, 111.0, num=int(n_5m)),
            "low": np.linspace(99.0, 109.0, num=int(n_5m)),
            "close": np.linspace(100.0, 110.0, num=int(n_5m)),
            "volume": np.full(int(n_5m), 100.0),
            "buy_volume": np.full(int(n_5m), 55.0),
            "sell_volume": np.full(int(n_5m), 45.0),
            "trades": np.full(int(n_5m), 500.0),
            "quote_volume": np.full(int(n_5m), 10000.0),
            "volume_ratio": np.full(int(n_5m), 1.0),
            "long_short_ratio": np.full(int(n_5m), 1.0),
        }
    )
    df_1d = pd.DataFrame(
        {
            "timestamp": ts_1d,
            "open": np.full(int(n_1d), 100.0),
            "high": np.full(int(n_1d), 102.0),
            "low": np.full(int(n_1d), 98.0),
            "close": np.full(int(n_1d), 100.0),
            "volume": np.full(int(n_1d), 1000.0),
        }
    )
    return df_5m, df_1d


def test_obs_quality_audit_runs_and_produces_key_summaries(monkeypatch) -> None:
    df_5m, df_1d = _make_minimal_df()

    def _fake_load_data():
        return df_5m.copy(), df_1d.copy()

    monkeypatch.setattr("Env.trading_env.load_data", _fake_load_data, raising=True)

    env = TradingEnvironment(env_id=0, random_start=False, window_size=32, window_size_1d=30, target_symbol="BTCUSDT")
    auditor = ObsQualityAuditor(rng_seed=0, sample_steps=64, max_flat_elems_per_key=512)
    rep = auditor.audit(env)
    env.close()

    # 至少要包含這些 key（代表 market/account/context 都有）
    keys = {s.key for s in rep.key_summaries}
    assert {"price_seq", "price_seq_1d", "account_state", "time_state", "rhythm_state", "cost_state"}.issubset(keys)

    # sampling 後應該都 finite（observer 會 np.nan_to_num）
    for s in rep.key_summaries:
        assert s.finite_frac == pytest.approx(1.0)
        assert s.nan_count == 0
        assert s.inf_count == 0

