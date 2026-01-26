from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from Env.Components.market_data import MarketData
from Env.Components.observer import TradingObserver
from LiveTradingRunner.obs_builder import LiveObsBuilder


def _prefix_df(df: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    out = df.copy()
    rename = {c: (f"{symbol}_{c}" if c != "timestamp" else c) for c in out.columns}
    return out.rename(columns=rename)


def _ensure_required_5m_cols(df_pref: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    """補齊 FeatureTransformer 需要的 5m 欄位（若無則填 0）。"""
    out = df_pref.copy()
    required = [
        f"{symbol}_buy_volume",
        f"{symbol}_sell_volume",
        f"{symbol}_volume_ratio",
        f"{symbol}_long_short_ratio",
        f"{symbol}_trades",
        f"{symbol}_quote_volume",
    ]
    for c in required:
        if c not in out.columns:
            out[c] = 0.0
    return out


def test_live_obs_builder_shapes_match_observation_space(make_synth_market) -> None:
    market = make_synth_market(n_5m=400, n_1d=120, step_5m=0.01, step_1d=0.5)

    # 模擬 live runner 的 df_5m：prefixed + dummy row
    df5 = _prefix_df(market.df_5m, symbol="BTCUSDT")
    df5 = _ensure_required_5m_cols(df5, symbol="BTCUSDT")
    dummy = df5.iloc[[-1]].copy()
    dummy.loc[:, "timestamp"] = pd.Timestamp(df5["timestamp"].iloc[-1]) + pd.Timedelta(minutes=5)
    df5 = pd.concat([df5, dummy], ignore_index=True)

    df1 = market.df_1d.copy()
    # 1d 只要求 timestamp 存在即可（缺的 coinglass/macro 欄位會自動補 0）
    assert "timestamp" in df1.columns

    builder = LiveObsBuilder(
        target_symbol="BTCUSDT",
        feature_symbols=("BTCUSDT",),
        window_size_5m=288,
        window_size_1d=14,
        leverage=5.0,
        obs_dtype="float32",
    )
    out = builder.build(df_5m=df5, df_1d=df1, equity_usdt=1000.0, current_position_qty=0.0)
    obs: Dict[str, np.ndarray] = out.obs

    # 以同一套 MarketData/Observer 的 observation_space 作為對照基準
    md = MarketData(
        df_5m=df5,
        df_1d=df1,
        window_size=288,
        window_size_1d=14,
        target_symbol="BTCUSDT",
        feature_symbols=["BTCUSDT"],
    )
    ob = TradingObserver(288, 14, md, obs_dtype="float32")
    space = ob.observation_space

    assert set(obs.keys()) == set(space.spaces.keys())
    for k, sp in space.spaces.items():
        assert obs[k].shape == sp.shape
        assert np.asarray(obs[k]).dtype == np.float32


