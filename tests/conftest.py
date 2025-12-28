from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import pytest


@dataclass(frozen=True)
class SyntheticMarket:
    """合成市場資料（5m / 1d）。"""

    df_5m: pd.DataFrame
    df_1d: pd.DataFrame


def _make_ohlcv_series(
    *,
    n: int,
    start_price: float = 100.0,
    step: float = 0.0,
    spread: float = 1.0,
    start_ts: str = "2020-01-01",
    freq: str = "5min",
) -> pd.DataFrame:
    """建立可控的 OHLCV 序列（timestamp 必為 datetime），供 Env 測試使用。"""
    if n < 5:
        raise ValueError("n must be >= 5")
    if start_price <= 0:
        raise ValueError("start_price must be > 0")
    if spread <= 0:
        raise ValueError("spread must be > 0")

    ts = pd.date_range(start=start_ts, periods=n, freq=freq)
    close = start_price + step * np.arange(n, dtype=np.float64)
    close = np.maximum(close, 1e-6)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = np.full(n, 100.0, dtype=np.float64)

    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_.astype(np.float64),
            "high": high.astype(np.float64),
            "low": low.astype(np.float64),
            "close": close.astype(np.float64),
            "volume": vol.astype(np.float64),
        }
    )


@pytest.fixture()
def make_synth_market() -> Callable[..., SyntheticMarket]:
    """回傳工廠：可建立合成的 5m 與 1d DataFrame。"""

    def _factory(
        *,
        n_5m: int = 200,
        n_1d: int = 60,
        start_price: float = 100.0,
        step_5m: float = 0.0,
        step_1d: float = 0.0,
        spread_5m: float = 1.0,
        spread_1d: float = 2.0,
    ) -> SyntheticMarket:
        df_5m = _make_ohlcv_series(
            n=n_5m,
            start_price=start_price,
            step=step_5m,
            spread=spread_5m,
            start_ts="2020-01-01",
            freq="5min",
        )
        df_1d = _make_ohlcv_series(
            n=n_1d,
            start_price=start_price,
            step=step_1d,
            spread=spread_1d,
            start_ts="2019-11-01",
            freq="1d",
        )
        return SyntheticMarket(df_5m=df_5m, df_1d=df_1d)

    return _factory


@pytest.fixture()
def patch_env_load_data(monkeypatch: pytest.MonkeyPatch, make_synth_market) -> Callable[..., Tuple[pd.DataFrame, pd.DataFrame]]:
    """monkeypatch Env.trading_env.load_data 讓 TradingEnvironment 不讀 Data/*.csv。"""

    def _patch(*, market: SyntheticMarket | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if market is None:
            market = make_synth_market()

        def _fake_load_data():
            return market.df_5m.copy(), market.df_1d.copy()

        monkeypatch.setattr("Env.trading_env.load_data", _fake_load_data, raising=True)
        return market.df_5m, market.df_1d

    return _patch


