from __future__ import annotations

import numpy as np
import pandas as pd

from Env.Components.market_data import MarketData


def _make_5m_ohlc(*, start: str = "2020-01-01", days: int = 80) -> pd.DataFrame:
    n = int(days) * 288
    ts = pd.date_range(start, periods=n, freq="5min")
    # 讓價格有緩慢趨勢 + 週期波動，避免回傳全 0
    t = np.linspace(0.0, 10.0, num=n)
    close = 100.0 + 5.0 * np.sin(t) + 0.01 * np.arange(n)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 100.0),
            "buy_volume": np.full(n, 55.0),
            "sell_volume": np.full(n, 45.0),
            "trades": np.full(n, 500.0),
            "quote_volume": np.full(n, 10000.0),
            "volume_ratio": np.full(n, 1.0),
            "long_short_ratio": np.full(n, 1.0),
        }
    )
    return df


def _make_1d_macro_misaligned(*, start: str = "2020-01-01", days: int = 200) -> pd.DataFrame:
    # 故意用 00:20:01，模擬真實 fear_greed 這種「同一天不同時間」來源
    ts = pd.date_range(start, periods=int(days), freq="1d") + pd.Timedelta(minutes=20, seconds=1)
    fear = np.clip(50.0 + 10.0 * np.sin(np.linspace(0.0, 3.0, num=int(days))), 0.0, 100.0)
    alt = np.clip(60.0 + 5.0 * np.cos(np.linspace(0.0, 4.0, num=int(days))), 0.0, 100.0)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            # 欄位名稱對齊 FeatureTransformer 的期待（load_file.py prefix 後也會長這樣）
            "fear_greed_fear_greed_index": fear,
            "altcoin_season_altcoin_index": alt,
        }
    )
    return df


def test_market_data_normalizes_1d_timestamps_and_fills_target_ohlc_from_5m() -> None:
    df_5m = _make_5m_ohlc(days=90)
    df_1d = _make_1d_macro_misaligned(days=120)

    # 注意：df_5m/df_1d 都是「未 prefixed」版本，MarketData/FeatureTransformer 都有 fallback 能處理
    md = MarketData(
        df_5m=df_5m,
        df_1d=df_1d,
        window_size=64,
        window_size_1d=30,
        target_symbol="BTCUSDT",
        feature_symbols=["BTCUSDT"],
    )

    # 1) timestamp 應被 normalize 到午夜（00:00:00）
    assert (pd.to_datetime(md.df_1d["timestamp"]).dt.hour == 0).all()
    assert (pd.to_datetime(md.df_1d["timestamp"]).dt.minute == 0).all()

    # 2) 應有 target_symbol 的 1d OHLC（由 5m 推導填補）
    for c in ("BTCUSDT_open", "BTCUSDT_high", "BTCUSDT_low", "BTCUSDT_close"):
        assert c in md.df_1d.columns
        # 至少要有一些非空值（覆蓋 5m 的日區間）
        assert float(md.df_1d[c].notna().mean()) > 0.2

    # 3) 1d 特徵不應全部為 0（至少 ret_1d_z 與 fear_greed_z 有變化）
    cols_1d = list(md.cols_1d)
    X = np.asarray(md.features_1d_arr, dtype=np.float64)
    assert X.ndim == 2 and X.shape[0] > 0 and X.shape[1] == len(cols_1d)

    for name in ("ret_1d_z", "fear_greed_z"):
        assert name in cols_1d
        v = X[:, cols_1d.index(name)]
        assert float(np.std(v)) > 1e-6

