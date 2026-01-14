from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from Env.Components.market_data import MarketData


def _read_prefixed_csv_head(path: Path, *, prefix: str, nrows: int) -> pd.DataFrame:
    """
    讀取真實 CSV 的前 nrows 筆，並模擬 Env/load_file.py 的 rename：
    - 除 timestamp 外，全部加上 {prefix}_ 前綴
    """
    df = pd.read_csv(path, nrows=int(nrows))
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"])
    else:
        raise ValueError(f"CSV missing timestamp/time: {path}")

    df = df.rename(columns=lambda c: f"{prefix}_{c}" if c != "timestamp" else c)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df


def _merge_on_timestamp(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    out = dfs[0]
    for d in dfs[1:]:
        out = pd.merge(out, d, on="timestamp", how="inner")
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return out


def _col_index(cols: List[str]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(cols)}


@pytest.mark.parametrize(
    "symbols",
    [
        ("BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT"),
    ],
)
def test_cnn_5m_features_are_normalized_on_real_data_head(symbols: Tuple[str, ...]) -> None:
    """
    integration-style：直接取 Data/ 內真實檔案的「前幾千行」做檢查。

    驗證重點（與合成資料測試不同）：
    - 真實資料的尺度/偏態更極端，若正規化或 clip 失效，這裡更容易爆掉。
    - 測試仍需快速，因此只取 head(nrows)。
    """
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "Data"
    if not data_dir.exists():
        pytest.skip("Data/ directory not found")

    nrows_5m = 6000
    five_min_files: Dict[str, Path] = {}
    for sym in symbols:
        # 專案目前的命名：{SYM}_futures_volume_5years_5min.csv
        p = data_dir / f"{sym}_futures_volume_5years_5min.csv"
        if not p.exists():
            pytest.skip(f"missing required 5m file: {p}")
        five_min_files[sym] = p

    dfs_5m = [_read_prefixed_csv_head(five_min_files[sym], prefix=sym, nrows=nrows_5m) for sym in symbols]
    df_5m = _merge_on_timestamp(dfs_5m)
    # 需要足夠長度讓 rolling z-score 有意義
    if len(df_5m) < 1200:
        pytest.skip(f"merged 5m too short after inner join: len={len(df_5m)}")

    # 1d：這裡用「從 5m timestamp 下採樣成日」建立最小可用 df，避免讀入巨大 1d 檔案。
    # FeatureTransformer 對缺欄會補 0；我們在此測試重點是 5m 的 CNN 特徵正規化。
    ts_1d = pd.to_datetime(df_5m["timestamp"].dt.floor("1d").unique())
    ts_1d = pd.Series(ts_1d).sort_values().reset_index(drop=True)
    n_1d = min(200, len(ts_1d))
    ts_1d = ts_1d.iloc[:n_1d]
    df_1d = pd.DataFrame(
        {
            "timestamp": ts_1d,
            "open": np.full(n_1d, 100.0),
            "high": np.full(n_1d, 102.0),
            "low": np.full(n_1d, 98.0),
            "close": np.full(n_1d, 100.0),
            "volume": np.full(n_1d, 1000.0),
        }
    )

    target = "BTCUSDT"
    md = MarketData(
        df_5m,
        df_1d,
        window_size=256,
        window_size_1d=21,
        target_symbol=target,
        feature_symbols=list(symbols),
    )

    feats = md.features_5m_arr
    cols = md.cols_5m
    idx = _col_index(cols)

    assert feats.dtype == np.float32
    assert feats.shape[0] == len(df_5m)
    assert feats.shape[1] == len(cols)
    assert np.isfinite(feats).all()

    # 1) *_z clip 範圍
    z_cols = [c for c in cols if c.endswith("_z")]
    for c in z_cols:
        x = feats[:, idx[c]]
        assert float(np.max(x)) <= 5.0001
        assert float(np.min(x)) >= -5.0001

    # 2) bounded 範圍（同合成測試）
    bounded_specs = {
        "price_pos_96": (-1.0, 1.0),
        "price_pos_288": (-1.0, 1.0),
        "bb_pos_48": (-2.0, 2.0),
        "rsi_14": (-1.0, 1.0),
        "macd_atr": (-10.0, 10.0),
        "macd_signal_atr": (-10.0, 10.0),
        "trend_strength_atr": (0.0, 10.0),
        "dir_persist_20": (-1.0, 1.0),
        "chop_48": (-5.0, 5.0),
        "trend_flip_rate_48": (-1.0, 1.0),
        "alts_trend_up_ratio": (-1.0, 1.0),
    }
    for c, (lo, hi) in bounded_specs.items():
        assert c in idx
        x = feats[:, idx[c]]
        assert float(np.max(x)) <= hi + 1e-6
        assert float(np.min(x)) >= lo - 1e-6

    # 3) 每個 alt symbol 的 4 通道存在
    for sym in symbols:
        if sym == target:
            continue
        prefix = sym.lower()
        for suffix in ("ret_15m_z", "rel_ret_15m_z", "volume_log_z", "trend_spread_z"):
            assert f"{prefix}_{suffix}" in idx


