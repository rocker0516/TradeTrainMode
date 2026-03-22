from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from Env.Components.market_data import MarketData
from Env.config import Config


def _make_multi_symbol_5m(
    *,
    n: int = 2500,
    symbols: Tuple[str, ...],
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """
    產生可重現的多幣 5m 資料（prefixed 欄位），用於測試特徵正規化/clip。
    """
    rng = np.random.default_rng(7)
    ts = pd.date_range(start=start, periods=n, freq="5min")

    out: pd.DataFrame | None = None
    for i, sym in enumerate(symbols):
        base = 100.0 * (1.0 + 0.4 * i)
        trend = np.linspace(0.0, 20.0 * (1.0 + 0.1 * i), n)
        cyc = np.sin(np.linspace(0.0, 30.0, n)) * (2.0 + 0.5 * i)
        noise = rng.normal(0.0, 0.5 + 0.2 * i, n)
        close = np.clip(base + trend + cyc + noise, 1.0, None)

        open_ = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(open_, close) + (0.5 + 0.2 * i)
        low = np.minimum(open_, close) - (0.5 + 0.2 * i)

        volume = np.clip(rng.lognormal(mean=4.5 + 0.2 * i, sigma=0.4, size=n), 1.0, None)
        buy_volume = volume * np.clip(0.55 + rng.normal(0.0, 0.05, n), 0.05, 0.95)
        sell_volume = np.clip(volume - buy_volume, 0.0, None)
        trades = np.clip(rng.normal(500.0 + 50.0 * i, 80.0, n), 1.0, None)
        quote_volume = close * volume

        volume_ratio = np.clip((buy_volume + 1.0) / (sell_volume + 1.0), 0.0, 10.0)
        long_short_ratio = np.clip((buy_volume + 10.0) / (sell_volume + 10.0), 0.0, 10.0)

        df = pd.DataFrame(
            {
                "timestamp": ts,
                f"{sym}_open": open_,
                f"{sym}_high": high,
                f"{sym}_low": low,
                f"{sym}_close": close,
                f"{sym}_volume": volume,
                f"{sym}_buy_volume": buy_volume,
                f"{sym}_sell_volume": sell_volume,
                f"{sym}_trades": trades,
                f"{sym}_quote_volume": quote_volume,
                f"{sym}_volume_ratio": volume_ratio,
                f"{sym}_long_short_ratio": long_short_ratio,
            }
        )

        out = df if out is None else pd.merge(out, df, on="timestamp", how="inner")

    assert out is not None
    return out


def _make_1d_minimal(*, n: int = 120, start: str = "2019-10-01") -> pd.DataFrame:
    ts = pd.date_range(start=start, periods=n, freq="1d")
    close = 100.0 + np.linspace(0.0, 10.0, n)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 2.0
    low = np.minimum(open_, close) - 2.0
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": 1000.0})


def _col_index(cols: List[str]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(cols)}


@pytest.mark.parametrize(
    "symbols",
    [
        ("BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT"),
    ],
)
def test_cnn_price_seq_target_features_are_normalized_and_bounded(symbols: Tuple[str, ...]) -> None:
    """
    驗證 5m `price_seq_target` 來源矩陣（`features_5m_target_arr`）：
    - dtype=float32、無 NaN/inf
    - rolling z 類通道 clip 到 [-5, 5]
    - scale / ratio 類在合理 bounded 範圍
    - 多幣時 `others` 矩陣含各 alt 前綴欄位
    """
    df_5m = _make_multi_symbol_5m(n=2500, symbols=symbols)
    df_1d = _make_1d_minimal(n=150)

    target = "BTCUSDT"
    md = MarketData(
        df_5m,
        df_1d,
        window_size=256,
        window_size_1d=21,
        target_symbol=target,
        feature_symbols=list(symbols),
    )
    assert md.cols_5m_target == list(Config.OBS_PRICE_SEQ_TARGET_COLS)

    feats = md.features_5m_target_arr
    cols = md.cols_5m_target

    assert feats.dtype == np.float32
    assert feats.shape[0] == len(df_5m)
    assert feats.shape[1] == len(cols)
    assert np.isfinite(feats).all()

    idx = _col_index(cols)

    z_cols = [c for c in cols if c.endswith("_z") or c.endswith("_z_short")]
    assert len(z_cols) >= 8
    for c in z_cols:
        x = feats[:, idx[c]]
        assert np.nanmax(x) <= 5.0001
        assert np.nanmin(x) >= -5.0001

    bounded_specs = {
        "ret_15m_scale": (-3.0, 3.0),
        "range_atr": (-3.0, 3.0),
        "volume_log_scale": (-3.0, 3.0),
        "alts_trend_up_ratio": (-1.0, 1.0),
        "alts_rel_ret_15m_abs_mean_scale": (0.0, 0.05),
        "alts_volume_log_mean_scale": (-3.0, 3.0),
        "ob_depth_imbalance_scale": (-1.0, 1.0),
        "rv_ratio_scale": (0.0, 0.05),
    }
    for c, (lo, hi) in bounded_specs.items():
        assert c in idx, f"missing expected feature col: {c}"
        x = feats[:, idx[c]]
        assert np.nanmax(x) <= hi + 1e-5
        assert np.nanmin(x) >= lo - 1e-5

    # 多幣：逐幣特徵在 others，欄位為 {SYMBOL}_*
    ocols = md.cols_5m_others
    for sym in symbols:
        if sym == target:
            continue
        assert any(str(col).startswith(f"{sym}_") for col in ocols), f"expected prefixed cols for {sym}"

    warmup = 800
    sample_cols = ["ret_1_z_short", "quote_volume_log_z", "amihud_z", "volume_log_scale"]
    for c in sample_cols:
        assert c in idx
        x = feats[warmup:, idx[c]]
        assert np.isfinite(x).all()
        assert float(np.std(x)) > 0.05
        assert float(np.std(x)) < 5.0
