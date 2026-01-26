from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from Env.Components.market_data import MarketData


def _make_multi_symbol_5m(
    *,
    n: int = 2500,
    symbols: Tuple[str, ...],
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """
    產生可重現的多幣 5m 資料（prefixed 欄位），用於測試特徵正規化/clip。

    注意：我們刻意讓每個 symbol 的 close/volume 有不同尺度與波動，
    以檢查 z-score + clip 是否能把尺度拉回穩定範圍、避免 inf/NaN。
    """
    rng = np.random.default_rng(7)
    ts = pd.date_range(start=start, periods=n, freq="5min")

    out: pd.DataFrame | None = None
    for i, sym in enumerate(symbols):
        # close: trend + sin + noise；不同 symbol 用不同尺度
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
    # 1d 特徵缺欄會自動補 0；這裡重點只在 5m 正規化測試
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
def test_cnn_price_seq_features_are_normalized_and_bounded(symbols: Tuple[str, ...]) -> None:
    """
    驗證 CNN 主要輸入 price_seq（5m）：
    - dtype=float32
    - 不含 NaN/inf
    - *_z 通道都已 clip 到 [-5, 5]
    - bounded 通道落在預期範圍（避免尺度爆炸導致 SAC 發散）
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
    feats = md.features_5m_arr
    cols = md.cols_5m

    # MarketData 內部使用 float32；env 輸出 obs 會轉成 float16 以省 RAM
    assert feats.dtype == np.float32
    assert feats.shape[0] == len(df_5m)
    assert feats.shape[1] == len(cols)
    assert np.isfinite(feats).all()

    idx = _col_index(cols)

    # ---- 1) 所有 *_z 必須在 [-5, 5] ----
    z_cols = [c for c in cols if c.endswith("_z")]
    assert len(z_cols) > 10  # 基本 sanity check
    for c in z_cols:
        x = feats[:, idx[c]]
        assert np.nanmax(x) <= 5.0001
        assert np.nanmin(x) >= -5.0001

    # ---- 2) bounded / ratio 類特徵 ----
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
        assert c in idx, f"missing expected feature col: {c}"
        x = feats[:, idx[c]]
        assert np.nanmax(x) <= hi + 1e-6
        assert np.nanmin(x) >= lo - 1e-6

    # ---- 3) 跨市場 per-symbol 4 通道應存在（每個 alt 幣）----
    for sym in symbols:
        if sym == target:
            continue
        prefix = sym.lower()
        for suffix in ("ret_15m_z", "rel_ret_15m_z", "volume_log_z", "trend_spread_z"):
            c = f"{prefix}_{suffix}"
            assert c in idx

    # ---- 4) z-score 品質：在暖機期之後，幾個關鍵 *_z 應該有合理變異 ----
    # 不要求 mean=0/std=1（因 clip/缺值補 0 會偏），但至少要「非全 0」且不爆。
    warmup = 800
    sample_cols = ["ret_1_z", "log_close_z", "volume_log_z", "ema_12_48_spread_z"]
    for c in sample_cols:
        assert c in idx
        x = feats[warmup:, idx[c]]
        assert np.isfinite(x).all()
        assert float(np.std(x)) > 0.05
        assert float(np.std(x)) < 5.0


