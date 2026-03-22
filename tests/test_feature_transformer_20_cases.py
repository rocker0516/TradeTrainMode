from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from Env.Components.market_data import MarketData
from Env.config import Config


@dataclass(frozen=True)
class Case:
    """特徵整測案例：用不同資料形態/缺欄/不同 symbol 來驗證固定 shape 與安全性。"""

    name: str
    target_symbol: str
    use_prefixed_cols: bool
    drop_cols: Tuple[str, ...] = ()
    n_5m: int = 200
    n_1d: int = 60


def _make_ts(start: str, n: int, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq=freq)


def _mk_base_5m_df(*, n: int, start: str = "2020-01-01") -> pd.DataFrame:
    ts = _make_ts(start, n, "5min")
    close = 100.0 + np.sin(np.linspace(0, 10, n)) * 2.0
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = np.full(n, 100.0)
    buy_volume = volume * 0.55
    sell_volume = volume - buy_volume
    trades = np.full(n, 500.0)
    quote_volume = close * volume
    volume_ratio = np.clip(buy_volume / (sell_volume + 1e-12), 0.0, 10.0)
    long_short_ratio = np.clip((buy_volume + 1.0) / (sell_volume + 1.0), 0.0, 10.0)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "trades": trades,
            "quote_volume": quote_volume,
            "volume_ratio": volume_ratio,
            "long_short_ratio": long_short_ratio,
        }
    )


def _mk_base_1d_df(*, n: int, start: str = "2019-11-01") -> pd.DataFrame:
    ts = _make_ts(start, n, "1d")
    close = 100.0 + np.linspace(0, 5, n)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 2.0
    low = np.minimum(open_, close) - 2.0
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    )


def _prefix(df: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    """模擬 load_file.py 的 rename 行為：除 timestamp 外全部加上 {symbol}_ 前綴。"""
    out = df.copy()
    rename_map: Dict[str, str] = {c: f"{symbol}_{c}" for c in out.columns if c != "timestamp"}
    out = out.rename(columns=rename_map)
    return out


def _build_case_market(case: Case) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_5m = _mk_base_5m_df(n=case.n_5m)
    df_1d = _mk_base_1d_df(n=case.n_1d)

    if case.use_prefixed_cols:
        df_5m = _prefix(df_5m, symbol=case.target_symbol)
        df_1d = _prefix(df_1d, symbol=case.target_symbol)

    # 依案例刪除欄位（驗證缺欄補 0 邏輯）
    for c in case.drop_cols:
        if c in df_5m.columns:
            df_5m = df_5m.drop(columns=[c])
        if c in df_1d.columns:
            df_1d = df_1d.drop(columns=[c])

    return df_5m, df_1d


def _cases() -> List[Case]:
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]
    cases: List[Case] = []

    # 每個 symbol 做 5 種變化 => 4 * 5 = 20 案例
    for sym in symbols:
        cases.extend(
            [
                Case(name=f"{sym}_prefixed_full", target_symbol=sym, use_prefixed_cols=True),
                Case(
                    name=f"{sym}_prefixed_missing_buy_sell",
                    target_symbol=sym,
                    use_prefixed_cols=True,
                    drop_cols=(f"{sym}_buy_volume", f"{sym}_sell_volume"),
                ),
                Case(
                    name=f"{sym}_prefixed_missing_quote_trades",
                    target_symbol=sym,
                    use_prefixed_cols=True,
                    drop_cols=(f"{sym}_quote_volume", f"{sym}_trades"),
                ),
                Case(name=f"{sym}_raw_full", target_symbol=sym, use_prefixed_cols=False),
                Case(
                    name=f"{sym}_raw_missing_volume_ratio_ls_ratio",
                    target_symbol=sym,
                    use_prefixed_cols=False,
                    drop_cols=("volume_ratio", "long_short_ratio"),
                ),
            ]
        )
    assert len(cases) == 20
    return cases


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.name)
def test_feature_shapes_are_fixed_and_safe(case: Case) -> None:
    df_5m, df_1d = _build_case_market(case)
    # 測試預設只使用單一 symbol（相容舊行為），但特徵集合可能擴充（例如結構化趨勢/廣度）。
    md = MarketData(
        df_5m,
        df_1d,
        window_size=32,
        window_size_1d=10,
        target_symbol=case.target_symbol,
        feature_symbols=[case.target_symbol],
    )

    # 5m target 必須固定 shape（對於同一組 feature_symbols），且 cols 與 Config 約定一致
    assert md.price_seq_target_features_dim == len(md.cols_5m_target)
    assert md.price_seq_target_features_dim > 0
    assert md.cols_5m_target == list(Config.OBS_PRICE_SEQ_TARGET_COLS)

    # 1d target 通道數 = OBS_PRICE_SEQ_1D_TARGET_COLS（macro 等在 others）
    n_1d_tgt = len(Config.OBS_PRICE_SEQ_1D_TARGET_COLS)
    assert md.features_1d_dim == n_1d_tgt
    assert md.cols_1d == list(Config.OBS_PRICE_SEQ_1D_TARGET_COLS)

    # 任意 step 的序列 shape 必須一致
    seq_5m_target, _ = md.get_price_seq(40)
    seq_1d_target, _ = md.get_1d_seq(40, window_size_1d=10)
    assert seq_5m_target.shape == (32, md.price_seq_target_features_dim)
    assert seq_1d_target.shape == (10, n_1d_tgt)
    # MarketData 內部特徵矩陣使用 float32（計算穩定）；env 輸出 obs 可能轉成 float16 以省 RAM
    assert seq_5m_target.dtype == np.float32
    assert seq_1d_target.dtype == np.float32

    # 不允許 NaN/inf（SAC 會直接爆）
    assert np.isfinite(seq_5m_target).all()
    assert np.isfinite(seq_1d_target).all()


def test_feature_symbols_expands_5m_dim_but_keeps_shape_fixed() -> None:
    """
    驗證：加入多幣 feature_symbols 會擴充 5m 特徵維度（包含跨市場摘要），但輸出仍然固定 shape 且無 NaN/inf。
    """
    df_5m = _mk_base_5m_df(n=400)
    df_1d = _mk_base_1d_df(n=80)
    # 模擬 multi-symbol 欄位（用同一份基礎資料複製並加前綴）
    sym_main = "BTCUSDT"
    alts = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT"]
    df_5m_pref = _prefix(df_5m, symbol=sym_main)
    for s in alts:
        df_5m_pref = pd.merge(df_5m_pref, _prefix(df_5m, symbol=s), on="timestamp", how="inner")
    df_1d_pref = _prefix(df_1d, symbol=sym_main)

    md_single = MarketData(
        df_5m_pref,
        df_1d_pref,
        window_size=64,
        window_size_1d=10,
        target_symbol=sym_main,
        feature_symbols=[sym_main],
    )
    md_multi = MarketData(
        df_5m_pref,
        df_1d_pref,
        window_size=64,
        window_size_1d=10,
        target_symbol=sym_main,
        feature_symbols=[sym_main] + alts,
    )

    # 多幣時 target 維度不變；跨市摘要在 target 內，逐幣細節在 others
    assert md_multi.price_seq_target_features_dim == md_single.price_seq_target_features_dim
    assert md_multi.price_seq_others_features_dim > md_single.price_seq_others_features_dim
    seq_target, _ = md_multi.get_price_seq(100)
    assert seq_target.shape == (64, md_multi.price_seq_target_features_dim)
    assert seq_target.dtype == np.float32
    assert np.isfinite(seq_target).all()


def test_1d_alignment_is_previous_closed_bar_B() -> None:
    """
    驗證策略 B：在 5m 當下只能看到「上一根已收盤」日線。

    範例：
    - 1d timestamps: 2020-01-01, 2020-01-02, 2020-01-03（皆為 00:00）
    - 5m 在 2020-01-02 12:00
      - idx_asof 應為 2020-01-02
      - idx_closed 應退一根到 2020-01-01
    """
    df_1d = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1000.0, 1000.0, 1000.0],
        }
    )
    df_5m = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-02 12:00:00", "2020-01-02 12:05:00", "2020-01-02 12:10:00", "2020-01-02 12:15:00", "2020-01-02 12:20:00"]),
            "open": [101.0, 101.0, 101.0, 101.0, 101.0],
            "high": [102.0, 102.0, 102.0, 102.0, 102.0],
            "low": [100.0, 100.0, 100.0, 100.0, 100.0],
            "close": [101.0, 101.0, 101.0, 101.0, 101.0],
            "volume": [100.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
    md = MarketData(df_5m, df_1d, window_size=2, window_size_1d=2, target_symbol="BTCUSDT")
    # 對第一根 5m（2020-01-02 12:00）：應該映射到 2020-01-01（index=0）
    assert int(md.map_5m_to_1d[0]) == 0


