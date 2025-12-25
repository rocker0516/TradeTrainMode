from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd


def _rolling_z(series: pd.Series, window: int, min_periods: int = 30) -> pd.Series:
    """以過去資料計算 rolling z-score，並將 NaN/inf 安全處理成 0.0。

    Args:
        series: 單一數值序列。
        window: rolling window 長度（以日為單位）。
        min_periods: 允許開始計算的最小樣本數。

    Returns:
        z-score 序列（float32）。
    """
    m = series.rolling(window=window, min_periods=min_periods).mean()
    s = series.rolling(window=window, min_periods=min_periods).std().replace(0.0, np.nan)
    z = (series - m) / s
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _to_daily_index(ts: pd.Series | pd.Index) -> pd.DatetimeIndex:
    """將 timestamp 轉成日頻 index（normalize 到 00:00:00）。"""
    if isinstance(ts, pd.DatetimeIndex):
        return pd.DatetimeIndex(ts).normalize()
    # Series or Index
    dt = pd.to_datetime(ts)
    if isinstance(dt, pd.Series):
        return pd.DatetimeIndex(dt.dt.normalize())
    return pd.DatetimeIndex(dt).normalize()


def _read_csv(path: str) -> pd.DataFrame:
    """讀取 CSV（不做 schema 假設；由各 parser 處理）。"""
    return pd.read_csv(path)


def _parse_altcoin_season(path: str) -> pd.DataFrame:
    """altcoin_season_index_history.csv -> altcoin_season_norm (0..1)."""
    df = _read_csv(path)
    if "timestamp" not in df.columns or "altcoin_index" not in df.columns:
        raise ValueError(f"Unexpected schema for {path}: need timestamp, altcoin_index")
    idx = _to_daily_index(df["timestamp"])
    df = df.copy()
    df.index = idx
    s = (pd.to_numeric(df["altcoin_index"], errors="coerce").astype(float) / 100.0).clip(0.0, 1.0)
    out = pd.DataFrame({"altcoin_season_norm": s.astype(np.float32)}, index=idx)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _parse_bitcoin_macro_oscillator(path: str) -> pd.DataFrame:
    """bitcoin_macro_oscillator_index_history.csv -> bmo_value_clipped."""
    df = _read_csv(path)
    if "timestamp" not in df.columns or "bmo_value" not in df.columns:
        raise ValueError(f"Unexpected schema for {path}: need timestamp, bmo_value")
    idx = _to_daily_index(df["timestamp"])
    df = df.copy()
    df.index = idx
    s = pd.to_numeric(df["bmo_value"], errors="coerce").astype(float).clip(-3.0, 3.0)
    out = pd.DataFrame({"bmo_value": s.astype(np.float32)}, index=idx)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _parse_bitcoin_sopr(path: str) -> pd.DataFrame:
    """bitcoin_sth_sopr_index_history.csv（目前檔內欄位是 lth_sopr）-> sopr_log_z."""
    df = _read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Unexpected schema for {path}: need timestamp")

    # 專案中檔名寫 sth_sopr，但實際欄位是 lth_sopr（以實際欄位為準，避免炸掉）
    val_col = "sth_sopr" if "sth_sopr" in df.columns else ("lth_sopr" if "lth_sopr" in df.columns else None)
    if val_col is None:
        raise ValueError(f"Unexpected schema for {path}: need sth_sopr or lth_sopr")

    idx = _to_daily_index(df["timestamp"])
    df = df.copy()
    df.index = idx
    raw = pd.to_numeric(df[val_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # SOPR 以 1 為基準，log 讓偏離變成較對稱；並做 clip 避免極端值主導
    sopr_log = np.log(np.clip(raw.astype(np.float64), 1e-8, None))
    sopr_log = pd.Series(sopr_log, index=idx).clip(-10.0, 10.0)
    out = pd.DataFrame({"sopr_log": sopr_log.astype(np.float32)}, index=idx)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _parse_fear_greed(path: str) -> pd.DataFrame:
    """fear_greed_index_history.csv -> fear_greed_norm (0..1)."""
    df = _read_csv(path)
    # 目前 schema: time,fear_greed_index,price
    time_col = "time" if "time" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    val_col = "fear_greed_index" if "fear_greed_index" in df.columns else ("value" if "value" in df.columns else None)
    if time_col is None or val_col is None:
        raise ValueError(f"Unexpected schema for {path}: need time/timestamp and fear_greed_index/value")
    idx = _to_daily_index(df[time_col])
    df = df.copy()
    df.index = idx
    s = (pd.to_numeric(df[val_col], errors="coerce").astype(float) / 100.0).clip(0.0, 1.0)
    out = pd.DataFrame({"fear_greed_norm": s.astype(np.float32)}, index=idx)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _parse_coinglass_btc_1d(path: str) -> pd.DataFrame:
    """BTCUSDT_futures_volume_coinglass_5years_1d.csv -> 一組日線/衍生特徵。"""
    df = _read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"Unexpected schema for {path}: need time")
    idx = _to_daily_index(df["time"])
    df = df.copy()
    df.index = idx
    out = pd.DataFrame(index=idx)

    def _num(col: str) -> pd.Series:
        s = df.get(col, pd.Series(np.nan, index=df.index))
        return pd.to_numeric(s, errors="coerce")

    open_ = _num("open")
    high = _num("high")
    low = _num("low")
    close = _num("close")
    vol_usd = _num("volume_usd")

    # 價格/量能基本面（以 coinglass 日線為基準）
    close_clip = close.replace(0.0, np.nan)
    daily_ret = np.log(close_clip / close_clip.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    daily_range = ((high - low) / close_clip).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    daily_vol_log = np.log1p(np.clip(vol_usd, 0.0, None)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["cg_daily_ret"] = pd.Series(daily_ret.to_numpy(), index=idx).astype(np.float32)
    out["cg_daily_range"] = pd.Series(daily_range.to_numpy(), index=idx).astype(np.float32)
    out["cg_daily_vol_log"] = pd.Series(daily_vol_log.to_numpy(), index=idx).astype(np.float32)

    # 衍生：常見衍生市場欄位（存在就用，不存在就忽略 -> NaN -> 0）
    out["cg_oi_close_log"] = np.log1p(np.clip(_num("open_interest_close"), 0.0, None)).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    out["cg_funding_close"] = _num("funding_rate_close").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    out["cg_global_ls_ratio"] = _num("global_long_short_account_ratio_global_account_long_short_ratio").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    out["cg_top_pos_ls_ratio"] = _num("top_long_short_position_ratio_top_position_long_short_ratio").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    out["cg_liq_long_usd_log"] = np.log1p(np.clip(_num("liquidation_long_liquidation_usd"), 0.0, None)).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    out["cg_liq_short_usd_log"] = np.log1p(np.clip(_num("liquidation_short_liquidation_usd"), 0.0, None)).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    # 另外提供 coinglass close 做對齊診斷（可被 downstream 忽略）
    out["cg_close"] = close.ffill().fillna(0.0).astype(np.float32)
    out["cg_open"] = open_.ffill().fillna(0.0).astype(np.float32)

    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


@dataclass(frozen=True)
class MacroDailyConfig:
    """宏觀日線資料設定。"""

    daily_window_size: int
    rolling_z_window: int = 365
    rolling_z_min_periods: int = 30


class MacroDailyDataset:
    """將多來源日線 CSV 合併成單一日線特徵表，供 env 組 `daily_seq` 使用。

    設計目標：
    - 不改變現有訓練主流程（env 只要回傳 `daily_seq`）
    - 盡可能 schema-robust（欄位缺就補 0）
    - 以 rolling z-score 穩定尺度，提升 CNN 收斂
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("MacroDailyDataset df.index must be DatetimeIndex")
        self.df = df.sort_index()

    @staticmethod
    def load(
        *,
        data_dir: str,
        files: Dict[str, str],
        cfg: MacroDailyConfig,
        date_range: Tuple[pd.Timestamp, pd.Timestamp],
    ) -> "MacroDailyDataset":
        """從指定 data_dir 讀取並合併日線資料。

        Args:
            data_dir: CSV 所在資料夾（例如 "Data"）。
            files: key->filename 映射；key 需包含: altcoin_season, bmo, sopr, fear_greed, coinglass_btc_1d。
            cfg: MacroDailyConfig。
            date_range: (start_ts, end_ts)；會 reindex 到日頻並 ffill/bfill。
        """
        start_ts, end_ts = date_range
        start_day = pd.Timestamp(start_ts).normalize()
        end_day = pd.Timestamp(end_ts).normalize()
        full_idx = pd.date_range(start=start_day, end=end_day, freq="D")

        parts: Sequence[pd.DataFrame] = []
        required_keys = ["altcoin_season", "bmo", "sopr", "fear_greed", "coinglass_btc_1d"]
        for k in required_keys:
            if k not in files:
                raise ValueError(f"files missing key: {k}")

        # load each
        base = Path(data_dir)
        parts = [
            _parse_altcoin_season(str(base / files["altcoin_season"])),
            _parse_bitcoin_macro_oscillator(str(base / files["bmo"])),
            _parse_bitcoin_sopr(str(base / files["sopr"])),
            _parse_fear_greed(str(base / files["fear_greed"])),
            _parse_coinglass_btc_1d(str(base / files["coinglass_btc_1d"])),
        ]

        df = pd.concat(parts, axis=1)
        # 對齊到 full_idx，避免切片越界；缺值用 ffill/bfill 再填 0
        df = df.reindex(full_idx).ffill().bfill().fillna(0.0)

        # rolling z-score：只對「非 0/1 類別」或連續量做（包含 cg_* 的連續變數、sopr_log、bmo_value 等）
        z_cols = [c for c in df.columns if c not in {"altcoin_season_norm", "fear_greed_norm"}]
        for c in z_cols:
            df[f"{c}_z"] = _rolling_z(df[c].astype(float), window=cfg.rolling_z_window, min_periods=cfg.rolling_z_min_periods)

        # 最終輸出：保留 norm + z 特徵（避免 raw scale 混雜）
        keep_cols = ["altcoin_season_norm", "fear_greed_norm"] + [f"{c}_z" for c in z_cols]
        df_out = df[keep_cols].astype(np.float32)
        return MacroDailyDataset(df_out)


