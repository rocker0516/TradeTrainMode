"""
特徵轉換模組（Feature Transformer）

目標：
- 將 `load_file.py` merge 後的超寬表，轉成固定 shape、固定欄位語意、可解釋、適合 SAC(含 Lagrangian) 訓練的特徵張量。
- 支援 `target_symbol` 可由 config 切換（BTCUSDT/ETHUSDT/SOLUSDT/...），但輸出 observation 維度不變。
- 避免 look-ahead：所有 rolling 統計只能使用「過去」資料。

輸出：
- 5m: price_seq  (T, 14)
- 1d: price_seq_1d (D, K)  (K = symbol-specific coinglass + global macro；固定順序)

設計原則（符合你專案 OOP + SOLID 需求）：
- SRP：本模組只負責「將 DataFrame → 特徵矩陣」與欄位定義，不負責環境 step/交易邏輯。
- DIP：`MarketData` 可依賴本模組的抽象輸出（features + cols），而非直接硬編碼大量特徵邏輯於 MarketData。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _safe_log(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    """安全 log：避免 0 或負值導致 inf/NaN。"""
    return np.log(np.clip(x.astype(float), eps, None))


def _rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """只用過去資料的 rolling z-score，NaN/inf 最終會轉成 0.0（中性）。"""
    s = series.astype(float)
    m = s.rolling(window=window, min_periods=min_periods).mean()
    sd = s.rolling(window=window, min_periods=min_periods).std()
    z = (s - m) / sd.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _clip(series: pd.Series, low: float = -5.0, high: float = 5.0) -> pd.Series:
    """裁切特徵範圍，避免極端值破壞 SAC 訓練穩定性。"""
    return series.clip(lower=low, upper=high)


def _get_col(df: pd.DataFrame, col: str, *, default: float = 0.0) -> pd.Series:
    """取得欄位；若不存在則回傳全 0（確保不同 symbol/資料版本仍輸出固定 shape）。"""
    if col in df.columns:
        return df[col].astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def _get_symbol_col(df: pd.DataFrame, *, target_symbol: str, suffix: str, default: float = 0.0) -> pd.Series:
    """
    取得「symbol 欄位」並提供 fallback。

    本專案存在兩種資料形態：
    - 真實資料（經 `load_file.py` rename）：`{symbol}_{suffix}`，例如 `BTCUSDT_close`
    - 測試/外部資料（無前綴）：`{suffix}`，例如 `close`

    為了讓特徵模組可重用且測試可跑，這裡採用：
    1) 先取 `{symbol}_{suffix}`
    2) 若不存在，改取 `{suffix}`
    3) 仍不存在，回傳全 0
    """
    prefixed = f"{target_symbol}_{suffix}"
    if prefixed in df.columns:
        return df[prefixed].astype(float)
    if suffix in df.columns:
        return df[suffix].astype(float)
    return pd.Series(default, index=df.index, dtype=float)


@dataclass(frozen=True)
class FeatureSpec:
    """特徵規格：固定欄位名稱與順序（用於解析與 debug）。"""

    price_seq_cols: Tuple[str, ...]
    price_seq_1d_cols: Tuple[str, ...]


class FeatureTransformer:
    """
    特徵轉換器。

    注意：
    - 本類不依賴特定環境/模型，單純將資料轉成可訓練的 feature tensor。
    - 所有特徵皆為 float32，且保證不含 NaN/inf。
    """

    # ---- 5m：固定 14 通道（你已選定 14 通道版本）----
    PRICE_SEQ_COLS: Final[Tuple[str, ...]] = (
        "ret_1_z",
        "ret_15m_z",
        "ret_1h_z",
        "range_z",
        "body_z",
        "volume_log_z",
        "quote_volume_log_z",
        "trades_z",
        "vol_imbalance_z",
        "amihud_z",
        "volume_ratio_z",
        "long_short_ratio_z",
        "atr_ratio_z",
        "rv_ratio_z",
    )

    # ---- 1d：固定通道順序（symbol-specific coinglass + global macro）----
    PRICE_SEQ_1D_SYMBOL_COLS: Final[Tuple[str, ...]] = (
        "oi_close_z",
        "funding_close_z",
        "ls_account_ratio_z",
        "ls_position_ratio_z",
        "liq_long_log_z",
        "liq_short_log_z",
        "ob_imbalance_z",
    )
    PRICE_SEQ_1D_MACRO_COLS: Final[Tuple[str, ...]] = (
        "fear_greed_z",
        "altcoin_season_z",
        "bmo_z",
        "sopr_z",
    )

    def get_spec(self) -> FeatureSpec:
        """回傳固定欄位規格（給外部做解析/打印/一致性檢查）。"""
        return FeatureSpec(
            price_seq_cols=self.PRICE_SEQ_COLS,
            price_seq_1d_cols=self.PRICE_SEQ_1D_SYMBOL_COLS + self.PRICE_SEQ_1D_MACRO_COLS,
        )

    def build_5m_features(
        self,
        df_5m: pd.DataFrame,
        *,
        target_symbol: str,
        atr_ratio_arr: np.ndarray,
        rv_ratio_arr: np.ndarray,
        z_window: int = 288,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        建立 5m 特徵矩陣。

        Args:
            df_5m: merge 後的 5m DataFrame（含 `{symbol}_open/high/low/close/...` 欄位）
            target_symbol: 例如 "BTCUSDT"
            atr_ratio_arr: MarketData 以 target_symbol 價格計算出的 ATR/close（長度需等於 df_5m）
            rv_ratio_arr: MarketData 以 target_symbol 價格計算出的 RV ratio（長度需等於 df_5m）
            z_window: rolling z-score window（5m 下 288 約 1 天）

        Returns:
            (features_arr, cols)
        """
        if len(df_5m) != len(atr_ratio_arr) or len(df_5m) != len(rv_ratio_arr):
            raise ValueError("atr_ratio_arr/rv_ratio_arr 長度必須與 df_5m 相同")

        # 讀取（缺欄補 0）
        c = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="close")
        o = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="open")
        h = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="high")
        l = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="low")
        v = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="volume")
        buy_v = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="buy_volume")
        sell_v = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="sell_volume")
        trades = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="trades")
        quote_v = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="quote_volume")
        volume_ratio = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="volume_ratio")
        long_short_ratio = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="long_short_ratio")

        # 基礎序列：ret / range / body
        log_c = _safe_log(c)
        ret_1 = log_c.diff().fillna(0.0)
        ret_15m = ret_1.rolling(window=3, min_periods=1).sum()
        ret_1h = ret_1.rolling(window=12, min_periods=1).sum()

        prev_c = c.shift(1).bfill()
        range_raw = (h - l) / np.clip(prev_c, 1e-12, None)
        body_raw = (c - o) / np.clip(prev_c, 1e-12, None)

        # 成交量：先做相對量，再 z
        v_ma20 = v.rolling(window=20, min_periods=1).mean().replace(0.0, np.nan).fillna(1.0)
        volume_log = np.log(np.clip(v / v_ma20, 1e-12, None))

        # 不平衡/流動性
        vol_imb = (buy_v - sell_v) / (buy_v + sell_v + 1e-12)
        amihud = np.abs(ret_1) / (quote_v + 1e-12)

        # rolling z（只用過去）
        minp = max(20, z_window // 10)
        feats = pd.DataFrame(
            {
                "ret_1_z": _clip(_rolling_zscore(ret_1, z_window, minp)),
                "ret_15m_z": _clip(_rolling_zscore(ret_15m, z_window, minp)),
                "ret_1h_z": _clip(_rolling_zscore(ret_1h, z_window, minp)),
                "range_z": _clip(_rolling_zscore(range_raw, z_window, minp)),
                "body_z": _clip(_rolling_zscore(body_raw, z_window, minp)),
                "volume_log_z": _clip(_rolling_zscore(volume_log, z_window, minp)),
                "quote_volume_log_z": _clip(_rolling_zscore(np.log1p(np.clip(quote_v, 0.0, None)), z_window, minp)),
                "trades_z": _clip(_rolling_zscore(trades, z_window, minp)),
                "vol_imbalance_z": _clip(_rolling_zscore(vol_imb, z_window, minp)),
                "amihud_z": _clip(_rolling_zscore(amihud, z_window, minp)),
                "volume_ratio_z": _clip(_rolling_zscore(volume_ratio, z_window, minp)),
                "long_short_ratio_z": _clip(_rolling_zscore(long_short_ratio, z_window, minp)),
                "atr_ratio_z": _clip(_rolling_zscore(pd.Series(atr_ratio_arr, index=df_5m.index), z_window, minp)),
                "rv_ratio_z": _clip(_rolling_zscore(pd.Series(rv_ratio_arr, index=df_5m.index), z_window, minp)),
            },
            index=df_5m.index,
        )

        # 固定順序 + float32 + NaN/inf 清理
        feats = feats.reindex(columns=list(self.PRICE_SEQ_COLS)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return feats.astype(np.float32).to_numpy(copy=True), list(self.PRICE_SEQ_COLS)

    def build_1d_features(
        self,
        df_1d: pd.DataFrame,
        *,
        target_symbol: str,
        z_window_1d: int = 60,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        建立 1d 特徵矩陣（包含 symbol-specific + macro 共用）。

        Args:
            df_1d: merge 後的 1d DataFrame（含 `{symbol}_open_interest_close` 等欄位，以及 macro 前綴欄位）
            target_symbol: 例如 "BTCUSDT"
            z_window_1d: rolling z-score window（1d 下 60 約 2 個月）

        Returns:
            (features_arr, cols)
        """
        minp = max(10, z_window_1d // 6)

        # ---- symbol-specific（coinglass 1d）----
        # coinglass 1d 欄位通常有前綴；若測試資料沒有，會自動補 0。
        oi_close = _get_col(df_1d, f"{target_symbol}_open_interest_close")
        funding_close = _get_col(df_1d, f"{target_symbol}_funding_rate_close")
        ls_account_ratio = _get_col(df_1d, f"{target_symbol}_global_long_short_account_ratio_global_account_long_short_ratio")
        ls_position_ratio = _get_col(df_1d, f"{target_symbol}_top_long_short_position_ratio_top_position_long_short_ratio")
        liq_long = _get_col(df_1d, f"{target_symbol}_liquidation_long_liquidation_usd")
        liq_short = _get_col(df_1d, f"{target_symbol}_liquidation_short_liquidation_usd")
        bids_usd = _get_col(df_1d, f"{target_symbol}_ask_bids_bids_usd")
        asks_usd = _get_col(df_1d, f"{target_symbol}_ask_bids_asks_usd")
        ob_imbalance = (bids_usd - asks_usd) / (bids_usd + asks_usd + 1e-12)

        # ---- global macro（共用，不跟 target_symbol 綁死）----
        # 注意：這些欄位名由 load_file.py 依檔名加前綴（例如 fear_greed_...）
        fear_greed = _get_col(df_1d, "fear_greed_fear_greed_index")
        altcoin_season = _get_col(df_1d, "altcoin_season_altcoin_index")
        bmo_value = _get_col(df_1d, "bitcoin_macro_oscillator_bmo_value")
        # 檔名是 bitcoin_sth_sopr，但欄位叫 lth_sopr：以實際欄位為準
        sopr_value = _get_col(df_1d, "bitcoin_sth_sopr_lth_sopr")

        feats = pd.DataFrame(
            {
                "oi_close_z": _clip(_rolling_zscore(oi_close, z_window_1d, minp)),
                "funding_close_z": _clip(_rolling_zscore(funding_close, z_window_1d, minp)),
                "ls_account_ratio_z": _clip(_rolling_zscore(ls_account_ratio, z_window_1d, minp)),
                "ls_position_ratio_z": _clip(_rolling_zscore(ls_position_ratio, z_window_1d, minp)),
                "liq_long_log_z": _clip(_rolling_zscore(np.log1p(np.clip(liq_long, 0.0, None)), z_window_1d, minp)),
                "liq_short_log_z": _clip(_rolling_zscore(np.log1p(np.clip(liq_short, 0.0, None)), z_window_1d, minp)),
                "ob_imbalance_z": _clip(_rolling_zscore(ob_imbalance, z_window_1d, minp)),
                "fear_greed_z": _clip(_rolling_zscore(fear_greed, z_window_1d, minp)),
                "altcoin_season_z": _clip(_rolling_zscore(altcoin_season, z_window_1d, minp)),
                "bmo_z": _clip(_rolling_zscore(bmo_value, z_window_1d, minp)),
                "sopr_z": _clip(_rolling_zscore(sopr_value, z_window_1d, minp)),
            },
            index=df_1d.index,
        )

        cols = list(self.PRICE_SEQ_1D_SYMBOL_COLS + self.PRICE_SEQ_1D_MACRO_COLS)
        feats = feats.reindex(columns=cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return feats.astype(np.float32).to_numpy(copy=True), cols


