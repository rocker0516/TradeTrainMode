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

    # ---- 5m：固定通道（主市場 + 結構化趨勢 + 跨市場摘要）----
    #
    # 注意：
    # - 這裡的欄位名是「順序規格」，實際輸出會依 feature_symbols 展開跨市場欄位。
    # - 測試與訓練端不應再假設固定為 14 通道；請以 MarketData.price_seq_features_dim 為準。
    BASE_5M_COLS: Final[Tuple[str, ...]] = (
        # ---- 原本的 14 通道（穩定、可解釋）----
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
        # ---- 主市場結構化趨勢/震盪特徵（約 20 條）----
        "log_close_z",
        "close_over_ema_12_z",
        "close_over_ema_48_z",
        "ema_12_48_spread_z",
        "ema_12_slope_z",
        "ema_48_slope_z",
        "price_pos_96",
        "price_pos_288",
        "bb_width_48_z",
        "bb_pos_48",
        "rsi_14",
        "macd_atr",
        "macd_signal_atr",
        "trend_strength_atr",
        "dir_persist_20",
        "abs_ret_1_z",
        "ret_4h_z",
        "hl_range_20_z",
        "chop_48",
        "trend_flip_rate_48",
        # ---- 跨市場廣度摘要（固定 6 條；依 feature_symbols 中「非 target」集合計算）----
        "alts_ret_15m_mean_z",
        "alts_ret_15m_std_z",
        "alts_rel_ret_15m_abs_mean_z",
        "alts_trend_up_ratio",
        "alts_volume_log_mean_z",
        "alts_trend_spread_std_z",
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
            # 5m 的完整欄位需要知道 feature_symbols 才能展開；這裡回傳 base 規格供 debug。
            price_seq_cols=self.BASE_5M_COLS,
            price_seq_1d_cols=self.PRICE_SEQ_1D_SYMBOL_COLS + self.PRICE_SEQ_1D_MACRO_COLS,
        )

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        """EWMA/EMA（只用過去）。"""
        return series.astype(float).ewm(span=int(span), adjust=False).mean()

    @staticmethod
    def _compute_atr_ratio_from_ohlc(
        close: pd.Series, high: pd.Series, low: pd.Series, *, window: int = 14, min_periods: int = 5
    ) -> pd.Series:
        """以 OHLC 計算 ATR/close（與 MarketData 的 atr_ratio 定義一致）。"""
        c = close.astype(float)
        h = high.astype(float)
        l = low.astype(float)
        prev_close = c.shift(1)
        tr = pd.concat(
            [
                (h - l),
                (h - prev_close).abs(),
                (l - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(int(window), min_periods=int(min_periods)).mean().fillna(0.0)
        return (atr / np.maximum(c, 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _compute_rv_ratio_from_close(close: pd.Series) -> pd.Series:
        """以 close 計算 RV ratio（與 MarketData._compute_rv_ratio 行為一致）。"""
        c = close.astype(float)
        log_c = np.log(np.clip(c, 1e-12, None))
        log_ret = log_c.diff().fillna(0.0)
        rv_20 = log_ret.rolling(20, min_periods=5).std().fillna(0.0)
        rv_288 = (
            log_ret.rolling(288, min_periods=20)
            .std()
            .replace(0.0, np.nan)
            .bfill()
            .fillna(1e-8)
        )
        rv_ratio = (rv_20 / rv_288).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return rv_ratio

    def build_5m_features(
        self,
        df_5m: pd.DataFrame,
        *,
        target_symbol: str,
        atr_ratio_arr: np.ndarray,
        rv_ratio_arr: np.ndarray,
        z_window: int = 288,
        feature_symbols: Iterable[str] | None = None,
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

        # feature_symbols：固定順序、固定維度（Gym observation_space 需要固定 shape）
        # - 若未傳入，預設只用 target_symbol（等同舊行為）
        symbols = [str(target_symbol)]
        if feature_symbols is not None:
            # 去重但保留順序，且保證 target_symbol 在第一個位置（主市場）
            seen = set()
            ordered = []
            for s in feature_symbols:
                ss = str(s)
                if ss not in seen:
                    seen.add(ss)
                    ordered.append(ss)
            if str(target_symbol) in ordered:
                ordered = [str(target_symbol)] + [s for s in ordered if s != str(target_symbol)]
            else:
                ordered = [str(target_symbol)] + ordered
            symbols = ordered

        # ---- 讀取 target_symbol 的 5m 原始欄位（缺欄補 0）----
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

        # ---- 主市場：基礎 14 通道（維持原定義）----
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

        # ---- 主市場：結構化趨勢/震盪特徵（約 20 條）----
        log_close = _safe_log(c)
        ema12 = self._ema(c, 12)
        ema26 = self._ema(c, 26)
        ema48 = self._ema(c, 48)
        ema96 = self._ema(c, 96)

        # 價格相對均線（log ratio 更穩）
        close_over_ema_12 = np.log(np.clip(c / np.clip(ema12, 1e-12, None), 1e-12, None))
        close_over_ema_48 = np.log(np.clip(c / np.clip(ema48, 1e-12, None), 1e-12, None))
        ema_12_48_spread = (ema12 - ema48) / np.clip(ema48, 1e-12, None)

        # 斜率（用 log-EMA diff，尺度較一致）
        ema_12_slope = _safe_log(ema12).diff().fillna(0.0)
        ema_48_slope = _safe_log(ema48).diff().fillna(0.0)

        # 價格位置：在 rolling high/low 區間的相對位置（轉到 [-1,1]）
        hi_96 = h.rolling(96, min_periods=10).max()
        lo_96 = l.rolling(96, min_periods=10).min()
        pos_96 = (c - lo_96) / np.clip((hi_96 - lo_96), 1e-12, None)
        pos_96 = (pos_96.clip(0.0, 1.0) * 2.0) - 1.0

        hi_288 = h.rolling(288, min_periods=20).max()
        lo_288 = l.rolling(288, min_periods=20).min()
        pos_288 = (c - lo_288) / np.clip((hi_288 - lo_288), 1e-12, None)
        pos_288 = (pos_288.clip(0.0, 1.0) * 2.0) - 1.0

        # Bollinger (48)
        mid_48 = c.rolling(48, min_periods=10).mean()
        sd_48 = c.rolling(48, min_periods=10).std().replace(0.0, np.nan).fillna(1e-8)
        bb_width_48 = (2.0 * sd_48) / np.clip(mid_48, 1e-12, None)
        bb_pos_48 = ((c - mid_48) / (2.0 * sd_48)).clip(-2.0, 2.0)

        # RSI(14) -> [-1,1]
        delta = c.diff().fillna(0.0)
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        roll_up = up.rolling(14, min_periods=5).mean()
        roll_down = down.rolling(14, min_periods=5).mean()
        rs = roll_up / np.clip(roll_down, 1e-12, None)
        rsi = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)
        rsi_14 = (((rsi - 50.0) / 50.0).clip(-1.0, 1.0)).astype(float)

        # MACD / ATR（避免價位尺度）
        macd = (ema12 - ema26)
        macd_signal = self._ema(macd, 9)
        # 以 target 的 ATR estimate 做分母：atr_ratio_arr * close
        atr_est = pd.Series(atr_ratio_arr, index=df_5m.index).astype(float) * np.clip(c, 1e-12, None)
        atr_est = atr_est.replace(0.0, np.nan).fillna(1e-8)
        macd_atr = (macd / atr_est).clip(-10.0, 10.0)
        macd_signal_atr = (macd_signal / atr_est).clip(-10.0, 10.0)
        trend_strength_atr = (macd.abs() / atr_est).clip(0.0, 10.0)

        # 方向一致性：近 20 根正報酬比例 -> [-1,1]
        pos_frac_20 = (ret_1 > 0.0).astype(float).rolling(20, min_periods=10).mean().fillna(0.5)
        dir_persist_20 = ((pos_frac_20 * 2.0) - 1.0).clip(-1.0, 1.0)

        # 其他穩健補強：abs_ret、4h 報酬、HL range、chop、trend flip rate
        abs_ret_1 = ret_1.abs()
        ret_4h = ret_1.rolling(window=48, min_periods=1).sum()
        hl_range_20 = ((h.rolling(20, min_periods=10).max() - l.rolling(20, min_periods=10).min()) / np.clip(c.shift(1).bfill(), 1e-12, None))

        # Choppiness (48)：log10( sum(TR) / (HH-LL) )，轉到 [0,1] 後再映射 [-1,1]
        tr1 = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1).fillna(0.0)
        sum_tr_48 = tr1.rolling(48, min_periods=10).sum()
        hh_48 = h.rolling(48, min_periods=10).max()
        ll_48 = l.rolling(48, min_periods=10).min()
        chop = np.log10(np.clip(sum_tr_48 / np.clip((hh_48 - ll_48), 1e-12, None), 1e-12, None))
        chop_48 = ((chop - chop.rolling(z_window, min_periods=minp).mean()) / chop.rolling(z_window, min_periods=minp).std().replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        chop_48 = chop_48.clip(-5.0, 5.0)

        # 趨勢翻轉率：EMA12-EMA48 的 sign 在 48 內翻轉的比例（0~1 -> [-1,1]）
        trend_sign = np.sign(ema12 - ema48).replace(0.0, np.nan).ffill().fillna(0.0)
        flips = (trend_sign != trend_sign.shift(1)).astype(float).fillna(0.0)
        flip_rate_48 = flips.rolling(48, min_periods=10).mean().fillna(0.0)
        trend_flip_rate_48 = ((flip_rate_48 * 2.0) - 1.0).clip(-1.0, 1.0)

        feats_extra = pd.DataFrame(
            {
                "log_close_z": _clip(_rolling_zscore(log_close, z_window, minp)),
                "close_over_ema_12_z": _clip(_rolling_zscore(pd.Series(close_over_ema_12, index=df_5m.index), z_window, minp)),
                "close_over_ema_48_z": _clip(_rolling_zscore(pd.Series(close_over_ema_48, index=df_5m.index), z_window, minp)),
                "ema_12_48_spread_z": _clip(_rolling_zscore(pd.Series(ema_12_48_spread, index=df_5m.index), z_window, minp)),
                "ema_12_slope_z": _clip(_rolling_zscore(ema_12_slope, z_window, minp)),
                "ema_48_slope_z": _clip(_rolling_zscore(ema_48_slope, z_window, minp)),
                "price_pos_96": pos_96.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0),
                "price_pos_288": pos_288.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0),
                "bb_width_48_z": _clip(_rolling_zscore(bb_width_48, z_window, minp)),
                "bb_pos_48": bb_pos_48.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-2.0, 2.0),
                "rsi_14": rsi_14.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0),
                "macd_atr": macd_atr.replace([np.inf, -np.inf], np.nan).fillna(0.0),
                "macd_signal_atr": macd_signal_atr.replace([np.inf, -np.inf], np.nan).fillna(0.0),
                "trend_strength_atr": trend_strength_atr.replace([np.inf, -np.inf], np.nan).fillna(0.0),
                "dir_persist_20": dir_persist_20.replace([np.inf, -np.inf], np.nan).fillna(0.0),
                "abs_ret_1_z": _clip(_rolling_zscore(abs_ret_1, z_window, minp)),
                "ret_4h_z": _clip(_rolling_zscore(ret_4h, z_window, minp)),
                "hl_range_20_z": _clip(_rolling_zscore(hl_range_20, z_window, minp)),
                "chop_48": chop_48.replace([np.inf, -np.inf], np.nan).fillna(0.0),
                "trend_flip_rate_48": trend_flip_rate_48.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            },
            index=df_5m.index,
        )
        feats = pd.concat([feats, feats_extra], axis=1)

        # ---- 跨市場摘要（非 target symbols）----
        alt_symbols = [s for s in symbols if s != str(target_symbol)]
        alt_ret_15m_list: list[pd.Series] = []
        alt_rel_ret_15m_abs_list: list[pd.Series] = []
        alt_trend_up_list: list[pd.Series] = []
        alt_vol_log_list: list[pd.Series] = []
        alt_trend_spread_list: list[pd.Series] = []

        # target 的 ret_15m（用於 rel_ret）
        target_ret_15m = ret_15m

        for sym in alt_symbols:
            cs = _get_symbol_col(df_5m, target_symbol=sym, suffix="close")
            os_ = _get_symbol_col(df_5m, target_symbol=sym, suffix="open")
            hs = _get_symbol_col(df_5m, target_symbol=sym, suffix="high")
            ls = _get_symbol_col(df_5m, target_symbol=sym, suffix="low")
            vs = _get_symbol_col(df_5m, target_symbol=sym, suffix="volume")

            # per-symbol returns
            log_cs = _safe_log(cs)
            ret1_s = log_cs.diff().fillna(0.0)
            ret15_s = ret1_s.rolling(window=3, min_periods=1).sum()

            # volume impulse（同主市場邏輯）
            v_ma20_s = vs.rolling(window=20, min_periods=1).mean().replace(0.0, np.nan).fillna(1.0)
            vol_log_s = np.log(np.clip(vs / v_ma20_s, 1e-12, None))

            # trend spread（EMA12-EMA48 相對化）
            ema12_s = self._ema(cs, 12)
            ema48_s = self._ema(cs, 48)
            spread_s = (ema12_s - ema48_s) / np.clip(ema48_s, 1e-12, None)

            # 4 條/幣：ret_15m_z, rel_ret_15m_z, volume_log_z, trend_spread_z
            feats[f"{sym.lower()}_ret_15m_z"] = _clip(_rolling_zscore(ret15_s, z_window, minp))
            rel = (ret15_s - target_ret_15m)
            feats[f"{sym.lower()}_rel_ret_15m_z"] = _clip(_rolling_zscore(rel, z_window, minp))
            feats[f"{sym.lower()}_volume_log_z"] = _clip(_rolling_zscore(vol_log_s, z_window, minp))
            feats[f"{sym.lower()}_trend_spread_z"] = _clip(_rolling_zscore(spread_s, z_window, minp))

            # breadth accumulators（用未 z 的原序列，最後再 z）
            alt_ret_15m_list.append(ret15_s)
            alt_rel_ret_15m_abs_list.append(rel.abs())
            alt_trend_up_list.append((ema12_s > ema48_s).astype(float))
            alt_vol_log_list.append(vol_log_s)
            alt_trend_spread_list.append(spread_s)

        # 6 條廣度：若沒有 alt_symbols，維持中性 0
        if alt_symbols:
            alts_ret_15m_mean = pd.concat(alt_ret_15m_list, axis=1).mean(axis=1)
            alts_ret_15m_std = pd.concat(alt_ret_15m_list, axis=1).std(axis=1).fillna(0.0)
            alts_rel_ret_15m_abs_mean = pd.concat(alt_rel_ret_15m_abs_list, axis=1).mean(axis=1)
            alts_trend_up_ratio = pd.concat(alt_trend_up_list, axis=1).mean(axis=1).fillna(0.5)
            alts_trend_up_ratio = ((alts_trend_up_ratio * 2.0) - 1.0).clip(-1.0, 1.0)
            alts_volume_log_mean = pd.concat(alt_vol_log_list, axis=1).mean(axis=1)
            alts_trend_spread_std = pd.concat(alt_trend_spread_list, axis=1).std(axis=1).fillna(0.0)
        else:
            alts_ret_15m_mean = pd.Series(0.0, index=df_5m.index)
            alts_ret_15m_std = pd.Series(0.0, index=df_5m.index)
            alts_rel_ret_15m_abs_mean = pd.Series(0.0, index=df_5m.index)
            alts_trend_up_ratio = pd.Series(0.0, index=df_5m.index)
            alts_volume_log_mean = pd.Series(0.0, index=df_5m.index)
            alts_trend_spread_std = pd.Series(0.0, index=df_5m.index)

        feats["alts_ret_15m_mean_z"] = _clip(_rolling_zscore(alts_ret_15m_mean, z_window, minp))
        feats["alts_ret_15m_std_z"] = _clip(_rolling_zscore(alts_ret_15m_std, z_window, minp))
        feats["alts_rel_ret_15m_abs_mean_z"] = _clip(_rolling_zscore(alts_rel_ret_15m_abs_mean, z_window, minp))
        feats["alts_trend_up_ratio"] = alts_trend_up_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        feats["alts_volume_log_mean_z"] = _clip(_rolling_zscore(alts_volume_log_mean, z_window, minp))
        feats["alts_trend_spread_std_z"] = _clip(_rolling_zscore(alts_trend_spread_std, z_window, minp))

        # ---- 最終欄位順序 ----
        cols: list[str] = list(self.BASE_5M_COLS)
        # 展開每個 alt symbol 的 4 條，順序固定
        for sym in alt_symbols:
            cols.extend(
                [
                    f"{sym.lower()}_ret_15m_z",
                    f"{sym.lower()}_rel_ret_15m_z",
                    f"{sym.lower()}_volume_log_z",
                    f"{sym.lower()}_trend_spread_z",
                ]
            )

        # 固定順序 + float32 + NaN/inf 清理
        feats = feats.reindex(columns=cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return feats.astype(np.float32).to_numpy(copy=True), cols

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


