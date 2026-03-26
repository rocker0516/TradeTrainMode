"""
特徵轉換模組（Feature Transformer）

目標：
- 將 `load_file.py` merge 後的超寬表，轉成固定 shape、固定欄位語意、可解釋、適合 SAC(含 Lagrangian) 訓練的特徵張量。
- 支援 `target_symbol` 可由 config 切換（BTCUSDT/ETHUSDT/SOLUSDT/...），但輸出 observation 維度不變。
- 避免 look-ahead：所有 rolling 統計只能使用「過去」資料。

輸出：
- 5m: price_seq  (T, 14)
- 1d: price_seq_1d (D, K)  (K = symbol-specific price+coinglass + global macro；固定順序)

設計原則（符合你專案 OOP + SOLID 需求）：
- SRP：本模組只負責「將 DataFrame → 特徵矩陣」與欄位定義，不負責環境 step/交易邏輯。
- DIP：`MarketData` 可依賴本模組的抽象輸出（features + cols），而非直接硬編碼大量特徵邏輯於 MarketData。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, List, Tuple

import numpy as np
import pandas as pd

from Env.config import Config


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


def _heavy_field_features(
    series: pd.Series,
    *,
    feature_name: str,
    z_window: int,
    z_window_short: int,
    minp: int,
    minp_short: int,
) -> dict[str, pd.Series]:
    """
    對單一原始欄位做重度工程化（多尺度 + 變化率 + EMA 結構）。

    產出 5 個欄位：
    - {name}_z_short
    - {name}_z_long
    - {name}_roc_1
    - {name}_roc_12
    - {name}_ema_12_48_spread
    """
    s = series.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    prev_1 = s.shift(1).replace(0.0, np.nan)
    prev_12 = s.shift(12).replace(0.0, np.nan)
    ema_12 = s.ewm(span=12, adjust=False).mean()
    ema_48 = s.ewm(span=48, adjust=False).mean()
    out = {
        f"{feature_name}_z_short": _clip(_rolling_zscore(s, z_window_short, minp_short)),
        f"{feature_name}_z_long": _clip(_rolling_zscore(s, z_window, minp)),
        f"{feature_name}_roc_1": ((s - s.shift(1)) / np.clip(prev_1.abs(), 1e-12, None))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(-5.0, 5.0),
        f"{feature_name}_roc_12": ((s - s.shift(12)) / np.clip(prev_12.abs(), 1e-12, None))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(-5.0, 5.0),
        f"{feature_name}_ema_12_48_spread": ((ema_12 - ema_48) / np.clip(ema_48.abs(), 1e-12, None))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(-5.0, 5.0),
    }
    return out


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


def _get_symbol_col_first_of(
    df: pd.DataFrame,
    *,
    target_symbol: str,
    suffixes: Iterable[str],
    default: float = 0.0,
) -> pd.Series:
    """
    依序嘗試多個 suffix，回傳第一個「欄位存在」的序列；若皆不存在則回傳全 default。
    用於 volume / quote_volume 等在不同資料源有不同命名（volume vs volume_usd）時避免 5m 全 0。
    """
    for suf in suffixes:
        prefixed = f"{target_symbol}_{suf}"
        if prefixed in df.columns:
            return df[prefixed].astype(float)
        if suf in df.columns:
            return df[suf].astype(float)
    return pd.Series(default, index=df.index, dtype=float)


@dataclass(frozen=True)
class FeatureSpec:
    """特徵規格：固定欄位名稱與順序（用於解析與 debug）。"""

    price_seq_cols: Tuple[str, ...]
    price_seq_1d_cols: Tuple[str, ...]


_DEFAULT_1D_MACRO_COLS: Tuple[str, ...] = (
    "fear_greed_scale",
    "fear_greed_z",
    "altcoin_season_z",
    "bmo_z",
    "sopr_z",
    "fear_greed_z_short",
    "fear_greed_z_long",
    "fear_greed_roc_1",
    "fear_greed_roc_12",
    "fear_greed_ema_12_48_spread",
    "altcoin_season_z_short",
    "altcoin_season_z_long",
    "altcoin_season_roc_1",
    "altcoin_season_roc_12",
    "altcoin_season_ema_12_48_spread",
    "bmo_z_short",
    "bmo_z_long",
    "bmo_roc_1",
    "bmo_roc_12",
    "bmo_ema_12_48_spread",
    "sopr_z_short",
    "sopr_z_long",
    "sopr_roc_1",
    "sopr_roc_12",
    "sopr_ema_12_48_spread",
)


class FeatureTransformer:
    """
    特徵轉換器。

    注意：
    - 本類不依賴特定環境/模型，單純將資料轉成可訓練的 feature tensor。
    - 所有特徵皆為 float32，且保證不含 NaN/inf。
    """

    # ---- 5m Target：與 Config.OBS_PRICE_SEQ_TARGET_COLS 一致 ----
    OPTIMIZED_TARGET_5M_COLS: Final[Tuple[str, ...]] = Config.OBS_PRICE_SEQ_TARGET_COLS

    # ---- 5m Others：與 Config.OBS_PRICE_SEQ_OTHERS_COLS 一致 ----
    OTHERS_5M_COLS_PER_SYMBOL: Final[Tuple[str, ...]] = Config.OBS_PRICE_SEQ_OTHERS_COLS

    # ---- 兼容性：保留 BASE_5M_COLS 供旧代码使用 ----
    BASE_5M_COLS: Final[Tuple[str, ...]] = OPTIMIZED_TARGET_5M_COLS

    # ---- 1d Target：與 Config.OBS_PRICE_SEQ_1D_TARGET_COLS 一致 ----
    TARGET_1D_COLS: Final[Tuple[str, ...]] = Config.OBS_PRICE_SEQ_1D_TARGET_COLS

    # ---- 1d Others：每標的 + macro ----
    # 以「名稱集合」拆分 macro，避免固定切片造成配置欄位數擴充時失真。
    DEFAULT_1D_MACRO_COLS: Final[Tuple[str, ...]] = _DEFAULT_1D_MACRO_COLS
    OTHERS_1D_COLS_PER_SYMBOL: Final[Tuple[str, ...]] = tuple(
        c for c in Config.OBS_PRICE_SEQ_1D_OTHERS_COLS if c not in _DEFAULT_1D_MACRO_COLS
    )
    PRICE_SEQ_1D_MACRO_COLS: Final[Tuple[str, ...]] = tuple(
        c for c in Config.OBS_PRICE_SEQ_1D_OTHERS_COLS if c in _DEFAULT_1D_MACRO_COLS
    )

    # ---- 兼容性：保留 PRICE_SEQ_1D_SYMBOL_COLS 供旧代码使用 ----
    PRICE_SEQ_1D_SYMBOL_COLS: Final[Tuple[str, ...]] = TARGET_1D_COLS

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

    def build_5m_features_split(
        self,
        df_5m: pd.DataFrame,
        *,
        target_symbol: str,
        atr_ratio_arr: np.ndarray,
        rv_ratio_arr: np.ndarray,
        z_window: int = 288,
        z_window_short: int = 96,
        feature_symbols: Iterable[str] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        建立分离的 5m 特征矩阵（target 和 others）。

        支援雙尺度 z：長期 z_window（預設 288）與短期 z_window_short（預設 96），
        以及部分特徵僅 scale/clip（ret_1_atr, range_atr, body_atr）保留原始對比。

        Returns:
            (target_features, others_features, target_cols, others_cols)
        """
        if len(df_5m) != len(atr_ratio_arr) or len(df_5m) != len(rv_ratio_arr):
            raise ValueError("atr_ratio_arr/rv_ratio_arr 長度必須與 df_5m 相同")

        # 确定符号列表
        symbols = [str(target_symbol)]
        if feature_symbols is not None:
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

        alt_symbols = [s for s in symbols if s != str(target_symbol)]
        minp = max(20, z_window // 10)
        minp_short = max(12, z_window_short // 4)
        
        # === 读取 target_symbol 数据 ===
        c = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="close")
        o = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="open")
        h = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="high")
        l = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="low")
        v = _get_symbol_col_first_of(df_5m, target_symbol=target_symbol, suffixes=("volume", "volume_usd"))
        buy_v = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="buy_volume")
        sell_v = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="sell_volume")
        trades = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="trades")
        quote_v = _get_symbol_col_first_of(df_5m, target_symbol=target_symbol, suffixes=("quote_volume", "quote_volume_usd"))
        volume_ratio = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="volume_ratio")
        long_short_ratio = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="long_short_ratio")

        # === Target 特征计算 ===
        log_c = _safe_log(c)
        ret_1 = log_c.diff().fillna(0.0)
        ret_15m = ret_1.rolling(window=3, min_periods=1).sum()
        prev_c = c.shift(1).bfill()
        range_raw = (h - l) / np.clip(prev_c, 1e-12, None)
        body_raw = (c - o) / np.clip(prev_c, 1e-12, None)
        
        v_ma20 = v.rolling(window=20, min_periods=1).mean().replace(0.0, np.nan).fillna(1.0)
        volume_log = np.log(np.clip(v / v_ma20, 1e-12, None))
        vol_imb = (buy_v - sell_v) / (buy_v + sell_v + 1e-12)
        amihud = np.abs(ret_1) / (quote_v + 1e-12)
        
        # EMA48（僅供 ema_48_slope_z）
        ema48 = self._ema(c, 48)
        ema_48_slope = _safe_log(ema48).diff().fillna(0.0)

        # 支撑阻力位（96 窗口，供 dist_to_support_96_atr / dist_to_resistance_96_atr_scale）
        hi_96 = h.rolling(96, min_periods=10).max()
        lo_96 = l.rolling(96, min_periods=10).min()
        support_96 = lo_96
        resistance_96 = hi_96
        atr_est = pd.Series(atr_ratio_arr, index=df_5m.index).astype(float) * np.clip(c, 1e-12, None)
        atr_est = atr_est.replace(0.0, np.nan).fillna(1e-8)
        dist_to_support_96_atr = (c - support_96) / atr_est
        dist_to_resistance_96_atr = (resistance_96 - c) / atr_est

        # 部分特徵僅 scale/clip（不做 rolling z-score），保留短期尖峰對比
        ret_1_atr = ((c - c.shift(1)) / atr_est).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0)
        range_atr = ((h - l) / atr_est).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0)
        body_atr = ((c - o) / atr_est).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0)

        # 雙尺度 z：短期 z（window=96）
        ret_1_z_short = _clip(_rolling_zscore(ret_1, z_window_short, minp_short))
        ret_15m_z_short = _clip(_rolling_zscore(ret_15m, z_window_short, minp_short))
        volume_log_z_short = _clip(_rolling_zscore(volume_log, z_window_short, minp_short))

        # 价格跳跃检测
        price_jump = ret_1.abs() / ret_1.rolling(window=20, min_periods=5).std().replace(0.0, np.nan).fillna(1e-8)
        price_jump_z = _rolling_zscore(price_jump, z_window, minp)
        
        # 订单簿（僅 ob_depth_imbalance_scale、ob_slope_ask_z）
        bids_qty = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="bids_quantity")
        asks_qty = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="asks_quantity")
        ask_price_1 = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="ask_price_1")
        ask_price_5 = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="ask_price_5")
        has_ob = (bids_qty.abs().sum() > 0 or asks_qty.abs().sum() > 0)
        if has_ob:
            ob_depth_bid = bids_qty.rolling(5, min_periods=1).sum()
            ob_depth_ask = asks_qty.rolling(5, min_periods=1).sum()
            ob_depth_imbalance = (ob_depth_bid - ob_depth_ask) / (ob_depth_bid + ob_depth_ask + 1e-12)
            ob_slope_ask = (ask_price_5 - ask_price_1) / 5.0 if (ask_price_1.abs().sum() > 0 and ask_price_5.abs().sum() > 0) else pd.Series(0.0, index=df_5m.index)
        else:
            ob_depth_imbalance = pd.Series(0.0, index=df_5m.index)
            ob_slope_ask = pd.Series(0.0, index=df_5m.index)

        # 跨市场摘要（僅 alts_rel_ret_15m_abs_mean_scale, alts_trend_up_ratio, alts_volume_log_mean_scale）
        target_ret_15m = ret_15m
        alt_ret_15m_list: list[pd.Series] = []
        alt_rel_ret_15m_abs_list: list[pd.Series] = []
        alt_trend_up_list: list[pd.Series] = []
        alt_vol_log_list: list[pd.Series] = []

        for sym in alt_symbols:
            cs = _get_symbol_col(df_5m, target_symbol=sym, suffix="close")
            vs = _get_symbol_col(df_5m, target_symbol=sym, suffix="volume")
            log_cs = _safe_log(cs)
            ret1_s = log_cs.diff().fillna(0.0)
            ret15_s = ret1_s.rolling(window=3, min_periods=1).sum()
            v_ma20_s = vs.rolling(window=20, min_periods=1).mean().replace(0.0, np.nan).fillna(1.0)
            vol_log_s = np.log(np.clip(vs / v_ma20_s, 1e-12, None))
            ema12_s = self._ema(cs, 12)
            ema48_s = self._ema(cs, 48)

            alt_ret_15m_list.append(ret15_s)
            alt_rel_ret_15m_abs_list.append((ret15_s - target_ret_15m).abs())
            alt_trend_up_list.append((ema12_s > ema48_s).astype(float))
            alt_vol_log_list.append(vol_log_s)

        if alt_symbols:
            alts_rel_ret_15m_abs_mean = pd.concat(alt_rel_ret_15m_abs_list, axis=1).mean(axis=1) if alt_rel_ret_15m_abs_list else pd.Series(0.0, index=df_5m.index)
            alts_trend_up_ratio = pd.concat(alt_trend_up_list, axis=1).mean(axis=1).fillna(0.5) if alt_trend_up_list else pd.Series(0.5, index=df_5m.index)
            alts_trend_up_ratio = ((alts_trend_up_ratio * 2.0) - 1.0).clip(-1.0, 1.0)
            alts_volume_log_mean = pd.concat(alt_vol_log_list, axis=1).mean(axis=1) if alt_vol_log_list else pd.Series(0.0, index=df_5m.index)
        else:
            alts_rel_ret_15m_abs_mean = pd.Series(0.0, index=df_5m.index)
            alts_trend_up_ratio = pd.Series(0.0, index=df_5m.index)
            alts_volume_log_mean = pd.Series(0.0, index=df_5m.index)
        
        # === 近期做多流動性區（僅 dist_to_long_liq_zone_atr）===
        left_swing, right_swing = 2, 2
        swing_low = (l.shift(1).rolling(left_swing, min_periods=1).min() > l) & (l < l.shift(-1).rolling(right_swing, min_periods=1).min())
        recent_swing_low = l.where(swing_low).ffill().bfill()
        dist_to_long_liq_zone_atr = ((c - recent_swing_low) / atr_est).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)

        # scale/clip 版本（僅輸出用到的）
        ret_15m_scale = ((c - c.shift(3)) / atr_est).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0)
        volume_log_scale = volume_log.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0)
        dist_to_resistance_96_atr_scale = dist_to_resistance_96_atr.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)
        ob_depth_imbalance_scale = ob_depth_imbalance.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        alts_rel_ret_15m_abs_mean_scale = alts_rel_ret_15m_abs_mean.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 0.05)
        alts_volume_log_mean_scale = alts_volume_log_mean.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0)

        trades_per_volume = trades / (v + 1e-12)
        volume_impact = np.abs(ret_1) / (v + 1e-12)
        volume_impact_scale = volume_impact.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 0.001)
        rv_ratio_scale = pd.Series(rv_ratio_arr, index=df_5m.index).clip(0.0, 0.05)

        # === 构建 Target 特征 DataFrame（先建全量，後續再依 Config 篩選）===
        target_feats = pd.DataFrame({
            "ret_15m_scale": ret_15m_scale,
            "body_z": _clip(_rolling_zscore(body_raw, z_window, minp)),
            "range_atr": range_atr,
            "ret_1_z_short": ret_1_z_short,
            "ret_15m_z_short": ret_15m_z_short,
            "volume_log_z_short": volume_log_z_short,
            "volume_log_scale": volume_log_scale,
            "quote_volume_log_z": _clip(_rolling_zscore(np.log1p(np.clip(quote_v, 0.0, None)), z_window, minp)),
            "amihud_z": _clip(_rolling_zscore(amihud, z_window, minp)),
            "long_short_ratio_z": _clip(_rolling_zscore(long_short_ratio, z_window, minp)),
            "trades_per_volume_z": _clip(_rolling_zscore(trades_per_volume, z_window, minp)),
            "volume_impact_scale": volume_impact_scale,
            "ema_48_slope_z": _clip(_rolling_zscore(ema_48_slope, z_window, minp)),
            "rv_ratio_scale": rv_ratio_scale,
            "dist_to_support_96_atr": _clip(_rolling_zscore(dist_to_support_96_atr, z_window, minp)),
            "dist_to_resistance_96_atr_scale": dist_to_resistance_96_atr_scale,
            "price_jump_z": _clip(price_jump_z),
            "ob_depth_imbalance_scale": ob_depth_imbalance_scale,
            "ob_slope_ask_z": _clip(_rolling_zscore(ob_slope_ask, z_window, minp)),
            "alts_rel_ret_15m_abs_mean_scale": alts_rel_ret_15m_abs_mean_scale,
            "alts_trend_up_ratio": alts_trend_up_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "alts_volume_log_mean_scale": alts_volume_log_mean_scale,
            "dist_to_long_liq_zone_atr": dist_to_long_liq_zone_atr.replace([np.inf, -np.inf], np.nan).fillna(0.0),
        }, index=df_5m.index)

        # 重度工程化：把 5m 原始 11 欄位全部展開為多尺度特徵
        target_raw_fields = {
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "buy_volume": buy_v,
            "sell_volume": sell_v,
            "volume_ratio": volume_ratio,
            "long_short_ratio": long_short_ratio,
            "trades": trades,
            "quote_volume": quote_v,
        }
        target_heavy: dict[str, pd.Series] = {}
        for raw_name, raw_series in target_raw_fields.items():
            target_heavy.update(
                _heavy_field_features(
                    raw_series,
                    feature_name=raw_name,
                    z_window=z_window,
                    z_window_short=z_window_short,
                    minp=minp,
                    minp_short=minp_short,
                )
            )
        target_feats = pd.concat([target_feats, pd.DataFrame(target_heavy, index=df_5m.index)], axis=1)

        configured_target_cols = list(self.OPTIMIZED_TARGET_5M_COLS)
        if configured_target_cols:
            missing_target = [c for c in configured_target_cols if c not in target_feats.columns]
            if missing_target:
                raise ValueError(f"OBS_PRICE_SEQ_TARGET_COLS 含未知欄位: {missing_target[:8]}")
            target_cols = configured_target_cols
        else:
            target_cols = list(target_feats.columns)
        target_feats = target_feats.reindex(columns=target_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # === 构建 Others 特征 ===
        others_feats_list = []
        others_cols = []
        
        for sym in alt_symbols:
            cs = _get_symbol_col(df_5m, target_symbol=sym, suffix="close")
            hs = _get_symbol_col(df_5m, target_symbol=sym, suffix="high")
            ls = _get_symbol_col(df_5m, target_symbol=sym, suffix="low")
            os = _get_symbol_col(df_5m, target_symbol=sym, suffix="open")
            vs = _get_symbol_col_first_of(df_5m, target_symbol=sym, suffixes=("volume", "volume_usd"))
            buy_vs = _get_symbol_col(df_5m, target_symbol=sym, suffix="buy_volume")
            sell_vs = _get_symbol_col(df_5m, target_symbol=sym, suffix="sell_volume")
            trades_s = _get_symbol_col(df_5m, target_symbol=sym, suffix="trades")
            quote_vs = _get_symbol_col_first_of(df_5m, target_symbol=sym, suffixes=("quote_volume", "quote_volume_usd"))
            volume_ratio_s = _get_symbol_col(df_5m, target_symbol=sym, suffix="volume_ratio")
            long_short_ratio_s = _get_symbol_col(df_5m, target_symbol=sym, suffix="long_short_ratio")

            log_cs = _safe_log(cs)
            ret1_s = log_cs.diff().fillna(0.0)
            ret15_s = ret1_s.rolling(window=3, min_periods=1).sum()
            v_ma20_s = vs.rolling(window=20, min_periods=1).mean().replace(0.0, np.nan).fillna(1.0)
            vol_log_s = np.log(np.clip(vs / v_ma20_s, 1e-12, None))
            ema12_s = self._ema(cs, 12)
            ema48_s = self._ema(cs, 48)
            ema_12_48_spread_raw_s = ((ema12_s - ema48_s) / np.clip(ema48_s, 1e-12, None)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.3, 0.3)
            ret_12_raw_s = log_cs.diff(12).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.05, 0.05)
            atr_ratio_s = self._compute_atr_ratio_from_ohlc(cs, hs, ls)
            atr_est_s = (atr_ratio_s * np.clip(cs, 1e-12, None)).replace(0.0, np.nan).fillna(1e-8)
            ret_15m_scale_s = ((cs - cs.shift(3)) / atr_est_s).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0)
            ret_1h_scale_s = ((cs - cs.shift(12)) / atr_est_s).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0)
            ret_1_atr_s = ((cs - cs.shift(1)) / atr_est_s).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0)
            delta_s = cs.diff().fillna(0.0)
            up_s = delta_s.clip(lower=0.0)
            down_s = (-delta_s).clip(lower=0.0)
            rs_s = up_s.rolling(14, min_periods=5).mean() / np.clip(down_s.rolling(14, min_periods=5).mean(), 1e-12, None)
            rsi_14_s = (((100.0 - (100.0 / (1.0 + rs_s))).fillna(50.0) - 50.0) / 50.0).clip(-1.0, 1.0).astype(float)
            body_raw_s = (cs - os) / np.clip(cs.shift(1).bfill(), 1e-12, None)
            range_raw_s = (hs - ls) / np.clip(cs.shift(1).bfill(), 1e-12, None)
            body_range_ratio_s = (body_raw_s / (range_raw_s + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
            volume_impact_s = np.abs(ret1_s) / (vs + 1e-12)
            volume_impact_scale_s = volume_impact_s.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 0.001)
            up_bar_s = (cs > os).astype(float)
            up_vol_12_s = (up_bar_s * vs).rolling(12, min_periods=1).sum()
            total_vol_12_s = vs.rolling(12, min_periods=1).sum().replace(0.0, np.nan).fillna(1.0)
            up_volume_ratio_12_s = (up_vol_12_s / total_vol_12_s * 2.0 - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
            left_swing_s, right_swing_s = 2, 2
            swing_high_s = (hs.shift(1).rolling(left_swing_s, min_periods=1).max() < hs) & (hs > hs.shift(-1).rolling(right_swing_s, min_periods=1).max())
            swing_low_s = (ls.shift(1).rolling(left_swing_s, min_periods=1).min() > ls) & (ls < ls.shift(-1).rolling(right_swing_s, min_periods=1).min())
            recent_swing_high_s = hs.where(swing_high_s).ffill().bfill()
            recent_swing_low_s = ls.where(swing_low_s).ffill().bfill()
            dist_to_long_liq_zone_atr_s = ((cs - recent_swing_low_s) / atr_est_s).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)
            dist_to_short_liq_zone_atr_s = ((recent_swing_high_s - cs) / atr_est_s).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)

            base_feats: dict[str, pd.Series] = {
                "ret_15m_scale": ret_15m_scale_s,
                "ret_1h_scale": ret_1h_scale_s,
                "ret_1_atr": ret_1_atr_s,
                "volume_log_z": _clip(_rolling_zscore(vol_log_s, z_window, minp)),
                "rsi_14": rsi_14_s.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0),
                "body_range_ratio": body_range_ratio_s,
                "volume_impact_scale": volume_impact_scale_s,
                "ema_12_48_spread_raw": ema_12_48_spread_raw_s,
                "ret_12_raw": ret_12_raw_s,
                "up_volume_ratio_12": up_volume_ratio_12_s,
                "dist_to_long_liq_zone_atr": dist_to_long_liq_zone_atr_s.replace([np.inf, -np.inf], np.nan).fillna(0.0),
                "dist_to_short_liq_zone_atr": dist_to_short_liq_zone_atr_s.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            }

            # 重度工程化：others 每個 symbol 也納入 11 個原始欄位的多尺度特徵
            others_raw_fields = {
                "open": os,
                "high": hs,
                "low": ls,
                "close": cs,
                "volume": vs,
                "buy_volume": buy_vs,
                "sell_volume": sell_vs,
                "volume_ratio": volume_ratio_s,
                "long_short_ratio": long_short_ratio_s,
                "trades": trades_s,
                "quote_volume": quote_vs,
            }
            for raw_name, raw_series in others_raw_fields.items():
                base_feats.update(
                    _heavy_field_features(
                        raw_series,
                        feature_name=raw_name,
                        z_window=z_window,
                        z_window_short=z_window_short,
                        minp=minp,
                        minp_short=minp_short,
                    )
                )

            configured_others_base_cols = list(self.OTHERS_5M_COLS_PER_SYMBOL)
            if configured_others_base_cols:
                missing_others = [c for c in configured_others_base_cols if c not in base_feats]
                if missing_others:
                    raise ValueError(f"OBS_PRICE_SEQ_OTHERS_COLS 含未知欄位: {missing_others[:8]}")
                selected_base_cols = configured_others_base_cols
            else:
                selected_base_cols = list(base_feats.keys())

            sym_feats = pd.DataFrame(
                {f"{sym}_{col}": base_feats[col] for col in selected_base_cols},
                index=df_5m.index,
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32, copy=False)
            others_feats_list.append(sym_feats)
            others_cols.extend([f"{sym}_{col}" for col in selected_base_cols])
        
        if others_feats_list:
            others_feats = pd.concat(others_feats_list, axis=1, copy=False)
            if list(others_feats.columns) != others_cols:
                others_feats = others_feats.reindex(columns=others_cols)
            others_feats = others_feats.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32, copy=False)
        else:
            others_feats = pd.DataFrame(index=df_5m.index, columns=others_cols, dtype=np.float32).fillna(0.0)
        
        return (
            target_feats.astype(np.float32, copy=False).to_numpy(copy=False),
            others_feats.to_numpy(copy=False),
            target_cols,
            others_cols,
        )

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
        建立 1d 特徵矩陣（包含 symbol-specific price+coinglass + macro 共用）。

        Args:
            df_1d: merge 後的 1d DataFrame（含 `{symbol}_open_interest_close` 等欄位，以及 macro 前綴欄位）
            target_symbol: 例如 "BTCUSDT"
            z_window_1d: rolling z-score window（1d 下 60 約 2 個月）

        Returns:
            (features_arr, cols)
        """
        minp = max(10, z_window_1d // 6)

        # ---- symbol-specific（price + coinglass 1d）----
        # 1) price-based regime
        close_1d = _get_symbol_col(df_1d, target_symbol=target_symbol, suffix="close")
        high_1d = _get_symbol_col(df_1d, target_symbol=target_symbol, suffix="high")
        low_1d = _get_symbol_col(df_1d, target_symbol=target_symbol, suffix="low")
        log_close_1d = _safe_log(close_1d)
        ret_1d = log_close_1d.diff().fillna(0.0)
        range_1d = (high_1d - low_1d) / np.maximum(close_1d, 1e-12)
        ema_20 = self._ema(close_1d, span=20)
        ema_60 = self._ema(close_1d, span=60)
        close_over_ema_20 = (close_1d / np.maximum(ema_20, 1e-12)) - 1.0
        ema_20_60_spread = (ema_20 - ema_60) / np.maximum(ema_60, 1e-12)

        # 2) coinglass 1d
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
                "ret_1d_z": _clip(_rolling_zscore(ret_1d, z_window_1d, minp)),
                "range_1d_z": _clip(_rolling_zscore(range_1d, z_window_1d, minp)),
                "close_over_ema_20_z": _clip(_rolling_zscore(close_over_ema_20, z_window_1d, minp)),
                "ema_20_60_spread_z": _clip(_rolling_zscore(ema_20_60_spread, z_window_1d, minp)),
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

    def build_1d_features_split(
        self,
        df_1d: pd.DataFrame,
        df_5m: pd.DataFrame,
        *,
        target_symbol: str,
        z_window_1d: int = 60,
        feature_symbols: Iterable[str] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        建立分离的 1d 特征矩阵（target 和 others）。
        
        Args:
            df_1d: merge 后的 1d DataFrame
            df_5m: merge 后的 5m DataFrame（用于计算长期关键价位和 VWAP，基于 288 窗口）
            target_symbol: 例如 "BTCUSDT"
            z_window_1d: rolling z-score window（1d 下 60 约 2 个月）
            feature_symbols: 其他币种列表
            
        Returns:
            (target_features, others_features, target_cols, others_cols)
        """
        minp = max(10, z_window_1d // 6)
        z_window_1d_short = max(20, z_window_1d // 2)
        minp_short = max(8, z_window_1d_short // 4)

        # 确定符号列表
        symbols = [str(target_symbol)]
        if feature_symbols is not None:
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
        alt_symbols = [s for s in symbols if s != str(target_symbol)]

        # === 读取 target_symbol 数据 ===
        close_1d = _get_symbol_col(df_1d, target_symbol=target_symbol, suffix="close")
        open_1d = _get_symbol_col(df_1d, target_symbol=target_symbol, suffix="open")
        high_1d = _get_symbol_col(df_1d, target_symbol=target_symbol, suffix="high")
        low_1d = _get_symbol_col(df_1d, target_symbol=target_symbol, suffix="low")
        volume_1d = _get_symbol_col_first_of(df_1d, target_symbol=target_symbol, suffixes=("volume", "volume_usd"))
        buy_volume_1d = _get_symbol_col(df_1d, target_symbol=target_symbol, suffix="buy_volume")
        sell_volume_1d = _get_symbol_col(df_1d, target_symbol=target_symbol, suffix="sell_volume")
        trades_1d = _get_symbol_col(df_1d, target_symbol=target_symbol, suffix="trades")
        quote_volume_1d = _get_symbol_col_first_of(df_1d, target_symbol=target_symbol, suffixes=("quote_volume", "quote_volume_usd"))
        volume_ratio_1d = _get_symbol_col(df_1d, target_symbol=target_symbol, suffix="volume_ratio")
        long_short_ratio_1d = _get_symbol_col(df_1d, target_symbol=target_symbol, suffix="long_short_ratio")

        # Coinglass 数据
        oi_close = _get_col(df_1d, f"{target_symbol}_open_interest_close")
        funding_close = _get_col(df_1d, f"{target_symbol}_funding_rate_close")
        ls_account_ratio = _get_col(df_1d, f"{target_symbol}_global_long_short_account_ratio_global_account_long_short_ratio")
        liq_long = _get_col(df_1d, f"{target_symbol}_liquidation_long_liquidation_usd")
        liq_short = _get_col(df_1d, f"{target_symbol}_liquidation_short_liquidation_usd")
        bids_usd = _get_col(df_1d, f"{target_symbol}_ask_bids_bids_usd")
        asks_usd = _get_col(df_1d, f"{target_symbol}_ask_bids_asks_usd")
        ob_imbalance = (bids_usd - asks_usd) / (bids_usd + asks_usd + 1e-12)

        # 1d 基礎序列
        log_close_1d = _safe_log(close_1d)
        ret_1d = log_close_1d.diff().fillna(0.0)
        ret_3d = ret_1d.rolling(window=3, min_periods=1).sum()
        ret_7d = ret_1d.rolling(window=7, min_periods=1).sum()
        ret_14d = ret_1d.rolling(window=14, min_periods=1).sum()
        range_1d = (high_1d - low_1d) / np.maximum(close_1d.shift(1).bfill(), 1e-12)
        body_1d = (close_1d - open_1d) / np.maximum(close_1d.shift(1).bfill(), 1e-12)

        vol_ma20 = volume_1d.rolling(20, min_periods=1).mean().replace(0.0, np.nan).fillna(1.0)
        volume_log_1d = np.log(np.clip(volume_1d / vol_ma20, 1e-12, None))
        trades_per_volume_1d = trades_1d / (volume_1d + 1e-12)
        volume_impact_1d = np.abs(ret_1d) / (volume_1d + 1e-12)
        volume_impact_scale_1d = volume_impact_1d.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 0.05)

        ema12_1d = self._ema(close_1d, 12)
        ema48_1d = self._ema(close_1d, 48)
        ema_12_48_spread_1d = ((ema12_1d - ema48_1d) / np.clip(ema48_1d, 1e-12, None)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ema_12_slope_1d = _safe_log(ema12_1d).diff().fillna(0.0)

        # 长期关键价位（1d = 288×5m），對齊到 1d 時間軸
        c_5m = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="close")
        h_5m = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="high")
        l_5m = _get_symbol_col(df_5m, target_symbol=target_symbol, suffix="low")
        resistance_1d = h_5m.rolling(288, min_periods=20).max()
        support_1d = l_5m.rolling(288, min_periods=20).min()
        prev_close_5m = c_5m.shift(1)
        tr_5m = pd.concat([(h_5m - l_5m), (h_5m - prev_close_5m).abs(), (l_5m - prev_close_5m).abs()], axis=1).max(axis=1).fillna(0.0)
        atr_1d = tr_5m.rolling(14, min_periods=5).mean().fillna(1e-8)
        if len(df_1d) > 0 and len(df_5m) > 0:
            times_1d = df_1d["timestamp"].values if "timestamp" in df_1d.columns else df_1d.index
            times_5m = df_5m["timestamp"].values if "timestamp" in df_5m.columns else df_5m.index
            resistance_1d_aligned = pd.Series(index=df_1d.index, dtype=float)
            support_1d_aligned = pd.Series(index=df_1d.index, dtype=float)
            atr_1d_aligned = pd.Series(index=df_1d.index, dtype=float)
            for i, t_1d in enumerate(times_1d):
                idx_5m = np.searchsorted(times_5m, t_1d, side="right") - 1
                if 0 <= idx_5m < len(resistance_1d):
                    resistance_1d_aligned.iloc[i] = resistance_1d.iloc[idx_5m]
                    support_1d_aligned.iloc[i] = support_1d.iloc[idx_5m]
                    atr_1d_aligned.iloc[i] = atr_1d.iloc[idx_5m]
                else:
                    resistance_1d_aligned.iloc[i] = close_1d.iloc[i]
                    support_1d_aligned.iloc[i] = close_1d.iloc[i]
                    atr_1d_aligned.iloc[i] = close_1d.iloc[i] * 0.01
        else:
            resistance_1d_aligned = close_1d.copy()
            support_1d_aligned = close_1d.copy()
            atr_1d_aligned = close_1d * 0.01
        atr_1d_aligned = np.maximum(atr_1d_aligned, 1e-12)
        dist_to_resistance_1d_atr = (resistance_1d_aligned - close_1d) / atr_1d_aligned
        dist_to_support_1d_atr = (close_1d - support_1d_aligned) / atr_1d_aligned

        # 时间特征
        if "timestamp" in df_1d.columns:
            timestamps = pd.to_datetime(df_1d["timestamp"])
        elif hasattr(df_1d.index, "dtype") and df_1d.index.dtype == "datetime64[ns]":
            timestamps = df_1d.index
        else:
            timestamps = pd.date_range(start="2020-01-01", periods=len(df_1d), freq="D")
        day_of_week = timestamps.dt.dayofweek if isinstance(timestamps, pd.Series) else timestamps.dayofweek
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)

        # liq_sweep（1d 尺度）
        n_sweep_1d = max(3, z_window_1d // 10)
        recent_max_h_1d = high_1d.rolling(n_sweep_1d, min_periods=2).max()
        recent_min_l_1d = low_1d.rolling(n_sweep_1d, min_periods=2).min()
        liq_sweep_up_1d = ((high_1d > recent_max_h_1d.shift(1)) & (close_1d < recent_max_h_1d.shift(1))).astype(np.float32)
        liq_sweep_down_1d = ((low_1d < recent_min_l_1d.shift(1)) & (close_1d > recent_min_l_1d.shift(1))).astype(np.float32)

        # 跨市場摘要（1d）
        alt_ret_1d_list: list[pd.Series] = []
        alt_rel_ret_1d_abs_list: list[pd.Series] = []
        alt_trend_up_list: list[pd.Series] = []
        alt_volume_log_list: list[pd.Series] = []
        for sym in alt_symbols:
            cs = _get_symbol_col(df_1d, target_symbol=sym, suffix="close")
            vs = _get_symbol_col_first_of(df_1d, target_symbol=sym, suffixes=("volume", "volume_usd"))
            log_cs = _safe_log(cs)
            ret1_s = log_cs.diff().fillna(0.0)
            v_ma20_s = vs.rolling(20, min_periods=1).mean().replace(0.0, np.nan).fillna(1.0)
            vol_log_s = np.log(np.clip(vs / v_ma20_s, 1e-12, None))
            ema12_s = self._ema(cs, 12)
            ema48_s = self._ema(cs, 48)
            alt_ret_1d_list.append(ret1_s)
            alt_rel_ret_1d_abs_list.append((ret1_s - ret_1d).abs())
            alt_trend_up_list.append((ema12_s > ema48_s).astype(float))
            alt_volume_log_list.append(vol_log_s)
        if alt_symbols:
            alts_rel_ret_1d_abs_mean = pd.concat(alt_rel_ret_1d_abs_list, axis=1).mean(axis=1)
            alts_trend_up_ratio = pd.concat(alt_trend_up_list, axis=1).mean(axis=1).fillna(0.5)
            alts_trend_up_ratio = ((alts_trend_up_ratio * 2.0) - 1.0).clip(-1.0, 1.0)
            alts_volume_log_mean = pd.concat(alt_volume_log_list, axis=1).mean(axis=1)
            alts_ret_1d_std = pd.concat(alt_ret_1d_list, axis=1).std(axis=1).fillna(0.0)
        else:
            alts_rel_ret_1d_abs_mean = pd.Series(0.0, index=df_1d.index)
            alts_trend_up_ratio = pd.Series(0.0, index=df_1d.index)
            alts_volume_log_mean = pd.Series(0.0, index=df_1d.index)
            alts_ret_1d_std = pd.Series(0.0, index=df_1d.index)

        oi_ma20 = oi_close.rolling(20, min_periods=5).mean().replace(0.0, np.nan).fillna(oi_close)
        oi_close_scale = ((oi_close / oi_ma20) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-2.0, 2.0)
        funding_close_scale = funding_close.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.01, 0.01) * 500.0
        liq_long_log_scale = np.log1p(np.clip(liq_long, 0.0, None)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)
        liq_short_log_scale = np.log1p(np.clip(liq_short, 0.0, None)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)

        # === 构建 Target 特征（全量），後續按 Config 篩選 ===
        target_feats = pd.DataFrame(
            {
                "ret_1d_z_short": _clip(_rolling_zscore(ret_1d, z_window_1d_short, minp_short)),
                "ret_1d_z_long": _clip(_rolling_zscore(ret_1d, z_window_1d, minp)),
                "ret_3d_z_short": _clip(_rolling_zscore(ret_3d, z_window_1d_short, minp_short)),
                "ret_7d_z_long": _clip(_rolling_zscore(ret_7d, z_window_1d, minp)),
                "ret_14d_z_long": _clip(_rolling_zscore(ret_14d, z_window_1d, minp)),
                "range_1d_z": _clip(_rolling_zscore(range_1d, z_window_1d, minp)),
                "body_1d_z": _clip(_rolling_zscore(body_1d, z_window_1d, minp)),
                "volume_log_z_short": _clip(_rolling_zscore(volume_log_1d, z_window_1d_short, minp_short)),
                "volume_log_z_long": _clip(_rolling_zscore(volume_log_1d, z_window_1d, minp)),
                "trades_per_volume_z": _clip(_rolling_zscore(trades_per_volume_1d, z_window_1d, minp)),
                "long_short_ratio_z": _clip(_rolling_zscore(long_short_ratio_1d, z_window_1d, minp)),
                "ob_imbalance_z": _clip(_rolling_zscore(ob_imbalance, z_window_1d, minp)),
                "oi_close_scale": oi_close_scale,
                "funding_close_scale": funding_close_scale,
                "ls_account_ratio_z": _clip(_rolling_zscore(ls_account_ratio, z_window_1d, minp)),
                "liq_long_log_scale": liq_long_log_scale,
                "liq_short_log_z": _clip(_rolling_zscore(np.log1p(np.clip(liq_short, 0.0, None)), z_window_1d, minp)),
                "ema_12_48_spread_z": _clip(_rolling_zscore(ema_12_48_spread_1d, z_window_1d, minp)),
                "ema_12_slope_z": _clip(_rolling_zscore(ema_12_slope_1d, z_window_1d, minp)),
                "volume_impact_scale": volume_impact_scale_1d,
                "dist_to_resistance_1d_atr": _clip(_rolling_zscore(dist_to_resistance_1d_atr, z_window_1d, minp)),
                "dist_to_support_1d_atr": _clip(_rolling_zscore(dist_to_support_1d_atr, z_window_1d, minp)),
                "day_of_week_cos": day_cos.values if hasattr(day_cos, "values") else day_cos,
                "day_of_week_sin": day_sin.values if hasattr(day_sin, "values") else day_sin,
                "liq_sweep_up": liq_sweep_up_1d,
                "liq_sweep_down": liq_sweep_down_1d,
                "alts_rel_ret_1d_abs_mean_z": _clip(_rolling_zscore(alts_rel_ret_1d_abs_mean, z_window_1d, minp)),
                "alts_trend_up_ratio": alts_trend_up_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0),
                "alts_volume_log_mean_z": _clip(_rolling_zscore(alts_volume_log_mean, z_window_1d, minp)),
                "alts_ret_1d_std_z": _clip(_rolling_zscore(alts_ret_1d_std, z_window_1d, minp)),
            },
            index=df_1d.index,
        )

        target_raw_fields = {
            "open": open_1d,
            "high": high_1d,
            "low": low_1d,
            "close": close_1d,
            "volume": volume_1d,
            "buy_volume": buy_volume_1d,
            "sell_volume": sell_volume_1d,
            "volume_ratio": volume_ratio_1d,
            "long_short_ratio": long_short_ratio_1d,
            "trades": trades_1d,
            "quote_volume": quote_volume_1d,
            "oi_close": oi_close,
            "funding_close": funding_close,
            "liq_long": liq_long,
            "liq_short": liq_short,
            "ob_imbalance": ob_imbalance,
        }
        target_heavy: dict[str, pd.Series] = {}
        for raw_name, raw_series in target_raw_fields.items():
            target_heavy.update(
                _heavy_field_features(
                    raw_series,
                    feature_name=raw_name,
                    z_window=z_window_1d,
                    z_window_short=z_window_1d_short,
                    minp=minp,
                    minp_short=minp_short,
                )
            )
        target_feats = pd.concat([target_feats, pd.DataFrame(target_heavy, index=df_1d.index)], axis=1)

        configured_target_cols = list(self.TARGET_1D_COLS)
        if configured_target_cols:
            missing_target = [c for c in configured_target_cols if c not in target_feats.columns]
            if missing_target:
                raise ValueError(f"OBS_PRICE_SEQ_1D_TARGET_COLS 含未知欄位: {missing_target[:8]}")
            target_cols = configured_target_cols
        else:
            target_cols = list(target_feats.columns)
        target_feats = target_feats.reindex(columns=target_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # === 构建 Others 特征 ===
        others_feats_list: list[pd.DataFrame] = []
        others_cols: list[str] = []
        configured_others_cols_per_symbol = list(self.OTHERS_1D_COLS_PER_SYMBOL)
        selected_others_cols_per_symbol: list[str] | None = None

        for sym in alt_symbols:
            cs_1d = _get_symbol_col(df_1d, target_symbol=sym, suffix="close")
            os_1d = _get_symbol_col(df_1d, target_symbol=sym, suffix="open")
            high_1d_s = _get_symbol_col(df_1d, target_symbol=sym, suffix="high")
            low_1d_s = _get_symbol_col(df_1d, target_symbol=sym, suffix="low")
            vol_1d_s = _get_symbol_col_first_of(df_1d, target_symbol=sym, suffixes=("volume", "volume_usd"))
            buy_vol_1d_s = _get_symbol_col(df_1d, target_symbol=sym, suffix="buy_volume")
            sell_vol_1d_s = _get_symbol_col(df_1d, target_symbol=sym, suffix="sell_volume")
            trades_1d_s = _get_symbol_col(df_1d, target_symbol=sym, suffix="trades")
            quote_vol_1d_s = _get_symbol_col_first_of(df_1d, target_symbol=sym, suffixes=("quote_volume", "quote_volume_usd"))
            volume_ratio_1d_s = _get_symbol_col(df_1d, target_symbol=sym, suffix="volume_ratio")
            long_short_ratio_1d_s = _get_symbol_col(df_1d, target_symbol=sym, suffix="long_short_ratio")

            log_cs_1d = _safe_log(cs_1d)
            ret_1d_s = log_cs_1d.diff().fillna(0.0)
            ret_3d_s = ret_1d_s.rolling(window=3, min_periods=1).sum()
            vol_ma20_s = vol_1d_s.rolling(20, min_periods=1).mean().replace(0.0, np.nan).fillna(1.0)
            vol_log_s = np.log(np.clip(vol_1d_s / vol_ma20_s, 1e-12, None))
            ema12_s = self._ema(cs_1d, 12)
            ema48_s = self._ema(cs_1d, 48)
            ema_12_48_spread_raw_s = ((ema12_s - ema48_s) / np.clip(ema48_s, 1e-12, None)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.5, 0.5)
            range_raw_s = (high_1d_s - low_1d_s) / np.clip(cs_1d.shift(1).bfill(), 1e-12, None)
            body_raw_s = (cs_1d - os_1d) / np.clip(cs_1d.shift(1).bfill(), 1e-12, None)
            body_range_ratio_s = (body_raw_s / (range_raw_s + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)

            delta_s = cs_1d.diff().fillna(0.0)
            up_s = delta_s.clip(lower=0.0)
            down_s = (-delta_s).clip(lower=0.0)
            rs_s = up_s.rolling(14, min_periods=5).mean() / np.clip(down_s.rolling(14, min_periods=5).mean(), 1e-12, None)
            rsi_14_s = (((100.0 - (100.0 / (1.0 + rs_s))).fillna(50.0) - 50.0) / 50.0).clip(-1.0, 1.0).astype(float)

            prev_c_1d_s = cs_1d.shift(1)
            tr_1d_s = pd.concat(
                [
                    (high_1d_s - low_1d_s),
                    (high_1d_s - prev_c_1d_s).abs(),
                    (low_1d_s - prev_c_1d_s).abs(),
                ],
                axis=1,
            ).max(axis=1).fillna(0.0)
            atr_1d_s = tr_1d_s.rolling(14, min_periods=3).mean().replace(0.0, np.nan).fillna(cs_1d * 0.01)
            atr_1d_s = np.maximum(atr_1d_s, 1e-12)
            ret_1d_scale_s = ((cs_1d - cs_1d.shift(1)) / atr_1d_s).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)
            ret_3d_scale_s = ((cs_1d - cs_1d.shift(3)) / atr_1d_s).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)

            volume_impact_s = np.abs(ret_1d_s) / (vol_1d_s + 1e-12)
            volume_impact_scale_s = volume_impact_s.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 0.05)
            up_bar_s = (cs_1d > os_1d).astype(float)
            up_vol_12_s = (up_bar_s * vol_1d_s).rolling(12, min_periods=1).sum()
            total_vol_12_s = vol_1d_s.rolling(12, min_periods=1).sum().replace(0.0, np.nan).fillna(1.0)
            up_volume_ratio_12_s = (up_vol_12_s / total_vol_12_s * 2.0 - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)

            left_swing_1d_s, right_swing_1d_s = 2, 2
            swing_high_1d_s = (high_1d_s.shift(1).rolling(left_swing_1d_s, min_periods=1).max() < high_1d_s) & (
                high_1d_s > high_1d_s.shift(-1).rolling(right_swing_1d_s, min_periods=1).max()
            )
            swing_low_1d_s = (low_1d_s.shift(1).rolling(left_swing_1d_s, min_periods=1).min() > low_1d_s) & (
                low_1d_s < low_1d_s.shift(-1).rolling(right_swing_1d_s, min_periods=1).min()
            )
            recent_swing_high_1d_s = high_1d_s.where(swing_high_1d_s).ffill().bfill()
            recent_swing_low_1d_s = low_1d_s.where(swing_low_1d_s).ffill().bfill()
            dist_to_long_liq_zone_atr_1d_s = ((cs_1d - recent_swing_low_1d_s) / atr_1d_s).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)
            dist_to_short_liq_zone_atr_1d_s = ((recent_swing_high_1d_s - cs_1d) / atr_1d_s).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)
            n_sweep_1d_s = max(3, z_window_1d // 10)
            recent_min_l_1d_s = low_1d_s.rolling(n_sweep_1d_s, min_periods=2).min()
            liq_sweep_down_1d_s = ((low_1d_s < recent_min_l_1d_s.shift(1)) & (cs_1d > recent_min_l_1d_s.shift(1))).astype(np.float32)

            oi_close_s = _get_col(df_1d, f"{sym}_open_interest_close")
            funding_close_s = _get_col(df_1d, f"{sym}_funding_rate_close")
            ls_account_ratio_s = _get_col(df_1d, f"{sym}_global_long_short_account_ratio_global_account_long_short_ratio")
            liq_long_s = _get_col(df_1d, f"{sym}_liquidation_long_liquidation_usd")
            liq_short_s = _get_col(df_1d, f"{sym}_liquidation_short_liquidation_usd")
            bids_usd_s = _get_col(df_1d, f"{sym}_ask_bids_bids_usd")
            asks_usd_s = _get_col(df_1d, f"{sym}_ask_bids_asks_usd")
            ob_imbalance_s = (bids_usd_s - asks_usd_s) / (bids_usd_s + asks_usd_s + 1e-12)

            oi_close_scale_s = (
                (oi_close_s / oi_close_s.rolling(20, min_periods=5).mean().replace(0.0, np.nan).fillna(oi_close_s)) - 1.0
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-2.0, 2.0)
            funding_close_scale_s = funding_close_s.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.01, 0.01) * 500.0

            base_feats: dict[str, pd.Series] = {
                "ret_1d_scale": ret_1d_scale_s,
                "ret_3d_scale": ret_3d_scale_s,
                "ret_3d_z": _clip(_rolling_zscore(ret_3d_s, z_window_1d, minp)),
                "volume_log_z": _clip(_rolling_zscore(vol_log_s, z_window_1d, minp)),
                "rsi_14": rsi_14_s.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0),
                "body_range_ratio": body_range_ratio_s,
                "volume_impact_scale": volume_impact_scale_s,
                "ema_12_48_spread_raw": ema_12_48_spread_raw_s,
                "up_volume_ratio_12": up_volume_ratio_12_s,
                "dist_to_long_liq_zone_atr": dist_to_long_liq_zone_atr_1d_s,
                "dist_to_short_liq_zone_atr": dist_to_short_liq_zone_atr_1d_s,
                "liq_sweep_down": liq_sweep_down_1d_s,
                "oi_close_scale": oi_close_scale_s,
                "funding_close_scale": funding_close_scale_s,
                "ls_account_ratio_z": _clip(_rolling_zscore(ls_account_ratio_s, z_window_1d, minp)),
                "ob_imbalance_z": _clip(_rolling_zscore(ob_imbalance_s, z_window_1d, minp)),
                "liq_long_log_z": _clip(_rolling_zscore(np.log1p(np.clip(liq_long_s, 0.0, None)), z_window_1d, minp)),
                "liq_short_log_z": _clip(_rolling_zscore(np.log1p(np.clip(liq_short_s, 0.0, None)), z_window_1d, minp)),
            }

            others_raw_fields = {
                "open": os_1d,
                "high": high_1d_s,
                "low": low_1d_s,
                "close": cs_1d,
                "volume": vol_1d_s,
                "buy_volume": buy_vol_1d_s,
                "sell_volume": sell_vol_1d_s,
                "volume_ratio": volume_ratio_1d_s,
                "long_short_ratio": long_short_ratio_1d_s,
                "trades": trades_1d_s,
                "quote_volume": quote_vol_1d_s,
                "oi_close": oi_close_s,
                "funding_close": funding_close_s,
                "liq_long": liq_long_s,
                "liq_short": liq_short_s,
                "ob_imbalance": ob_imbalance_s,
            }
            for raw_name, raw_series in others_raw_fields.items():
                base_feats.update(
                    _heavy_field_features(
                        raw_series,
                        feature_name=raw_name,
                        z_window=z_window_1d,
                        z_window_short=z_window_1d_short,
                        minp=minp,
                        minp_short=minp_short,
                    )
                )

            if selected_others_cols_per_symbol is None:
                if configured_others_cols_per_symbol:
                    missing_others = [c for c in configured_others_cols_per_symbol if c not in base_feats]
                    if missing_others:
                        raise ValueError(f"OBS_PRICE_SEQ_1D_OTHERS_COLS(per-symbol) 含未知欄位: {missing_others[:8]}")
                    selected_others_cols_per_symbol = configured_others_cols_per_symbol
                else:
                    selected_others_cols_per_symbol = list(base_feats.keys())

            sym_feats = pd.DataFrame(
                {f"{sym}_{col}": base_feats[col] for col in selected_others_cols_per_symbol},
                index=df_1d.index,
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            others_feats_list.append(sym_feats)
            others_cols.extend([f"{sym}_{col}" for col in selected_others_cols_per_symbol])

        # Macro 指标：空配置時採用全量
        fear_greed = _get_col(df_1d, "fear_greed_fear_greed_index")
        altcoin_season = _get_col(df_1d, "altcoin_season_altcoin_index")
        bmo_value = _get_col(df_1d, "bitcoin_macro_oscillator_bmo_value")
        sopr_value = _get_col(df_1d, "bitcoin_sth_sopr_lth_sopr")
        fear_greed_scale = fear_greed.replace([np.inf, -np.inf], np.nan).fillna(50.0).clip(0.0, 100.0) / 50.0 - 1.0

        macro_feats = pd.DataFrame(
            {
                "fear_greed_scale": fear_greed_scale,
                "fear_greed_z": _clip(_rolling_zscore(fear_greed, z_window_1d, minp)),
                "altcoin_season_z": _clip(_rolling_zscore(altcoin_season, z_window_1d, minp)),
                "bmo_z": _clip(_rolling_zscore(bmo_value, z_window_1d, minp)),
                "sopr_z": _clip(_rolling_zscore(sopr_value, z_window_1d, minp)),
            },
            index=df_1d.index,
        )
        macro_heavy: dict[str, pd.Series] = {}
        for raw_name, raw_series in {
            "fear_greed": fear_greed,
            "altcoin_season": altcoin_season,
            "bmo": bmo_value,
            "sopr": sopr_value,
        }.items():
            macro_heavy.update(
                _heavy_field_features(
                    raw_series,
                    feature_name=raw_name,
                    z_window=z_window_1d,
                    z_window_short=z_window_1d_short,
                    minp=minp,
                    minp_short=minp_short,
                )
            )
        macro_feats = pd.concat([macro_feats, pd.DataFrame(macro_heavy, index=df_1d.index)], axis=1)
        others_feats_list.append(macro_feats)

        configured_macro_cols = list(self.PRICE_SEQ_1D_MACRO_COLS)
        if configured_macro_cols:
            missing_macro = [c for c in configured_macro_cols if c not in macro_feats.columns]
            if missing_macro:
                raise ValueError(f"OBS_PRICE_SEQ_1D_OTHERS_COLS(macro) 含未知欄位: {missing_macro[:8]}")
            selected_macro_cols = configured_macro_cols
        else:
            selected_macro_cols = list(macro_feats.columns)
        others_cols.extend(selected_macro_cols)

        if others_feats_list:
            others_feats = pd.concat(others_feats_list, axis=1, copy=False)
            others_feats = others_feats.reindex(columns=others_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            others_feats = pd.DataFrame(index=df_1d.index, columns=others_cols).fillna(0.0)

        return (
            target_feats.astype(np.float32).to_numpy(copy=True),
            others_feats.astype(np.float32).to_numpy(copy=True),
            target_cols,
            others_cols,
        )


