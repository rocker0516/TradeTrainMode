"""LiveTradingRunner.render_history_utils：GATE regime 與 conviction 公式。"""

from __future__ import annotations

import numpy as np

from LiveTradingRunner.render_history_utils import (
    atr_ratio_from_ohlc,
    build_sideway_mask_from_series,
    conviction_strength_from_trend_score,
    gate_flags_to_regime_indicator,
    trend_tanh_signed_from_trend_score,
)


def test_gate_flags_to_regime_indicator_bull_bear_neutral() -> None:
    assert gate_flags_to_regime_indicator(np.array([1.0, 0.0, 0.0])) == 1.0
    assert gate_flags_to_regime_indicator(np.array([0.0, 1.0, -1.0])) == -1.0
    assert gate_flags_to_regime_indicator(np.array([0.0, 1.0, 0.0])) == 0.0


def test_conviction_strength_from_trend_score_clip_and_tanh() -> None:
    s = conviction_strength_from_trend_score(0.5, scale=10.0)
    assert 0.0 <= s <= 1.0
    assert abs(s - float(abs(np.tanh(5.0)))) < 1e-9


def test_trend_tanh_signed_matches_abs_of_conviction() -> None:
    ts = 0.03
    sc = 10.0
    signed = trend_tanh_signed_from_trend_score(ts, scale=sc)
    strength = conviction_strength_from_trend_score(ts, scale=sc)
    assert abs(abs(signed) - strength) < 1e-12


def test_atr_ratio_from_ohlc_returns_finite_series() -> None:
    high = [101.0, 102.0, 103.0, 103.5, 104.0]
    low = [99.0, 100.0, 101.0, 102.0, 102.5]
    close = [100.0, 101.0, 102.0, 103.0, 103.2]
    out = atr_ratio_from_ohlc(high, low, close)
    assert len(out) == len(close)
    assert all(np.isfinite(v) and v >= 0.0 for v in out)


def test_build_sideway_mask_from_series_marks_flat_regime() -> None:
    n = 80
    close = [100.0 + 0.01 * np.sin(i / 3.0) for i in range(n)]
    atr_ratio = [0.002] * n
    mask = build_sideway_mask_from_series(close, atr_ratio, lookback_l=24)
    assert len(mask) == n
    # 前段因 rolling 不足可為 False；後段應有盤整判定成立
    assert any(mask[30:])


def test_build_sideway_mask_from_series_rejects_trending_regime() -> None:
    n = 80
    close = [100.0 + 0.8 * i for i in range(n)]
    atr_ratio = [0.002] * n
    mask = build_sideway_mask_from_series(close, atr_ratio, lookback_l=24)
    assert len(mask) == n
    # 明顯單向趨勢時，後段盤整比例應很低
    assert sum(1 for x in mask[30:] if x) <= 5
