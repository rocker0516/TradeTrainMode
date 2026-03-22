"""LiveTradingRunner.render_history_utils：GATE regime 與 conviction 公式。"""

from __future__ import annotations

import numpy as np

from LiveTradingRunner.render_history_utils import (
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
