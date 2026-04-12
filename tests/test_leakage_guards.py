from __future__ import annotations

import numpy as np
import pandas as pd

from Env.Components.market_data import _past_only_long_std as market_long_std
from Env.Components.market_data import _past_only_rolling_quantile
from Env.feature_transformer import (
    _past_only_long_std as feature_long_std,
    _past_only_reference_level,
    _prev_or_current,
)


def test_prev_or_current_does_not_pull_future_value() -> None:
    series = pd.Series([10.0, 11.0, 12.0], dtype=float)

    prev = _prev_or_current(series)

    assert prev.tolist() == [10.0, 10.0, 11.0]


def test_past_only_reference_level_uses_expanding_fallback_before_first_event() -> None:
    price = pd.Series([5.0, 4.0, 6.0, 3.0], dtype=float)
    event_mask = pd.Series([False, False, True, False])

    recent_low = _past_only_reference_level(price, event_mask, mode="low")

    assert recent_low.tolist() == [5.0, 4.0, 6.0, 6.0]


def test_past_only_long_std_does_not_require_future_fill() -> None:
    series = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0], dtype=float)

    ft_std = feature_long_std(series, window=10, min_periods=3)
    md_std = market_long_std(series, window=10, min_periods=3)

    assert np.isfinite(ft_std).all()
    assert np.isfinite(md_std).all()
    assert float(ft_std.iloc[0]) > 0.0 or float(ft_std.iloc[0]) == 1e-8
    assert float(md_std.iloc[0]) > 0.0 or float(md_std.iloc[0]) == 1e-8


def test_past_only_rolling_quantile_is_shifted_from_history() -> None:
    arr = np.array([1.0, 10.0, 2.0, 20.0], dtype=np.float64)

    threshold = _past_only_rolling_quantile(arr, quantile=0.5, window=3, min_periods=2)

    assert threshold[0] == arr[0]
    assert threshold[1] == arr[1]
    assert threshold[2] <= 10.0
    assert threshold[3] <= 10.0
