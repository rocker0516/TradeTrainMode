import os
import sys

import pytest

# Add project root to sys.path so `Env.*` imports work under pytest on Windows
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from Env.reward import create_default_calculator


def test_conviction_bonus_only_when_strong_signal_and_large_aligned_position():
    calc = create_default_calculator(
        base_log_ret_weight=1.0,
        conviction_trend_bonus_weight=1.0,
        conviction_trend_min_strength=0.8,
        conviction_min_abs_pos=0.15,
    )

    # base log return (1.0 -> 1.01) > 0
    base = calc.compute(last_equity=1.0, new_equity=1.01, position_pct=0.0, abs_position_pct=0.0, trend_score=0.0)

    # Strong positive trend (tanh(5) ~ 0.9999), large aligned long position -> bonus should increase reward
    shaped = calc.compute(last_equity=1.0, new_equity=1.01, position_pct=0.50, abs_position_pct=0.50, trend_score=5.0)
    assert shaped > base


def test_no_bonus_for_small_position_even_if_signal_strong():
    calc = create_default_calculator(
        base_log_ret_weight=1.0,
        conviction_trend_bonus_weight=1.0,
        conviction_trend_min_strength=0.8,
        conviction_min_abs_pos=0.15,
    )

    base = calc.compute(last_equity=1.0, new_equity=1.01, trend_score=5.0, position_pct=0.0, abs_position_pct=0.0)
    # Small position below p0 => no bonus
    small = calc.compute(last_equity=1.0, new_equity=1.01, trend_score=5.0, position_pct=0.05, abs_position_pct=0.05)
    assert small == pytest.approx(base)


def test_no_bonus_for_misaligned_position():
    calc = create_default_calculator(
        base_log_ret_weight=1.0,
        conviction_trend_bonus_weight=1.0,
        conviction_trend_min_strength=0.8,
        conviction_min_abs_pos=0.15,
    )

    base = calc.compute(last_equity=1.0, new_equity=1.01, trend_score=5.0, position_pct=0.0, abs_position_pct=0.0)
    # Strong positive trend but short position => misaligned => no bonus
    misaligned = calc.compute(last_equity=1.0, new_equity=1.01, trend_score=5.0, position_pct=-0.50, abs_position_pct=0.50)
    assert misaligned == pytest.approx(base)


def test_no_bonus_if_signal_weak_even_with_large_position():
    calc = create_default_calculator(
        base_log_ret_weight=1.0,
        conviction_trend_bonus_weight=1.0,
        conviction_trend_min_strength=0.8,
        conviction_min_abs_pos=0.15,
    )

    base = calc.compute(last_equity=1.0, new_equity=1.01, trend_score=0.0, position_pct=0.0, abs_position_pct=0.0)
    weak = calc.compute(last_equity=1.0, new_equity=1.01, trend_score=0.1, position_pct=0.80, abs_position_pct=0.80)
    assert weak == pytest.approx(base)


