from __future__ import annotations

import pandas as pd

from Eval.search_risk_buffer import _is_success, _risk_buffer_grid, _score_row


def test_risk_buffer_grid_default_size() -> None:
    g = _risk_buffer_grid(full=False)
    assert len(g) == 16


def test_risk_buffer_grid_full_size() -> None:
    g = _risk_buffer_grid(full=True)
    assert len(g) == 36


def test_is_success_all_met() -> None:
    row = pd.Series(
        {
            "profit_mean": 1.0,
            "profit_per_trade_mean": 0.5,
            "fees_profit_ratio": 0.5,
            "episode_max_dd_mean": 0.1,
            "trade_count_mean": 20.0,
            "termination_balance_insufficient_count": 0.0,
            "termination_liq_count": 0.0,
        }
    )
    assert _is_success(row, max_dd=0.35, max_fees_profit_ratio=1.0, min_eval_trades=12.0) is True


def test_is_success_fails_on_death() -> None:
    row = pd.Series(
        {
            "profit_mean": 1.0,
            "profit_per_trade_mean": 0.5,
            "fees_profit_ratio": 0.5,
            "episode_max_dd_mean": 0.1,
            "trade_count_mean": 20.0,
            "termination_balance_insufficient_count": 2.0,
            "termination_liq_count": 0.0,
        }
    )
    assert _is_success(row, max_dd=0.35, max_fees_profit_ratio=1.0, min_eval_trades=12.0) is False


def test_score_row_ordering() -> None:
    good = pd.Series(
        {
            "profit_mean": 2.0,
            "profit_per_trade_mean": 1.0,
            "fees_profit_ratio": 0.2,
            "episode_max_dd_mean": 0.1,
            "trade_count_mean": 20.0,
            "termination_balance_insufficient_count": 0.0,
            "termination_liq_count": 0.0,
        }
    )
    bad = pd.Series(
        {
            "profit_mean": -1.0,
            "profit_per_trade_mean": -0.5,
            "fees_profit_ratio": 2.0,
            "episode_max_dd_mean": 0.9,
            "trade_count_mean": 20.0,
            "termination_balance_insufficient_count": 5.0,
            "termination_liq_count": 1.0,
        }
    )
    assert _score_row(good, max_dd=0.35, max_fees_profit_ratio=1.0, min_eval_trades=12.0) < _score_row(
        bad, max_dd=0.35, max_fees_profit_ratio=1.0, min_eval_trades=12.0
    )
