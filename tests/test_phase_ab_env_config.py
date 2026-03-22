"""
契約測試：Phase AB `get_phase_ab_env_kwargs` 回傳鍵完整、train/eval 行為與常數一致。
"""

from __future__ import annotations

import sys

_PROJECT_ROOT = __import__("os").path.abspath(
    __import__("os").path.join(__import__("os").path.dirname(__file__), "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Eval.phase_ab_env_config import PhaseABEnvConfig  # noqa: E402
from Train.run_sac_phase_ab import get_phase_ab_env_kwargs  # noqa: E402

_PHASE_AB_KWARGS_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "env_id",
        "random_start",
        "window_size",
        "window_size_1d",
        "max_episode_steps",
        "target_symbol",
        "feature_symbols",
        "data_split_enabled",
        "holdout_months",
        "data_mode",
        "no_trade_entry_threshold",
        "no_trade_exit_threshold",
        "max_step_pos_change_pct",
        "min_position_change",
        "trade_freq_window_steps",
        "trade_freq_cost_limit",
        "cost_fric_scale",
        "regime_alignment_bonus_weight",
        "conviction_trend_bonus_weight",
        "neutral_trade_penalty_weight",
        "conviction_min_abs_pos",
        "conviction_trend_min_strength",
    }
)


def _kwargs_train_eval() -> tuple[dict, dict]:
    common = dict(
        regime_bonus_weight=0.01,
        conviction_bonus_weight=0.02,
        conviction_min_abs_pos=0.3,
        conviction_trend_min_strength=0.4,
        data_split_enabled=True,
        holdout_months=2,
        cost_fric_scale=123.0,
        max_episode_steps=None,
    )
    train_kw = get_phase_ab_env_kwargs(data_mode="train", **common)
    eval_kw = get_phase_ab_env_kwargs(data_mode="eval", **common)
    return train_kw, eval_kw


def test_phase_ab_env_kwargs_train_has_required_keys() -> None:
    train_kw, _ = _kwargs_train_eval()
    assert _PHASE_AB_KWARGS_REQUIRED_KEYS <= train_kw.keys()
    assert "min_episode_steps" not in train_kw


def test_phase_ab_env_kwargs_eval_has_required_keys_and_min_steps() -> None:
    _, eval_kw = _kwargs_train_eval()
    assert _PHASE_AB_KWARGS_REQUIRED_KEYS <= eval_kw.keys()
    assert eval_kw["min_episode_steps"] == int(PhaseABEnvConfig.EVAL_MIN_EPISODE_STEPS)


def test_phase_ab_numeric_anchors() -> None:
    train_kw, eval_kw = _kwargs_train_eval()
    assert train_kw["window_size"] == int(PhaseABEnvConfig.WINDOW_SIZE)
    assert train_kw["no_trade_entry_threshold"] == float(PhaseABEnvConfig.NO_TRADE_ENTRY_THRESHOLD)
    assert train_kw["max_episode_steps"] == int(PhaseABEnvConfig.DEFAULT_MAX_EPISODE_STEPS)
    assert eval_kw["max_episode_steps"] == int(PhaseABEnvConfig.DEFAULT_MAX_EPISODE_STEPS)


def test_phase_ab_env_config_fields_mirror_documented_recipe() -> None:
    assert PhaseABEnvConfig.WINDOW_SIZE == 14
    assert PhaseABEnvConfig.WINDOW_SIZE_1D == 12
    assert PhaseABEnvConfig.NO_TRADE_ENTRY_THRESHOLD == 0.2
    assert PhaseABEnvConfig.MAX_STEP_POS_CHANGE_PCT == 0.4
    assert PhaseABEnvConfig.DEFAULT_MAX_EPISODE_STEPS == 288 * 31
    assert PhaseABEnvConfig.NEUTRAL_TRADE_PENALTY_WEIGHT == 0.0
