"""輕量驗證 Phase A/B 訓練腳本組出的 env kwargs 是否帶入 TradingEnvironment 所需參數。"""

from __future__ import annotations

from Train.run_sac_phase_ab import get_phase_ab_env_kwargs


def test_get_phase_ab_env_kwargs_passes_turnover_anchor_params() -> None:
    """rolling turnover anchor 參數應原樣進入 env_kwargs。"""
    kw = get_phase_ab_env_kwargs(
        turnover_anchor_update_steps=144,
        turnover_anchor_source="equity",
    )
    assert kw["turnover_anchor_update_steps"] == 144
    assert kw["turnover_anchor_source"] == "equity"
