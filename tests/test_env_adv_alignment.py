from __future__ import annotations

import math

import numpy as np

from Env.trading_env import TradingEnvironment


def test_adv_notional_rebuilds_after_eval_split(patch_env_load_data, make_synth_market) -> None:
    market = make_synth_market(
        n_5m=20_000,
        n_1d=120,
        start_price=100.0,
        step_5m=0.0,
        step_1d=0.0,
        spread_5m=1.0,
        spread_1d=2.0,
    )
    market.df_5m.loc[:, "volume"] = np.linspace(100.0, 20_000.0, len(market.df_5m), dtype=np.float64)
    patch_env_load_data(market=market)

    env = TradingEnvironment(
        env_id=0,
        random_start=False,
        window_size=10,
        window_size_1d=7,
        min_episode_steps=30,
        max_episode_steps=100,
        data_split_enabled=True,
        holdout_months=1,
        data_mode="eval",
        adv_lookback_days=1,
    )

    first_eval_row = env.market_data.df_5m.iloc[0]
    expected_first_adv = float(first_eval_row["close"]) * float(first_eval_row["volume"])

    assert len(env._adv_notional_arr) == len(env.market_data.df_5m)
    assert len(env.market_data.df_5m) < len(market.df_5m)
    assert math.isclose(float(env._adv_notional_arr[0]), expected_first_adv, rel_tol=1e-9)
