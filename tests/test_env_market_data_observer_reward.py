from __future__ import annotations

import numpy as np

from Env.Components.market_data import MarketData
from Env.Components.observer import TradingObserver
from Env.Executors.trade_executor import TradeExecutor
from Env.Rewards.reward import RewardCalculator


def test_market_data_sequences_have_expected_shapes(make_synth_market) -> None:
    market = make_synth_market(n_5m=120, n_1d=50)
    md = MarketData(market.df_5m, market.df_1d, window_size=10, window_size_1d=7, target_symbol="BTCUSDT")

    seq_5m_target, seq_5m_others = md.get_price_seq(10)
    seq_1d_target, seq_1d_others = md.get_1d_seq(10, window_size_1d=7)
    assert seq_5m_target.shape == (10, md.price_seq_target_features_dim)
    assert seq_1d_target.shape == (7, md.price_seq_1d_target_features_dim)
    # MarketData 內部特徵矩陣用 float32（計算穩定）；env 輸出 obs 可能轉成 float16 以省 RAM
    assert seq_5m_target.dtype == np.float32
    assert seq_1d_target.dtype == np.float32

    m = md.get_market_metrics(10)
    assert set(m.keys()) >= {"close", "high", "low", "atr_ratio", "rv_ratio", "trend_score"}
    assert m["close"] > 0.0
    assert m["atr_ratio"] >= 0.0


def test_observer_risk_signals_are_zero_when_flat(make_synth_market) -> None:
    market = make_synth_market(n_5m=120, n_1d=50)
    md = MarketData(market.df_5m, market.df_1d, window_size=10, window_size_1d=7, target_symbol="BTCUSDT")
    obs = TradingObserver(window_size=10, window_size_1d=7, market_data=md)

    ex = TradeExecutor(
        initial_balance=1000.0,
        fee_rate=0.0,
        leverage=10.0,
        min_trade_qty=0.001,
        maintenance_margin_rate=0.005,
        margin_mode="isolated",
        stop_loss_atr=2.0,
        stop_loss_liq_buffer_pct=0.0,
        min_position_change=0.0,
    )

    price = md.get_market_metrics(10)["close"]
    atr_est = float(md.get_market_metrics(10)["atr_ratio"]) * float(price)
    risk = obs.compute_risk_signals(ex, float(price), atr_est, 10, len(md.df_5m))
    assert risk["liq_price"] == 0.0
    assert risk["near_liq"] is False
    assert risk["stop_loss_missing"] == 0.0


def test_observer_observation_shapes(make_synth_market) -> None:
    market = make_synth_market(n_5m=120, n_1d=50)
    md = MarketData(market.df_5m, market.df_1d, window_size=10, window_size_1d=7, target_symbol="BTCUSDT")
    # Observer 預設使用 Config.OBS_DTYPE（現為 float16）
    obs = TradingObserver(window_size=10, window_size_1d=7, market_data=md)

    ex = TradeExecutor(
        initial_balance=1000.0,
        fee_rate=0.0,
        leverage=10.0,
        min_trade_qty=0.001,
        maintenance_margin_rate=0.005,
        margin_mode="isolated",
        stop_loss_atr=2.0,
        stop_loss_liq_buffer_pct=0.0,
        min_position_change=0.0,
    )

    step_idx = 10
    m = md.get_market_metrics(step_idx)
    price = float(m["close"])
    atr_est = float(m["atr_ratio"]) * float(price)
    risk = obs.compute_risk_signals(ex, price, atr_est, step_idx, len(md.df_5m))

    account_metrics = {
        "initial_balance": 1000.0,
        "max_equity_so_far": 1000.0,
        "episode_stop_loss_count": 0,
        "episode_liq_count": 0,
        "risk_budget": 1.0,
        "steps_since_trade": 10.0,
        "holding_steps": 0.0,
        "last_step_fee": 0.0,
        "rolling_fee_sum": 0.0,
        "recent_flat_ratio": 0.5,
    }
    last_action_effects = {
        "expected_fee_if_trade": 0.0,
        "predicted_used_margin_after_action": 0.0,
        "predicted_available_balance_after_action": 0.0,
        "predicted_liq_distance_after_action": 0.0,
        "predicted_stop_distance_after_action": 0.0,
        # action vs execution discrepancy (new)
        "cooldown_remaining_norm": 0.0,
        "action_overridden_flag": 0.0,
        "last_action_raw": 0.0,
        "last_action_used": 0.0,
        "last_target_pos_pct": 0.0,
        "last_final_pos_pct": 0.0,
        "trade_executed_flag": 0.0,
    }

    out = obs.get_observation(
        step_idx=step_idx,
        executor=ex,
        market_data=md,
        account_metrics=account_metrics,
        risk_signals=risk,
        last_action_effects=last_action_effects,
    )

    assert out["price_seq_target"].shape == (10, md.price_seq_target_features_dim)
    assert out["price_seq_1d_target"].shape == (7, md.price_seq_1d_target_features_dim)
    assert out["account_state"].shape == (23,)
    # dtype 必須與 observation_space 一致（預設 float16）
    assert out["price_seq_target"].dtype == obs.observation_space["price_seq_target"].dtype
    assert out["account_state"].dtype == obs.observation_space["account_state"].dtype


def test_reward_calculator_log_return_only() -> None:
    rc = RewardCalculator(base_log_ret_weight=1.0)
    r = rc.compute(last_equity=1000.0, new_equity=1100.0)
    assert r == np.log(1100.0 / 1000.0)


