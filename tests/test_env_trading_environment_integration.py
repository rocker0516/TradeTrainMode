from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pytest

from Env.trading_env import TradingEnvironment


@dataclass(frozen=True)
class Scenario:
    """整合測試情境：用固定行情與 action 序列驗證交易環境行為。"""

    name: str
    # 在 reset 後，依序 step 的 actions
    actions: List[float]
    window_size: int = 10
    # 若需要：在建立 env 前對 df_5m 做微調（用來觸發 SL/LIQ/資料結束）
    tweak_5m: Optional[Callable[[object], None]] = None
    # 結束後檢查條件
    expect: Dict[str, object] = None


def _mk_env(patch_env_load_data, *, window_size: int = 10, window_size_1d: int = 7, **kwargs) -> TradingEnvironment:
    return TradingEnvironment(
        env_id=0,
        random_start=False,
        window_size=window_size,
        window_size_1d=window_size_1d,
        min_episode_steps=30,
        # 若測試想覆寫 max_episode_steps，讓 kwargs 覆寫即可
        max_episode_steps=int(kwargs.pop("max_episode_steps", 1000)),
        **kwargs,
    )


def _run_actions(env: TradingEnvironment, actions: List[float]):
    obs, _ = env.reset(seed=1)
    last = None
    for a in actions:
        last = env.step(np.array([a], dtype=np.float32))
        # Gymnasium 使用方式：terminated/truncated 後不應再 step。
        if bool(last[2] or last[3]):
            break
    assert last is not None
    return last


def _scenarios() -> List[Scenario]:
    # reset 後 current_step = window_size(=10)，因此第 1 次 step 用 idx=10，下一次 idx=11 ...
    return [
        Scenario(
            name="open_long_and_hold",
            actions=[1.0, 1.0],
            expect={"pos_sign": 1, "done": False},
        ),
        Scenario(
            name="open_short_and_hold",
            actions=[-1.0, -1.0],
            expect={"pos_sign": -1, "done": False},
        ),
        Scenario(
            name="open_long_then_reduce",
            actions=[1.0, 0.5],
            expect={"pos_nonzero": True, "done": False},
        ),
        Scenario(
            name="open_long_then_close",
            actions=[1.0, 0.0],
            expect={"pos_zero": True, "done": False},
        ),
        Scenario(
            name="flip_allowed",
            actions=[1.0, -1.0],
            # Flip budget 機制移除後：至少應標記 is_flip，且不應因 budget 強制阻擋。
            #（此情境會另外在 env_kwargs 把 stop_loss_atr=0，避免止損干擾）
            # 注意：由於 ActionProcessor 的 max_step_pos_change_pct 限制，flip 可能會先「縮倉到 0」，
            # 需要下一步才會真正開到反向倉位；因此這裡只驗證會平倉而不是被 budget 阻擋。
            expect={"is_flip": True, "pos_zero": True, "done": False},
        ),
        Scenario(
            name="max_step_change_limits_build_up",
            actions=[1.0],
            expect={"pos_less_than_full": True},
        ),
        Scenario(
            name="min_position_change_deadband_skips_tiny_trade",
            actions=[0.1],
            expect={"pos_zero": True, "fee_zero": True},
        ),
        Scenario(
            name="stop_loss_triggers",
            actions=[1.0, 1.0],
            window_size=20,
            expect={"stop_loss_triggered": True, "terminated": False},
        ),
        Scenario(
            name="stop_loss_cooldown_forces_no_trade_next_step",
            actions=[1.0, 1.0, 1.0],
            window_size=20,
            expect={"stop_loss_triggered": True, "cooldown_forced_flat": True},
        ),
        Scenario(
            name="liquidation_triggers_terminated",
            actions=[1.0, 1.0],
            window_size=20,
            expect={"liq_triggered": True, "terminated": True, "termination_reason": "liq_triggered"},
        ),
        Scenario(
            name="balance_insufficient_terminated",
            actions=[1.0],
            expect={"terminated": True, "termination_reason": "balance_insufficient"},
        ),
        Scenario(
            name="data_exhausted_truncated",
            actions=[0.0, 0.0],
            # 此 env 設計 episode_max_steps 由剩餘資料推導，資料走完時通常也同時 max_steps_reached。
            expect={"truncated": True, "termination_reason": "max_steps_reached"},
        ),
        Scenario(
            name="max_steps_reached_truncated",
            actions=[0.0, 0.0, 0.0],
            expect={"truncated": True, "termination_reason": "max_steps_reached"},
        ),
        Scenario(
            name="fee_tracking_updates_last_step_fee",
            actions=[1.0, 0.0],
            expect={"last_step_fee_nonneg": True, "rolling_fee_nonneg": True},
        ),
        Scenario(
            name="action_effects_cached_in_next_observation",
            actions=[1.0, 1.0],
            expect={"next_obs_expected_fee_positive": True},
        ),
        Scenario(
            name="reward_positive_when_next_close_up",
            actions=[1.0],
            expect={"reward_positive": True},
        ),
        Scenario(
            name="risk_signals_zero_when_flat",
            actions=[0.0],
            expect={"risk_liq_zero": True},
        ),
        Scenario(
            name="info_contains_required_keys",
            actions=[0.0],
            expect={"info_keys": True},
        ),
        Scenario(
            name="cost_stop_missing_when_stop_disabled",
            actions=[1.0],
            expect={"cost_stop_missing": True},
        ),
        Scenario(
            name="mark_to_market_uses_next_close",
            actions=[1.0],
            expect={"equity_matches_next_close": True},
        ),
    ]


@pytest.mark.parametrize("sc", _scenarios(), ids=lambda s: s.name)
def test_trading_environment_integration_scenarios(sc: Scenario, patch_env_load_data, make_synth_market, monkeypatch) -> None:
    # 預設行情：close=100，high=101，low=99，ATR≈2（足以穩定產生 stop distance）
    market = make_synth_market(n_5m=120, n_1d=60, start_price=100.0, step_5m=0.0, spread_5m=1.0)

    # 情境調整（用 index 11 觸發 stop/liq 等）
    trigger_idx = int(sc.window_size + 1)
    if sc.name == "stop_loss_triggers" or sc.name == "stop_loss_cooldown_forces_no_trade_next_step":
        # entry=100, atr≈2, stop_loss_atr=2 => stop≈96，下一根 low 觸發
        market.df_5m.loc[trigger_idx, "low"] = 50.0
    if sc.name == "liquidation_triggers_terminated":
        # 先關閉止損，避免「止損優先於強平」讓 liq 永遠觸發不到
        market.df_5m.loc[trigger_idx, "low"] = 1.0
    if sc.name == "reward_positive_when_next_close_up" or sc.name == "mark_to_market_uses_next_close":
        # step=10 後 mark-to-market 用 next close(idx=11)，提高 next close
        market.df_5m.loc[trigger_idx, "close"] = 110.0
        market.df_5m.loc[trigger_idx, "high"] = 111.0
        market.df_5m.loc[trigger_idx, "low"] = 109.0
    if sc.name == "data_exhausted_truncated":
        market = make_synth_market(n_5m=13, n_1d=60, start_price=100.0, step_5m=0.0, spread_5m=1.0)
    if sc.name == "balance_insufficient_terminated":
        # next close 下跌造成 equity <= min_balance
        market.df_5m.loc[11, "close"] = 50.0
        market.df_5m.loc[11, "high"] = 51.0
        market.df_5m.loc[11, "low"] = 49.0

    patch_env_load_data(market=market)

    # 依情境調整環境參數
    env_kwargs = {}
    if sc.name == "max_step_change_limits_build_up":
        env_kwargs["max_step_pos_change_pct"] = 0.05
    if sc.name == "min_position_change_deadband_skips_tiny_trade":
        env_kwargs["min_position_change"] = 0.2
    if sc.name == "flip_allowed":
        # 避免止損/追蹤止損干擾 flip 行為測試
        env_kwargs["stop_loss_atr"] = 0.0
    if sc.name == "stop_loss_triggers" or sc.name == "stop_loss_cooldown_forces_no_trade_next_step":
        env_kwargs["stop_loss_atr"] = 2.0
    # 注意：TradingEnvironment.__init__ 目前不吃 kwargs 的 max_episode_steps（固定取 Config.MAX_EPISODE_STEPS）。
    # 所以這個情境用「建立後覆寫 env.max_episode_steps」來觸發 truncated。
    if sc.name == "balance_insufficient_terminated":
        env_kwargs["min_balance"] = 800.0  # 高門檻，讓小虧也能終止
    if sc.name == "liquidation_triggers_terminated":
        env_kwargs["stop_loss_atr"] = 0.0
    if sc.name == "cost_stop_missing_when_stop_disabled":
        # 關閉止損：讓環境回報 stop_loss_missing cost（有倉但沒有 stop）
        env_kwargs["stop_loss_atr"] = 0.0

    # stop loss cooldown 需要 Config 常數
    if sc.name == "stop_loss_cooldown_forces_no_trade_next_step":
        from Env import config as env_config

        monkeypatch.setattr(env_config.Config, "STOP_LOSS_COOLDOWN_STEPS", 2, raising=True)

    env = _mk_env(patch_env_load_data, window_size=sc.window_size, **env_kwargs)
    if sc.name == "max_steps_reached_truncated":
        env.max_episode_steps = 2
    obs0, _ = env.reset(seed=1)

    obs, reward, terminated, truncated, info = _run_actions(env, sc.actions)

    done = bool(terminated or truncated)

    # --- episode summary fields when done ---
    if done:
        # Env 只在回合結束時提供 episode_max_dd
        assert "episode_max_dd" in info
        assert 0.0 <= float(info["episode_max_dd"]) <= 1.0
        # Overtrading / fee diagnostics
        assert "episode_turnover_notional" in info
        assert float(info["episode_turnover_notional"]) >= 0.0
        assert "episode_holding_steps" in info
        assert int(info["episode_holding_steps"]) >= 0
        assert "episode_trade_count" in info
        assert int(info["episode_trade_count"]) >= 0
        assert "fees_to_equity_ratio" in info
        assert float(info["fees_to_equity_ratio"]) >= 0.0

    # --- common required info keys ---
    if sc.expect.get("info_keys"):
        for k in (
            "equity",
            "profit",
            "stop_loss_triggered",
            "liq_triggered",
            "step_fee_ratio",
            "is_flip",
            "risk_budget",
            "current_dd",
            # cost / breakdown（供 Lagrangian 或解析使用）
            "cost",
            # multi-lambda channels（訓練端會用到）
            "cost_risk",
            "cost_fric",
            "cost_sl_buf",
            "cost_breakdown",
        ):
            assert k in info
        assert isinstance(info["cost_breakdown"], dict)
        # 核心分項 key
        for k in ("death_cost", "fric_cost", "sl_buf_cost", "stop_missing_cost"):
            assert k in info["cost_breakdown"]

    # --- position checks ---
    pos = float(env.executor.position.size)
    if sc.expect.get("pos_sign") == 1:
        assert pos > 0.0
    if sc.expect.get("pos_sign") == -1:
        assert pos < 0.0
    if sc.expect.get("pos_nonzero"):
        assert abs(pos) > 0.0
    if sc.expect.get("pos_zero"):
        assert abs(pos) <= 1e-12
    if sc.expect.get("pos_less_than_full"):
        # 理論滿倉 size ≈ base*lev/price = 1000*10/100=100
        assert 0.0 < abs(pos) < 100.0

    # --- done checks ---
    if "done" in sc.expect:
        assert done is bool(sc.expect["done"])

    # --- flip / budget checks ---
    if sc.expect.get("is_flip"):
        assert info.get("is_flip", False) is True
    # Flip budget 機制已移除：不再檢查 risk_budget 變動

    # --- fees checks ---
    if sc.expect.get("fee_zero"):
        assert env.executor.total_fees == 0.0
    if sc.expect.get("last_step_fee_nonneg"):
        assert env.tracker.last_step_fee >= 0.0
    if sc.expect.get("rolling_fee_nonneg"):
        assert env.tracker.rolling_fee_sum >= 0.0

    # --- stop loss / cooldown ---
    if sc.expect.get("stop_loss_triggered"):
        if sc.name == "stop_loss_cooldown_forces_no_trade_next_step":
            # 這個情境最後一步是 cooldown 強制 action=0 的那一步，info 可能不再標 stop_loss_triggered；
            # 用 episode counter 檢查更符合「回合中曾發生過止損」的語意。
            assert int(env.episode_stop_loss_count) >= 1
        else:
            assert info.get("stop_loss_triggered", False) is True
    if sc.expect.get("cooldown_forced_flat"):
        # cooldown 觸發後，下一步 action 會被強制成 0，所以最後仍應保持空倉
        assert abs(env.executor.position.size) <= 1e-12

    # --- liquidation ---
    if sc.expect.get("liq_triggered"):
        assert info.get("liq_triggered", False) is True
    if "terminated" in sc.expect:
        assert terminated is bool(sc.expect["terminated"])
    if "truncated" in sc.expect:
        assert truncated is bool(sc.expect["truncated"])
    if "termination_reason" in sc.expect:
        assert info.get("termination_reason") == sc.expect["termination_reason"]

    # --- death cost consistency (terminated episodes) ---
    # 重要：若回合是因為 liq_triggered / balance_insufficient 終止，
    # 則 death_cost 與 cost_risk 應該在該 step 直接為 1.0，避免「有死亡但 risk cost=0」。
    if info.get("termination_reason") in ("liq_triggered", "balance_insufficient"):
        assert float(info.get("cost_risk", 0.0)) == 1.0
        assert float(info.get("cost_breakdown", {}).get("death_cost", 0.0)) == 1.0

    # --- reward ---
    if sc.expect.get("reward_positive"):
        assert reward > 0.0

    # --- next observation (action effects cache) ---
    if sc.expect.get("next_obs_expected_fee_positive"):
        # cost_state[14] = expected_fee_if_trade (下一個 observation)
        assert float(obs["cost_state"][14]) >= 0.0
        assert float(obs["cost_state"][14]) > 0.0

    # --- risk signals in obs when flat ---
    if sc.expect.get("risk_liq_zero"):
        # cost_state[6:14] 有 risk_signals；平倉時 compute_risk_signals 會全部歸零
        assert float(obs["cost_state"][6]) == 0.0
        assert float(obs["cost_state"][7]) == 0.0

    # --- mark-to-market uses next close ---
    if sc.expect.get("equity_matches_next_close"):
        # new_equity 是用 idx+1 close 計算；此情境把 idx=11 close 設成 110
        assert float(info["equity"]) == pytest.approx(env.executor.equity(110.0), abs=1e-6)

    # --- cost: stop missing when stop disabled ---
    if sc.expect.get("cost_stop_missing"):
        assert float(info["cost"]) >= 0.0
        assert float(info["cost_breakdown"]["stop_missing_cost"]) == 1.0


