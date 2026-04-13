"""
Live 端 observation 與 `TradingEnvironment` 對齊的輔助函式。

將 `last_action_effects`、account_metrics 相關欄位在 tick 之間持久化，
使推論時的 `context_state` / `account_state` 分佈接近訓練逐步演化（而非每步全零）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Env.config import Config
from Env.Executors.trade_executor import TradeExecutor
from Eval.phase_ab_env_config import PhaseABEnvConfig


def default_last_action_effects() -> Dict[str, float]:
    """與 `TradingEnvironment.reset()` 內 `_last_action_effects` 初始值一致。"""
    return {
        "expected_fee_if_trade": 0.0,
        "predicted_used_margin_after_action": 0.0,
        "predicted_available_balance_after_action": 0.0,
        "predicted_liq_distance_after_action": 0.0,
        "predicted_stop_distance_after_action": 0.0,
        "cooldown_remaining_norm": 0.0,
        "action_overridden_flag": 0.0,
        "last_action_raw": 0.0,
        "last_action_used": 0.0,
        "last_target_pos_pct": 0.0,
        "last_final_pos_pct": 0.0,
        "trade_executed_flag": 0.0,
        "trade_freq_blocked": 0.0,
    }


def paper_position_qty(*, equity_usdt: float, leverage: float, pos_pct: float, price: float) -> float:
    """由權益與目標曝險比例推算合約數量（與 env 名義口徑一致）。"""
    eq = float(max(equity_usdt, 1e-8))
    lev = float(max(leverage, 1e-8))
    px = float(price)
    if px <= 0.0:
        return 0.0
    max_nominal = eq * lev
    return float(float(pos_pct) * max_nominal / px)


def estimate_expected_fee_usdt(
    *,
    final_pos_pct: float,
    last_equity: float,
    current_price: float,
    current_size: float,
    leverage: float,
    fee_rate_pct: float,
) -> float:
    """對齊 `TradingEnvironment._estimate_expected_fee`：調倉名目 × 費率。"""
    px = float(current_price)
    if px <= 0.0:
        return 0.0
    desired_notional = float(final_pos_pct) * float(last_equity) * float(leverage)
    desired_size = desired_notional / px
    fee_rate_pct_f = float(fee_rate_pct)
    return abs(float(desired_size) - float(current_size)) * px * (fee_rate_pct_f / 100.0)


def sync_executor_snapshot_for_obs_cache(
    executor: TradeExecutor,
    *,
    wallet_balance: float,
    position_qty: float,
    entry_price: float,
) -> None:
    """
    在不呼叫 `execute()` 的前提下，同步 wallet / 倉位 / used_margin，
    供 `_update_predicted_action_effects` 與強平距離估算使用。

    Args:
        executor: 本輪用於風險預測的執行器（通常為 live 的輕量副本）。
        wallet_balance: 錢包餘額（USDT）。
        position_qty: 持倉數量（資產單位）。
        entry_price: 進場均價；無倉時應為 0。
    """
    executor.wallet_balance = float(wallet_balance)
    sz = float(position_qty)
    ep = float(entry_price)
    executor.position.size = sz
    if abs(sz) <= 1e-12:
        executor.position.entry_price = 0.0
        executor.used_margin = 0.0
        return
    executor.position.entry_price = float(ep) if ep > 0.0 else 0.0
    if executor.position.entry_price <= 0.0:
        executor.used_margin = 0.0
        return
    executor.used_margin = float(max(0.0, abs(sz) * float(executor.position.entry_price) / float(executor.leverage)))


def _update_predicted_action_effects(
    executor: TradeExecutor,
    *,
    expected_fee: float,
    current_price: float,
    effects_out: Dict[str, float],
) -> None:
    """對齊 `TradingEnvironment._update_action_effects_cache` 的預測欄位寫入 effects_out。"""
    try:
        used_margin_after = float(getattr(executor, "used_margin", 0.0))
        available_after = float(executor.available_balance())
        liq_after = float(executor.get_liquidation_price(current_price))
        liq_dist_after = float(abs(current_price - liq_after) / current_price) if liq_after > 0 else 0.0

        entry_after = float(executor.position.entry_price)
        stop_after = float(executor.position.stop_loss_price)
        stop_dist_after = (
            float(abs(entry_after - stop_after) / entry_after)
            if (entry_after > 0 and stop_after > 0)
            else 0.0
        )

        effects_out["expected_fee_if_trade"] = float(expected_fee)
        effects_out["predicted_used_margin_after_action"] = used_margin_after
        effects_out["predicted_available_balance_after_action"] = available_after
        effects_out["predicted_liq_distance_after_action"] = float(np.clip(liq_dist_after, 0.0, 5.0))
        effects_out["predicted_stop_distance_after_action"] = float(np.clip(stop_dist_after, 0.0, 5.0))
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        pass


def build_account_metrics_live(
    *,
    session_initial_balance: float,
    max_equity_so_far: float,
    episode_stop_loss_count: int,
    episode_liq_count: int,
    risk_budget: float,
    decision_step_completed: int,
    episode_max_steps_cap: int,
    last_trade_decision_step: int,
    position_entry_decision_step: Optional[int],
    last_step_fee: float,
    rolling_fee_sum: float,
    cooldown_remaining: float,
    min_balance: float,
    window_size_5m: int,
    recent_flat_ratio: float,
    trade_freq_blocked_last: float,
) -> Dict[str, Any]:
    """
    建構與 `TradingEnvironment._get_observation` 相同 key 的 account_metrics。

    Args:
        decision_step_completed: 本 tick 開始前「已完成」的決策步數（對齊 env 在 step 尾端 episode_steps 再取 obs 的語意）。
        episode_max_steps_cap: `steps_since_trade_norm` 分母用；預設與 Phase AB `DEFAULT_MAX_EPISODE_STEPS` 一致。
        last_trade_decision_step: 上次成交時的 decision_step（未成交用 -999999）。
        position_entry_decision_step: 當前有倉時的進場 decision_step，否則 None。

    Returns:
        可直接餵給 `TradingObserver.get_observation(..., account_metrics=...)` 的 dict。
    """
    init_b = float(max(session_initial_balance, 1e-8))
    steps_since_trade = (
        float(int(decision_step_completed) - int(last_trade_decision_step))
        if int(last_trade_decision_step) > -10**6
        else float(window_size_5m)
    )
    holding_steps = 0.0
    if position_entry_decision_step is not None:
        holding_steps = float(max(0, int(decision_step_completed) - int(position_entry_decision_step)))

    return {
        "initial_balance": float(init_b),
        "max_equity_so_far": float(max_equity_so_far),
        "episode_stop_loss_count": int(episode_stop_loss_count),
        "episode_liq_count": int(episode_liq_count),
        "risk_budget": float(risk_budget),
        "steps_since_trade": float(steps_since_trade),
        "holding_steps": float(holding_steps),
        "last_step_fee": float(last_step_fee),
        "rolling_fee_sum": float(rolling_fee_sum),
        "cooldown_remaining": float(cooldown_remaining),
        "min_balance": float(min_balance),
        "episode_steps": int(decision_step_completed),
        "episode_max_steps": int(max(1, episode_max_steps_cap)),
        "trade_freq_remaining_ratio": 1.0,
        "trade_freq_blocked_last": float(trade_freq_blocked_last),
        "recent_flat_ratio": float(recent_flat_ratio),
    }


def update_fee_rolling_live(
    *,
    fee_history: List[Tuple[int, float]],
    rolling_fee_sum: float,
    decision_step: int,
    step_fee: float,
    fee_window: int,
) -> Tuple[List[Tuple[int, float]], float, float]:
    """
    對齊 `Tracker.update_fee_tracking` 的滾動手續費視窗（以 decision_step 當時間索引）。

    Args:
        fee_history: 目前儲存的 (decision_step, step_fee) 序列（會就地修改）。
        rolling_fee_sum: 目前滾動加總。
        decision_step: 本步索引（通常為「本步完成後」的 decision_step）。
        step_fee: 本步新增手續費（USDT）。
        fee_window: 視窗寬（步）；使用 `Config.FEE_ROLLING_WINDOW`。

    Returns:
        (fee_history, new_rolling_fee_sum, last_step_fee)。
    """
    hist = list(fee_history)
    ds = int(decision_step)
    sf = float(max(0.0, step_fee))
    roll = float(rolling_fee_sum) + sf
    hist.append((ds, sf))
    win = int(max(1, fee_window))
    while hist and (ds - int(hist[0][0])) > win:
        _, old_fee = hist.pop(0)
        roll -= float(old_fee)
    return hist, float(max(0.0, roll)), sf


def append_flat_window(
    flat_history: List[float],
    *,
    is_flat: bool,
    max_len: int,
) -> Tuple[List[float], float]:
    """
    維護空倉比例滑窗（對齊 `TradingEnvironment` 的 `_flat_deque`）。

    Args:
        flat_history: 歷史 0/1 旗標（會回傳新 list）。
        is_flat: 本步是否視為空倉（|final_pos_pct| < flat_threshold）。
        max_len: 視窗長度（預設 864）。

    Returns:
        (new_history, recent_flat_ratio)。
    """
    ml = int(max(1, max_len))
    cur = list(flat_history)
    cur.append(1.0 if is_flat else 0.0)
    if len(cur) > ml:
        cur = cur[-ml:]
    n = len(cur)
    ratio = float(sum(cur) / n) if n > 0 else 0.5
    return cur, float(ratio)


def merge_completed_last_action_effects(
    *,
    base: Dict[str, float],
    executor: TradeExecutor,
    expected_fee: float,
    current_price: float,
    action_raw: float,
    action_used_scalar: float,
    target_pos_pct: float,
    final_pos_pct: float,
    traded: bool,
    action_overridden_flag: bool,
    trade_freq_blocked: bool,
    cooldown_remaining_norm: float,
) -> Dict[str, float]:
    """
    在本步決策完成後，產生「供下一 tick obs 使用」的 last_action_effects（新 dict）。

    Args:
        base: 通常為上一狀態的 effects（用於保留欄位）；會複製後更新。
        executor: 已同步至本步 post-action 快照的 TradeExecutor。
        expected_fee: `_estimate_expected_fee` 同口徑。
        action_used_scalar: 進入 ActionProcessor 前的動作（live 取 clip 後純量）。
        cooldown_remaining_norm: 止損冷卻正規化 [0,1]；live 無冷卻時為 0。

    Returns:
        新的 last_action_effects dict（可存入 LiveRunnerState）。
    """
    out = dict(base)
    _update_predicted_action_effects(
        executor,
        expected_fee=float(expected_fee),
        current_price=float(current_price),
        effects_out=out,
    )
    out["last_action_raw"] = float(action_raw)
    out["last_action_used"] = float(action_used_scalar)
    out["last_target_pos_pct"] = float(target_pos_pct)
    out["last_final_pos_pct"] = float(final_pos_pct)
    out["action_overridden_flag"] = 1.0 if bool(action_overridden_flag) else 0.0
    out["trade_freq_blocked"] = 1.0 if bool(trade_freq_blocked) else 0.0
    out["trade_executed_flag"] = 1.0 if bool(traded) else 0.0
    out["cooldown_remaining_norm"] = float(np.clip(float(cooldown_remaining_norm), 0.0, 1.0))
    return out


def default_episode_max_steps_cap() -> int:
    """訓練 Phase AB 預設回合長度上限（供 steps_since_trade_norm 分母）。"""
    return int(PhaseABEnvConfig.DEFAULT_MAX_EPISODE_STEPS)


def default_min_balance_usdt(*, session_initial_balance: float) -> float:
    """對齊 env：若無獨立 min_balance，採用 Config 比例。"""
    init_b = float(max(session_initial_balance, 1e-8))
    # TradingEnvironment 預設 min_balance 與 initial 成比例；live 以 session 初始為錨點
    return float(init_b * (float(Config.MIN_BALANCE) / float(max(Config.INITIAL_BALANCE, 1e-8))))


def default_flat_window_steps() -> int:
    """與 `TradingEnvironment` kwargs 預設 `flat_window_steps` 一致。"""
    return int(864)
