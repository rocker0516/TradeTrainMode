from __future__ import annotations

import numpy as np
from collections import Counter
from typing import Any, Dict, List


def compute_trade_stats(ep_infos: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    計算「TRADE STATS」視窗統計（純計算，方便測試）。

    Args:
        ep_infos: Callback 收集到的 episode 結束 info（VecMonitor 會在結束時注入 "episode" key）。

    Returns:
        dict，包含：
        - avg_fee
        - avg_liq
        - avg_dd
        - max_dd: 視窗內最大 drawdown（不排除任何回合）
        - max_dd_excl_liq: 排除「曾發生強平」回合後的視窗內最大 drawdown
        - max_dd_excl_stop_loss: 排除「曾發生止損」回合後的視窗內最大 drawdown
        - avg_long_entries
        - avg_short_entries
        - avg_long_closes
        - avg_short_closes
        - avg_stop_loss
        - avg_holding_steps
        - avg_flat_steps: 平均空倉步數（與 cost_flat 同口徑：|final_pos_pct| < flat_threshold）
        - avg_trade_count
        - avg_active_exits
        - stop_loss_rate_pct
        - active_exit_rate_pct
        - exit_coverage_rate_pct: (active_exit + stop_loss) / total_closes
        - stop_loss_per_entry_pct: stop_loss / total_entries（越高通常代表止損偏緊或訊號品質偏差）
    """
    if not ep_infos:
        return {
            "avg_fee": 0.0,
            "avg_liq": 0.0,
            "avg_dd": 0.0,
            "max_dd": 0.0,
            "max_dd_excl_liq": 0.0,
            "max_dd_excl_stop_loss": 0.0,
            "avg_long_entries": 0.0,
            "avg_short_entries": 0.0,
            "avg_long_closes": 0.0,
            "avg_short_closes": 0.0,
            "avg_stop_loss": 0.0,
            "avg_holding_steps": 0.0,
            "avg_flat_steps": 0.0,
            "avg_trade_count": 0.0,
            "avg_active_exits": 0.0,
            "stop_loss_rate_pct": 0.0,
            "active_exit_rate_pct": 0.0,
            "exit_coverage_rate_pct": 0.0,
            "stop_loss_per_entry_pct": 0.0,
        }

    total_fees = [float(x.get("total_fees", 0.0)) for x in ep_infos]
    liq_counts = [int(x.get("episode_liq_count", 0)) for x in ep_infos]
    max_dds = [float(x.get("episode_max_dd", 0.0)) for x in ep_infos]
    long_entries = [int(x.get("long_entry_count", 0)) for x in ep_infos]
    short_entries = [int(x.get("short_entry_count", 0)) for x in ep_infos]
    long_closes = [int(x.get("long_close_count", 0)) for x in ep_infos]
    short_closes = [int(x.get("short_close_count", 0)) for x in ep_infos]
    stop_losses = [int(x.get("episode_stop_loss_count", 0)) for x in ep_infos]
    active_exits = [int(x.get("episode_active_exit_count", 0)) for x in ep_infos]
    holding_steps = [int(x.get("episode_holding_steps", 0)) for x in ep_infos]
    flat_steps = [int(x.get("episode_flat_steps", 0)) for x in ep_infos]
    trade_counts = [int(x.get("episode_trade_count", 0)) for x in ep_infos]

    # --- max drawdown variants ---
    # 「排除」定義：整個 episode 只要曾發生過該事件，就把該 episode 從 max_dd 計算樣本排除。
    dd_pairs = list(zip(max_dds, liq_counts, stop_losses))
    def _safe_max(xs: List[float]) -> float:
        if not xs:
            return 0.0
        return float(np.max(xs))

    max_dd_all = _safe_max(list(max_dds))
    max_dd_excl_liq = _safe_max([dd for dd, liq, _sl in dd_pairs if int(liq) <= 0])
    # 你要求的口徑：排除止損「同時也排除強平」=> 僅保留 (stop_loss==0 且 liq==0) 的回合
    max_dd_excl_stop_loss = _safe_max([dd for dd, liq, sl in dd_pairs if int(sl) <= 0 and int(liq) <= 0])

    total_stop_losses = int(np.sum(stop_losses)) if stop_losses else 0
    total_active_exits = int(np.sum(active_exits)) if active_exits else 0
    total_exits = int(total_stop_losses + total_active_exits)
    if total_exits > 0:
        stop_loss_rate_pct = (total_stop_losses / total_exits) * 100.0
        active_exit_rate_pct = (total_active_exits / total_exits) * 100.0
    else:
        stop_loss_rate_pct = 0.0
        active_exit_rate_pct = 0.0

    total_closes = int(np.sum(long_closes) + np.sum(short_closes)) if (long_closes or short_closes) else 0
    if total_closes > 0:
        exit_coverage_rate_pct = (total_exits / total_closes) * 100.0
    else:
        exit_coverage_rate_pct = 0.0
    
    total_entries = int(np.sum(long_entries) + np.sum(short_entries)) if (long_entries or short_entries) else 0
    if total_entries > 0:
        stop_loss_per_entry_pct = (total_stop_losses / total_entries) * 100.0
    else:
        stop_loss_per_entry_pct = 0.0

    return {
        "avg_fee": float(np.mean(total_fees)) if total_fees else 0.0,
        "avg_liq": float(np.mean(liq_counts)) if liq_counts else 0.0,
        "avg_dd": float(np.mean(max_dds)) if max_dds else 0.0,
        "max_dd": float(max_dd_all),
        "max_dd_excl_liq": float(max_dd_excl_liq) if np.isfinite(max_dd_excl_liq) else 0.0,
        "max_dd_excl_stop_loss": float(max_dd_excl_stop_loss) if np.isfinite(max_dd_excl_stop_loss) else 0.0,
        "avg_long_entries": float(np.mean(long_entries)) if long_entries else 0.0,
        "avg_short_entries": float(np.mean(short_entries)) if short_entries else 0.0,
        "avg_long_closes": float(np.mean(long_closes)) if long_closes else 0.0,
        "avg_short_closes": float(np.mean(short_closes)) if short_closes else 0.0,
        "avg_stop_loss": float(np.mean(stop_losses)) if stop_losses else 0.0,
        "avg_holding_steps": float(np.mean(holding_steps)) if holding_steps else 0.0,
        "avg_flat_steps": float(np.mean(flat_steps)) if flat_steps else 0.0,
        "avg_trade_count": float(np.mean(trade_counts)) if trade_counts else 0.0,
        "avg_active_exits": float(np.mean(active_exits)) if active_exits else 0.0,
        "stop_loss_rate_pct": float(stop_loss_rate_pct),
        "active_exit_rate_pct": float(active_exit_rate_pct),
        "exit_coverage_rate_pct": float(exit_coverage_rate_pct),
        "stop_loss_per_entry_pct": float(stop_loss_per_entry_pct),
    }


def compute_end_result_stats(ep_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    計算「回合結束結果」視窗統計（純計算，方便測試）。

    統計內容：
    - terminated / truncated 次數與比例
    - termination_reason 各類型次數與比例
    - Avg Episode Length（VecMonitor: info["episode"]["l】，macro/agent 步數）
    - avg_episode_steps_inner: 環境內層步數（TradingEnv 的 episode_steps），與 holding_steps/flat_steps 同口徑，供比例計算
    - Avg Final Balance（TradingEnv: info["final_balance"]）

    Args:
        ep_infos: Callback 收集到的 episode 結束 info。

    Returns:
        dict（結構穩定，適合印出/寫 TB）。
    """
    n = int(len(ep_infos))
    if n == 0:
        return {
            "n": 0,
            "terminated_count": 0,
            "truncated_count": 0,
            "terminated_rate": 0.0,
            "truncated_rate": 0.0,
            "reason_counts": {},
            "reason_rates": {},
            "avg_episode_len": 0.0,
            "avg_episode_steps_inner": 0.0,
            "avg_final_balance": 0.0,
            "avg_simple_return": None,
        }

    terminated_flags = [bool(x.get("terminated", False)) for x in ep_infos]
    truncated_flags = [bool(x.get("truncated", False)) for x in ep_infos]
    terminated_count = int(sum(terminated_flags))
    truncated_count = int(sum(truncated_flags))

    # termination_reason（缺失則歸類 unknown）
    reasons: List[str] = []
    for x in ep_infos:
        r = x.get("termination_reason", "unknown")
        r = str(r) if r is not None else "unknown"
        r = r.strip() or "unknown"
        reasons.append(r)

    reason_counts = dict(Counter(reasons))
    reason_rates = {k: (v / n) * 100.0 for k, v in reason_counts.items()}

    # VecMonitor episode length（macro/agent 步數）
    ep_lens: List[int] = []
    # 環境內層步數（與 episode_holding_steps / episode_flat_steps 同口徑，供比例分母）
    ep_steps_inner: List[int] = []
    for x in ep_infos:
        ep = x.get("episode", {})
        if isinstance(ep, dict):
            try:
                ep_lens.append(int(ep.get("l", 0)))
            except (TypeError, ValueError):
                ep_lens.append(0)
        else:
            ep_lens.append(0)
        try:
            inner = int(x.get("episode_steps", 0))
            if inner <= 0 and isinstance(ep, dict):
                inner = int(ep.get("l", 0))  # fallback to macro 步數
            ep_steps_inner.append(max(0, inner))
        except (TypeError, ValueError):
            ep_steps_inner.append(ep_lens[-1] if ep_lens else 0)

    final_balances: List[float] = []
    simple_returns: List[float] = []
    for x in ep_infos:
        try:
            final_balances.append(float(x.get("final_balance", 0.0)))
        except (TypeError, ValueError):
            final_balances.append(0.0)
        try:
            init_bal = float(x.get("initial_balance", 0.0))
            fin_bal = float(x.get("final_balance", 0.0))
            if init_bal > 0:
                simple_returns.append(fin_bal / init_bal - 1.0)
        except (TypeError, ValueError):
            pass

    return {
        "n": n,
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
        "terminated_rate": (terminated_count / n) * 100.0,
        "truncated_rate": (truncated_count / n) * 100.0,
        "reason_counts": reason_counts,
        "reason_rates": reason_rates,
        "avg_episode_len": float(np.mean(ep_lens)) if ep_lens else 0.0,
        "avg_episode_steps_inner": float(np.mean(ep_steps_inner)) if ep_steps_inner else 0.0,
        "avg_final_balance": float(np.mean(final_balances)) if final_balances else 0.0,
        "avg_simple_return": float(np.mean(simple_returns)) if simple_returns else None,
    }

