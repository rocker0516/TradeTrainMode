"""
評估結果的數據模型。
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from Train.eval.config import EvalConstraints
from Train.eval.utils import (
    to_float_or_none,
    to_int_or_none,
    mean_or_none,
)


@dataclass(frozen=True)
class EpisodeEval:
    """單一 episode 的評估結果（由 episode 結束時 info + 本地累加值構成）。"""

    final_balance: Optional[float]
    episode_max_dd: Optional[float]
    episode_steps: Optional[int]
    episode_start_step: Optional[int]
    episode_start_timestamp: Optional[str]
    stop_loss_count: Optional[int]
    liq_count: Optional[int]
    trade_count: Optional[int]
    long_entry_count: Optional[int]
    short_entry_count: Optional[int]
    holding_steps: Optional[int]
    holding_ratio: Optional[float]
    trades_per_step: Optional[float]
    mean_cost: Optional[float]
    ret: Optional[float]
    termination_reason: Optional[str]
    is_death_event: bool
    
    # 新增：可執行性指標
    total_fees_ratio: Optional[float] = None
    turnover_notional: Optional[float] = None
    final_position_size: Optional[float] = None
    episode_active_exit_count: Optional[int] = None
    
    # 新增：成本分解 (Mean per step)
    mean_cost_risk: Optional[float] = None
    mean_cost_fric: Optional[float] = None
    
    # 新增：是否通過上線門檻
    passed: bool = False

    # 額外：episode slice 的 regime 統計（每個 feature: mean/std/q10/q50/q90）
    regime_stats: Optional[Dict[str, Dict[str, float]]] = None

    @staticmethod
    def from_episode_end_info(
        info: Dict[str, Any],
        episode_steps: int,
        cost_sum: float,
        initial_balance: float,
        constraints: EvalConstraints,
        *,
        cost_breakdown_sums: Dict[str, float] = None,
        episode_start_timestamp: Optional[str] = None,
        episode_start_step: Optional[int] = None,
        regime_stats: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> "EpisodeEval":
        final_balance = to_float_or_none(info.get("final_balance", None))
        episode_max_dd = to_float_or_none(info.get("episode_max_dd", None))
        termination_reason = info.get("termination_reason", None)
        termination_reason = str(termination_reason) if termination_reason is not None else None

        stop_loss_count = to_int_or_none(info.get("episode_stop_loss_count", None))
        liq_count = to_int_or_none(info.get("episode_liq_count", None))
        trade_count = to_int_or_none(info.get("episode_trade_count", None))
        long_entry_count = to_int_or_none(info.get("long_entry_count", None))
        short_entry_count = to_int_or_none(info.get("short_entry_count", None))
        holding_steps = to_int_or_none(info.get("episode_holding_steps", None))
        
        # 新增欄位讀取
        total_fees_ratio = to_float_or_none(info.get("total_fees_ratio", None))
        turnover_notional = to_float_or_none(info.get("episode_turnover_notional", None))
        final_position_size = to_float_or_none(info.get("final_position_size", None))
        episode_active_exit_count = to_int_or_none(info.get("episode_active_exit_count", None))

        steps = max(1, int(episode_steps))
        holding_ratio = None
        if holding_steps is not None:
            holding_ratio = float(max(0, holding_steps)) / float(steps)

        trades_per_step = None
        if trade_count is not None:
            trades_per_step = float(max(0, trade_count)) / float(steps)

        mean_cost = float(cost_sum) / float(steps) if steps > 0 else None
        
        # 成本分解均值
        c_sums = cost_breakdown_sums or {}
        mean_cost_risk = float(c_sums.get("risk", 0.0)) / float(steps) if steps > 0 else 0.0
        mean_cost_fric = float(c_sums.get("fric", 0.0)) / float(steps) if steps > 0 else 0.0

        ret = None
        if (final_balance is not None) and (initial_balance > 0):
            ret = float(final_balance) / float(initial_balance) - 1.0

        # 死亡事件判定（雙保險）：termination_reason or liq_count>0
        is_death_event = False
        if termination_reason in {"liq_triggered", "balance_insufficient"}:
            is_death_event = True
        if (liq_count is not None) and (liq_count > 0):
            is_death_event = True
            
        # Pass 判定
        # 1. 不死
        # 2. MaxDD <= limit
        # 3. MeanCost <= limit
        # 4. Ret >= min_mean_return（預設 0.20 = +20%）
        min_mean_return = getattr(constraints, "min_mean_return", 0.0)
        passed = True
        if is_death_event:
            passed = False
        if (episode_max_dd is not None) and (episode_max_dd > constraints.max_dd_limit):
            passed = False
        if (mean_cost is not None) and (mean_cost > constraints.mean_cost_limit):
            passed = False
        if (ret is not None) and (ret < min_mean_return):
            passed = False

        return EpisodeEval(
            final_balance=final_balance,
            episode_max_dd=episode_max_dd,
            episode_steps=int(episode_steps),
            episode_start_step=to_int_or_none(info.get("episode_start_step", episode_start_step)),
            episode_start_timestamp=str(info.get("episode_start_timestamp", episode_start_timestamp)) if (info.get("episode_start_timestamp", episode_start_timestamp) is not None) else None,
            stop_loss_count=stop_loss_count,
            liq_count=liq_count,
            trade_count=trade_count,
            long_entry_count=long_entry_count,
            short_entry_count=short_entry_count,
            holding_steps=holding_steps,
            holding_ratio=holding_ratio,
            trades_per_step=trades_per_step,
            mean_cost=mean_cost,
            ret=ret,
            termination_reason=termination_reason,
            is_death_event=bool(is_death_event),
            total_fees_ratio=total_fees_ratio,
            turnover_notional=turnover_notional,
            final_position_size=final_position_size,
            episode_active_exit_count=episode_active_exit_count,
            mean_cost_risk=mean_cost_risk,
            mean_cost_fric=mean_cost_fric,
            passed=passed,
            regime_stats=regime_stats,
        )


@dataclass(frozen=True)
class EvalResults:
    """多個 episodes 聚合後的評估結果。"""

    any_death_event: bool
    death_event_count: int
    mean_final_balance: Optional[float]
    mean_return: Optional[float]
    mean_episode_max_dd: Optional[float]
    mean_cost: Optional[float]
    mean_stop_loss_count: Optional[float]
    mean_liq_count: Optional[float]
    mean_trades_per_step: Optional[float]
    mean_long_entry_count: Optional[float]
    mean_short_entry_count: Optional[float]
    mean_holding_ratio: Optional[float]
    
    # 新增：分位數與尾部風險
    ret_p10: Optional[float]
    ret_median: Optional[float]
    ret_p90: Optional[float]
    dd_p90: Optional[float]
    dd_max: Optional[float]
    
    # 新增：Gate 統計
    pass_rate: float
    pass_count: int
    total_count: int
    
    # 新增：執行性
    mean_total_fees_ratio: Optional[float]
    mean_turnover_notional: Optional[float]
    
    # 新增：成本分解
    mean_cost_risk: Optional[float]
    mean_cost_fric: Optional[float]
    
    # 用於顯示
    worst_episodes: List[EpisodeEval]

    @staticmethod
    def aggregate(per_episode: List[EpisodeEval], any_death_event: bool) -> "EvalResults":
        death_event_count = int(sum(1 for ep in per_episode if ep.is_death_event))
        pass_count = int(sum(1 for ep in per_episode if ep.passed))
        total_count = len(per_episode)
        pass_rate = float(pass_count) / float(total_count) if total_count > 0 else 0.0

        # 計算分位數
        rets = [ep.ret for ep in per_episode if ep.ret is not None]
        dds = [ep.episode_max_dd for ep in per_episode if ep.episode_max_dd is not None]
        
        def _quantile(data, q):
            if not data: return None
            return float(np.quantile(data, q))
            
        ret_p10 = _quantile(rets, 0.1)
        ret_median = _quantile(rets, 0.5)
        ret_p90 = _quantile(rets, 0.9)
        dd_p90 = _quantile(dds, 0.9)
        dd_max = float(np.max(dds)) if dds else None
        
        # 找出最差 3 個 episode (依 ret 排序)
        valid_eps = [ep for ep in per_episode if ep.ret is not None]
        worst_episodes = sorted(valid_eps, key=lambda x: x.ret)[:3]

        return EvalResults(
            any_death_event=bool(any_death_event),
            death_event_count=int(death_event_count),
            mean_final_balance=mean_or_none([ep.final_balance for ep in per_episode]),
            mean_return=mean_or_none([ep.ret for ep in per_episode]),
            mean_episode_max_dd=mean_or_none([ep.episode_max_dd for ep in per_episode]),
            mean_cost=mean_or_none([ep.mean_cost for ep in per_episode]),
            mean_stop_loss_count=mean_or_none([ep.stop_loss_count for ep in per_episode]),
            mean_liq_count=mean_or_none([ep.liq_count for ep in per_episode]),
            mean_trades_per_step=mean_or_none([ep.trades_per_step for ep in per_episode]),
            mean_long_entry_count=mean_or_none([ep.long_entry_count for ep in per_episode]),
            mean_short_entry_count=mean_or_none([ep.short_entry_count for ep in per_episode]),
            mean_holding_ratio=mean_or_none([ep.holding_ratio for ep in per_episode]),
            ret_p10=ret_p10,
            ret_median=ret_median,
            ret_p90=ret_p90,
            dd_p90=dd_p90,
            dd_max=dd_max,
            pass_rate=pass_rate,
            pass_count=pass_count,
            total_count=total_count,
            mean_total_fees_ratio=mean_or_none([ep.total_fees_ratio for ep in per_episode]),
            mean_turnover_notional=mean_or_none([ep.turnover_notional for ep in per_episode]),
            mean_cost_risk=mean_or_none([ep.mean_cost_risk for ep in per_episode]),
            mean_cost_fric=mean_or_none([ep.mean_cost_fric for ep in per_episode]),
            worst_episodes=worst_episodes,
        )

