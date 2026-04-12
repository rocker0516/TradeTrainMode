from __future__ import annotations

"""
評估後的上線 blocker 判定模組。

用途：
- 將 `PhaseABEvaluator.evaluate()` 回傳的 payload 轉成標準化 metrics。
- 根據 `P0 / P1 / P2` 規則輸出結構化判定結果。
- 提供終端摘要文字，供訓練入口或 eval-only 模式直接顯示。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class JudgmentStatus(str, Enum):
    """判定狀態。"""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class JudgmentItem:
    """單一 blocker 的判定結果。"""

    status: JudgmentStatus
    reason: str
    evidence: dict[str, float | str | bool | int]


@dataclass(frozen=True)
class JudgmentResult:
    """完整的 blocker 判定結果。"""

    p0_economics: JudgmentItem
    p1_risk: JudgmentItem
    p1_objective: JudgmentItem
    p2_observability: JudgmentItem
    recommended_next_action: str
    should_block_reward_redesign: bool
    should_request_more_diagnostics: bool


def _safe_float(value: Any, default: float = 0.0) -> float:
    """將任意值轉成 float；失敗時回傳預設值。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    """將任意值轉成 int；失敗時回傳預設值。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


TRADES_PER_DAY_WARN_THRESHOLD = 80.0
TRADE_RATE_WARN_THRESHOLD = TRADES_PER_DAY_WARN_THRESHOLD / 288.0


def _summary_stat(payload: dict[str, Any], key: str, stat: str = "mean") -> float:
    """讀取 payload.summary[key][stat]；缺值時回傳 0。"""

    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        return 0.0
    item = summary.get(key, {})
    if not isinstance(item, dict):
        return 0.0
    return _safe_float(item.get(stat, 0.0), default=0.0)


def _guardrail_pass(payload: dict[str, Any], check_name: str) -> bool | None:
    """讀取 guardrail pass 狀態；guardrail 不可用時回傳 None。"""

    guardrails = payload.get("guardrails", {})
    if not isinstance(guardrails, dict):
        return None
    if not bool(guardrails.get("enabled", False)):
        return None
    checks = guardrails.get("checks", {})
    if not isinstance(checks, dict):
        return None
    item = checks.get(check_name, {})
    if not isinstance(item, dict) or "pass" not in item:
        return None
    return bool(item.get("pass"))


def _quantile(values: list[float], q: float) -> float:
    """簡單分位數近似。"""

    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    idx = int(round((len(vals) - 1) * max(0.0, min(1.0, q))))
    return float(vals[idx])


def extract_judgment_metrics(payload: dict[str, Any]) -> dict[str, float | bool | int]:
    """
    將 evaluator payload 轉成 judgment 規則直接使用的平面 metrics dict。

    Args:
        payload: `PhaseABEvaluator.evaluate()` 回傳的 payload。

    Returns:
        平面化後的 metrics。
    """

    results = payload.get("results", [])
    if not isinstance(results, list):
        results = []
    term_counts = payload.get("termination_reason_counts", {})
    if not isinstance(term_counts, dict):
        term_counts = {}

    profits: list[float] = []
    trade_counts: list[float] = []
    max_dds: list[float] = []
    positive_count = 0
    negative_count = 0

    for item in results:
        if not isinstance(item, dict):
            continue
        profit = _safe_float(item.get("profit", 0.0))
        trade_count = _safe_float(item.get("episode_trade_count", 0.0))
        max_dd = _safe_float(item.get("episode_max_dd", 0.0))
        profits.append(profit)
        trade_counts.append(trade_count)
        max_dds.append(max_dd)
        if profit > 0.0:
            positive_count += 1
        else:
            negative_count += 1

    low_trade_threshold = _quantile(trade_counts, 0.25)
    high_trade_threshold = _quantile(trade_counts, 0.75)
    low_churn_positive = 0
    high_churn_negative = 0
    for item in results:
        if not isinstance(item, dict):
            continue
        profit = _safe_float(item.get("profit", 0.0))
        trade_count = _safe_float(item.get("episode_trade_count", 0.0))
        if profit > 0.0 and trade_count <= low_trade_threshold:
            low_churn_positive += 1
        if profit <= 0.0 and trade_count >= high_trade_threshold:
            high_churn_negative += 1

    economics_guardrail_pass = _guardrail_pass(payload, "profit_guardrail")
    dd_guardrail_pass = _guardrail_pass(payload, "max_dd_guardrail")
    trade_guardrail_pass = _guardrail_pass(payload, "trade_count_guardrail")
    episode_steps_mean = _summary_stat(payload, "episode_steps")
    trade_count_mean = _summary_stat(payload, "episode_trade_count")
    trade_rate = trade_count_mean / max(episode_steps_mean, 1e-8)
    trades_per_day = trade_count_mean / max(episode_steps_mean / 288.0, 1e-8)

    return {
        "episodes": _safe_int(payload.get("episodes", len(results)), default=len(results)),
        "result_count": len(results),
        "profit_mean": _summary_stat(payload, "profit"),
        "profit_std": _summary_stat(payload, "profit", stat="std"),
        "profit_per_trade_mean": _summary_stat(payload, "profit_per_trade"),
        "episode_trade_count_mean": trade_count_mean,
        "episode_trade_count_std": _summary_stat(payload, "episode_trade_count", stat="std"),
        "episode_steps_mean": episode_steps_mean,
        "trade_rate": trade_rate,
        "trades_per_day": trades_per_day,
        "total_fees_mean": _summary_stat(payload, "total_fees"),
        "episode_max_dd_mean": _summary_stat(payload, "episode_max_dd"),
        "episode_max_dd_std": _summary_stat(payload, "episode_max_dd", stat="std"),
        "episode_cost_risk_sum_mean": _summary_stat(payload, "episode_cost_risk_sum"),
        "episode_cost_risk_dense_sum_mean": _summary_stat(payload, "episode_cost_risk_dense_sum"),
        "override_rate_mean": _summary_stat(payload, "override_rate"),
        "execution_rate_mean": _summary_stat(payload, "execution_rate"),
        "guardrail_profit_pass": economics_guardrail_pass if economics_guardrail_pass is not None else False,
        "guardrail_max_dd_pass": dd_guardrail_pass if dd_guardrail_pass is not None else False,
        "guardrail_trade_count_pass": trade_guardrail_pass if trade_guardrail_pass is not None else False,
        "guardrails_enabled": bool(payload.get("guardrails", {}).get("enabled", False))
        if isinstance(payload.get("guardrails", {}), dict)
        else False,
        "termination_liq_count": _safe_int(term_counts.get("liq_triggered", 0)),
        "termination_balance_insufficient_count": _safe_int(term_counts.get("balance_insufficient", 0)),
        "positive_profit_episode_count": positive_count,
        "negative_profit_episode_count": negative_count,
        "low_churn_positive_episode_count": low_churn_positive,
        "high_churn_negative_episode_count": high_churn_negative,
        "low_trade_count_threshold": low_trade_threshold,
        "high_trade_count_threshold": high_trade_threshold,
    }


def _judge_p0_economics(metrics: dict[str, float | bool | int]) -> JudgmentItem:
    """判定 P0 經濟不可行。"""

    profit_mean = _safe_float(metrics.get("profit_mean", 0.0))
    profit_per_trade_mean = _safe_float(metrics.get("profit_per_trade_mean", 0.0))
    total_fees_mean = _safe_float(metrics.get("total_fees_mean", 0.0))
    trade_count_mean = _safe_float(metrics.get("episode_trade_count_mean", 0.0))
    guardrails_enabled = bool(metrics.get("guardrails_enabled", False))
    guardrail_profit_pass = bool(metrics.get("guardrail_profit_pass", False))
    guardrail_trade_count_pass = bool(metrics.get("guardrail_trade_count_pass", False))
    fees_profit_ratio = total_fees_mean / max(abs(profit_mean), 1e-8)

    if (guardrails_enabled and (not guardrail_profit_pass) and (not guardrail_trade_count_pass)) or profit_mean <= 0.0 or profit_per_trade_mean <= 0.0:
        return JudgmentItem(
            status=JudgmentStatus.FAIL,
            reason="成本後整體收益效率不足，尚不適合進一步談上線。",
            evidence={
                "profit_mean": profit_mean,
                "profit_per_trade_mean": profit_per_trade_mean,
                "trade_count_mean": trade_count_mean,
                "total_fees_mean": total_fees_mean,
                "fees_profit_ratio": fees_profit_ratio,
                "guardrail_profit_pass": guardrail_profit_pass,
                "guardrail_trade_count_pass": guardrail_trade_count_pass,
            },
        )
    if profit_mean > 0.0 and (profit_per_trade_mean <= 0.01 or fees_profit_ratio >= 1.0 or (guardrails_enabled and not guardrail_trade_count_pass)):
        return JudgmentItem(
            status=JudgmentStatus.WARN,
            reason="雖然平均收益為正，但交易效率偏弱或交易次數偏高，實盤經濟性仍有疑慮。",
            evidence={
                "profit_mean": profit_mean,
                "profit_per_trade_mean": profit_per_trade_mean,
                "trade_count_mean": trade_count_mean,
                "total_fees_mean": total_fees_mean,
                "fees_profit_ratio": fees_profit_ratio,
                "guardrail_trade_count_pass": guardrail_trade_count_pass,
            },
        )
    return JudgmentItem(
        status=JudgmentStatus.PASS,
        reason="平均收益與每筆交易效率均達到基本要求。",
        evidence={
            "profit_mean": profit_mean,
            "profit_per_trade_mean": profit_per_trade_mean,
            "trade_count_mean": trade_count_mean,
            "total_fees_mean": total_fees_mean,
            "fees_profit_ratio": fees_profit_ratio,
        },
    )


def _judge_p1_risk(metrics: dict[str, float | bool | int]) -> JudgmentItem:
    """判定 P1 風險與穩定性。"""

    episode_max_dd_mean = _safe_float(metrics.get("episode_max_dd_mean", 0.0))
    episode_max_dd_std = _safe_float(metrics.get("episode_max_dd_std", 0.0))
    cost_risk_dense_mean = _safe_float(metrics.get("episode_cost_risk_dense_sum_mean", 0.0))
    liq_count = _safe_int(metrics.get("termination_liq_count", 0))
    balance_count = _safe_int(metrics.get("termination_balance_insufficient_count", 0))
    guardrails_enabled = bool(metrics.get("guardrails_enabled", False))
    guardrail_max_dd_pass = bool(metrics.get("guardrail_max_dd_pass", False))

    if (guardrails_enabled and not guardrail_max_dd_pass) or liq_count > 0 or balance_count > 0:
        return JudgmentItem(
            status=JudgmentStatus.FAIL,
            reason="回撤或死亡事件超出可接受範圍，策略穩定性不足。",
            evidence={
                "episode_max_dd_mean": episode_max_dd_mean,
                "episode_max_dd_std": episode_max_dd_std,
                "episode_cost_risk_dense_sum_mean": cost_risk_dense_mean,
                "termination_liq_count": liq_count,
                "termination_balance_insufficient_count": balance_count,
                "guardrail_max_dd_pass": guardrail_max_dd_pass,
            },
        )
    if episode_max_dd_mean >= 0.25 or episode_max_dd_std >= 0.15 or cost_risk_dense_mean >= 0.2:
        return JudgmentItem(
            status=JudgmentStatus.WARN,
            reason="平均回撤或風險貼線程度偏高，仍需觀察穩定性。",
            evidence={
                "episode_max_dd_mean": episode_max_dd_mean,
                "episode_max_dd_std": episode_max_dd_std,
                "episode_cost_risk_dense_sum_mean": cost_risk_dense_mean,
                "termination_liq_count": liq_count,
                "termination_balance_insufficient_count": balance_count,
            },
        )
    return JudgmentItem(
        status=JudgmentStatus.PASS,
        reason="回撤與風險事件均處於可接受範圍。",
        evidence={
            "episode_max_dd_mean": episode_max_dd_mean,
            "episode_max_dd_std": episode_max_dd_std,
            "episode_cost_risk_dense_sum_mean": cost_risk_dense_mean,
            "termination_liq_count": liq_count,
            "termination_balance_insufficient_count": balance_count,
        },
    )


def _judge_p1_objective(
    metrics: dict[str, float | bool | int], p0_item: JudgmentItem
) -> JudgmentItem:
    """判定 P1 目標函數錯配。"""

    low_churn_positive = _safe_int(metrics.get("low_churn_positive_episode_count", 0))
    high_churn_negative = _safe_int(metrics.get("high_churn_negative_episode_count", 0))
    execution_rate_mean = _safe_float(metrics.get("execution_rate_mean", 0.0))
    trade_count_mean = _safe_float(metrics.get("episode_trade_count_mean", 0.0))
    trade_rate = _safe_float(metrics.get("trade_rate", 0.0))
    trades_per_day = _safe_float(metrics.get("trades_per_day", 0.0))
    total_fees_mean = _safe_float(metrics.get("total_fees_mean", 0.0))
    profit_mean = _safe_float(metrics.get("profit_mean", 0.0))

    if (
        p0_item.status == JudgmentStatus.FAIL
        and low_churn_positive > 0
        and high_churn_negative > 0
    ):
        return JudgmentItem(
            status=JudgmentStatus.FAIL,
            reason="存在低 churn 正報酬樣本，但整體仍偏向高 churn 失敗，顯示主目標可能在選錯行為。",
            evidence={
                "profit_mean": profit_mean,
                "trade_count_mean": trade_count_mean,
                "trade_rate": trade_rate,
                "trades_per_day": trades_per_day,
                "total_fees_mean": total_fees_mean,
                "execution_rate_mean": execution_rate_mean,
                "low_churn_positive_episode_count": low_churn_positive,
                "high_churn_negative_episode_count": high_churn_negative,
            },
        )
    if profit_mean > 0.0 and (
        trades_per_day >= TRADES_PER_DAY_WARN_THRESHOLD
        or trade_rate >= TRADE_RATE_WARN_THRESHOLD
        or execution_rate_mean >= 0.25
    ):
        return JudgmentItem(
            status=JudgmentStatus.WARN,
            reason="雖有獲利，但策略仍高度依賴頻繁成交，可能與上線目標不一致。",
            evidence={
                "profit_mean": profit_mean,
                "trade_count_mean": trade_count_mean,
                "trade_rate": trade_rate,
                "trades_per_day": trades_per_day,
                "trade_rate_warn_threshold": TRADE_RATE_WARN_THRESHOLD,
                "trades_per_day_warn_threshold": TRADES_PER_DAY_WARN_THRESHOLD,
                "execution_rate_mean": execution_rate_mean,
                "low_churn_positive_episode_count": low_churn_positive,
                "high_churn_negative_episode_count": high_churn_negative,
            },
        )
    return JudgmentItem(
        status=JudgmentStatus.PASS,
        reason="目前尚未觀察到明確的目標函數錯配證據。",
        evidence={
            "profit_mean": profit_mean,
            "trade_count_mean": trade_count_mean,
            "trade_rate": trade_rate,
            "trades_per_day": trades_per_day,
            "execution_rate_mean": execution_rate_mean,
            "low_churn_positive_episode_count": low_churn_positive,
            "high_churn_negative_episode_count": high_churn_negative,
        },
    )


def _judge_p2_observability(metrics: dict[str, float | bool | int]) -> JudgmentItem:
    """判定 P2 可觀測性不足。"""

    result_count = _safe_int(metrics.get("result_count", 0))
    positive_count = _safe_int(metrics.get("positive_profit_episode_count", 0))
    negative_count = _safe_int(metrics.get("negative_profit_episode_count", 0))
    low_churn_positive = _safe_int(metrics.get("low_churn_positive_episode_count", 0))
    high_churn_negative = _safe_int(metrics.get("high_churn_negative_episode_count", 0))

    if result_count <= 0:
        return JudgmentItem(
            status=JudgmentStatus.FAIL,
            reason="缺少 episode 級結果，無法支持 blocker 根因判讀。",
            evidence={"result_count": result_count},
        )
    if positive_count <= 0 or negative_count <= 0:
        return JudgmentItem(
            status=JudgmentStatus.FAIL,
            reason="結果分布缺少正負樣本對照，暫時無法區分主因。",
            evidence={
                "result_count": result_count,
                "positive_profit_episode_count": positive_count,
                "negative_profit_episode_count": negative_count,
            },
        )
    if low_churn_positive <= 0 or high_churn_negative <= 0:
        return JudgmentItem(
            status=JudgmentStatus.WARN,
            reason="已有初步方向，但低 churn 正例或高 churn 負例證據仍不足。",
            evidence={
                "positive_profit_episode_count": positive_count,
                "negative_profit_episode_count": negative_count,
                "low_churn_positive_episode_count": low_churn_positive,
                "high_churn_negative_episode_count": high_churn_negative,
            },
        )
    return JudgmentItem(
        status=JudgmentStatus.PASS,
        reason="現有欄位已足以支持主要 blocker 假設的初步判讀。",
        evidence={
            "result_count": result_count,
            "positive_profit_episode_count": positive_count,
            "negative_profit_episode_count": negative_count,
            "low_churn_positive_episode_count": low_churn_positive,
            "high_churn_negative_episode_count": high_churn_negative,
        },
    )


def _recommended_next_action(
    p0_item: JudgmentItem,
    p1_risk_item: JudgmentItem,
    p1_objective_item: JudgmentItem,
    p2_item: JudgmentItem,
) -> str:
    """依 blocker 狀態推導下一步動作。"""

    if p2_item.status == JudgmentStatus.FAIL:
        return "補強 diagnostics / decomposition，先提升根因可觀測性。"
    if p0_item.status == JudgmentStatus.FAIL:
        return "先分析 churn 與交易經濟性，不進入 reward redesign。"
    if p1_risk_item.status == JudgmentStatus.FAIL:
        return "先降低回撤與死亡事件，再討論 production readiness。"
    if p1_objective_item.status == JudgmentStatus.FAIL:
        return "優先檢查 objective mismatch，再評估是否重設 reward。"
    if p0_item.status == JudgmentStatus.WARN or p1_risk_item.status == JudgmentStatus.WARN:
        return "先做針對性的驗收與穩定性觀察，避免過早改動主線。"
    return "目前 blocker 已初步受控，可進入下一階段驗證。"


def judge_payload(payload: dict[str, Any]) -> JudgmentResult:
    """
    對 evaluator payload 執行 `P0 / P1 / P2` 判定。

    Args:
        payload: `PhaseABEvaluator.evaluate()` 回傳的 payload。

    Returns:
        結構化的判定結果。
    """

    metrics = extract_judgment_metrics(payload)
    p0_item = _judge_p0_economics(metrics)
    p1_risk_item = _judge_p1_risk(metrics)
    p1_objective_item = _judge_p1_objective(metrics, p0_item)
    p2_item = _judge_p2_observability(metrics)
    recommended_next_action = _recommended_next_action(
        p0_item=p0_item,
        p1_risk_item=p1_risk_item,
        p1_objective_item=p1_objective_item,
        p2_item=p2_item,
    )
    return JudgmentResult(
        p0_economics=p0_item,
        p1_risk=p1_risk_item,
        p1_objective=p1_objective_item,
        p2_observability=p2_item,
        recommended_next_action=recommended_next_action,
        should_block_reward_redesign=(p0_item.status == JudgmentStatus.FAIL),
        should_request_more_diagnostics=(p2_item.status == JudgmentStatus.FAIL),
    )


def format_judgment_summary(result: JudgmentResult) -> str:
    """
    將判定結果格式化為終端可讀摘要。

    Args:
        result: `judge_payload()` 回傳的判定結果。

    Returns:
        多行摘要字串。
    """

    lines = [
        (
            "[Judgment] "
            f"P0={result.p0_economics.status.value.upper()} "
            f"P1_risk={result.p1_risk.status.value.upper()} "
            f"P1_obj={result.p1_objective.status.value.upper()} "
            f"P2={result.p2_observability.status.value.upper()}"
        ),
        f"[Judgment] P0: {result.p0_economics.reason}",
        f"[Judgment] P1_risk: {result.p1_risk.reason}",
        f"[Judgment] P1_obj: {result.p1_objective.reason}",
        f"[Judgment] P2: {result.p2_observability.reason}",
        f"[Judgment] Recommended next action: {result.recommended_next_action}",
    ]
    return "\n".join(lines)
