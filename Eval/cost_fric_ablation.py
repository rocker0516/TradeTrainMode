from __future__ import annotations

"""
cost_fric 三輪實驗報告比較工具。

用途：
- 讀取多個 eval JSON（由 `PhaseABEvaluator` 產生）
- 輸出主 KPI（profit_per_trade）與 guardrail 結果
- 挑出「通過 guardrail 且 profit_per_trade 最佳」的候選
"""

import argparse
import json
from pathlib import Path
from typing import Any


def _safe_mean(payload: dict[str, Any], key: str) -> float:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    item = summary.get(key, {}) if isinstance(summary, dict) else {}
    if not isinstance(item, dict):
        return 0.0
    try:
        return float(item.get("mean", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _guardrail_status(payload: dict[str, Any]) -> str:
    guardrails = payload.get("guardrails", {}) if isinstance(payload, dict) else {}
    if not isinstance(guardrails, dict) or not guardrails:
        return "unknown"
    return str(guardrails.get("status", "unknown"))


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="比較 cost_fric ablation eval JSON")
    parser.add_argument("--reports", nargs="+", required=True, help="eval JSON 路徑列表")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for raw_path in args.reports:
        p = Path(raw_path)
        payload = _load(p)
        row = {
            "report": str(p),
            "profit_per_trade": _safe_mean(payload, "profit_per_trade"),
            "profit": _safe_mean(payload, "profit"),
            "max_dd": _safe_mean(payload, "episode_max_dd"),
            "trade_count": _safe_mean(payload, "episode_trade_count"),
            "guardrail": _guardrail_status(payload),
        }
        rows.append(row)

    if not rows:
        print("[Ablation] no reports")
        return

    print("[Ablation] report | guardrail | profit_per_trade | profit | max_dd | trade_count")
    for row in rows:
        print(
            f"[Ablation] {row['report']} | {row['guardrail']} | "
            f"{row['profit_per_trade']:.6f} | {row['profit']:.6f} | "
            f"{row['max_dd']:.6f} | {row['trade_count']:.6f}"
        )

    passed = [r for r in rows if str(r["guardrail"]).lower() == "pass"]
    candidates = passed if passed else rows
    best = max(candidates, key=lambda x: float(x["profit_per_trade"]))
    print(
        f"[Ablation] best={best['report']} "
        f"(guardrail={best['guardrail']}, profit_per_trade={best['profit_per_trade']:.6f})"
    )


if __name__ == "__main__":
    main()
