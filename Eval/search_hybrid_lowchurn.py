from __future__ import annotations

"""
針對 hybrid_minimal_signal 的低換手參數搜尋器。

策略：
- 固定特徵組合，不碰 reward / risk 權重。
- 先用 seed=42 搜尋；首次達標後再用 seed=43 驗證。
- 若無完全達標組合，輸出目前最佳候選（依條件違反程度與經濟性排序）。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _candidate_grid() -> list[dict[str, float]]:
    """依經驗排序的低換手參數候選，從最溫和到較強。"""
    return [
        {"no_trade_entry_threshold": 0.24, "no_trade_exit_threshold": 0.11, "min_position_change": 0.22, "max_step_pos_change_pct": 0.35, "lambda_turnover": 0.0001},
        {"no_trade_entry_threshold": 0.26, "no_trade_exit_threshold": 0.12, "min_position_change": 0.22, "max_step_pos_change_pct": 0.35, "lambda_turnover": 0.0001},
        {"no_trade_entry_threshold": 0.28, "no_trade_exit_threshold": 0.14, "min_position_change": 0.25, "max_step_pos_change_pct": 0.30, "lambda_turnover": 0.0002},
        {"no_trade_entry_threshold": 0.24, "no_trade_exit_threshold": 0.12, "min_position_change": 0.25, "max_step_pos_change_pct": 0.30, "lambda_turnover": 0.0002},
        {"no_trade_entry_threshold": 0.22, "no_trade_exit_threshold": 0.11, "min_position_change": 0.22, "max_step_pos_change_pct": 0.35, "lambda_turnover": 0.0002},
        {"no_trade_entry_threshold": 0.23, "no_trade_exit_threshold": 0.11, "min_position_change": 0.22, "max_step_pos_change_pct": 0.35, "lambda_turnover": 0.0002},
        {"no_trade_entry_threshold": 0.24, "no_trade_exit_threshold": 0.11, "min_position_change": 0.22, "max_step_pos_change_pct": 0.33, "lambda_turnover": 0.0002},
        {"no_trade_entry_threshold": 0.25, "no_trade_exit_threshold": 0.12, "min_position_change": 0.22, "max_step_pos_change_pct": 0.35, "lambda_turnover": 0.0002},
        {"no_trade_entry_threshold": 0.22, "no_trade_exit_threshold": 0.11, "min_position_change": 0.25, "max_step_pos_change_pct": 0.30, "lambda_turnover": 0.0003},
        {"no_trade_entry_threshold": 0.26, "no_trade_exit_threshold": 0.13, "min_position_change": 0.25, "max_step_pos_change_pct": 0.35, "lambda_turnover": 0.0002},
        {"no_trade_entry_threshold": 0.24, "no_trade_exit_threshold": 0.12, "min_position_change": 0.20, "max_step_pos_change_pct": 0.35, "lambda_turnover": 0.0002},
    ]


def _is_success(row: pd.Series, *, max_trades_per_day: float, max_dd: float) -> bool:
    return (
        float(row["profit_mean"]) > 0.0
        and float(row["profit_per_trade_mean"]) > 0.0
        and float(row["fees_profit_ratio"]) < 1.0
        and float(row["trades_per_day"]) < float(max_trades_per_day)
        and float(row["episode_max_dd_mean"]) < float(max_dd)
        and float(row["termination_balance_insufficient_count"]) <= 0.0
    )


def _score_row(row: pd.Series, *, max_trades_per_day: float, max_dd: float) -> tuple[float, float, float, float, float]:
    """
    越小越好：
    1) 條件違反數
    2) fees_profit_ratio 超標幅度
    3) trades_per_day 超標幅度
    4) 負 profit_per_trade 的懲罰
    5) 負 profit_mean 的懲罰
    """
    violations = 0.0
    fees_penalty = max(0.0, float(row["fees_profit_ratio"]) - 1.0)
    trades_penalty = max(0.0, float(row["trades_per_day"]) - float(max_trades_per_day))
    ppt_penalty = max(0.0, -float(row["profit_per_trade_mean"]))
    profit_penalty = max(0.0, -float(row["profit_mean"]))
    dd_penalty = max(0.0, float(row["episode_max_dd_mean"]) - float(max_dd))
    for cond in (
        float(row["profit_mean"]) > 0.0,
        float(row["profit_per_trade_mean"]) > 0.0,
        float(row["fees_profit_ratio"]) < 1.0,
        float(row["trades_per_day"]) < float(max_trades_per_day),
        float(row["episode_max_dd_mean"]) < float(max_dd),
        float(row["termination_balance_insufficient_count"]) <= 0.0,
    ):
        if not cond:
            violations += 1.0
    return (violations, fees_penalty, trades_penalty, ppt_penalty + dd_penalty, profit_penalty)


def _run_one_trial(
    *,
    repo_root: Path,
    candidate_json: str,
    timesteps: int,
    n_envs: int,
    eval_episodes: int,
    holdout_months: int,
    seed: int,
    out_dir: Path,
    params: dict[str, float],
) -> pd.Series:
    cmd = [
        sys.executable,
        "-m",
        "Eval.feature_candidate_retrain",
        "--candidate-json",
        candidate_json,
        "--candidate-names",
        "hybrid_minimal_signal",
        "--phase",
        "B",
        "--timesteps",
        str(int(timesteps)),
        "--n-envs",
        str(int(n_envs)),
        "--eval-episodes",
        str(int(eval_episodes)),
        "--seeds",
        str(int(seed)),
        "--holdout-months",
        str(int(holdout_months)),
        "--no-trade-entry-threshold",
        str(float(params["no_trade_entry_threshold"])),
        "--no-trade-exit-threshold",
        str(float(params["no_trade_exit_threshold"])),
        "--min-position-change",
        str(float(params["min_position_change"])),
        "--max-step-pos-change-pct",
        str(float(params["max_step_pos_change_pct"])),
        "--lambda-turnover",
        str(float(params["lambda_turnover"])),
        "--out-dir",
        str(out_dir),
    ]
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(2):
        try:
            subprocess.run(cmd, cwd=str(repo_root), check=True)
            last_error = None
            break
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt == 0:
                print(f"[Search] retry trial due to subprocess failure: {exc.returncode}", flush=True)
                continue
            raise
    summary = pd.read_csv(out_dir / "retrain_summary.csv")
    if summary.empty:
        raise ValueError(f"No summary rows found in {out_dir}")
    row = summary.iloc[0].copy()
    for key, value in params.items():
        row[key] = float(value)
    row["seed"] = int(seed)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="搜尋 hybrid_minimal_signal 的低換手最優參數")
    parser.add_argument("--candidate-json", type=str, default="logs/feature_hybrid_candidates_run1.json")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--holdout-months", type=int, default=1)
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--confirm-seed", type=int, default=43)
    parser.add_argument("--max-trades-per-day", type=float, default=80.0)
    parser.add_argument("--max-dd", type=float, default=0.25)
    parser.add_argument("--out-dir", type=str, default="logs/hybrid_lowchurn_search")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_row: pd.Series | None = None
    best_score: tuple[float, float, float, float, float] | None = None
    search_rows: list[dict[str, Any]] = []

    for idx, params in enumerate(_candidate_grid(), start=1):
        trial_dir = out_dir / f"trial_{idx:02d}_seed{int(args.search_seed)}"
        print(f"[Search] trial={idx} params={params}", flush=True)
        try:
            row = _run_one_trial(
                repo_root=repo_root,
                candidate_json=str(args.candidate_json),
                timesteps=int(args.timesteps),
                n_envs=int(args.n_envs),
                eval_episodes=int(args.eval_episodes),
                holdout_months=int(args.holdout_months),
                seed=int(args.search_seed),
                out_dir=trial_dir,
                params=params,
            )
        except subprocess.CalledProcessError as exc:
            failed_row = {**params, "seed": int(args.search_seed), "status": f"subprocess_failed_{exc.returncode}"}
            search_rows.append(failed_row)
            print(f"[Search] skip trial={idx} due to subprocess failure {exc.returncode}", flush=True)
            continue
        score = _score_row(row, max_trades_per_day=float(args.max_trades_per_day), max_dd=float(args.max_dd))
        row_dict = row.to_dict()
        row_dict["search_score"] = json.dumps(score)
        search_rows.append(row_dict)
        if best_row is None or score < best_score:
            best_row = row
            best_score = score

        if _is_success(row, max_trades_per_day=float(args.max_trades_per_day), max_dd=float(args.max_dd)):
            confirm_dir = out_dir / f"trial_{idx:02d}_seed{int(args.confirm_seed)}"
            confirm_row = _run_one_trial(
                repo_root=repo_root,
                candidate_json=str(args.candidate_json),
                timesteps=int(args.timesteps),
                n_envs=int(args.n_envs),
                eval_episodes=int(args.eval_episodes),
                holdout_months=int(args.holdout_months),
                seed=int(args.confirm_seed),
                out_dir=confirm_dir,
                params=params,
            )
            confirm_ok = _is_success(
                confirm_row,
                max_trades_per_day=float(args.max_trades_per_day),
                max_dd=float(args.max_dd),
            )
            confirm_dict = confirm_row.to_dict()
            confirm_dict["search_score"] = json.dumps(
                _score_row(confirm_row, max_trades_per_day=float(args.max_trades_per_day), max_dd=float(args.max_dd))
            )
            search_rows.append(confirm_dict)
            if confirm_ok:
                result = {
                    "status": "success_confirmed",
                    "search_seed_result": row.to_dict(),
                    "confirm_seed_result": confirm_row.to_dict(),
                    "params": params,
                }
                pd.DataFrame(search_rows).to_csv(out_dir / "search_history.csv", index=False)
                (out_dir / "best_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                print("[Search] success status=success_confirmed", flush=True)
                return
            print("[Search] seed42 passed but seed43 failed; continue searching", flush=True)

    result = {
        "status": "no_success_found",
        "best_result": best_row.to_dict() if best_row is not None else {},
        "best_score": list(best_score) if best_score is not None else [],
    }
    pd.DataFrame(search_rows).to_csv(out_dir / "search_history.csv", index=False)
    (out_dir / "best_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[Search] no_success_found", flush=True)


if __name__ == "__main__":
    main()
