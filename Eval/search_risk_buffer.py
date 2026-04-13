from __future__ import annotations

"""
Phase B：`lambda_risk`（事件型死亡懲罰）與 `lambda_buffer`（dense 緩衝懲罰）網格搜尋。

- 固定目前 Repo 的 observation / PhaseAB 環境參數（`--config-only` 重訓）。
- 先用 search seed 篩選；達標後以 confirm seed 驗證。
- 若無完全達標，輸出違反條件最少、經濟性較佳的候選。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _risk_buffer_grid(*, full: bool = False) -> list[dict[str, float]]:
    """
    預設為較疏網格以縮短搜尋時間；`full=True` 時為 6×6 完整掃描。

    排序：先掃較高 lambda_buffer（dense 緩衝），再掃 lambda_risk。
    """
    if full:
        risks = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0]
        buffers = [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01]
    else:
        risks = [1.0, 2.0, 5.0, 10.0]
        buffers = [0.0001, 0.0003, 0.001, 0.003]
    out: list[dict[str, float]] = []
    for lb in buffers:
        for lr in risks:
            out.append({"lambda_risk": float(lr), "lambda_buffer": float(lb)})
    return out


def _is_success(
    row: pd.Series,
    *,
    max_dd: float,
    max_fees_profit_ratio: float,
    min_eval_trades: float,
) -> bool:
    if float(row["trade_count_mean"]) < float(min_eval_trades):
        return False
    return (
        float(row["profit_mean"]) > 0.0
        and float(row["profit_per_trade_mean"]) > 0.0
        and float(row["fees_profit_ratio"]) < float(max_fees_profit_ratio)
        and float(row["episode_max_dd_mean"]) < float(max_dd)
        and float(row["termination_balance_insufficient_count"]) <= 0.0
        and float(row["termination_liq_count"]) <= 0.0
    )


def _score_row(
    row: pd.Series,
    *,
    max_dd: float,
    max_fees_profit_ratio: float,
    min_eval_trades: float,
) -> tuple[float, float, float, float, float, float, float]:
    """
    越小越好：
    1) 條件違反數
    2) 死亡／強平次數（餘額不足 + 強平）
    3) 評估成交筆數不足（短訓常見假陰性）
    4) max_dd 超標
    5) fees_profit_ratio 超標
    6) 負 PPT
    7) 負 profit_mean
    """
    violations = 0.0
    trade_shortfall = max(0.0, float(min_eval_trades) - float(row["trade_count_mean"]))
    death_ct = float(row["termination_balance_insufficient_count"]) + float(row["termination_liq_count"])
    dd_penalty = max(0.0, float(row["episode_max_dd_mean"]) - float(max_dd))
    fees_penalty = max(0.0, float(row["fees_profit_ratio"]) - float(max_fees_profit_ratio))
    ppt_penalty = max(0.0, -float(row["profit_per_trade_mean"]))
    profit_penalty = max(0.0, -float(row["profit_mean"]))
    for cond in (
        float(row["trade_count_mean"]) >= float(min_eval_trades),
        float(row["profit_mean"]) > 0.0,
        float(row["profit_per_trade_mean"]) > 0.0,
        float(row["fees_profit_ratio"]) < float(max_fees_profit_ratio),
        float(row["episode_max_dd_mean"]) < float(max_dd),
        float(row["termination_balance_insufficient_count"]) <= 0.0,
        float(row["termination_liq_count"]) <= 0.0,
    ):
        if not cond:
            violations += 1.0
    return (violations, death_ct, trade_shortfall, dd_penalty, fees_penalty, ppt_penalty, profit_penalty)


def _run_one_trial(
    *,
    repo_root: Path,
    timesteps: int,
    n_envs: int,
    eval_episodes: int,
    holdout_months: int,
    seed: int,
    out_dir: Path,
    params: dict[str, float],
    lambda_turnover: float,
    reward_scale: float,
) -> pd.Series:
    cmd = [
        sys.executable,
        "-m",
        "Eval.feature_candidate_retrain",
        "--config-only",
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
        "--lambda-risk",
        str(float(params["lambda_risk"])),
        "--lambda-buffer",
        str(float(params["lambda_buffer"])),
        "--lambda-turnover",
        str(float(lambda_turnover)),
        "--reward-scale",
        str(float(reward_scale)),
        "--out-dir",
        str(out_dir),
    ]
    for attempt in range(3):
        try:
            subprocess.run(cmd, cwd=str(repo_root), check=True)
            break
        except subprocess.CalledProcessError as exc:
            if attempt < 2:
                print(f"[RiskSearch] retry trial due to subprocess failure: {exc.returncode}", flush=True)
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
    parser = argparse.ArgumentParser(description="搜尋 Phase B lambda_risk / lambda_buffer")
    parser.add_argument("--timesteps", type=int, default=12_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--holdout-months", type=int, default=1)
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--confirm-seed", type=int, default=43)
    parser.add_argument("--max-dd", type=float, default=0.35)
    parser.add_argument("--max-fees-profit-ratio", type=float, default=1.0)
    parser.add_argument(
        "--min-eval-trades",
        type=float,
        default=12.0,
        help="成功條件：eval 的 episode_trade_count.mean 至少為此（避免短訓零成交誤判）",
    )
    parser.add_argument("--lambda-turnover", type=float, default=0.0002)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default="logs/risk_buffer_search")
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="使用 6×6 完整網格（較慢）；預設為 4×4 疏網格",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_row: pd.Series | None = None
    best_score: tuple[float, float, float, float, float, float, float] | None = None
    search_rows: list[dict[str, Any]] = []

    grid = _risk_buffer_grid(full=bool(args.full_grid))
    for idx, params in enumerate(grid, start=1):
        trial_dir = out_dir / f"trial_{idx:03d}_lr{params['lambda_risk']}_lb{params['lambda_buffer']}_s{int(args.search_seed)}"
        print(f"[RiskSearch] trial={idx} params={params}", flush=True)
        try:
            row = _run_one_trial(
                repo_root=repo_root,
                timesteps=int(args.timesteps),
                n_envs=int(args.n_envs),
                eval_episodes=int(args.eval_episodes),
                holdout_months=int(args.holdout_months),
                seed=int(args.search_seed),
                out_dir=trial_dir,
                params=params,
                lambda_turnover=float(args.lambda_turnover),
                reward_scale=float(args.reward_scale),
            )
        except subprocess.CalledProcessError as exc:
            failed_row = {
                **params,
                "seed": int(args.search_seed),
                "status": f"subprocess_failed_{exc.returncode}",
            }
            search_rows.append(failed_row)
            print(f"[RiskSearch] skip trial={idx} subprocess failure {exc.returncode}", flush=True)
            continue
        score = _score_row(
            row,
            max_dd=float(args.max_dd),
            max_fees_profit_ratio=float(args.max_fees_profit_ratio),
            min_eval_trades=float(args.min_eval_trades),
        )
        row_dict = row.to_dict()
        row_dict["search_score"] = json.dumps(score)
        search_rows.append(row_dict)
        if best_score is None or score < best_score:
            best_row = row
            best_score = score

        if _is_success(
            row,
            max_dd=float(args.max_dd),
            max_fees_profit_ratio=float(args.max_fees_profit_ratio),
            min_eval_trades=float(args.min_eval_trades),
        ):
            confirm_dir = out_dir / f"trial_{idx:03d}_lr{params['lambda_risk']}_lb{params['lambda_buffer']}_s{int(args.confirm_seed)}"
            confirm_row = _run_one_trial(
                repo_root=repo_root,
                timesteps=int(args.timesteps),
                n_envs=int(args.n_envs),
                eval_episodes=int(args.eval_episodes),
                holdout_months=int(args.holdout_months),
                seed=int(args.confirm_seed),
                out_dir=confirm_dir,
                params=params,
                lambda_turnover=float(args.lambda_turnover),
                reward_scale=float(args.reward_scale),
            )
            confirm_ok = _is_success(
                confirm_row,
                max_dd=float(args.max_dd),
                max_fees_profit_ratio=float(args.max_fees_profit_ratio),
                min_eval_trades=float(args.min_eval_trades),
            )
            confirm_dict = confirm_row.to_dict()
            confirm_dict["search_score"] = json.dumps(
                _score_row(
                    confirm_row,
                    max_dd=float(args.max_dd),
                    max_fees_profit_ratio=float(args.max_fees_profit_ratio),
                    min_eval_trades=float(args.min_eval_trades),
                )
            )
            search_rows.append(confirm_dict)
            if confirm_ok:
                result = {
                    "status": "success_confirmed",
                    "search_seed_result": row.to_dict(),
                    "confirm_seed_result": confirm_row.to_dict(),
                    "params": params,
                    "criteria": {
                        "max_dd": float(args.max_dd),
                        "max_fees_profit_ratio": float(args.max_fees_profit_ratio),
                        "min_eval_trades": float(args.min_eval_trades),
                    },
                }
                pd.DataFrame(search_rows).to_csv(out_dir / "search_history.csv", index=False)
                (out_dir / "best_result.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print("[RiskSearch] success_confirmed", flush=True)
                return
            print("[RiskSearch] search seed passed but confirm seed failed; continue", flush=True)

    result = {
        "status": "no_success_found",
        "best_result": best_row.to_dict() if best_row is not None else {},
        "best_score": list(best_score) if best_score is not None else [],
        "criteria": {
            "max_dd": float(args.max_dd),
            "max_fees_profit_ratio": float(args.max_fees_profit_ratio),
            "min_eval_trades": float(args.min_eval_trades),
        },
    }
    pd.DataFrame(search_rows).to_csv(out_dir / "search_history.csv", index=False)
    (out_dir / "best_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[RiskSearch] no_success_found (see best_result.json)", flush=True)


if __name__ == "__main__":
    main()
