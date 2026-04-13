from __future__ import annotations

"""
低成本 observation 候選重訓驗證器。

用途：
- 以相同 seed / split / cost setting 對 baseline 與候選 observation 配置做短版重訓。
- 驗證離線與 mask 分析挑出的特徵組合，是否在 RL holdout 上方向一致。
"""

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Eval.phase_ab_env_config import PhaseABEnvConfig
from Eval.feature_selection_utils import temporary_observation_config
from Eval.phase_ab_evaluator import PhaseABEvaluator, build_single_env_builder
from Train.run_sac_phase_ab import get_phase_ab_env_kwargs, make_env


def _parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _load_candidates(candidate_json: str, candidate_names: list[str]) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(candidate_json).read_text(encoding="utf-8"))
    candidate_sets = payload.get("candidate_sets", {}) if isinstance(payload, dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for name in candidate_names:
        item = candidate_sets.get(name)
        if isinstance(item, dict):
            out[name] = item
    if not out:
        raise ValueError("No candidate sets loaded from candidate JSON.")
    return out


def _build_train_env(
    phase: str,
    env_kwargs_train: dict[str, Any],
    n_envs: int,
    seed: int,
    *,
    lambda_risk: float,
    lambda_buffer: float,
    lambda_turnover: float,
    reward_scale: float,
) -> VecEnv:
    env = DummyVecEnv(
        [
            make_env(
                phase=phase,
                lambda_risk=float(lambda_risk),
                lambda_buffer=float(lambda_buffer),
                lambda_turnover=float(lambda_turnover),
                reward_scale=float(reward_scale),
                anneal_steps=0,
                seed=int(seed + idx),
                **env_kwargs_train,
            )
            for idx in range(int(n_envs))
        ]
    )
    return VecMonitor(env)


def _build_model(vec_env: VecEnv, device: str) -> SAC:
    return SAC(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=300_000,
        train_freq=1,
        gradient_steps=1,
        verbose=0,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="低成本 observation 候選重訓驗證")
    parser.add_argument(
        "--candidate-json",
        type=str,
        default="",
        help="candidate_feature_sets.json 路徑；與 --config-only 擇一：後者不需此檔",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="不覆寫 observation，直接使用 Env.config.Config 與 PhaseABEnvConfig 當前設定重訓",
    )
    parser.add_argument(
        "--candidate-names",
        type=str,
        default="baseline_current,ultra_minimal,minimal_signal,minimal_plus_entry_filter",
    )
    parser.add_argument("--phase", choices=("A", "B"), default="B")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seeds", type=str, default="42,43")
    parser.add_argument("--holdout-months", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-trade-entry-threshold", type=float, default=None)
    parser.add_argument("--no-trade-exit-threshold", type=float, default=None)
    parser.add_argument("--min-position-change", type=float, default=None)
    parser.add_argument("--max-step-pos-change-pct", type=float, default=None)
    parser.add_argument("--lambda-risk", type=float, default=0.05)
    parser.add_argument("--lambda-buffer", type=float, default=0.1)
    parser.add_argument("--lambda-turnover", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--out-dir", type=str, default="logs/feature_candidate_retrain")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.config_only):
        candidate_names = ["config_default"]
        candidates: dict[str, dict[str, Any]] = {"config_default": {}}
    else:
        if not str(args.candidate_json or "").strip():
            raise SystemExit("需要 --candidate-json，或使用 --config-only")
        candidate_names = _parse_csv(args.candidate_names)
        candidates = _load_candidates(str(args.candidate_json), candidate_names)

    seeds = _parse_int_csv(args.seeds)

    rows: list[dict[str, Any]] = []

    for candidate_name in candidate_names:
        candidate = candidates.get(candidate_name)
        if candidate is None:
            continue
        for seed in seeds:
            print(f"[Retrain] candidate={candidate_name} seed={seed}", flush=True)
            obs_ctx: contextlib.AbstractContextManager[Any] = (
                contextlib.nullcontext()
                if bool(args.config_only)
                else temporary_observation_config(candidate)
            )
            with obs_ctx:
                env_kwargs_train = get_phase_ab_env_kwargs(
                    data_split_enabled=True,
                    holdout_months=int(args.holdout_months),
                    data_mode="train",
                )
                env_kwargs_eval = get_phase_ab_env_kwargs(
                    data_split_enabled=True,
                    holdout_months=int(args.holdout_months),
                    data_mode="eval",
                )
                if args.no_trade_entry_threshold is not None:
                    env_kwargs_train["no_trade_entry_threshold"] = float(args.no_trade_entry_threshold)
                    env_kwargs_eval["no_trade_entry_threshold"] = float(args.no_trade_entry_threshold)
                if args.no_trade_exit_threshold is not None:
                    env_kwargs_train["no_trade_exit_threshold"] = float(args.no_trade_exit_threshold)
                    env_kwargs_eval["no_trade_exit_threshold"] = float(args.no_trade_exit_threshold)
                if args.min_position_change is not None:
                    env_kwargs_train["min_position_change"] = float(args.min_position_change)
                    env_kwargs_eval["min_position_change"] = float(args.min_position_change)
                if args.max_step_pos_change_pct is not None:
                    env_kwargs_train["max_step_pos_change_pct"] = float(args.max_step_pos_change_pct)
                    env_kwargs_eval["max_step_pos_change_pct"] = float(args.max_step_pos_change_pct)
                vec_env = _build_train_env(
                    phase=args.phase,
                    env_kwargs_train=env_kwargs_train,
                    n_envs=int(args.n_envs),
                    seed=int(seed),
                    lambda_risk=float(args.lambda_risk),
                    lambda_buffer=float(args.lambda_buffer),
                    lambda_turnover=float(args.lambda_turnover),
                    reward_scale=float(args.reward_scale),
                )
                try:
                    model = _build_model(vec_env, args.device)
                    model.learn(total_timesteps=int(args.timesteps), progress_bar=False)
                    eval_thunk = make_env(
                        phase=args.phase,
                        lambda_risk=float(args.lambda_risk),
                        lambda_buffer=float(args.lambda_buffer),
                        lambda_turnover=float(args.lambda_turnover),
                        reward_scale=float(args.reward_scale),
                        anneal_steps=0,
                        seed=None,
                        **env_kwargs_eval,
                    )
                    evaluator = PhaseABEvaluator(
                        env_builder=build_single_env_builder(eval_thunk),
                        n_episodes=int(args.eval_episodes),
                        deterministic=True,
                        seed=int(seed),
                    )
                    payload = evaluator.evaluate(
                        model=model,
                        reason="feature_candidate_retrain",
                        step=int(args.timesteps),
                        print_result=False,
                        print_per_episode=False,
                    )
                    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
                    term_counts = payload.get("termination_reason_counts") or {}
                    rows.append(
                        {
                            "candidate": candidate_name,
                            "seed": int(seed),
                            "lambda_risk": float(args.lambda_risk),
                            "lambda_buffer": float(args.lambda_buffer),
                            "window_size": int(candidate.get("WINDOW_SIZE", PhaseABEnvConfig.WINDOW_SIZE)),
                            "window_size_1d": int(candidate.get("WINDOW_SIZE_1D", PhaseABEnvConfig.WINDOW_SIZE_1D)),
                            "no_trade_entry_threshold": float(env_kwargs_train.get("no_trade_entry_threshold", 0.0)),
                            "no_trade_exit_threshold": float(env_kwargs_train.get("no_trade_exit_threshold", 0.0)),
                            "min_position_change": float(env_kwargs_train.get("min_position_change", 0.0)),
                            "max_step_pos_change_pct": float(env_kwargs_train.get("max_step_pos_change_pct", 0.0)),
                            "lambda_turnover": float(args.lambda_turnover),
                            "profit_mean": float(((summary.get("profit") or {}).get("mean")) or 0.0),
                            "profit_per_trade_mean": float(((summary.get("profit_per_trade") or {}).get("mean")) or 0.0),
                            "trade_count_mean": float(((summary.get("episode_trade_count") or {}).get("mean")) or 0.0),
                            "episode_steps_mean": float(((summary.get("episode_steps") or {}).get("mean")) or 0.0),
                            "total_fees_mean": float(((summary.get("total_fees") or {}).get("mean")) or 0.0),
                            "episode_max_dd_mean": float(((summary.get("episode_max_dd") or {}).get("mean")) or 0.0),
                            "termination_balance_insufficient_count": int(
                                term_counts.get("balance_insufficient", 0) or 0
                            ),
                            "termination_liq_count": int(term_counts.get("liq_triggered", 0) or 0),
                        }
                    )
                finally:
                    vec_env.close()

    df_runs = pd.DataFrame(rows)
    df_runs["fees_profit_ratio"] = df_runs["total_fees_mean"] / df_runs["profit_mean"].abs().clip(lower=1e-8)
    df_runs["trade_rate"] = df_runs["trade_count_mean"] / df_runs["episode_steps_mean"].clip(lower=1e-8)
    df_runs["trades_per_day"] = df_runs["trade_count_mean"] / (df_runs["episode_steps_mean"] / 288.0).clip(lower=1e-8)
    group_cols = ["candidate"]
    for col in ("lambda_risk", "lambda_buffer"):
        if col in df_runs.columns and df_runs[col].nunique(dropna=False) > 1:
            group_cols.append(col)
    df_summary = (
        df_runs.groupby(group_cols, as_index=False)
        .agg(
            profit_mean=("profit_mean", "mean"),
            profit_per_trade_mean=("profit_per_trade_mean", "mean"),
            trade_count_mean=("trade_count_mean", "mean"),
            episode_steps_mean=("episode_steps_mean", "mean"),
            total_fees_mean=("total_fees_mean", "mean"),
            fees_profit_ratio=("fees_profit_ratio", "mean"),
            trade_rate=("trade_rate", "mean"),
            trades_per_day=("trades_per_day", "mean"),
            episode_max_dd_mean=("episode_max_dd_mean", "mean"),
            termination_balance_insufficient_count=("termination_balance_insufficient_count", "mean"),
            termination_liq_count=("termination_liq_count", "mean"),
        )
        .sort_values(by=["profit_per_trade_mean", "fees_profit_ratio"], ascending=[False, True])
    )

    df_runs.to_csv(out_dir / "retrain_runs.csv", index=False)
    df_summary.to_csv(out_dir / "retrain_summary.csv", index=False)
    print(f"[Saved] {out_dir / 'retrain_runs.csv'}", flush=True)
    print(f"[Saved] {out_dir / 'retrain_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
