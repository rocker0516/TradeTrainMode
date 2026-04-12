from __future__ import annotations

"""
已訓練模型的 observation mask / ablation 評估。

用途：
- 對市場特徵群組做整組遮罩。
- 對候選特徵子集做「保持 shape 不變、未選欄位清零」的近似驗證。
- 輸出 paired delta 指標，檢查是 alpha 還是噪音交易來源。
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Eval.feature_selection_utils import bootstrap_mean_ci, tag_redundancy
from Train.run_sac_phase_ab import get_phase_ab_env_kwargs, make_env


OBS_TO_CONFIG_FIELD: dict[str, str] = {
    "price_seq_target": "OBS_PRICE_SEQ_TARGET_COLS",
    "price_seq_others": "OBS_PRICE_SEQ_OTHERS_COLS",
    "price_seq_1d_target": "OBS_PRICE_SEQ_1D_TARGET_COLS",
    "price_seq_1d_others": "OBS_PRICE_SEQ_1D_OTHERS_COLS",
}

DEFAULT_MASK_GROUPS: tuple[str, ...] = (
    "price_seq_target",
    "price_seq_others",
    "price_seq_1d_target",
    "price_seq_1d_others",
    "gate_flags",
    "regime_score",
)


@dataclass(frozen=True)
class MaskScenario:
    """單一遮罩情境。"""

    name: str
    zero_keys: tuple[str, ...] = ()
    keep_cols_by_key: dict[str, tuple[str, ...]] | None = None


class ObservationMaskWrapper(gym.ObservationWrapper):
    """保持 observation shape 不變，但將指定 key 或欄位清零。"""

    def __init__(
        self,
        env: gym.Env,
        *,
        zero_keys: tuple[str, ...] = (),
        keep_cols_by_key: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        super().__init__(env)
        self.zero_keys = tuple(zero_keys)
        self.keep_cols_by_key = keep_cols_by_key or {}
        self._visible_cols = self._build_visible_cols()

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        out = dict(observation)
        for key in self.zero_keys:
            if key in out and isinstance(out[key], np.ndarray):
                out[key] = np.zeros_like(out[key])

        for key, keep_cols in self.keep_cols_by_key.items():
            if key not in out or not isinstance(out[key], np.ndarray):
                continue
            keep_set = set(keep_cols)
            visible_cols = list(self._visible_cols.get(key, ()))
            if not visible_cols:
                continue
            arr = np.array(out[key], copy=True)
            keep_mask = np.array(
                [
                    (col in keep_set) or (tag_redundancy(col).normalized_name in keep_set)
                    for col in visible_cols
                ],
                dtype=bool,
            )
            if arr.ndim == 2 and arr.shape[1] == len(keep_mask):
                arr[:, ~keep_mask] = 0.0
            elif arr.ndim == 1 and arr.shape[0] == len(keep_mask):
                arr[~keep_mask] = 0.0
            out[key] = arr
        return out

    def _build_visible_cols(self) -> dict[str, tuple[str, ...]]:
        base = _unwrap_base_env(self.env)
        observer = getattr(base, "observer", None)
        market_data = getattr(base, "market_data", None)
        if observer is None or market_data is None:
            return {}
        return {
            "price_seq_target": tuple(market_data.cols_5m_target[i] for i in observer._obs_price_seq_target_idx),
            "price_seq_others": tuple(market_data.cols_5m_others[i] for i in observer._obs_price_seq_others_idx),
            "price_seq_1d_target": tuple(market_data.cols_1d_target[i] for i in observer._obs_price_seq_1d_target_idx),
            "price_seq_1d_others": tuple(market_data.cols_1d_others[i] for i in observer._obs_price_seq_1d_others_idx),
        }


def _unwrap_base_env(env: gym.Env) -> gym.Env:
    current = env
    while hasattr(current, "env"):
        current = current.env
    return current


def _build_vec_env(thunk: Callable[[], gym.Env]) -> Callable[[], VecEnv]:
    def builder() -> VecEnv:
        env = DummyVecEnv([thunk])
        return VecMonitor(env)

    return builder


def _parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _load_candidate_scenarios(candidate_json: str) -> list[MaskScenario]:
    path = Path(candidate_json)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidate_sets", {}) if isinstance(payload, dict) else {}
    scenarios: list[MaskScenario] = []
    for name, item in candidates.items():
        keep_cols_by_key: dict[str, tuple[str, ...]] = {}
        if not isinstance(item, dict):
            continue
        for obs_key, field_name in OBS_TO_CONFIG_FIELD.items():
            cols = item.get(field_name, ())
            if cols:
                keep_cols_by_key[obs_key] = tuple(str(c) for c in cols)
        if keep_cols_by_key:
            scenarios.append(MaskScenario(name=f"candidate__{name}", keep_cols_by_key=keep_cols_by_key))
    return scenarios


def _make_scenarios(mask_groups: list[str], candidate_json: str | None) -> list[MaskScenario]:
    scenarios = [MaskScenario(name="baseline")]
    for key in mask_groups:
        scenarios.append(MaskScenario(name=f"mask__{key}", zero_keys=(key,)))
    if candidate_json:
        scenarios.extend(_load_candidate_scenarios(candidate_json))
    return scenarios


def _evaluate_scenario(
    model: SAC,
    env_builder: Callable[[], VecEnv],
    *,
    n_episodes: int,
    deterministic: bool,
    seed: int,
) -> list[dict[str, Any]]:
    env = env_builder()
    try:
        rows: list[dict[str, Any]] = []
        for episode_idx in range(int(n_episodes)):
            try:
                env.seed(int(seed + episode_idx))
            except Exception:
                pass
            obs = env.reset()
            done = False
            row: dict[str, Any] = {
                "episode_index": int(episode_idx),
                "profit": 0.0,
                "episode_trade_count": 0,
                "total_fees": 0.0,
                "episode_max_dd": 0.0,
                "termination_reason": "",
                "profit_per_trade": 0.0,
            }
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, _, dones, infos = env.step(action)
                info = infos[0] if isinstance(infos, (list, tuple)) and infos else {}
                if isinstance(info, dict) and (info.get("terminated") or info.get("truncated")):
                    profit = float(info.get("final_balance", 0.0)) - float(info.get("initial_balance", 0.0))
                    trade_count = int(info.get("episode_trade_count", 0) or 0)
                    row["profit"] = profit
                    row["episode_trade_count"] = trade_count
                    row["total_fees"] = float(info.get("total_fees", 0.0))
                    row["episode_max_dd"] = float(info.get("episode_max_dd", 0.0))
                    row["termination_reason"] = str(info.get("termination_reason") or "")
                    row["profit_per_trade"] = float(profit / max(1, trade_count))
                done = bool(dones[0]) if isinstance(dones, np.ndarray) else bool(dones)
            rows.append(row)
        return rows
    finally:
        try:
            env.close()
        except Exception:
            pass


def _summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    df = pd.DataFrame(rows)
    profit_mean = float(df["profit"].mean()) if not df.empty else 0.0
    total_fees_mean = float(df["total_fees"].mean()) if not df.empty else 0.0
    return {
        "profit_mean": profit_mean,
        "profit_per_trade_mean": float(df["profit_per_trade"].mean()) if not df.empty else 0.0,
        "trade_count_mean": float(df["episode_trade_count"].mean()) if not df.empty else 0.0,
        "total_fees_mean": total_fees_mean,
        "episode_max_dd_mean": float(df["episode_max_dd"].mean()) if not df.empty else 0.0,
        "fees_profit_ratio": float(total_fees_mean / max(abs(profit_mean), 1e-8)),
    }


def _paired_delta_frame(
    base_rows: list[dict[str, Any]],
    other_rows: list[dict[str, Any]],
    scenario_name: str,
) -> pd.DataFrame:
    base = pd.DataFrame(base_rows).rename(
        columns={
            "profit": "profit_base",
            "profit_per_trade": "profit_per_trade_base",
            "episode_trade_count": "episode_trade_count_base",
            "total_fees": "total_fees_base",
            "episode_max_dd": "episode_max_dd_base",
        }
    )
    other = pd.DataFrame(other_rows).rename(
        columns={
            "profit": "profit_other",
            "profit_per_trade": "profit_per_trade_other",
            "episode_trade_count": "episode_trade_count_other",
            "total_fees": "total_fees_other",
            "episode_max_dd": "episode_max_dd_other",
        }
    )
    merged = base.merge(other, on="episode_index", how="inner")
    merged["scenario"] = scenario_name
    merged["delta_profit"] = merged["profit_other"] - merged["profit_base"]
    merged["delta_profit_per_trade"] = (
        merged["profit_per_trade_other"] - merged["profit_per_trade_base"]
    )
    merged["delta_trade_count"] = (
        merged["episode_trade_count_other"] - merged["episode_trade_count_base"]
    )
    merged["delta_total_fees"] = merged["total_fees_other"] - merged["total_fees_base"]
    merged["delta_episode_max_dd"] = (
        merged["episode_max_dd_other"] - merged["episode_max_dd_base"]
    )
    return merged


def _paired_delta_summary(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}
    low, high = bootstrap_mean_ci(df["delta_profit_per_trade"].tolist(), n_bootstrap=500)
    return {
        "delta_profit_mean": float(df["delta_profit"].mean()),
        "delta_profit_per_trade_mean": float(df["delta_profit_per_trade"].mean()),
        "delta_trade_count_mean": float(df["delta_trade_count"].mean()),
        "delta_total_fees_mean": float(df["delta_total_fees"].mean()),
        "delta_episode_max_dd_mean": float(df["delta_episode_max_dd"].mean()),
        "delta_profit_per_trade_ci_low": low,
        "delta_profit_per_trade_ci_high": high,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="已訓練模型的 feature mask / ablation eval")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--phase", choices=("A", "B"), default="B")
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--holdout-months", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=288 * 7)
    parser.add_argument("--mask-groups", type=str, default="price_seq_target,price_seq_others,price_seq_1d_target,price_seq_1d_others,gate_flags,regime_score")
    parser.add_argument("--candidate-json", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="logs/feature_mask_ablation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SAC.load(args.model_path, device=args.device)
    base_env_kwargs = get_phase_ab_env_kwargs(
        data_split_enabled=True,
        holdout_months=int(args.holdout_months),
        data_mode="eval",
    )
    base_env_kwargs["max_episode_steps"] = int(args.max_episode_steps)

    scenarios = _make_scenarios(_parse_csv(args.mask_groups), args.candidate_json or None)
    scenario_rows: list[dict[str, Any]] = []
    paired_frames: list[pd.DataFrame] = []
    baseline_rows: list[dict[str, Any]] | None = None

    for scenario in scenarios:
        print(f"[MaskEval] scenario={scenario.name}", flush=True)
        base_thunk = make_env(
            phase=args.phase,
            anneal_steps=0,
            seed=None,
            **base_env_kwargs,
        )

        def masked_thunk(base_thunk: Callable[[], gym.Env] = base_thunk, scenario: MaskScenario = scenario) -> gym.Env:
            env = base_thunk()
            if scenario.name == "baseline":
                return env
            return ObservationMaskWrapper(
                env,
                zero_keys=scenario.zero_keys,
                keep_cols_by_key=scenario.keep_cols_by_key,
            )

        env_builder = _build_vec_env(masked_thunk)
        rows = _evaluate_scenario(
            model,
            env_builder,
            n_episodes=args.eval_episodes,
            deterministic=True,
            seed=args.eval_seed,
        )
        summary = _summary_from_rows(rows)
        scenario_rows.append({"scenario": scenario.name, **summary})
        print(
            f"[MaskEval] done={scenario.name} profit_mean={summary['profit_mean']:.4f} "
            f"ppt={summary['profit_per_trade_mean']:.6f} trades={summary['trade_count_mean']:.2f}",
            flush=True,
        )
        if scenario.name == "baseline":
            baseline_rows = rows
        elif baseline_rows is not None:
            paired = _paired_delta_frame(baseline_rows, rows, scenario.name)
            paired_frames.append(paired)

    df_summary = pd.DataFrame(scenario_rows)
    df_summary.to_csv(out_dir / "mask_summary.csv", index=False)

    delta_rows: list[dict[str, Any]] = []
    if paired_frames:
        df_paired = pd.concat(paired_frames, ignore_index=True)
        df_paired.to_csv(out_dir / "paired_episode_deltas.csv", index=False)
        for scenario_name, group in df_paired.groupby("scenario"):
            delta_rows.append({"scenario": scenario_name, **_paired_delta_summary(group)})
        pd.DataFrame(delta_rows).to_csv(out_dir / "paired_delta_summary.csv", index=False)

    print(f"[Saved] {out_dir / 'mask_summary.csv'}", flush=True)
    if paired_frames:
        print(f"[Saved] {out_dir / 'paired_episode_deltas.csv'}", flush=True)
        print(f"[Saved] {out_dir / 'paired_delta_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
