from __future__ import annotations

"""
Observation 特徵重選主腳本。

功能：
1. 對多組 window_size / window_size_1d / label horizon 做公平離線評分。
2. 產出各 CNN channel 的 binary AUC、3-class Macro-F1、importance 與穩定度。
3. 掃描 FeatureTransformer / MarketData / E2 腳本中的 leakage 風險。
4. 依冗餘規則輸出候選 observation 配置（含 ultra_minimal）。
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Env.config import Config
from Eval.e2_direction_proxy import (
    SUMMARY_STATS_TEMPORAL,
    build_labels,
    collect_obs_and_indices,
    obs_to_summary_features_one_cnn,
    train_valid_test_split_time_ordered,
)
from Eval.e2_feature_signal_analysis import (
    get_summary_feature_names,
    run_importance,
    run_per_channel,
)
from Eval.feature_selection_utils import (
    WindowHorizonSpec,
    bootstrap_mean_ci,
    build_redundancy_frame,
    build_window_horizon_specs,
    capture_observation_config_snapshot,
    findings_to_frame,
    market_full_universe_overrides,
    scan_source_for_leakage,
    select_representatives,
    tag_redundancy,
    temporary_observation_config,
)


MARKET_CNN_KEYS: tuple[str, ...] = ("5m_target", "5m_others", "1d_target", "1d_others")
CNN_TO_OBS_KEY: dict[str, str] = {
    "5m_target": "price_seq_target",
    "5m_others": "price_seq_others",
    "1d_target": "price_seq_1d_target",
    "1d_others": "price_seq_1d_others",
}
CNN_TO_CONFIG_FIELD: dict[str, str] = {
    "5m_target": "OBS_PRICE_SEQ_TARGET_COLS",
    "5m_others": "OBS_PRICE_SEQ_OTHERS_COLS",
    "1d_target": "OBS_PRICE_SEQ_1D_TARGET_COLS",
    "1d_others": "OBS_PRICE_SEQ_1D_OTHERS_COLS",
}


def _parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_float_csv(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _mean_or_zero(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.mean())


def _signal_score(auc_binary: float, macro_f1: float, importance_pct: float) -> float:
    auc_component = max(0.0, float(auc_binary) - 0.5) * 2.0
    f1_component = max(0.0, float(macro_f1))
    imp_component = max(0.0, float(importance_pct))
    return float((0.45 * auc_component) + (0.40 * f1_component) + (0.15 * imp_component))


def _collect_one_spec(
    spec: WindowHorizonSpec,
    *,
    max_steps: int | None,
    device: str,
    selected_cnn_keys: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    obs_list, valid_indices, close_arr, atr_ratio_arr, cnn_cols = collect_obs_and_indices(
        window_size=spec.window_size,
        window_size_1d=spec.window_size_1d,
        horizon_k=spec.horizon_k,
        max_steps=max_steps,
        return_cnn_cols=True,
    )
    n_valid = len(valid_indices)
    y_binary, y_3class = build_labels(
        close_arr,
        atr_ratio_arr,
        valid_indices,
        k=spec.horizon_k,
        delta_coef=10.0,
    )
    train_idx, valid_idx, test_idx = train_valid_test_split_time_ordered(n_valid)

    rows: list[dict[str, Any]] = []
    meta = {
        "window_size": spec.window_size,
        "window_size_1d": spec.window_size_1d,
        "horizon_k": spec.horizon_k,
        "recent_bars": spec.recent_bars,
        "n_valid": n_valid,
        "n_train": int(len(train_idx)),
        "n_valid_split": int(len(valid_idx)),
        "n_test": int(len(test_idx)),
        "binary_positive_rate_train": float(np.mean(y_binary[train_idx])) if len(train_idx) else 0.0,
        "binary_positive_rate_test": float(np.mean(y_binary[test_idx])) if len(test_idx) else 0.0,
    }

    for cnn_key in selected_cnn_keys:
        x_list = [
            obs_to_summary_features_one_cnn(
                obs,
                cnn_key,
                summary_mode="temporal",
                recent_bars=spec.recent_bars,
            )
            for obs in obs_list
        ]
        if not x_list:
            continue
        x = np.stack(x_list, axis=0)
        stats_per_channel = len(SUMMARY_STATS_TEMPORAL)
        f_count = x.shape[1] // stats_per_channel
        channel_names = list(cnn_cols.get(cnn_key, ()))
        if f_count <= 0 or len(channel_names) != f_count:
            continue
        feature_names = get_summary_feature_names(channel_names, stats=SUMMARY_STATS_TEMPORAL)
        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train_bin = y_binary[train_idx]
        y_test_bin = y_binary[test_idx]
        y_train_3 = y_3class[train_idx]
        y_test_3 = y_3class[test_idx]

        _, imp = run_importance(
            x_train,
            y_train_bin,
            x_test,
            y_test_bin,
            "binary",
            feature_names,
            device=device,
        )
        imp = np.asarray(imp, dtype=np.float64)
        total_imp = float(np.sum(imp)) if imp.size > 0 else 0.0
        _, auc_metrics = run_per_channel(
            x,
            y_binary,
            y_3class,
            train_idx,
            test_idx,
            "binary",
            f_count,
            channel_names,
            stats_per_channel=stats_per_channel,
            device=device,
        )
        _, macro_f1_metrics = run_per_channel(
            x,
            y_binary,
            y_3class,
            train_idx,
            test_idx,
            "3class",
            f_count,
            channel_names,
            stats_per_channel=stats_per_channel,
            device=device,
        )

        for ch, channel in enumerate(channel_names):
            start = ch * stats_per_channel
            end = (ch + 1) * stats_per_channel
            imp_sum = float(np.sum(imp[start:end]))
            importance_pct = imp_sum / total_imp if total_imp > 0.0 else 0.0
            tag = tag_redundancy(channel)
            rows.append(
                {
                    **meta,
                    "cnn_key": cnn_key,
                    "obs_key": CNN_TO_OBS_KEY[cnn_key],
                    "channel": channel,
                    "auc_binary": float(auc_metrics[ch]),
                    "macro_f1": float(macro_f1_metrics[ch]),
                    "importance_sum": imp_sum,
                    "importance_pct": float(importance_pct),
                    "signal_score": _signal_score(
                        auc_binary=float(auc_metrics[ch]),
                        macro_f1=float(macro_f1_metrics[ch]),
                        importance_pct=float(importance_pct),
                    ),
                    "concept_key": tag.concept_key,
                    "variant": tag.variant,
                    "symbol_scope": tag.symbol_scope,
                    "normalized_name": tag.normalized_name,
                }
            )
    return pd.DataFrame(rows), meta


def _aggregate_channel_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return pd.DataFrame()
    grouped = (
        df_all.groupby(
            ["cnn_key", "obs_key", "channel", "concept_key", "variant", "symbol_scope", "normalized_name"],
            as_index=False,
        )
        .agg(
            mean_auc_binary=("auc_binary", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_importance_pct=("importance_pct", "mean"),
            mean_signal_score=("signal_score", "mean"),
            std_signal_score=("signal_score", "std"),
            specs_count=("signal_score", "count"),
            positive_signal_specs=("signal_score", lambda s: int(np.sum(np.asarray(s) > 0.05))),
            mean_window_size=("window_size", "mean"),
            mean_horizon_k=("horizon_k", "mean"),
        )
        .sort_values(
            by=["mean_signal_score", "mean_auc_binary", "mean_macro_f1"],
            ascending=[False, False, False],
        )
    )
    grouped["std_signal_score"] = grouped["std_signal_score"].fillna(0.0)
    lows: list[float] = []
    highs: list[float] = []
    for _, row in grouped.iterrows():
        mask = (
            (df_all["cnn_key"] == row["cnn_key"])
            & (df_all["channel"] == row["channel"])
        )
        low, high = bootstrap_mean_ci(df_all.loc[mask, "signal_score"].tolist(), n_bootstrap=500)
        lows.append(low)
        highs.append(high)
    grouped["signal_score_ci_low"] = lows
    grouped["signal_score_ci_high"] = highs
    return grouped


def _aggregate_spec_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (window_size, window_size_1d, horizon_k, recent_bars), group in df_all.groupby(
        ["window_size", "window_size_1d", "horizon_k", "recent_bars"]
    ):
        ordered = group.sort_values("signal_score", ascending=False)
        top_slice = ordered.head(max(4, min(12, len(ordered))))
        rows.append(
            {
                "window_size": int(window_size),
                "window_size_1d": int(window_size_1d),
                "horizon_k": int(horizon_k),
                "recent_bars": int(recent_bars),
                "rows_count": int(len(group)),
                "mean_signal_score": _mean_or_zero(group["signal_score"]),
                "top_signal_mean": _mean_or_zero(top_slice["signal_score"]),
                "mean_auc_binary": _mean_or_zero(group["auc_binary"]),
                "mean_macro_f1": _mean_or_zero(group["macro_f1"]),
                "mean_importance_pct": _mean_or_zero(group["importance_pct"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["top_signal_mean", "mean_signal_score", "mean_auc_binary"],
        ascending=[False, False, False],
    )


def _candidate_payload(
    spec_summary: pd.DataFrame,
    channel_summary: pd.DataFrame,
    *,
    baseline_snapshot: dict[str, Any],
    universe_mode: str,
) -> dict[str, Any]:
    current_state_keys = list(getattr(Config, "OBS_STATE_KEYS", ()))
    non_market_state_keys = [
        key
        for key in current_state_keys
        if key not in {"price_seq_target", "price_seq_others", "price_seq_1d_target", "price_seq_1d_others"}
    ]
    best_spec_row = spec_summary.iloc[0] if not spec_summary.empty else None
    best_spec = {
        "window_size": int(best_spec_row["window_size"]) if best_spec_row is not None else int(Config.WINDOW_SIZE),
        "window_size_1d": int(best_spec_row["window_size_1d"]) if best_spec_row is not None else int(Config.WINDOW_SIZE_1D),
        "horizon_k": int(best_spec_row["horizon_k"]) if best_spec_row is not None else 24,
        "recent_bars": int(best_spec_row["recent_bars"]) if best_spec_row is not None else min(int(Config.WINDOW_SIZE), 24),
    }

    stable = channel_summary.copy()
    if stable.empty:
        stable = pd.DataFrame(
            columns=["cnn_key", "channel", "concept_key", "mean_signal_score", "signal_score_ci_low"]
        )
    stable = stable[
        (stable["mean_signal_score"] >= 0.05)
        & (stable["signal_score_ci_low"] >= 0.0)
    ].copy()
    if stable.empty:
        stable = channel_summary.head(12).copy()

    candidate_sets: dict[str, dict[str, Any]] = {}

    def top_channels(cnn_key: str, limit: int, top_k_per_concept: int) -> list[str]:
        subset = stable[stable["cnn_key"] == cnn_key].copy()
        if subset.empty:
            subset = channel_summary[channel_summary["cnn_key"] == cnn_key].copy()
        if subset.empty:
            return []
        redundancy_df = build_redundancy_frame(
            subset["normalized_name"].tolist(),
            metric_by_feature=dict(zip(subset["normalized_name"], subset["mean_signal_score"])),
        )
        redundancy_df["cnn_key"] = cnn_key
        reps = select_representatives(redundancy_df, top_k_per_concept=top_k_per_concept)
        merged = reps.merge(
            subset[["normalized_name", "mean_signal_score"]],
            left_on="raw_name",
            right_on="normalized_name",
            how="left",
        ).sort_values("mean_signal_score", ascending=False)
        return merged["raw_name"].head(limit).tolist()

    ultra_minimal_market = {
        "price_seq_target": tuple(top_channels("5m_target", limit=0, top_k_per_concept=1)),
        "price_seq_others": tuple(),
        "price_seq_1d_target": tuple(top_channels("1d_target", limit=6, top_k_per_concept=1)),
        "price_seq_1d_others": tuple(top_channels("1d_others", limit=2, top_k_per_concept=1)),
    }
    minimal_signal_market = {
        "price_seq_target": tuple(top_channels("5m_target", limit=2, top_k_per_concept=1)),
        "price_seq_others": tuple(),
        "price_seq_1d_target": tuple(top_channels("1d_target", limit=8, top_k_per_concept=2)),
        "price_seq_1d_others": tuple(top_channels("1d_others", limit=4, top_k_per_concept=1)),
    }
    minimal_plus_entry_filter_market = {
        "price_seq_target": tuple(top_channels("5m_target", limit=4, top_k_per_concept=2)),
        "price_seq_others": tuple(top_channels("5m_others", limit=2, top_k_per_concept=1)),
        "price_seq_1d_target": tuple(top_channels("1d_target", limit=8, top_k_per_concept=2)),
        "price_seq_1d_others": tuple(top_channels("1d_others", limit=4, top_k_per_concept=1)),
    }

    market_candidates = {
        "baseline_current": {
            "price_seq_target": tuple(baseline_snapshot["OBS_PRICE_SEQ_TARGET_COLS"]),
            "price_seq_others": tuple(baseline_snapshot["OBS_PRICE_SEQ_OTHERS_COLS"]),
            "price_seq_1d_target": tuple(baseline_snapshot["OBS_PRICE_SEQ_1D_TARGET_COLS"]),
            "price_seq_1d_others": tuple(baseline_snapshot["OBS_PRICE_SEQ_1D_OTHERS_COLS"]),
            "window_size": int(baseline_snapshot["WINDOW_SIZE"]),
            "window_size_1d": int(baseline_snapshot["WINDOW_SIZE_1D"]),
        },
        "ultra_minimal": {
            **ultra_minimal_market,
            "window_size": best_spec["window_size"],
            "window_size_1d": best_spec["window_size_1d"],
        },
        "minimal_signal": {
            **minimal_signal_market,
            "window_size": best_spec["window_size"],
            "window_size_1d": best_spec["window_size_1d"],
        },
        "minimal_plus_entry_filter": {
            **minimal_plus_entry_filter_market,
            "window_size": best_spec["window_size"],
            "window_size_1d": best_spec["window_size_1d"],
        },
    }

    for name, item in market_candidates.items():
        state_keys = list(non_market_state_keys)
        for obs_key in ("price_seq_target", "price_seq_others", "price_seq_1d_target", "price_seq_1d_others"):
            if tuple(item.get(obs_key, ())):
                state_keys.insert(0, obs_key)
        state_keys = list(dict.fromkeys(state_keys))
        candidate_sets[name] = {
            "OBS_STATE_KEYS": tuple(state_keys),
            "OBS_PRICE_SEQ_TARGET_COLS": tuple(item["price_seq_target"]),
            "OBS_PRICE_SEQ_OTHERS_COLS": tuple(item["price_seq_others"]),
            "OBS_PRICE_SEQ_1D_TARGET_COLS": tuple(item["price_seq_1d_target"]),
            "OBS_PRICE_SEQ_1D_OTHERS_COLS": tuple(item["price_seq_1d_others"]),
            "OBS_ACCOUNT_STATE_COLS": tuple(getattr(Config, "OBS_ACCOUNT_STATE_COLS", ())),
            "OBS_CONTEXT_STATE_COLS": tuple(getattr(Config, "OBS_CONTEXT_STATE_COLS", ())),
            "WINDOW_SIZE": int(item["window_size"]),
            "WINDOW_SIZE_1D": int(item["window_size_1d"]),
        }
    return {
        "universe_mode": universe_mode,
        "best_spec": best_spec,
        "candidate_sets": candidate_sets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="重選 observation 特徵與 window/horizon 聯合評估")
    parser.add_argument("--window-sizes", type=str, default="12,24,48")
    parser.add_argument("--window-sizes-1d", type=str, default="6,12,24")
    parser.add_argument("--horizon-ratios", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--max-horizon", type=int, default=144)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, choices=("cpu", "gpu", "cuda"), default="cpu")
    parser.add_argument("--only-5m", action="store_true")
    parser.add_argument(
        "--feature-universe",
        type=str,
        choices=("current_subset", "full_market_universe"),
        default="current_subset",
        help="current_subset=沿用目前 Config 子集；full_market_universe=市場特徵四組設為空 tuple，用全部工程特徵重選",
    )
    parser.add_argument("--out-dir", type=str, default="logs/feature_reselection")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_snapshot = capture_observation_config_snapshot()

    selected_cnn_keys = (
        ["5m_target", "5m_others"]
        if args.only_5m
        else list(MARKET_CNN_KEYS)
    )
    specs = build_window_horizon_specs(
        _parse_int_csv(args.window_sizes),
        _parse_int_csv(args.window_sizes_1d),
        horizon_ratios=_parse_float_csv(args.horizon_ratios),
        max_horizon=int(args.max_horizon),
    )

    all_frames: list[pd.DataFrame] = []
    meta_rows: list[dict[str, Any]] = []
    overrides = market_full_universe_overrides() if args.feature_universe == "full_market_universe" else {}
    with temporary_observation_config(overrides):
        for spec in specs:
            print(
                f"[OfflineEval] universe={args.feature_universe} window={spec.window_size} "
                f"window_1d={spec.window_size_1d} horizon={spec.horizon_k} recent={spec.recent_bars}",
                flush=True,
            )
            df_spec, meta = _collect_one_spec(
                spec,
                max_steps=args.max_steps,
                device=args.device,
                selected_cnn_keys=selected_cnn_keys,
            )
            all_frames.append(df_spec)
            meta_rows.append(meta)

    df_all = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    df_meta = pd.DataFrame(meta_rows)
    df_channel_summary = _aggregate_channel_summary(df_all)
    df_spec_summary = _aggregate_spec_summary(df_all)

    leakage_findings = scan_source_for_leakage(
        [
            Path("Env/feature_transformer.py"),
            Path("Env/Components/market_data.py"),
            Path("Eval/e2_direction_proxy.py"),
        ]
    )
    df_leakage = findings_to_frame(leakage_findings)

    payload = _candidate_payload(
        df_spec_summary,
        df_channel_summary,
        baseline_snapshot=baseline_snapshot,
        universe_mode=str(args.feature_universe),
    )
    candidate_json = {
        "summary": {
            "spec_rows": int(len(df_spec_summary)),
            "channel_rows": int(len(df_channel_summary)),
            "leakage_findings": int(len(df_leakage)),
        },
        **payload,
    }

    df_all.to_csv(out_dir / "feature_signal_by_spec.csv", index=False)
    df_meta.to_csv(out_dir / "window_horizon_meta.csv", index=False)
    df_channel_summary.to_csv(out_dir / "feature_signal_summary.csv", index=False)
    df_spec_summary.to_csv(out_dir / "window_horizon_ranking.csv", index=False)
    df_leakage.to_csv(out_dir / "leakage_findings.csv", index=False)
    (out_dir / "candidate_feature_sets.json").write_text(
        json.dumps(candidate_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[Saved] {out_dir / 'feature_signal_by_spec.csv'}", flush=True)
    print(f"[Saved] {out_dir / 'feature_signal_summary.csv'}", flush=True)
    print(f"[Saved] {out_dir / 'window_horizon_ranking.csv'}", flush=True)
    print(f"[Saved] {out_dir / 'leakage_findings.csv'}", flush=True)
    print(f"[Saved] {out_dir / 'candidate_feature_sets.json'}", flush=True)


if __name__ == "__main__":
    main()
