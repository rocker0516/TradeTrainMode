"""
E2 特徵訊號分析：(1) LightGBM feature importance 對應欄位名；(2) 逐 channel 單變量 AUC/Macro-F1。

- 使用 market_only 特徵（5*F_5m），同一套時間切分與標籤。
- 輸出：importance 表/圖、per-channel 指標表/圖，存於 --out_dir。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Env.feature_transformer import FeatureTransformer
from Train.e2_direction_proxy import (
    DELTA_COEF,
    HORIZON_K,
    RANDOM_STATE,
    TEST_RATIO,
    TRAIN_RATIO,
    VALID_RATIO,
    build_labels,
    collect_obs_and_indices,
    obs_to_summary_features,
    train_valid_test_split_time_ordered,
    fit_predict_binary,
    fit_predict_3class,
)
import lightgbm as lgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATS = ("last", "mean", "std", "min", "max")


def get_market_5m_feature_names() -> List[str]:
    """5m target 每個 channel 的 5 個統計量對應的欄位名，順序與 obs_to_summary_features(market_only) 一致。"""
    cols = list(FeatureTransformer.OPTIMIZED_TARGET_5M_COLS)
    names: List[str] = []
    for c in cols:
        for s in STATS:
            names.append(f"{c}_{s}")
    return names


def run_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
    feature_names: List[str],
) -> Tuple[Any, np.ndarray]:
    """訓練 LightGBM，回傳 model 與 feature_importances_（與 feature_names 同序）。"""
    if task == "binary":
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=1,
        )
        model.fit(
            pd.DataFrame(X_train, columns=feature_names),
            y_train,
        )
    else:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_enc = le.fit_transform(y_train)
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=1,
        )
        model.fit(
            pd.DataFrame(X_train, columns=feature_names),
            y_enc,
        )
    imp = model.feature_importances_
    return model, imp


def run_per_channel(
    X: np.ndarray,
    y_binary: np.ndarray,
    y_3class: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    task: str,
    F: int,
) -> Tuple[List[str], List[float]]:
    """每個 channel 只用 5 維特徵訓練，回傳 channel 名列表與對應 Test 指標（AUC 或 Macro-F1）。"""
    cols = list(FeatureTransformer.OPTIMIZED_TARGET_5M_COLS)
    assert X.shape[1] == 5 * F
    channel_names: List[str] = []
    metrics: List[float] = []
    for ch in range(F):
        start, end = ch * 5, (ch + 1) * 5
        X_ch = X[:, start:end]
        X_tr = X_ch[train_idx]
        X_te = X_ch[test_idx]
        if task == "binary":
            y_tr = y_binary[train_idx]
            y_te = y_binary[test_idx]
            _, auc, _ = fit_predict_binary(X_tr, y_tr, X_te, y_te, use_lightgbm=True)
            metrics.append(auc)
        else:
            y_tr = y_3class[train_idx]
            y_te = y_3class[test_idx]
            _, macro_f1, _, _ = fit_predict_3class(X_tr, y_tr, X_te, y_te, use_lightgbm=True)
            metrics.append(macro_f1)
        channel_names.append(cols[ch])
    return channel_names, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 feature signal: importance + per-channel metric")
    parser.add_argument("--window_size", type=int, default=288)
    parser.add_argument("--k", type=int, default=HORIZON_K)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--task", type=str, choices=("binary", "3class"), default="binary")
    parser.add_argument("--out_dir", type=str, default="logs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Collecting obs (market_only)...")
    obs_list, valid_indices, close_arr, atr_ratio_arr = collect_obs_and_indices(
        window_size=args.window_size,
        horizon_k=args.k,
        max_steps=args.max_steps,
    )
    n_valid = len(valid_indices)
    y_binary, y_3class = build_labels(
        close_arr, atr_ratio_arr, valid_indices, k=args.k, delta_coef=DELTA_COEF
    )
    X_list = [
        obs_to_summary_features(o, use_1d=False, feature_mode="market_only")
        for o in obs_list
    ]
    X = np.stack(X_list, axis=0)
    F = len(FeatureTransformer.OPTIMIZED_TARGET_5M_COLS)
    assert X.shape[1] == 5 * F, f"Expected 5*F={5*F}, got {X.shape[1]}"

    train_idx, valid_idx, test_idx = train_valid_test_split_time_ordered(
        n_valid, TRAIN_RATIO, VALID_RATIO, TEST_RATIO
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train_bin = y_binary[train_idx]
    y_test_bin = y_binary[test_idx]
    y_train_3 = y_3class[train_idx]
    y_test_3 = y_3class[test_idx]

    feature_names = get_market_5m_feature_names()

    # ---------- (1) LightGBM feature importance ----------
    print("(1) Training LightGBM and computing feature importance...")
    if args.task == "binary":
        _, imp = run_importance(
            X_train, y_train_bin, X_test, y_test_bin, "binary", feature_names
        )
    else:
        _, imp = run_importance(
            X_train, y_train_3, X_test, y_test_3, "3class", feature_names
        )
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": imp,
    }).sort_values("importance", ascending=False)
    csv_imp = os.path.join(args.out_dir, f"e2_feature_importance_{args.task}.csv")
    df_imp.to_csv(csv_imp, index=False)
    print(f"  Saved {csv_imp}")

    fig1, ax1 = plt.subplots(figsize=(12, 10))
    top_n = min(60, len(df_imp))
    plot_df = df_imp.head(top_n)
    ax1.barh(range(len(plot_df)), plot_df["importance"].values, align="center")
    ax1.set_yticks(range(len(plot_df)))
    ax1.set_yticklabels(plot_df["feature"].values, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Feature importance (LightGBM)")
    ax1.set_title(f"E2 market_only top-{top_n} features ({args.task})")
    plt.tight_layout()
    fig1_path = os.path.join(args.out_dir, f"e2_feature_importance_{args.task}.png")
    plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig1_path}")

    # ---------- (2) Per-channel univariate metric ----------
    print("(2) Per-channel univariate metric...")
    ch_names, ch_metrics = run_per_channel(
        X, y_binary, y_3class, train_idx, test_idx, args.task, F
    )
    metric_name = "AUC" if args.task == "binary" else "MacroF1"
    df_ch = pd.DataFrame({
        "channel": ch_names,
        metric_name: ch_metrics,
    }).sort_values(metric_name, ascending=False)
    csv_ch = os.path.join(args.out_dir, f"e2_per_channel_metric_{args.task}.csv")
    df_ch.to_csv(csv_ch, index=False)
    print(f"  Saved {csv_ch}")

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.barh(range(len(ch_names)), ch_metrics, align="center")
    ax2.set_yticks(range(len(ch_names)))
    ax2.set_yticklabels(ch_names, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel(metric_name)
    ax2.set_title(f"E2 per-channel {metric_name} (market_only, {args.task})")
    if args.task == "binary":
        ax2.axvline(0.5, color="gray", linestyle="--", alpha=0.7)
    else:
        ax2.axvline(1.0 / 3.0, color="gray", linestyle="--", alpha=0.7)
    plt.tight_layout()
    fig2_path = os.path.join(args.out_dir, f"e2_per_channel_metric_{args.task}.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig2_path}")

    print("Done.")


if __name__ == "__main__":
    main()
