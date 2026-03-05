"""
E2 特徵訊號分析：(1) LightGBM feature importance 對應欄位名；(2) 逐 channel 單變量 AUC/Macro-F1。

- 對四路 CNN（5m_target, 5m_others, 1d_target, 1d_others）分別跑同一套時間切分與標籤。
- 輸出：importance 表/圖、per-channel 指標表/圖，存於 --out_dir，檔名含 cnn_key。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Train.e2_direction_proxy import (
    CNN_KEYS,
    DELTA_COEF,
    HORIZON_K,
    RANDOM_STATE,
    STATS_PER_CHANNEL_FIVE,
    STATS_PER_CHANNEL_TEMPORAL,
    STATS_PER_CHANNEL_THREE_SEGMENT,
    SUMMARY_STATS_FIVE,
    SUMMARY_STATS_TEMPORAL,
    SUMMARY_STATS_THREE_SEGMENT,
    TEST_RATIO,
    TRAIN_RATIO,
    VALID_RATIO,
    build_labels,
    collect_obs_and_indices,
    obs_to_summary_features_one_cnn,
    train_valid_test_split_time_ordered,
    fit_predict_binary,
    fit_predict_3class,
)
import lightgbm as lgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_summary_feature_names(
    channel_names: List[str],
    *,
    stats: Tuple[str, ...] = SUMMARY_STATS_TEMPORAL,
) -> List[str]:
    """依 channel 名產生 (len(stats)*F) 個彙總特徵名（預設時序彙總：recent_last, recent_mean, ...）。"""
    names: List[str] = []
    for c in channel_names:
        for s in stats:
            names.append(f"{c}_{s}")
    return names


def temporal_importance_breakdown(
    feature_names: List[str],
    importance: np.ndarray,
    summary_mode: str,
) -> pd.DataFrame:
    """
    依特徵名中的時段標記彙總 importance，得到「時間維度」重要性。
    temporal: 名稱含 _recent 或 _full 者分別加總
    three_segment: 名稱含 _early、_mid、_late 者分別加總
    five_stats: single_window
    """
    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    if summary_mode == "temporal":
        # 特徵名為 {channel}_{stat}，stat 為 recent_last, full_mean 等
        def _temporal_scale(name: str) -> str:
            if "_recent" in name:
                return "recent"
            if "_full" in name:
                return "full"
            return "other"
        df["time_scale"] = df["feature"].map(_temporal_scale)
    elif summary_mode == "three_segment":
        def _segment_scale(name: str) -> str:
            if "_early" in name:
                return "early"
            if "_mid" in name:
                return "mid"
            if "_late" in name:
                return "late"
            return "other"
        df["time_scale"] = df["feature"].map(_segment_scale)
    else:
        total = float(np.sum(importance))
        return pd.DataFrame([{"time_scale": "single_window", "importance_sum": total}])
    out = (
        df.groupby("time_scale", as_index=False)["importance"]
        .sum()
        .rename(columns={"importance": "importance_sum"})
    )
    return out


def run_interaction_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
    feature_names: List[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """
    取 importance 前 top_n 個特徵，建 pairwise 乘積交互項，再訓 LightGBM，
    回傳交互項的 importance 表（feat_a, feat_b, importance）。
    """
    from sklearn.preprocessing import LabelEncoder
    n = len(feature_names)
    if n < 2 or top_n < 2:
        return pd.DataFrame(columns=["feat_a", "feat_b", "importance"])
    # 先跑一次得到排序後的前 top_n 名索引（用全特徵 importance 排序）
    model0 = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=1,
    )
    df_train = pd.DataFrame(X_train, columns=feature_names)
    if task == "3class":
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        model0.fit(df_train, y_train_enc)
    else:
        model0.fit(df_train, y_train)
    imp0 = model0.feature_importances_
    order = np.argsort(-imp0)[:top_n]
    # 建交互特徵：X_i * X_j for i < j in order
    pairs: List[Tuple[int, int]] = []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            pairs.append((int(order[i]), int(order[j])))
    X_tr_ia = np.column_stack(
        [X_train[:, i] * X_train[:, j] for i, j in pairs]
    )
    X_te_ia = np.column_stack(
        [X_test[:, i] * X_test[:, j] for i, j in pairs]
    )
    ia_names = [f"ia__{feature_names[i]}__{feature_names[j]}" for i, j in pairs]
    X_tr_full = np.hstack([X_train, X_tr_ia])
    X_te_full = np.hstack([X_test, X_te_ia])
    all_names = list(feature_names) + ia_names
    model_ia = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=1,
    )
    df_tr = pd.DataFrame(X_tr_full, columns=all_names)
    if task == "3class":
        model_ia.fit(df_tr, y_train_enc)
    else:
        model_ia.fit(df_tr, y_train)
    imp_full = model_ia.feature_importances_
    n_orig = len(feature_names)
    ia_imp = imp_full[n_orig:]
    rows = [
        {"feat_a": feature_names[i], "feat_b": feature_names[j], "importance": float(ia_imp[k])}
        for k, (i, j) in enumerate(pairs)
    ]
    df_ia = pd.DataFrame(rows).sort_values("importance", ascending=False)
    return df_ia


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
    channel_names: List[str],
    *,
    stats_per_channel: int = STATS_PER_CHANNEL_TEMPORAL,
) -> Tuple[List[str], List[float]]:
    """每個 channel 只用該 channel 的 stats_per_channel 維訓練，回傳 channel 名列表與對應 Test 指標。"""
    assert X.shape[1] == stats_per_channel * F
    assert len(channel_names) == F
    metrics: List[float] = []
    for ch in range(F):
        start, end = ch * stats_per_channel, (ch + 1) * stats_per_channel
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
    return list(channel_names), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 feature signal: importance + per-channel metric (per CNN)")
    parser.add_argument("--window_size", type=int, default=288)
    parser.add_argument("--k", type=int, default=HORIZON_K)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--task", type=str, choices=("binary", "3class"), default="binary")
    parser.add_argument(
        "--summary_mode",
        type=str,
        choices=("temporal", "three_segment", "five_stats"),
        default="temporal",
        help="temporal=近期+全窗(7*F), three_segment=早/中/晚(6*F), five_stats=5*F",
    )
    parser.add_argument("--out_dir", type=str, default="logs")
    args = parser.parse_args()

    if args.summary_mode == "temporal":
        stats_tuple = SUMMARY_STATS_TEMPORAL
        stats_per_channel = STATS_PER_CHANNEL_TEMPORAL
    elif args.summary_mode == "three_segment":
        stats_tuple = SUMMARY_STATS_THREE_SEGMENT
        stats_per_channel = STATS_PER_CHANNEL_THREE_SEGMENT
    else:
        stats_tuple = SUMMARY_STATS_FIVE
        stats_per_channel = STATS_PER_CHANNEL_FIVE

    os.makedirs(args.out_dir, exist_ok=True)

    print("Collecting obs (return_cnn_cols=True)...")
    obs_list, valid_indices, close_arr, atr_ratio_arr, cnn_cols = collect_obs_and_indices(
        window_size=args.window_size,
        horizon_k=args.k,
        max_steps=args.max_steps,
        return_cnn_cols=True,
    )
    n_valid = len(valid_indices)
    y_binary, y_3class = build_labels(
        close_arr, atr_ratio_arr, valid_indices, k=args.k, delta_coef=DELTA_COEF
    )
    train_idx, valid_idx, test_idx = train_valid_test_split_time_ordered(
        n_valid, TRAIN_RATIO, VALID_RATIO, TEST_RATIO
    )
    y_train_bin = y_binary[train_idx]
    y_test_bin = y_binary[test_idx]
    y_train_3 = y_3class[train_idx]
    y_test_3 = y_3class[test_idx]

    metric_name = "AUC" if args.task == "binary" else "MacroF1"
    # 收集各 CNN 的繪圖資料，最後合成一張 4x2 圖
    plot_imp: List[Tuple[str, pd.DataFrame, int]] = []   # (cnn_key, plot_df, top_n)
    plot_ch: List[Tuple[str, List[str], List[float]]] = []  # (cnn_key, ch_names, ch_metrics)

    for cnn_key in CNN_KEYS:
        print(f"--- CNN: {cnn_key} ---")
        X_list = [
            obs_to_summary_features_one_cnn(o, cnn_key, summary_mode=args.summary_mode)
            for o in obs_list
        ]
        X = np.stack(X_list, axis=0)
        F = X.shape[1] // stats_per_channel
        if F == 0:
            print(f"  Skip (F=0).")
            continue
        channel_names = list(cnn_cols[cnn_key])
        assert len(channel_names) == F, f"cnn_cols[{cnn_key!r}] len {len(channel_names)} != F {F}"
        feature_names = get_summary_feature_names(channel_names, stats=stats_tuple)
        X_train, X_test = X[train_idx], X[test_idx]

        # ---------- (1) LightGBM feature importance ----------
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
        csv_imp = os.path.join(args.out_dir, f"e2_feature_importance_{cnn_key}_{args.task}.csv")
        df_imp.to_csv(csv_imp, index=False)
        print(f"  Saved {csv_imp}")

        # ---------- (1b) 時間維度彙總：依時段前綴加總 importance ----------
        df_temporal = temporal_importance_breakdown(
            feature_names, imp, args.summary_mode
        )
        csv_temporal = os.path.join(
            args.out_dir, f"e2_temporal_importance_{cnn_key}_{args.task}.csv"
        )
        df_temporal.to_csv(csv_temporal, index=False)
        print(f"  Saved {csv_temporal}")

        # ---------- (1c) 交互項重要性：top 15 兩兩乘積再訓，輸出交互 importance ----------
        y_train_task = y_train_bin if args.task == "binary" else y_train_3
        y_test_task = y_test_bin if args.task == "binary" else y_test_3
        df_ia = run_interaction_importance(
            X_train, y_train_task, X_test, y_test_task,
            args.task, feature_names, top_n=15,
        )
        csv_ia = os.path.join(
            args.out_dir, f"e2_interaction_importance_{cnn_key}_{args.task}.csv"
        )
        df_ia.to_csv(csv_ia, index=False)
        print(f"  Saved {csv_ia} (top {len(df_ia)} interactions)")

        top_n = min(60, len(df_imp))
        plot_df = df_imp.head(top_n)
        plot_imp.append((cnn_key, plot_df, top_n))

        # ---------- (2) Per-channel univariate metric ----------
        ch_names, ch_metrics = run_per_channel(
            X, y_binary, y_3class, train_idx, test_idx, args.task, F, channel_names,
            stats_per_channel=stats_per_channel,
        )
        df_ch = pd.DataFrame({
            "channel": ch_names,
            metric_name: ch_metrics,
        }).sort_values(metric_name, ascending=False)
        csv_ch = os.path.join(args.out_dir, f"e2_per_channel_metric_{cnn_key}_{args.task}.csv")
        df_ch.to_csv(csv_ch, index=False)
        print(f"  Saved {csv_ch}")

        # 依 sort 後順序供繪圖
        ch_names_sorted = df_ch["channel"].tolist()
        ch_metrics_sorted = df_ch[metric_name].tolist()
        plot_ch.append((cnn_key, ch_names_sorted, ch_metrics_sorted))

    # ---------- 合成一張 2x4 圖（第 1 列：4 個 Feature importance；第 2 列：4 個 Per-channel）----------
    n_cnns = len(plot_imp)
    if n_cnns > 0:
        fig, axes = plt.subplots(2, n_cnns, figsize=(5 * n_cnns, 10))
        if n_cnns == 1:
            axes = axes.reshape(-1, 1)
        for i in range(n_cnns):
            cnn_key = plot_imp[i][0]
            ax_imp = axes[0, i]
            ax_ch = axes[1, i]

            # 第 1 列：feature importance (top N)
            _, plot_df, top_n = plot_imp[i]
            ax_imp.barh(range(len(plot_df)), plot_df["importance"].values, align="center")
            ax_imp.set_yticks(range(len(plot_df)))
            ax_imp.set_yticklabels(plot_df["feature"].values, fontsize=6)
            ax_imp.invert_yaxis()
            ax_imp.set_xlabel("Feature importance (LightGBM)")
            ax_imp.set_title(f"Top-{top_n} features ({cnn_key})")

            # 第 2 列：per-channel metric
            _, ch_names, ch_metrics = plot_ch[i]
            ax_ch.barh(range(len(ch_names)), ch_metrics, align="center")
            ax_ch.set_yticks(range(len(ch_names)))
            ax_ch.set_yticklabels(ch_names, fontsize=6)
            ax_ch.invert_yaxis()
            ax_ch.set_xlabel(metric_name)
            ax_ch.set_title(f"Per-channel {metric_name} ({cnn_key})")
            if args.task == "binary":
                ax_ch.axvline(0.5, color="gray", linestyle="--", alpha=0.7)
            else:
                ax_ch.axvline(1.0 / 3.0, color="gray", linestyle="--", alpha=0.7)

        plt.suptitle(f"E2 Feature Signal Analysis ({args.task})", fontsize=12, y=1.002)
        plt.tight_layout()
        combined_path = os.path.join(args.out_dir, f"e2_feature_signal_combined_{args.task}.png")
        plt.savefig(combined_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved combined figure: {combined_path}")

    print("Done.")


if __name__ == "__main__":
    main()
