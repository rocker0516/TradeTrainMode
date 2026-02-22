"""
E2 多 window_size 版本：對多種 (window_size_5m, window_size_1d) × 四路 CNN × task 跑方向 proxy，並以圖表顯示結果。

- 迴圈 (ws_5m, ws_1d, cnn_key, task)，每組合：收集 obs → 標籤 → 單路 CNN 摘要特徵（5*F）→ 時間切分 → LR + LightGBM → 破壞測試。
- 產出：文字報告與圖表，檔名含 cnn_key（5m_target, 5m_others, 1d_target, 1d_others）。
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np

# 專案根目錄加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 從單一 window 版本複用邏輯
from Train.e2_direction_proxy import (
    CNN_KEYS,
    DELTA_COEF,
    HORIZON_K,
    RANDOM_STATE,
    TEST_RATIO,
    TRAIN_RATIO,
    VALID_RATIO,
    build_labels,
    collect_obs_and_indices,
    obs_to_summary_features_one_cnn,
    run_sanity_label_shuffle,
    run_sanity_market_shuffle,
    train_valid_test_split_time_ordered,
    fit_predict_binary,
    fit_predict_3class,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_one_combination(
    window_size_5m: int,
    window_size_1d: int,
    task: str,
    cnn_key: str,
    k: int = HORIZON_K,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    對單一 (window_size_5m, window_size_1d, cnn_key, task) 跑完整 E2 pipeline（單路 CNN 摘要特徵）。
    task: "binary" | "3class"
    cnn_key: 其一 CNN_KEYS（5m_target, 5m_others, 1d_target, 1d_others）
    """
    obs_list, valid_indices, close_arr, atr_ratio_arr = collect_obs_and_indices(
        window_size=window_size_5m,
        window_size_1d=window_size_1d,
        horizon_k=k,
        max_steps=max_steps,
    )
    n_valid = len(valid_indices)
    y_binary, y_3class = build_labels(
        close_arr, atr_ratio_arr, valid_indices, k=k, delta_coef=DELTA_COEF
    )
    X_list = [obs_to_summary_features_one_cnn(o, cnn_key) for o in obs_list]
    X = np.stack(X_list, axis=0)
    n_features_market = X.shape[1]
    if n_features_market == 0:
        nan = float("nan")
        return {
            "window_size_5m": window_size_5m,
            "window_size_1d": window_size_1d,
            "cnn_key": cnn_key,
            "task": task,
            "n_samples": n_valid,
            "n_feat": 0,
            "n_feat_market": 0,
            "auc_lr": nan,
            "auc_lgb": nan,
            "auc_label_shuf": nan,
            "auc_market_shuf": None,
            "macro_f1_lr": nan,
            "macro_f1_lgb": nan,
            "bal_acc_lgb": nan,
        }

    train_idx, valid_idx, test_idx = train_valid_test_split_time_ordered(
        n_valid, TRAIN_RATIO, VALID_RATIO, TEST_RATIO
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train_bin = y_binary[train_idx]
    y_test_bin = y_binary[test_idx]
    y_train_3 = y_3class[train_idx]
    y_test_3 = y_3class[test_idx]

    out: Dict[str, Any] = {
        "window_size_5m": window_size_5m,
        "window_size_1d": window_size_1d,
        "cnn_key": cnn_key,
        "task": task,
        "n_samples": n_valid,
        "n_feat": X.shape[1],
        "n_feat_market": n_features_market,
    }

    if task == "binary":
        _, auc_lr, _ = fit_predict_binary(
            X_train, y_train_bin, X_test, y_test_bin, use_lightgbm=False
        )
        _, auc_lgb, _ = fit_predict_binary(
            X_train, y_train_bin, X_test, y_test_bin, use_lightgbm=True
        )
        auc_label_shuf = run_sanity_label_shuffle(
            X_train, y_train_bin, X_test, y_test_bin, use_lightgbm=True
        )
        if n_features_market > 0:
            auc_market_shuf = run_sanity_market_shuffle(
                X, y_binary, train_idx, test_idx, n_features_market, use_lightgbm=True
            )
        else:
            auc_market_shuf = None  # N/A for account_s_only
        out["auc_lr"] = auc_lr
        out["auc_lgb"] = auc_lgb
        out["auc_label_shuf"] = auc_label_shuf
        out["auc_market_shuf"] = auc_market_shuf
    else:
        _, macro_f1_lr, bal_acc_lr, _ = fit_predict_3class(
            X_train, y_train_3, X_test, y_test_3, use_lightgbm=False
        )
        _, macro_f1_lgb, bal_acc_lgb, _ = fit_predict_3class(
            X_train, y_train_3, X_test, y_test_3, use_lightgbm=True
        )
        auc_label_shuf = run_sanity_label_shuffle(
            X_train, y_train_bin, X_test, y_test_bin, use_lightgbm=True
        )
        if n_features_market > 0:
            auc_market_shuf = run_sanity_market_shuffle(
                X, y_binary, train_idx, test_idx, n_features_market, use_lightgbm=True
            )
        else:
            auc_market_shuf = None
        out["macro_f1_lr"] = macro_f1_lr
        out["macro_f1_lgb"] = macro_f1_lgb
        out["bal_acc_lr"] = bal_acc_lr
        out["bal_acc_lgb"] = bal_acc_lgb
        out["auc_label_shuf"] = auc_label_shuf
        out["auc_market_shuf"] = auc_market_shuf
    return out


def _build_2d_grid(
    results: List[Dict[str, Any]],
    key_5m: str = "window_size_5m",
    key_1d: str = "window_size_1d",
    value_key: str = "auc_lgb",
) -> tuple:
    """從 results 建出 (ws_5m 排序, ws_1d 排序, 矩陣)。"""
    ws_5m = sorted({r[key_5m] for r in results})
    ws_1d = sorted({r[key_1d] for r in results})
    idx_5m = {w: i for i, w in enumerate(ws_5m)}
    idx_1d = {w: j for j, w in enumerate(ws_1d)}
    mat = np.full((len(ws_1d), len(ws_5m)), np.nan)
    for r in results:
        i = idx_5m[r[key_5m]]
        j = idx_1d[r[key_1d]]
        mat[j, i] = r[value_key]
    return ws_5m, ws_1d, mat


def plot_results(
    results: List[Dict[str, Any]],
    out_path: str,
    task: str = "binary",
) -> None:
    """
    若 (ws_5m × ws_1d) 有多種組合：畫 heatmap（主指標、Label shuffle）。
    若僅單一維度多值：畫折線圖。
    task: "binary" 時主指標為 auc_lgb；"3class" 時為 macro_f1_lgb。
    """
    main_key = "auc_lgb" if task == "binary" else "macro_f1_lgb"
    main_title = "Test AUC (LightGBM)" if task == "binary" else "Test Macro-F1 (LightGBM)"
    ws_5m_vals = sorted({r["window_size_5m"] for r in results})
    ws_1d_vals = sorted({r["window_size_1d"] for r in results})
    two_d = len(ws_5m_vals) > 1 and len(ws_1d_vals) > 1

    if two_d:
        ws_5m, ws_1d, mat_main = _build_2d_grid(results, value_key=main_key)
        _, _, mat_label = _build_2d_grid(results, value_key="auc_label_shuf")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        vmin, vmax = (0.4, 1.0) if task == "binary" else (0.0, 1.0)
        im1 = ax1.imshow(mat_main, aspect="auto", vmin=vmin, vmax=vmax, cmap="RdYlGn")
        ax1.set_xticks(range(len(ws_5m)))
        ax1.set_xticklabels(ws_5m)
        ax1.set_yticks(range(len(ws_1d)))
        ax1.set_yticklabels(ws_1d)
        ax1.set_xlabel("window_size_5m (bars)")
        ax1.set_ylabel("window_size_1d (days)")
        ax1.set_title(main_title)
        plt.colorbar(im1, ax=ax1)
        for j in range(len(ws_1d)):
            for i in range(len(ws_5m)):
                v = mat_main[j, i]
                if not np.isnan(v):
                    ax1.text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=8)
        im2 = ax2.imshow(mat_label, aspect="auto", vmin=0.4, vmax=1.0, cmap="RdYlGn")
        ax2.set_xticks(range(len(ws_5m)))
        ax2.set_xticklabels(ws_5m)
        ax2.set_yticks(range(len(ws_1d)))
        ax2.set_yticklabels(ws_1d)
        ax2.set_xlabel("window_size_5m (bars)")
        ax2.set_ylabel("window_size_1d (days)")
        ax2.set_title("Label shuffle AUC (expect ~0.5)")
        plt.colorbar(im2, ax=ax2)
        for j in range(len(ws_1d)):
            for i in range(len(ws_5m)):
                v = mat_label[j, i]
                if not np.isnan(v):
                    ax2.text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=8)
    else:
        if len(ws_5m_vals) > 1:
            x_vals = ws_5m_vals
            x_key = "window_size_5m"
            x_label = "window_size_5m (5m bars)"
        else:
            x_vals = ws_1d_vals
            x_key = "window_size_1d"
            x_label = "window_size_1d (days)"
        by_x = {}
        for r in results:
            x = r[x_key]
            if x not in by_x:
                by_x[x] = r
        ordered = [by_x[x] for x in x_vals if x in by_x]
        if not ordered:
            ordered = results
        ws = [r[x_key] for r in ordered]
        auc_label_shuf = [r["auc_label_shuf"] for r in ordered]
        auc_market_shuf = [r.get("auc_market_shuf") for r in ordered]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        if task == "binary":
            ax1.plot(ws, [r["auc_lr"] for r in ordered], "o-", label="LogisticRegression (L2)", color="C0")
            ax1.plot(ws, [r["auc_lgb"] for r in ordered], "s-", label="LightGBM", color="C1")
            ax1.set_ylabel("Test AUC")
        else:
            ax1.plot(ws, [r["macro_f1_lr"] for r in ordered], "o-", label="LogisticRegression (L2)", color="C0")
            ax1.plot(ws, [r["macro_f1_lgb"] for r in ordered], "s-", label="LightGBM", color="C1")
            ax1.set_ylabel("Test Macro-F1")
        ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
        ax1.set_title("E2 Direction Proxy vs " + x_key)
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.0, 1.0)
        ax2.plot(ws, [r[main_key] for r in ordered], "s-", label="LightGBM (normal)", color="C1")
        ax2.plot(ws, auc_label_shuf, "^-", label="Label shuffle", color="C2")
        mkt_vals = [v if v is not None else np.nan for v in auc_market_shuf]
        if any(v is not None for v in auc_market_shuf):
            ax2.plot(ws, mkt_vals, "v-", label="Market shuffle", color="C3")
        ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
        ax2.set_xlabel(x_label)
        ax2.set_ylabel("AUC (sanity)")
        ax2.set_title("Sanity Tests")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2 direction proxy: sweep (window_size_5m, window_size_1d), cnn_key, task"
    )
    parser.add_argument(
        "--window_sizes",
        type=str,
        default="96,144,288,432",
        help="Comma-separated window_size_5m list (5m bars)",
    )
    parser.add_argument(
        "--window_sizes_1d",
        type=str,
        default="14,30",
        help="Comma-separated window_size_1d list (days)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=("binary", "3class", "both"),
        default="binary",
        help="Task: binary, 3class, or both",
    )
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Run all 4 CNNs x 2 tasks (overrides --task)",
    )
    parser.add_argument("--k", type=int, default=HORIZON_K)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="logs")
    parser.add_argument("--out_report", type=str, default="e2_by_window_report.txt")
    args = parser.parse_args()

    window_sizes_5m = [int(x.strip()) for x in args.window_sizes.split(",")]
    window_sizes_1d = [int(x.strip()) for x in args.window_sizes_1d.split(",")]
    window_combos = list(product(window_sizes_5m, window_sizes_1d))

    if args.run_all:
        tasks = ["binary", "3class"]
    else:
        tasks = [args.task] if args.task != "both" else ["binary", "3class"]

    run_combos = list(product(window_combos, CNN_KEYS, tasks))
    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, args.out_report)

    results: List[Dict[str, Any]] = []
    for idx, ((ws_5m, ws_1d), cnn_key, task) in enumerate(run_combos):
        print(f"[{idx+1}/{len(run_combos)}] ws_5m={ws_5m}, ws_1d={ws_1d}, cnn={cnn_key}, task={task} ...")
        try:
            r = run_one_combination(
                window_size_5m=ws_5m,
                window_size_1d=ws_1d,
                task=task,
                cnn_key=cnn_key,
                k=args.k,
                max_steps=args.max_steps,
            )
            results.append(r)
            if task == "binary":
                mkt = r["auc_market_shuf"] if r["auc_market_shuf"] is not None else float("nan")
                print(f"  n={r['n_samples']}, feat={r['n_feat']}, AUC_LGB={r['auc_lgb']:.4f}, "
                      f"LblShuf={r['auc_label_shuf']:.4f}, MktShuf={mkt if not np.isnan(mkt) else 'N/A'}")
            else:
                mkt = r["auc_market_shuf"] if r["auc_market_shuf"] is not None else float("nan")
                print(f"  n={r['n_samples']}, feat={r['n_feat']}, MacroF1_LGB={r['macro_f1_lgb']:.4f}, "
                      f"LblShuf={r['auc_label_shuf']:.4f}, MktShuf={mkt if not np.isnan(mkt) else 'N/A'}")
        except Exception as e:
            print(f"  Error: {e}")
            continue

    if not results:
        print("No results to report or plot.")
        return

    # 文字報告（依 cnn_key, task 分組列出）
    lines = [
        "=" * 100,
        "E2 Direction Proxy by (window_size_5m, window_size_1d, cnn_key, task)",
        "=" * 100,
        f"window_sizes_5m: {window_sizes_5m}, window_sizes_1d: {window_sizes_1d}",
        f"tasks: {tasks}, cnn_keys: {list(CNN_KEYS)}, k={args.k}, max_steps={args.max_steps}",
        "",
    ]
    for (cnn_key, t) in product(CNN_KEYS, tasks):
        subset = [r for r in results if r["cnn_key"] == cnn_key and r["task"] == t]
        if not subset:
            continue
        lines.append(f"--- cnn_key={cnn_key}, task={t} ---")
        if t == "binary":
            lines.append(f"{'ws_5m':>6} {'ws_1d':>6} {'n':>8} {'feat':>6} {'AUC_LR':>8} {'AUC_LGB':>8} {'LblShuf':>8} {'MktShuf':>8}")
        else:
            lines.append(f"{'ws_5m':>6} {'ws_1d':>6} {'n':>8} {'feat':>6} {'F1_LR':>8} {'F1_LGB':>8} {'BalAcc':>8} {'LblShuf':>8} {'MktShuf':>8}")
        lines.append("-" * 80)
        def _fmt(v: Any) -> str:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return f"{v:.4f}"
        for r in subset:
            mkt_str = _fmt(r.get("auc_market_shuf"))
            if t == "binary":
                lines.append(
                    f"{r['window_size_5m']:>6} {r['window_size_1d']:>6} {r['n_samples']:>8} {r['n_feat']:>6} "
                    f"{_fmt(r.get('auc_lr')):>8} {_fmt(r.get('auc_lgb')):>8} "
                    f"{_fmt(r.get('auc_label_shuf')):>8} {mkt_str:>8}"
                )
            else:
                lines.append(
                    f"{r['window_size_5m']:>6} {r['window_size_1d']:>6} {r['n_samples']:>8} {r['n_feat']:>6} "
                    f"{_fmt(r.get('macro_f1_lr')):>8} {_fmt(r.get('macro_f1_lgb')):>8} {_fmt(r.get('bal_acc_lgb')):>8} "
                    f"{_fmt(r.get('auc_label_shuf')):>8} {mkt_str:>8}"
                )
        lines.append("")
    lines.append("=" * 100)
    report = "\n".join(lines)
    print("\n" + report)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 多張圖：每張對應一個 (cnn_key, task)
    for (cnn_key, t) in product(CNN_KEYS, tasks):
        subset = [r for r in results if r["cnn_key"] == cnn_key and r["task"] == t]
        if not subset:
            continue
        fig_name = f"e2_by_window_{cnn_key}_{t}.png"
        fig_path = os.path.join(args.out_dir, fig_name)
        plot_results(subset, fig_path, task=t)
        print(f"Figure: {fig_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
