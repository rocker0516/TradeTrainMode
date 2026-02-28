"""
E2 多 window_size 版本：對多種 (window_size_5m, window_size_1d) × 四路 CNN × task 跑方向 proxy，並以圖表顯示結果。

- 迴圈 (ws_5m, ws_1d, cnn_key, task)，每組合：收集 obs → 標籤 → 單路 CNN 時序彙總（7*F：近期+全窗）→ 時間切分 → LR + LightGBM → 破壞測試。
- 產出：文字報告與圖表，檔名含 cnn_key（5m_target, 5m_others, 1d_target, 1d_others）。
"""

from __future__ import annotations

import argparse
import gc
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
    fit_predict_binary_return_proba,
    fit_predict_3class,
)
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 報告保留樣本比例：固定 k=12、δ=0.2
KEEP_RATE_HORIZON_K = 12
KEEP_RATE_DELTA = 0.2
# 穩定性：3 seeds
STABILITY_SEEDS = [42, 43, 44]
# Horizon sweep 的 k 候選
HORIZON_SWEEP_KS = (12, 36, 72)


def _compute_keep_rate(
    close_arr: np.ndarray,
    atr_ratio_arr: np.ndarray,
    valid_indices: np.ndarray,
    k: int = KEEP_RATE_HORIZON_K,
    delta: float = KEEP_RATE_DELTA,
) -> float:
    """keep_rate = mean(|r_{t,k}| > delta * atr_ratio[t])，r_{t,k} = log(close[t+k]/close[t])。"""
    keeps = []
    for t in valid_indices:
        if t + k >= len(close_arr):
            continue
        close_t = close_arr[t]
        close_tk = close_arr[t + k]
        if close_t <= 0:
            log_ret = 0.0
        else:
            log_ret = np.log(close_tk / close_t)
        thresh = delta * float(atr_ratio_arr[t])
        keeps.append(np.abs(log_ret) > thresh)
    return float(np.mean(keeps)) if keeps else float("nan")


def _keep_rate_judge(keep_rate: float) -> str:
    """判斷保留樣本比例：15–40% 合理，<10% 太稀疏。"""
    if np.isnan(keep_rate):
        return "N/A"
    if keep_rate < 0.10:
        return "太稀疏"
    if 0.15 <= keep_rate <= 0.40:
        return "合理"
    if keep_rate > 0.40:
        return "偏多"
    return "偏少"


def _stability_judge(std_auc: float) -> str:
    """穩定性（3 seeds）：std ≤ 0.005 可做 feature pruning；std ≥ 0.01 先做 regime/horizon。"""
    if np.isnan(std_auc):
        return "N/A"
    if std_auc <= 0.005:
        return "可做 feature pruning"
    if std_auc >= 0.01:
        return "先做 regime 分桶與 horizon 掃描"
    return "尚可"


def _horizon_sweep_judge(
    k: int, auc_lgb: float, auc_label_shuf: float
) -> str:
    """Horizon sweep：k=36 或 72 > 0.52 且 shuffle≈0.5 → 找到更適合 horizon；僅 k=12 → 偏短線。"""
    if np.isnan(auc_lgb) or np.isnan(auc_label_shuf):
        return "N/A"
    shuf_ok = abs(auc_label_shuf - 0.5) < 0.02
    if k in (36, 72) and auc_lgb > 0.52 and shuf_ok:
        return "找到更適合的預測 horizon"
    if k == 12 and auc_lgb > 0.52 and shuf_ok:
        return "訊號偏短線，需更強 microstructure 特徵"
    return ""


def _regime_bucket_judge(auc_low: float, auc_mid: float, auc_high: float) -> str:
    """Regime 分桶：高波動 AUC > 0.54 且低波動≈0.5 → 建議 regime-gated。"""
    if np.isnan(auc_high) or np.isnan(auc_low):
        return ""
    if auc_high > 0.54 and abs(auc_low - 0.5) < 0.03:
        return "建議做 regime-gated（僅在可判斷 regime 交易/訓練）"
    return ""


def _regime_auc_per_bucket(
    atr_ratio_test: np.ndarray,
    y_test_bin: np.ndarray,
    proba_test: np.ndarray,
) -> tuple:
    """依 atr_ratio_test 三分位分桶，回傳 (auc_low, auc_mid, auc_high)。"""
    n = len(atr_ratio_test)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    q33 = np.percentile(atr_ratio_test, 33.33)
    q66 = np.percentile(atr_ratio_test, 66.67)
    low = atr_ratio_test <= q33
    mid = (atr_ratio_test > q33) & (atr_ratio_test <= q66)
    high = atr_ratio_test > q66
    auc_low = roc_auc_score(y_test_bin[low], proba_test[low]) if np.unique(y_test_bin[low]).size > 1 else 0.5
    auc_mid = roc_auc_score(y_test_bin[mid], proba_test[mid]) if np.unique(y_test_bin[mid]).size > 1 else 0.5
    auc_high = roc_auc_score(y_test_bin[high], proba_test[high]) if np.unique(y_test_bin[high]).size > 1 else 0.5
    return auc_low, auc_mid, auc_high


def run_one_combination(
    window_size_5m: int,
    window_size_1d: int,
    task: str,
    cnn_key: str,
    k: int = HORIZON_K,
    max_steps: Optional[int] = None,
    random_state: Optional[int] = None,
    delta_coef: Optional[float] = None,
    return_regime_data: bool = False,
) -> Dict[str, Any]:
    """
    對單一 (window_size_5m, window_size_1d, cnn_key, task) 跑完整 E2 pipeline（單路 CNN 摘要特徵）。
    task: "binary" | "3class"
    cnn_key: 其一 CNN_KEYS（5m_target, 5m_others, 1d_target, 1d_others）
    return_regime_data: 若 True 且 task==binary，out 內含 "regime_data"（atr_ratio_test, y_test_bin, proba_test）。
    """
    rs = RANDOM_STATE if random_state is None else random_state
    dc = DELTA_COEF if delta_coef is None else delta_coef
    obs_list, valid_indices, close_arr, atr_ratio_arr = collect_obs_and_indices(
        window_size=window_size_5m,
        window_size_1d=window_size_1d,
        horizon_k=k,
        max_steps=max_steps,
    )
    n_valid = len(valid_indices)
    keep_rate = _compute_keep_rate(close_arr, atr_ratio_arr, valid_indices, k=KEEP_RATE_HORIZON_K, delta=KEEP_RATE_DELTA)
    y_binary, y_3class = build_labels(
        close_arr, atr_ratio_arr, valid_indices, k=k, delta_coef=dc
    )
    X_list = [obs_to_summary_features_one_cnn(o, cnn_key) for o in obs_list]
    X = np.stack(X_list, axis=0)
    del X_list
    del obs_list
    gc.collect()
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
            "keep_rate": keep_rate,
            "keep_rate_judge": _keep_rate_judge(keep_rate),
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
        "keep_rate": keep_rate,
        "keep_rate_judge": _keep_rate_judge(keep_rate),
    }

    if task == "binary":
        _, auc_lr, _ = fit_predict_binary(
            X_train, y_train_bin, X_test, y_test_bin, use_lightgbm=False, random_state=rs
        )
        if return_regime_data:
            _, auc_lgb, _, proba_test = fit_predict_binary_return_proba(
                X_train, y_train_bin, X_test, y_test_bin, use_lightgbm=True, random_state=rs
            )
            atr_ratio_at_valid = np.array([atr_ratio_arr[t] for t in valid_indices], dtype=np.float64)
            atr_ratio_test = atr_ratio_at_valid[test_idx]
            out["regime_data"] = {
                "atr_ratio_test": atr_ratio_test,
                "y_test_bin": y_test_bin,
                "proba_test": proba_test,
            }
        else:
            _, auc_lgb, _ = fit_predict_binary(
                X_train, y_train_bin, X_test, y_test_bin, use_lightgbm=True, random_state=rs
            )
        auc_label_shuf = run_sanity_label_shuffle(
            X_train, y_train_bin, X_test, y_test_bin, use_lightgbm=True, random_state=rs
        )
        if n_features_market > 0:
            auc_market_shuf = run_sanity_market_shuffle(
                X, y_binary, train_idx, test_idx, n_features_market, use_lightgbm=True, random_state=rs
            )
        else:
            auc_market_shuf = None  # N/A for account_s_only
        out["auc_lr"] = auc_lr
        out["auc_lgb"] = auc_lgb
        out["auc_label_shuf"] = auc_label_shuf
        out["auc_market_shuf"] = auc_market_shuf
    else:
        _, macro_f1_lr, bal_acc_lr, _ = fit_predict_3class(
            X_train, y_train_3, X_test, y_test_3, use_lightgbm=False, random_state=rs
        )
        _, macro_f1_lgb, bal_acc_lgb, _ = fit_predict_3class(
            X_train, y_train_3, X_test, y_test_3, use_lightgbm=True, random_state=rs
        )
        auc_label_shuf = run_sanity_label_shuffle(
            X_train, y_train_bin, X_test, y_test_bin, use_lightgbm=True, random_state=rs
        )
        if n_features_market > 0:
            auc_market_shuf = run_sanity_market_shuffle(
                X, y_binary, train_idx, test_idx, n_features_market, use_lightgbm=True, random_state=rs
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
        single_point = len(ws) <= 1
        # 單一設定時只用 marker，不畫線，避免看起來像斷掉的折線
        fmt_lr = "o" if single_point else "o-"
        fmt_lgb = "s" if single_point else "s-"
        fmt_tri_up = "^" if single_point else "^-"
        fmt_tri_down = "v" if single_point else "v-"
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        if task == "binary":
            ax1.plot(ws, [r["auc_lr"] for r in ordered], fmt_lr, label="LogisticRegression (L2)", color="C0")
            ax1.plot(ws, [r["auc_lgb"] for r in ordered], fmt_lgb, label="LightGBM", color="C1")
            ax1.set_ylabel("Test AUC")
        else:
            ax1.plot(ws, [r["macro_f1_lr"] for r in ordered], fmt_lr, label="LogisticRegression (L2)", color="C0")
            ax1.plot(ws, [r["macro_f1_lgb"] for r in ordered], fmt_lgb, label="LightGBM", color="C1")
            ax1.set_ylabel("Test Macro-F1")
        ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
        title1 = "E2 Direction Proxy vs " + x_key
        if single_point:
            title1 += " (單一設定，僅一點；可加 --window_sizes 或 --window_sizes_1d 多組以得折線)"
        ax1.set_title(title1)
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.0, 1.0)
        if single_point and len(ws) == 1:
            ax1.set_xlim(ws[0] - 0.5, ws[0] + 0.5)
        ax2.plot(ws, [r[main_key] for r in ordered], fmt_lgb, label="LightGBM (normal)", color="C1")
        ax2.plot(ws, auc_label_shuf, fmt_tri_up, label="Label shuffle", color="C2")
        mkt_vals = [v if v is not None else np.nan for v in auc_market_shuf]
        if any(v is not None for v in auc_market_shuf):
            ax2.plot(ws, mkt_vals, fmt_tri_down, label="Market shuffle", color="C3")
        ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
        ax2.set_xlabel(x_label)
        ax2.set_ylabel("AUC (sanity)")
        ax2.set_title("Sanity Tests")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.4, 1.0)
        if single_point and len(ws) == 1:
            ax2.set_xlim(ws[0] - 0.5, ws[0] + 0.5)
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
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="每組合最多收集步數，可降低記憶體；None 表示不限制",
    )
    parser.add_argument("--out_dir", type=str, default="logs")
    parser.add_argument("--out_report", type=str, default="e2_by_window_report.txt")
    parser.add_argument(
        "--feature_mode",
        type=str,
        default="all",
        choices=("all", "market_only", "account_s_only", "market_plus_account_s", "market_plus_account_full"),
        help="Feature mode (by_window 僅用於報告；單路摘要不區分 market/account)",
    )
    parser.add_argument(
        "--use_1d",
        action="store_true",
        help="指定時跑全部四路 CNN（含 1d）；未指定僅跑 5m_target, 5m_others",
    )
    parser.add_argument(
        "--run_stability",
        action="store_true",
        help="跑穩定性（3 seeds）並輸出表格",
    )
    parser.add_argument(
        "--run_horizon_sweep",
        action="store_true",
        help="跑 Horizon sweep（k=12,36,72）並輸出表格",
    )
    parser.add_argument(
        "--run_regime_bucket",
        action="store_true",
        help="跑 Regime 分桶（test 依 atr_ratio 三分桶算 AUC）並輸出表格",
    )
    args = parser.parse_args()

    cnn_keys = list(CNN_KEYS) if args.use_1d else [k for k in CNN_KEYS if k.startswith("5m_")]
    window_sizes_5m = [int(x.strip()) for x in args.window_sizes.split(",")]
    window_sizes_1d = [int(x.strip()) for x in args.window_sizes_1d.split(",")]
    window_combos = list(product(window_sizes_5m, window_sizes_1d))

    if args.run_all:
        tasks = ["binary", "3class"]
    else:
        tasks = [args.task] if args.task != "both" else ["binary", "3class"]

    run_combos = list(product(window_combos, cnn_keys, tasks))
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
                return_regime_data=(args.run_regime_bucket and task == "binary"),
            )
            results.append(r)
            gc.collect()
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

    # 穩定性（3 seeds）：同設定跑 3 個 seed，收集 AUC list 與 std
    stability_results: List[Dict[str, Any]] = []
    if args.run_stability:
        for (ws_5m, ws_1d), cnn_key, task in run_combos:
            auc_list: List[float] = []
            for seed in STABILITY_SEEDS:
                try:
                    r = run_one_combination(
                        window_size_5m=ws_5m,
                        window_size_1d=ws_1d,
                        task=task,
                        cnn_key=cnn_key,
                        k=args.k,
                        max_steps=args.max_steps,
                        random_state=seed,
                    )
                    main_metric = r["auc_lgb"] if task == "binary" else r["macro_f1_lgb"]
                    auc_list.append(main_metric)
                except Exception:
                    auc_list.append(float("nan"))
            if len(auc_list) == 3:
                arr = np.array(auc_list)
                std_auc = float(np.nanstd(arr))
            else:
                std_auc = float("nan")
            stability_results.append({
                "window_size_5m": ws_5m,
                "window_size_1d": ws_1d,
                "cnn_key": cnn_key,
                "task": task,
                "auc_list": auc_list,
                "std_auc": std_auc,
                "judge": _stability_judge(std_auc),
            })
            gc.collect()

    # Horizon sweep：task=binary，k ∈ {12, 36, 72}，delta_coef=0.2
    horizon_sweep_results: List[Dict[str, Any]] = []
    if args.run_horizon_sweep:
        for (ws_5m, ws_1d), cnn_key in product(window_combos, cnn_keys):
            for k in HORIZON_SWEEP_KS:
                try:
                    r = run_one_combination(
                        window_size_5m=ws_5m,
                        window_size_1d=ws_1d,
                        task="binary",
                        cnn_key=cnn_key,
                        k=k,
                        max_steps=args.max_steps,
                        delta_coef=0.2,
                    )
                    judge = _horizon_sweep_judge(
                        k, r["auc_lgb"], r["auc_label_shuf"]
                    )
                    horizon_sweep_results.append({
                        "window_size_5m": ws_5m,
                        "window_size_1d": ws_1d,
                        "cnn_key": cnn_key,
                        "k": k,
                        "auc_lgb": r["auc_lgb"],
                        "auc_label_shuf": r["auc_label_shuf"],
                        "judge": judge,
                    })
                except Exception:
                    horizon_sweep_results.append({
                        "window_size_5m": ws_5m,
                        "window_size_1d": ws_1d,
                        "cnn_key": cnn_key,
                        "k": k,
                        "auc_lgb": float("nan"),
                        "auc_label_shuf": float("nan"),
                        "judge": "N/A",
                    })
                gc.collect()

    # 文字報告（依 cnn_key, task 分組列出）
    lines = [
        "=" * 100,
        "E2 Direction Proxy by (window_size_5m, window_size_1d, cnn_key, task)",
        "=" * 100,
        f"window_sizes_5m: {window_sizes_5m}, window_sizes_1d: {window_sizes_1d}",
        f"tasks: {tasks}, cnn_keys: {cnn_keys}, feature_mode={args.feature_mode}, use_1d={args.use_1d}, k={args.k}, max_steps={args.max_steps}",
        "",
    ]
    for (cnn_key, t) in product(cnn_keys, tasks):
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

    # 報告保留樣本比例（k=12, δ=0.2）
    lines.append("")
    lines.append("--- 報告保留樣本比例 ---")
    lines.append(f"{'ws_5m':>6} {'ws_1d':>6} {'cnn_key':>12} {'keep_rate':>10} {'判斷':>12}")
    lines.append("-" * 60)
    seen = set()
    for r in results:
        key = (r["window_size_5m"], r["window_size_1d"], r["cnn_key"])
        if key in seen:
            continue
        seen.add(key)
        kr = r.get("keep_rate", float("nan"))
        j = r.get("keep_rate_judge", "N/A")
        kr_str = f"{kr:.2%}" if not np.isnan(kr) else "N/A"
        lines.append(f"{r['window_size_5m']:>6} {r['window_size_1d']:>6} {r['cnn_key']:>12} {kr_str:>10} {j:>12}")

    # 穩定性（3 seeds）
    if stability_results:
        lines.append("")
        lines.append("--- 穩定性（3 seeds） ---")
        lines.append(f"{'ws_5m':>6} {'ws_1d':>6} {'cnn_key':>12} {'task':>8} {'AUC_s1':>8} {'AUC_s2':>8} {'AUC_s3':>8} {'std':>8} {'判斷':>28}")
        lines.append("-" * 100)
        for r in stability_results:
            a = r["auc_list"]
            a1 = f"{a[0]:.4f}" if not np.isnan(a[0]) else "N/A"
            a2 = f"{a[1]:.4f}" if not np.isnan(a[1]) else "N/A"
            a3 = f"{a[2]:.4f}" if not np.isnan(a[2]) else "N/A"
            s = r["std_auc"]
            s_str = f"{s:.4f}" if not np.isnan(s) else "N/A"
            lines.append(
                f"{r['window_size_5m']:>6} {r['window_size_1d']:>6} {r['cnn_key']:>12} {r['task']:>8} "
                f"{a1:>8} {a2:>8} {a3:>8} {s_str:>8} {r['judge']:>28}"
            )

    # Horizon sweep
    if horizon_sweep_results:
        lines.append("")
        lines.append("--- Horizon sweep ---")
        lines.append(f"{'ws_5m':>6} {'ws_1d':>6} {'cnn_key':>12} {'k':>4} {'AUC_LGB':>8} {'LblShuf':>8} {'判斷':>36}")
        lines.append("-" * 90)
        for r in horizon_sweep_results:
            auc_s = f"{r['auc_lgb']:.4f}" if not np.isnan(r["auc_lgb"]) else "N/A"
            shuf_s = f"{r['auc_label_shuf']:.4f}" if not np.isnan(r["auc_label_shuf"]) else "N/A"
            lines.append(
                f"{r['window_size_5m']:>6} {r['window_size_1d']:>6} {r['cnn_key']:>12} {r['k']:>4} "
                f"{auc_s:>8} {shuf_s:>8} {r['judge']:>36}"
            )

    # Regime 分桶（從 results 中 binary 且含 regime_data 者計算）
    regime_rows: List[Dict[str, Any]] = []
    if args.run_regime_bucket:
        for r in results:
            if r.get("task") != "binary" or "regime_data" not in r:
                continue
            rd = r["regime_data"]
            atr_test = rd["atr_ratio_test"]
            y_test = rd["y_test_bin"]
            proba = rd["proba_test"]
            auc_low, auc_mid, auc_high = _regime_auc_per_bucket(atr_test, y_test, proba)
            judge = _regime_bucket_judge(auc_low, auc_mid, auc_high)
            for bucket_name, auc in [("low", auc_low), ("mid", auc_mid), ("high", auc_high)]:
                regime_rows.append({
                    "window_size_5m": r["window_size_5m"],
                    "window_size_1d": r["window_size_1d"],
                    "cnn_key": r["cnn_key"],
                    "bucket": bucket_name,
                    "auc": auc,
                    "judge": judge if bucket_name == "high" else "",
                })
    if regime_rows:
        lines.append("")
        lines.append("--- Regime 分桶 ---")
        lines.append(f"{'ws_5m':>6} {'ws_1d':>6} {'cnn_key':>12} {'bucket':>6} {'AUC':>8} {'判斷':>42}")
        lines.append("-" * 90)
        for row in regime_rows:
            auc_s = f"{row['auc']:.4f}" if not np.isnan(row["auc"]) else "N/A"
            lines.append(
                f"{row['window_size_5m']:>6} {row['window_size_1d']:>6} {row['cnn_key']:>12} {row['bucket']:>6} "
                f"{auc_s:>8} {row['judge']:>42}"
            )

    lines.append("")
    lines.append("=" * 100)
    report = "\n".join(lines)
    print("\n" + report)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 多張圖：每張對應一個 (cnn_key, task)
    for (cnn_key, t) in product(cnn_keys, tasks):
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
