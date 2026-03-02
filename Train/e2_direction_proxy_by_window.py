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
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

    class _SimpleProgress:
        """無 tqdm 時：顯示 進度 X/Y (Z%) 與已用/預估時間。"""

        def __init__(self, total: int, desc: str = "E2"):
            self.total = total
            self.desc = desc
            self.done = 0
            self.start = time.perf_counter()

        def update(self, n: int = 1) -> None:
            self.done += n
            elapsed = time.perf_counter() - self.start
            pct = 100.0 * self.done / self.total if self.total else 0
            eta_min = (elapsed / self.done * (self.total - self.done) / 60) if self.done else 0
            print(
                f"\r>>> {self.desc}: {self.done}/{self.total} ({pct:.0f}%)  "
                f"已用 {elapsed / 60:.1f} 分  預估剩餘 {eta_min:.1f} 分  ",
                end="",
                flush=True,
            )

        def set_postfix_str(self, s: str) -> None:
            pass

        def close(self) -> None:
            print()

# 專案根目錄加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 從單一 window 版本複用邏輯
from Train.e2_direction_proxy import (
    CNN_KEYS,
    HORIZON_K,
    KEEP_RATE_DEFAULT,
    RANDOM_STATE,
    TEST_RATIO,
    TRAIN_RATIO,
    VALID_RATIO,
    build_labels_binary_quantile,
    build_labels_3class_quantile,
    collect_obs_and_features_one_cnn,
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
    """keep_rate = mean(|r_{t,k}| > delta * atr_ratio[t])，r_{t,k} = log(close[t+k]/close[t])。向量化實作。"""
    n = len(close_arr)
    mask = (valid_indices >= 0) & (valid_indices + k < n)
    t_valid = valid_indices[mask]
    if t_valid.size == 0:
        return float("nan")
    close_t = close_arr[t_valid]
    close_tk = close_arr[t_valid + k]
    log_ret = np.where(close_t <= 0, 0.0, np.log(close_tk / close_t))
    thresh = delta * atr_ratio_arr[t_valid]
    keeps = np.abs(log_ret) > thresh
    return float(np.mean(keeps))


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


def _run_one_combo(pack: tuple) -> tuple:
    """ProcessPoolExecutor 用：可 pickle 的模組級函式。pack = (idx, kwargs)。"""
    idx, kwargs = pack
    try:
        return (idx, run_one_combination(**kwargs))
    except Exception as e:
        return (idx, {"_error": str(e), **{k: v for k, v in kwargs.items() if k != "use_gpu"}})


def run_one_combination(
    window_size_5m: int,
    window_size_1d: int,
    task: str,
    cnn_key: str,
    k: int = HORIZON_K,
    max_steps: Optional[int] = None,
    random_state: Optional[int] = None,
    return_regime_data: bool = False,
    use_gpu: bool = False,
    keep_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """
    對單一 (window_size_5m, window_size_1d, cnn_key, task) 跑完整 E2 pipeline（單路 CNN 摘要特徵）。
    二分類與三分類皆用分位數門檻（定保留率 keep_rate）；不依賴 atr_ratio。
    """
    device = "gpu" if use_gpu else "cpu"
    rs = RANDOM_STATE if random_state is None else random_state
    keep_rate_val = KEEP_RATE_DEFAULT if keep_rate is None else keep_rate
    valid_indices, X, close_arr, atr_ratio_arr = collect_obs_and_features_one_cnn(
        window_size=window_size_5m,
        window_size_1d=window_size_1d,
        horizon_k=k,
        max_steps=max_steps,
        cnn_key=cnn_key,
    )
    n_valid = len(valid_indices)
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
            "keep_rate": keep_rate_val,
            "keep_rate_judge": _keep_rate_judge(keep_rate_val),
            "auc_lr": nan,
            "auc_lgb": nan,
            "auc_label_shuf": nan,
            "auc_market_shuf": None,
            "macro_f1_lr": nan,
            "macro_f1_lgb": nan,
            "bal_acc_lgb": nan,
        }

    if task == "binary":
        keep_mask, y_binary = build_labels_binary_quantile(
            close_arr, valid_indices, k=k, keep_rate=keep_rate_val
        )
        X_kept = X[keep_mask]
        y_kept = y_binary[keep_mask]
        n_kept = int(keep_mask.sum())
        reported_keep_rate = float(keep_mask.mean())
        train_idx, valid_idx, test_idx = train_valid_test_split_time_ordered(
            n_kept, TRAIN_RATIO, VALID_RATIO, TEST_RATIO
        )
        X_train, X_test = X_kept[train_idx], X_kept[test_idx]
        y_train_bin = y_kept[train_idx]
        y_test_bin = y_kept[test_idx]
        out: Dict[str, Any] = {
            "window_size_5m": window_size_5m,
            "window_size_1d": window_size_1d,
            "cnn_key": cnn_key,
            "task": task,
            "n_samples": n_kept,
            "n_feat": X.shape[1],
            "n_feat_market": n_features_market,
            "keep_rate": reported_keep_rate,
            "keep_rate_judge": _keep_rate_judge(reported_keep_rate),
        }
        _, auc_lr, _ = fit_predict_binary(
            X_train, y_train_bin, X_test, y_test_bin,
            use_lightgbm=False, random_state=rs, device=device,
        )
        if return_regime_data:
            _, auc_lgb, _, proba_test = fit_predict_binary_return_proba(
                X_train, y_train_bin, X_test, y_test_bin,
                use_lightgbm=True, random_state=rs, device=device,
            )
            atr_ratio_kept = atr_ratio_arr[valid_indices][keep_mask]
            atr_ratio_test = atr_ratio_kept[test_idx]
            out["regime_data"] = {
                "atr_ratio_test": atr_ratio_test,
                "y_test_bin": y_test_bin,
                "proba_test": proba_test,
            }
        else:
            _, auc_lgb, _ = fit_predict_binary(
                X_train, y_train_bin, X_test, y_test_bin,
                use_lightgbm=True, random_state=rs, device=device,
            )
        auc_label_shuf = run_sanity_label_shuffle(
            X_train, y_train_bin, X_test, y_test_bin,
            use_lightgbm=True, random_state=rs, device=device,
        )
        if n_features_market > 0:
            auc_market_shuf = run_sanity_market_shuffle(
                X_kept, y_kept, train_idx, test_idx, n_features_market,
                use_lightgbm=True, random_state=rs, device=device,
            )
        else:
            auc_market_shuf = None
        out["auc_lr"] = auc_lr
        out["auc_lgb"] = auc_lgb
        out["auc_label_shuf"] = auc_label_shuf
        out["auc_market_shuf"] = auc_market_shuf
        return out

    # task == "3class"
    y_3class, y_binary = build_labels_3class_quantile(
        close_arr, valid_indices, k=k, keep_rate=keep_rate_val
    )
    train_idx, valid_idx, test_idx = train_valid_test_split_time_ordered(
        n_valid, TRAIN_RATIO, VALID_RATIO, TEST_RATIO
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train_bin = y_binary[train_idx]
    y_test_bin = y_binary[test_idx]
    y_train_3 = y_3class[train_idx]
    y_test_3 = y_3class[test_idx]
    out = {
        "window_size_5m": window_size_5m,
        "window_size_1d": window_size_1d,
        "cnn_key": cnn_key,
        "task": task,
        "n_samples": n_valid,
        "n_feat": X.shape[1],
        "n_feat_market": n_features_market,
        "keep_rate": keep_rate_val,
        "keep_rate_judge": _keep_rate_judge(keep_rate_val),
    }
    _, macro_f1_lr, bal_acc_lr, _ = fit_predict_3class(
        X_train, y_train_3, X_test, y_test_3,
        use_lightgbm=False, random_state=rs, device=device,
    )
    _, macro_f1_lgb, bal_acc_lgb, _ = fit_predict_3class(
        X_train, y_train_3, X_test, y_test_3,
        use_lightgbm=True, random_state=rs, device=device,
    )
    auc_label_shuf = run_sanity_label_shuffle(
        X_train, y_train_bin, X_test, y_test_bin,
        use_lightgbm=True, random_state=rs, device=device,
    )
    if n_features_market > 0:
        auc_market_shuf = run_sanity_market_shuffle(
            X, y_binary, train_idx, test_idx, n_features_market,
            use_lightgbm=True, random_state=rs, device=device,
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
) -> Tuple[List[int], List[int], np.ndarray]:
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


def _build_2d_grids(
    results: List[Dict[str, Any]],
    key_main: str,
    key_label: str = "auc_label_shuf",
) -> Tuple[List[int], List[int], np.ndarray, np.ndarray]:
    """一次遍歷 results 建出主指標與 Label shuffle 兩張矩陣。"""
    ws_5m = sorted({r["window_size_5m"] for r in results})
    ws_1d = sorted({r["window_size_1d"] for r in results})
    idx_5m = {w: i for i, w in enumerate(ws_5m)}
    idx_1d = {w: j for j, w in enumerate(ws_1d)}
    mat_main = np.full((len(ws_1d), len(ws_5m)), np.nan)
    mat_label = np.full((len(ws_1d), len(ws_5m)), np.nan)
    for r in results:
        i = idx_5m[r["window_size_5m"]]
        j = idx_1d[r["window_size_1d"]]
        mat_main[j, i] = r[key_main]
        mat_label[j, i] = r[key_label]
    return ws_5m, ws_1d, mat_main, mat_label


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
        ws_5m, ws_1d, mat_main, mat_label = _build_2d_grids(results, main_key)
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


def _fmt_report(v: Any) -> str:
    """報告用數值格式化，None/nan → N/A。"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.4f}"


def _write_main_report_chunk(
    report_path: str,
    results: List[Dict[str, Any]],
    cnn_keys: List[str],
    tasks: List[str],
    window_sizes_5m: List[int],
    window_sizes_1d: List[int],
    args: Any,
    regime_rows: List[Dict[str, Any]],
) -> str:
    """寫入報告檔：標題、主迴圈表格、保留樣本比例、Regime 分桶。回傳已寫入內容供印出。"""
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
        for r in subset:
            mkt_str = _fmt_report(r.get("auc_market_shuf"))
            if t == "binary":
                lines.append(
                    f"{r['window_size_5m']:>6} {r['window_size_1d']:>6} {r['n_samples']:>8} {r['n_feat']:>6} "
                    f"{_fmt_report(r.get('auc_lr')):>8} {_fmt_report(r.get('auc_lgb')):>8} "
                    f"{_fmt_report(r.get('auc_label_shuf')):>8} {mkt_str:>8}"
                )
            else:
                lines.append(
                    f"{r['window_size_5m']:>6} {r['window_size_1d']:>6} {r['n_samples']:>8} {r['n_feat']:>6} "
                    f"{_fmt_report(r.get('macro_f1_lr')):>8} {_fmt_report(r.get('macro_f1_lgb')):>8} {_fmt_report(r.get('bal_acc_lgb')):>8} "
                    f"{_fmt_report(r.get('auc_label_shuf')):>8} {mkt_str:>8}"
                )
        lines.append("")

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

    if regime_rows:
        lines.append("")
        lines.append("--- Regime 分桶 ---")
        lines.append(f"{'ws_5m':>6} {'ws_1d':>6} {'cnn_key':>12} {'bucket':>6} {'AUC':>8} {'判斷':>42}")
        lines.append("-" * 90)
        for row in regime_rows:
            auc_s = _fmt_report(row["auc"])
            lines.append(
                f"{row['window_size_5m']:>6} {row['window_size_1d']:>6} {row['cnn_key']:>12} {row['bucket']:>6} "
                f"{auc_s:>8} {row['judge']:>42}"
            )
    text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


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
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="並行數（ProcessPoolExecutor），1 為單行程序執行",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="LightGBM 使用 GPU（device='gpu'）；未指定則用 CPU",
    )
    parser.add_argument(
        "--keep_rate",
        type=float,
        default=KEEP_RATE_DEFAULT,
        help="分位數門檻之保留率 p（二分類：保留 |r|>q 的樣本；三分類：非 0 比例）。預設 0.3",
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

    use_gpu = getattr(args, "use_gpu", False)
    n_jobs = max(1, int(getattr(args, "n_jobs", 1)))
    keep_rate_arg = args.keep_rate
    combo_kwargs_list = [
        {
            "window_size_5m": ws_5m,
            "window_size_1d": ws_1d,
            "task": task,
            "cnn_key": cnn_key,
            "k": args.k,
            "max_steps": args.max_steps,
            "return_regime_data": (args.run_regime_bucket and task == "binary"),
            "use_gpu": use_gpu,
            "keep_rate": keep_rate_arg,
        }
        for (ws_5m, ws_1d), cnn_key, task in run_combos
    ]

    total_tasks = len(run_combos)
    start_wall = time.perf_counter()
    print("")
    print("=" * 60)
    print(f"  E2 by window: 共 {total_tasks} 個組合  n_jobs={n_jobs}  use_gpu={use_gpu}")
    print("=" * 60)

    results_order: List[Optional[Dict[str, Any]]] = [None] * len(run_combos)
    if n_jobs <= 1:
        if _TQDM_AVAILABLE:
            it = tqdm(enumerate(combo_kwargs_list), total=total_tasks, desc="E2 主迴圈", unit="組合")
        else:
            pbar = _SimpleProgress(total_tasks, desc="E2 主迴圈")
            it = enumerate(combo_kwargs_list)
        for idx, kwargs in it:
            (ws_5m, ws_1d), cnn_key, task = run_combos[idx]
            try:
                r = run_one_combination(**kwargs)
                results_order[idx] = r
                if not _TQDM_AVAILABLE:
                    pbar.update(1)
                if "_error" in r:
                    print(f"  [{idx+1}/{total_tasks}] Error: {r['_error']}")
                elif task == "binary":
                    print(f"  [{idx+1}/{total_tasks}] ws_5m={ws_5m} ws_1d={ws_1d} {cnn_key} AUC_LGB={r['auc_lgb']:.4f}")
                else:
                    print(f"  [{idx+1}/{total_tasks}] ws_5m={ws_5m} ws_1d={ws_1d} {cnn_key} F1_LGB={r['macro_f1_lgb']:.4f}")
            except Exception as e:
                if not _TQDM_AVAILABLE:
                    pbar.update(1)
                print(f"  [{idx+1}/{total_tasks}] Error: {e}")
            gc.collect()
        if not _TQDM_AVAILABLE:
            pbar.close()
        results = [r for r in results_order if r is not None and "_error" not in r]
    else:
        if _TQDM_AVAILABLE:
            pbar = tqdm(total=total_tasks, desc="E2 主迴圈", unit="組合")
        else:
            pbar = _SimpleProgress(total_tasks, desc="E2 主迴圈")
        inputs = [(i, combo_kwargs_list[i]) for i in range(len(run_combos))]
        executor = ProcessPoolExecutor(max_workers=n_jobs)
        try:
            futures = {executor.submit(_run_one_combo, inp): inp[0] for inp in inputs}
            for fut in as_completed(futures):
                idx = futures[fut]
                (ws_5m, ws_1d), cnn_key, task = run_combos[idx]
                pbar.update(1)
                pbar.set_postfix_str(f"ws_5m={ws_5m} {cnn_key}")
                try:
                    i, r = fut.result()
                    results_order[i] = r
                    if r.get("_error"):
                        print(f"\n  [{idx+1}/{total_tasks}] Error: {r['_error']}")
                    elif task == "binary":
                        print(f"\n  [{idx+1}/{total_tasks}] ws_5m={ws_5m} ws_1d={ws_1d} {cnn_key} AUC_LGB={r['auc_lgb']:.4f}")
                    else:
                        print(f"\n  [{idx+1}/{total_tasks}] ws_5m={ws_5m} ws_1d={ws_1d} {cnn_key} F1_LGB={r['macro_f1_lgb']:.4f}")
                except Exception as e:
                    print(f"\n  [{idx+1}/{total_tasks}] Error: {e}")
        finally:
            pbar.close()
            print("\n正在等待 worker 行程結束（若卡在此處請用 --n_jobs 1）...", flush=True)
            executor.shutdown(wait=True)
            print("Worker 已結束.", flush=True)
        results = [r for r in results_order if r is not None and r.get("_error") is None]

    elapsed_min = (time.perf_counter() - start_wall) / 60
    print("")
    print("=" * 60)
    print(f"  主迴圈完成: {len(results)}/{total_tasks} 成功  總耗時 {elapsed_min:.1f} 分")
    print("=" * 60)

    if not results:
        print("No results to report or plot.")
        return

    # Regime 分桶（從 results 計算，供主報告使用）
    regime_rows: List[Dict[str, Any]] = []
    if args.run_regime_bucket:
        for r in results:
            if r.get("task") != "binary" or "regime_data" not in r:
                continue
            rd = r["regime_data"]
            auc_low, auc_mid, auc_high = _regime_auc_per_bucket(
                rd["atr_ratio_test"], rd["y_test_bin"], rd["proba_test"]
            )
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

    # 主迴圈結束後立即寫報告並畫圖，釋放記憶體
    print("正在寫報告（主迴圈）...", flush=True)
    main_report_text = _write_main_report_chunk(
        report_path, results, cnn_keys, tasks,
        window_sizes_5m, window_sizes_1d, args, regime_rows,
    )
    print("\n" + main_report_text)
    print(f"報告已寫入: {report_path}", flush=True)
    del regime_rows
    gc.collect()

    # 先產生圖，再跑後續階段（依 (cnn_key, task) 分組，只掃 results 一次）
    by_cnn_task: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_cnn_task[(r["cnn_key"], r["task"])].append(r)
    fig_combos = [(ck, t) for (ck, t) in product(cnn_keys, tasks) if (ck, t) in by_cnn_task]
    for fig_idx, (cnn_key, t) in enumerate(fig_combos, 1):
        print(f"正在畫圖 ({fig_idx}/{len(fig_combos)}): {cnn_key} {t}...", flush=True)
        subset = by_cnn_task[(cnn_key, t)]
        fig_name = f"e2_by_window_{cnn_key}_{t}.png"
        fig_path = os.path.join(args.out_dir, fig_name)
        plot_results(subset, fig_path, task=t)
        print(f"  已存: {fig_path}", flush=True)
    gc.collect()

    # 穩定性（3 seeds）：跑完即追加寫入報告並釋放
    stability_results: List[Dict[str, Any]] = []
    if args.run_stability:
        total_stab = len(run_combos) * len(STABILITY_SEEDS)
        if _TQDM_AVAILABLE:
            pbar_stab = tqdm(total=total_stab, desc="穩定性 (3 seeds)", unit="run")
        else:
            pbar_stab = _SimpleProgress(total_stab, desc="穩定性 (3 seeds)")
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
                        use_gpu=use_gpu,
                        keep_rate=keep_rate_arg,
                    )
                    main_metric = r["auc_lgb"] if task == "binary" else r["macro_f1_lgb"]
                    auc_list.append(main_metric)
                except Exception:
                    auc_list.append(float("nan"))
                pbar_stab.update(1)
                pbar_stab.set_postfix_str(f"ws_5m={ws_5m} {cnn_key}")
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
        pbar_stab.close()
        # 穩定性跑完即寫入報告並釋放
        stab_lines = [
            "",
            "--- 穩定性（3 seeds） ---",
            f"{'ws_5m':>6} {'ws_1d':>6} {'cnn_key':>12} {'task':>8} {'AUC_s1':>8} {'AUC_s2':>8} {'AUC_s3':>8} {'std':>8} {'判斷':>28}",
            "-" * 100,
        ]
        for r in stability_results:
            a = r["auc_list"]
            stab_lines.append(
                f"{r['window_size_5m']:>6} {r['window_size_1d']:>6} {r['cnn_key']:>12} {r['task']:>8} "
                f"{_fmt_report(a[0]):>8} {_fmt_report(a[1]):>8} {_fmt_report(a[2]):>8} {_fmt_report(r['std_auc']):>8} {r['judge']:>28}"
            )
        stab_text = "\n".join(stab_lines)
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(stab_text)
        print(stab_text)
        del stability_results
        gc.collect()

    # Horizon sweep：task=binary，k ∈ {12, 36, 72}，keep_rate 同主迴圈
    horizon_sweep_results: List[Dict[str, Any]] = []
    if args.run_horizon_sweep:
        horizon_tasks = [
            ((ws_5m, ws_1d), cnn_key, k)
            for (ws_5m, ws_1d), cnn_key in product(window_combos, cnn_keys)
            for k in HORIZON_SWEEP_KS
        ]
        total_hor = len(horizon_tasks)
        if _TQDM_AVAILABLE:
            it_hor = tqdm(horizon_tasks, desc="Horizon sweep", unit="run")
        else:
            pbar_hor = _SimpleProgress(total_hor, desc="Horizon sweep")
            it_hor = horizon_tasks
        for ((ws_5m, ws_1d), cnn_key, k) in it_hor:
            if not _TQDM_AVAILABLE:
                pbar_hor.update(1)
                pbar_hor.set_postfix_str(f"ws_5m={ws_5m} {cnn_key} k={k}")
            try:
                r = run_one_combination(
                    window_size_5m=ws_5m,
                    window_size_1d=ws_1d,
                    task="binary",
                    cnn_key=cnn_key,
                    k=k,
                    max_steps=args.max_steps,
                    use_gpu=use_gpu,
                    keep_rate=keep_rate_arg,
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
        if not _TQDM_AVAILABLE:
            pbar_hor.close()
        # Horizon 跑完即追加寫入報告並釋放
        hor_lines = [
            "",
            "--- Horizon sweep ---",
            f"{'ws_5m':>6} {'ws_1d':>6} {'cnn_key':>12} {'k':>4} {'AUC_LGB':>8} {'LblShuf':>8} {'判斷':>36}",
            "-" * 90,
        ]
        for r in horizon_sweep_results:
            hor_lines.append(
                f"{r['window_size_5m']:>6} {r['window_size_1d']:>6} {r['cnn_key']:>12} {r['k']:>4} "
                f"{_fmt_report(r.get('auc_lgb')):>8} {_fmt_report(r.get('auc_label_shuf')):>8} {r['judge']:>36}"
            )
        hor_text = "\n".join(hor_lines)
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(hor_text)
        print(hor_text)
        del horizon_sweep_results
        gc.collect()

    # 報告結尾
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 100)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
