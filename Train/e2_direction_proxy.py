"""
E2 最小可驗證規格：方向 proxy（bar close 決策、統計彙總特徵、時間切分、LR + LightGBM、破壞測試）。

- 標籤：二分類 y = 1[log(close_{t+k}/close_t) > 0]，三分類 ±1/0 用 δ = 0.1*atr_ratio[t]。
- 特徵：預設時序彙總（近期+全窗）每 channel 7 統計量 → 7*F；或 five_stats 模式 5*F。可 concat account_state(22)。
- 時間切分：Train 70% / Valid 15% / Test 15%，禁止 shuffle。
- 模型：LogisticRegression(L2) + LightGBM。
- 破壞測試：label shuffle（AUC→0.5）、market shuffle（AUC 顯著下降）。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# 專案根目錄加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Env.load_file import load_data
from Env.trading_env import TradingEnvironment
from Train.train_config import TrainConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb


# ---------- 常數 ----------
HORIZON_K = 6
DELTA_COEF = 10
KEEP_RATE_DEFAULT = 0.3  # 分位數門檻之預設保留率 p
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42
# Account state: S = 前 12 維（state-only），H = 後 10 維（history/behavior）
ACCOUNT_S_NDIM = 12
# 時序彙總：近期視窗 bar 數（與 HORIZON_K 對齊，保留短期結構）
RECENT_WINDOW_BARS = 12
# 雙時間尺度彙總：每 channel 7 個統計量（近期 3 + 全窗 4）
SUMMARY_STATS_TEMPORAL: Tuple[str, ...] = (
    "recent_last",
    "recent_mean",
    "recent_std",
    "full_mean",
    "full_std",
    "full_min",
    "full_max",
)
STATS_PER_CHANNEL_TEMPORAL = len(SUMMARY_STATS_TEMPORAL)  # 7
# 三段時序彙總：early / mid / late，每段 mean+std → 6*F
SUMMARY_STATS_THREE_SEGMENT: Tuple[str, ...] = (
    "early_mean",
    "early_std",
    "mid_mean",
    "mid_std",
    "late_mean",
    "late_std",
)
STATS_PER_CHANNEL_THREE_SEGMENT = len(SUMMARY_STATS_THREE_SEGMENT)  # 6
# 舊版單一視窗彙總（相容用）
SUMMARY_STATS_FIVE: Tuple[str, ...] = ("last", "mean", "std", "min", "max")
STATS_PER_CHANNEL_FIVE = 5

FEATURE_MODES = ("market_only", "account_s_only", "market_plus_account_s", "market_plus_account_full")

# 四路 CNN 對應 obs key，供單路摘要與依 CNN 輸出用
CNN_KEYS = ("5m_target", "5m_others", "1d_target", "1d_others")
OBS_KEY_BY_CNN = {
    "5m_target": "price_seq_target",
    "5m_others": "price_seq_others",
    "1d_target": "price_seq_1d_target",
    "1d_others": "price_seq_1d_others",
}


def n_market_dims_from_feature_mode(feature_mode: str, n_total: int) -> int:
    """
    依 feature_mode 與總特徵維度 n_total 回傳「市場」特徵數（供 market shuffle 用）。
    account_s_only 時為 0；market_only 時為 n_total；其餘為 n_total - account 維數。
    """
    if feature_mode == "account_s_only":
        return 0
    if feature_mode == "market_only":
        return n_total
    if feature_mode == "market_plus_account_s":
        return n_total - ACCOUNT_S_NDIM
    return n_total - 22  # market_plus_account_full


def build_labels(
    close_arr: np.ndarray,
    atr_ratio_arr: np.ndarray,
    valid_indices: np.ndarray,
    k: int = HORIZON_K,
    delta_coef: float = DELTA_COEF,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    依 close[t], close[t+k], atr_ratio[t] 計算二分類與三分類標籤（僅對 valid_indices）。

    Returns:
        y_binary: 0/1，shape (n_valid,)
        y_3class: -1/0/1，shape (n_valid,)
    """
    n = len(valid_indices)
    y_binary = np.zeros(n, dtype=np.int32)
    y_3class = np.zeros(n, dtype=np.int32)

    for i, t in enumerate(valid_indices):
        close_t = close_arr[t]
        close_tk = close_arr[t + k]
        if close_t <= 0:
            log_ret = 0.0
        else:
            log_ret = np.log(close_tk / close_t)

        # 二分類
        y_binary[i] = 1 if log_ret > 0 else 0

        # 三分類：δ = delta_coef * atr_ratio[t]
        delta = delta_coef * float(atr_ratio_arr[t])
        if log_ret > delta:
            y_3class[i] = 1
        elif log_ret < -delta:
            y_3class[i] = -1
        else:
            y_3class[i] = 0

    return y_binary, y_3class


def _log_returns_for_valid(
    close_arr: np.ndarray, valid_indices: np.ndarray, k: int
) -> np.ndarray:
    """向量化計算 valid_indices 對應的 r_{t,k} = log(close[t+k]/close[t])。長度 n_valid。"""
    n = len(close_arr)
    close_t = close_arr[valid_indices]
    close_tk = close_arr[valid_indices + k]
    r = np.where(close_t <= 0, 0.0, np.log(close_tk / close_t))
    return r.astype(np.float64)


def build_labels_binary_quantile(
    close_arr: np.ndarray,
    valid_indices: np.ndarray,
    k: int = HORIZON_K,
    keep_rate: float = KEEP_RATE_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    分位數門檻二分類：q = quantile(|r|, 1-p)，只保留 |r| > q 的樣本，label = sign(r)。
    回傳 (keep_mask, y_binary)，長度皆 n_valid；caller 以 X[keep_mask], y_binary[keep_mask] 得 X_kept, y_kept。
    """
    r = _log_returns_for_valid(close_arr, valid_indices, k)
    abs_r = np.abs(r)
    q = np.nanpercentile(abs_r, (1.0 - keep_rate) * 100.0)
    keep_mask = abs_r > q
    y_binary = (r > 0).astype(np.int32)
    return keep_mask, y_binary


def build_labels_3class_quantile(
    close_arr: np.ndarray,
    valid_indices: np.ndarray,
    k: int = HORIZON_K,
    keep_rate: float = KEEP_RATE_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    分位數門檻三分類（定保留率）：q_high = quantile(r, 1-p/2)，q_low = quantile(r, p/2)；
    y_3class = 1 if r > q_high, -1 if r < q_low, else 0。不刪樣本，全體 n_valid 參與。
    回傳 (y_3class, y_binary)，y_binary 供 sanity 用。
    """
    r = _log_returns_for_valid(close_arr, valid_indices, k)
    q_high = np.nanpercentile(r, (1.0 - keep_rate / 2.0) * 100.0)
    q_low = np.nanpercentile(r, (keep_rate / 2.0) * 100.0)
    y_3class = np.where(r > q_high, 1, np.where(r < q_low, -1, 0)).astype(np.int32)
    y_binary = (r > 0).astype(np.int32)
    return y_3class, y_binary


def _seq_to_five_stats(seq: np.ndarray) -> np.ndarray:
    """對 (T, F) 序列取 last, mean, std, min, max，回傳 (5*F,) float64。"""
    seq = np.asarray(seq, dtype=np.float64)
    last = seq[-1]
    mean = np.mean(seq, axis=0)
    std = np.std(seq, axis=0)
    np.place(std, std <= 0, 1e-12)
    min_ = np.min(seq, axis=0)
    max_ = np.max(seq, axis=0)
    return np.concatenate([last, mean, std, min_, max_])


def _seq_to_temporal_stats(
    seq: np.ndarray,
    recent_bars: int = RECENT_WINDOW_BARS,
) -> np.ndarray:
    """
    雙時間尺度彙總：保留「近期 vs 全窗」時序，回傳 (7*F,) float64。

    - 近期（last recent_bars）：recent_last, recent_mean, recent_std
    - 全窗（all T）：full_mean, full_std, full_min, full_max

    當 T < recent_bars 時，近期用整段序列計算。
    """
    seq = np.asarray(seq, dtype=np.float64)
    T, F = seq.shape
    R = min(max(1, int(recent_bars)), T)
    recent_slice = seq[-R:]
    recent_last = recent_slice[-1]
    recent_mean = np.mean(recent_slice, axis=0)
    recent_std = np.std(recent_slice, axis=0)
    np.place(recent_std, recent_std <= 0, 1e-12)
    full_mean = np.mean(seq, axis=0)
    full_std = np.std(seq, axis=0)
    np.place(full_std, full_std <= 0, 1e-12)
    full_min = np.min(seq, axis=0)
    full_max = np.max(seq, axis=0)
    return np.concatenate(
        [recent_last, recent_mean, recent_std, full_mean, full_std, full_min, full_max]
    )


def _seq_to_three_segment_stats(seq: np.ndarray) -> np.ndarray:
    """
    三段時序彙總：將視窗均分為 early / mid / late，每段取 mean 與 std，回傳 (6*F,) float64。
    用於評估「視窗內哪一時段」對預測較重要。
    """
    seq = np.asarray(seq, dtype=np.float64)
    T, F = seq.shape
    if T < 3:
        seg = seq
        one_mean = np.mean(seg, axis=0)
        one_std = np.std(seg, axis=0)
        np.place(one_std, one_std <= 0, 1e-12)
        return np.concatenate([one_mean, one_std] * 3)
    n1 = T // 3
    n2 = (T - n1) // 2
    n3 = T - n1 - n2
    early = seq[:n1]
    mid = seq[n1 : n1 + n2]
    late = seq[n1 + n2 :]
    early_mean = np.mean(early, axis=0)
    early_std = np.std(early, axis=0)
    np.place(early_std, early_std <= 0, 1e-12)
    mid_mean = np.mean(mid, axis=0)
    mid_std = np.std(mid, axis=0)
    np.place(mid_std, mid_std <= 0, 1e-12)
    late_mean = np.mean(late, axis=0)
    late_std = np.std(late, axis=0)
    np.place(late_std, late_std <= 0, 1e-12)
    return np.concatenate(
        [early_mean, early_std, mid_mean, mid_std, late_mean, late_std]
    )


def obs_to_summary_features(
    obs: Dict[str, Any],
    use_1d: bool = False,
    feature_mode: str = "market_plus_account_full",
    *,
    summary_mode: str = "temporal",
    recent_bars: int = RECENT_WINDOW_BARS,
) -> np.ndarray:
    """
    從單步 obs 抽出統計彙總特徵；依 feature_mode 決定市場與帳戶組合。

    feature_mode:
        market_only: 僅市場彙總（5m + 可選 1d）
        account_s_only: 僅 account_state[0:ACCOUNT_S_NDIM]
        market_plus_account_s: 市場彙總 + account_state[0:ACCOUNT_S_NDIM]
        market_plus_account_full: 市場彙總 + account_state 全 22 維（預設）

    summary_mode: "temporal"（近期+全窗）或 "five_stats"（單一視窗 5 統計量）
    recent_bars: summary_mode=="temporal" 時近期視窗 bar 數

    Returns:
        一維 float32 向量，維度依 feature_mode 與 summary_mode 不同。
    """
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}, got {feature_mode!r}")
    acc = obs["account_state"]
    if hasattr(acc, "numpy"):
        acc = acc.numpy()
    acc = np.asarray(acc, dtype=np.float64).ravel()
    parts: List[np.ndarray] = []
    if summary_mode == "temporal":
        _seq_summary = lambda s: _seq_to_temporal_stats(s, recent_bars=recent_bars)
    else:
        _seq_summary = _seq_to_five_stats
    if feature_mode in ("market_only", "market_plus_account_s", "market_plus_account_full"):
        seq = obs["price_seq_target"]
        if hasattr(seq, "numpy"):
            seq = seq.numpy()
        feats_5m = _seq_summary(np.asarray(seq, dtype=np.float64))
        parts.append(feats_5m)
        if use_1d and "price_seq_1d_target" in obs:
            seq_1d = obs["price_seq_1d_target"]
            if hasattr(seq_1d, "numpy"):
                seq_1d = seq_1d.numpy()
            parts.append(_seq_summary(np.asarray(seq_1d, dtype=np.float64)))
    if feature_mode == "account_s_only":
        parts.append(acc[0:ACCOUNT_S_NDIM].copy())
    elif feature_mode == "market_plus_account_s":
        parts.append(acc[0:ACCOUNT_S_NDIM].copy())
    elif feature_mode == "market_plus_account_full":
        parts.append(acc)
    return np.concatenate(parts).astype(np.float32)


def obs_to_summary_features_one_cnn(
    obs: Dict[str, Any],
    cnn_key: str,
    *,
    summary_mode: str = "temporal",
    recent_bars: int = RECENT_WINDOW_BARS,
) -> np.ndarray:
    """
    從單步 obs 抽出單一 CNN 的統計彙總特徵。

    Args:
        obs: 單步觀察 dict
        cnn_key: 其一 CNN_KEYS（5m_target, 5m_others, 1d_target, 1d_others）
        summary_mode: "temporal"（近期+全窗，7*F）、"three_segment"（早/中/晚，6*F）或 "five_stats"（5*F）
        recent_bars: summary_mode=="temporal" 時近期視窗 bar 數

    Returns:
        一維 float32 向量，長度依 summary_mode：7*F / 6*F / 5*F。
    """
    if cnn_key not in OBS_KEY_BY_CNN:
        raise ValueError(f"cnn_key must be one of {CNN_KEYS}, got {cnn_key!r}")
    obs_key = OBS_KEY_BY_CNN[cnn_key]
    if obs_key not in obs:
        raise KeyError(f"obs missing key {obs_key!r}")
    seq = obs[obs_key]
    if hasattr(seq, "numpy"):
        seq = seq.numpy()
    seq = np.asarray(seq, dtype=np.float64)
    if summary_mode == "temporal":
        return _seq_to_temporal_stats(seq, recent_bars=recent_bars).astype(np.float32)
    if summary_mode == "three_segment":
        return _seq_to_three_segment_stats(seq).astype(np.float32)
    if summary_mode == "five_stats":
        return _seq_to_five_stats(seq).astype(np.float32)
    raise ValueError(
        f"summary_mode must be 'temporal', 'three_segment' or 'five_stats', got {summary_mode!r}"
    )


def train_valid_test_split_time_ordered(
    n: int,
    train_ratio: float = TRAIN_RATIO,
    valid_ratio: float = VALID_RATIO,
    test_ratio: float = TEST_RATIO,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    按時間序切分索引：前 train_ratio、中 valid_ratio、後 test_ratio。
    返回 train_idx, valid_idx, test_idx（各為 0..n-1 的子集）。
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6
    indices = np.arange(n)
    n_train = int(round(n * train_ratio))
    n_valid = int(round(n * valid_ratio))
    n_test = n - n_train - n_valid
    train_idx = indices[:n_train]
    valid_idx = indices[n_train : n_train + n_valid]
    test_idx = indices[n_train + n_valid :]
    return train_idx, valid_idx, test_idx


def collect_obs_and_indices(
    window_size: int = 288,
    window_size_1d: Optional[int] = None,
    horizon_k: int = HORIZON_K,
    max_steps: Optional[int] = None,
    return_cnn_cols: bool = False,
) -> Union[
    Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray],
    Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, Dict[str, Tuple[str, ...]]],
]:
    """
    建立 env（全量資料、不切分），以固定策略 action=0 沿時間軸收集 obs，僅保留可算 label 的 step_idx。

    Args:
        window_size: 5m 視窗長度（bar 數）
        window_size_1d: 1d 視窗長度（日數）；None 時使用 Env 預設
        horizon_k: 標籤 horizon（bar 數）
        max_steps: 最多收集步數，None 表示不限制
        return_cnn_cols: 若 True，多回傳 cnn_cols（四路 CNN 的 channel 名稱）

    Returns:
        obs_list, valid_indices, close_arr, atr_ratio_arr；若 return_cnn_cols 則再回傳 cnn_cols。
        cnn_cols: Dict[cnn_key, tuple of channel names]，cnn_key in CNN_KEYS。
    """
    kwargs = {
        "env_id": 0,
        "window_size": window_size,
        "data_split_enabled": False,
        "random_start": False,
        "max_episode_steps": 500000,
        "target_symbol": TrainConfig.SYMBOL,
        "feature_symbols": list(TrainConfig.FEATURE_SYMBOLS),
    }
    if window_size_1d is not None:
        kwargs["window_size_1d"] = window_size_1d
    env = TradingEnvironment(**kwargs)
    close_arr = np.array(env.market_data.close_arr, copy=True)
    atr_ratio_arr = np.array(env.market_data.atr_ratio_arr, copy=True)
    N = len(close_arr)
    last_valid_step = N - 1 - horizon_k
    if last_valid_step < window_size:
        env.close()
        raise ValueError(
            f"Data too short: need at least window_size + k = {window_size + horizon_k}, got N={N}"
        )

    obs, _ = env.reset()
    done = False
    truncated = False
    steps = 0
    obs_list: List[Dict[str, Any]] = []
    step_indices: List[int] = []
    limit = (N - 1) if max_steps is None else min(N - 1, max_steps)

    while not (done or truncated) and env.current_step <= last_valid_step and steps < limit:
        step_idx = int(env.get_current_step())
        if step_idx >= window_size and step_idx <= last_valid_step:
            obs_list.append(obs)
            step_indices.append(step_idx)
        action = np.array([0.0], dtype=np.float32)  # 固定平倉
        obs, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        steps += 1

    if return_cnn_cols:
        # 使用 observer 的有效欄位索引，使 cnn_cols 與 obs 實際 channel 數一致（Config 可能只納入部分欄位）
        obs = env.observer
        cols_5m_t = env.market_data.cols_5m_target
        cols_5m_o = env.market_data.cols_5m_others
        cols_1d_t = env.market_data.cols_1d_target
        cols_1d_o = env.market_data.cols_1d_others
        cnn_cols = {
            "5m_target": tuple(cols_5m_t[i] for i in obs._obs_price_seq_target_idx),
            "5m_others": tuple(cols_5m_o[i] for i in obs._obs_price_seq_others_idx),
            "1d_target": tuple(cols_1d_t[i] for i in obs._obs_price_seq_1d_target_idx),
            "1d_others": tuple(cols_1d_o[i] for i in obs._obs_price_seq_1d_others_idx),
        }
    env.close()
    valid_indices = np.array(step_indices, dtype=np.int64)
    if return_cnn_cols:
        return obs_list, valid_indices, close_arr, atr_ratio_arr, cnn_cols
    return obs_list, valid_indices, close_arr, atr_ratio_arr


def collect_obs_and_features_one_cnn(
    window_size: int = 288,
    window_size_1d: Optional[int] = None,
    horizon_k: int = HORIZON_K,
    max_steps: Optional[int] = None,
    cnn_key: str = "5m_target",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    串流收集：沿時間軸 step 時只計算單路 CNN 的彙總特徵並累積，不保留 obs，降低大 window 時的記憶體。

    Returns:
        valid_indices, X, close_arr, atr_ratio_arr。X 為 (n_valid, n_feat) float32。
    """
    if cnn_key not in CNN_KEYS:
        raise ValueError(f"cnn_key must be in {CNN_KEYS}, got {cnn_key!r}")
    kwargs = {
        "env_id": 0,
        "window_size": window_size,
        "data_split_enabled": False,
        "random_start": False,
        "max_episode_steps": 500000,
        "target_symbol": TrainConfig.SYMBOL,
        "feature_symbols": list(TrainConfig.FEATURE_SYMBOLS),
    }
    if window_size_1d is not None:
        kwargs["window_size_1d"] = window_size_1d
    env = TradingEnvironment(**kwargs)
    close_arr = np.array(env.market_data.close_arr, copy=True)
    atr_ratio_arr = np.array(env.market_data.atr_ratio_arr, copy=True)
    N = len(close_arr)
    last_valid_step = N - 1 - horizon_k
    if last_valid_step < window_size:
        env.close()
        raise ValueError(
            f"Data too short: need at least window_size + k = {window_size + horizon_k}, got N={N}"
        )

    obs, _ = env.reset()
    done = False
    truncated = False
    steps = 0
    step_indices: List[int] = []
    feature_list: List[np.ndarray] = []
    limit = (N - 1) if max_steps is None else min(N - 1, max_steps)

    while not (done or truncated) and env.current_step <= last_valid_step and steps < limit:
        step_idx = int(env.get_current_step())
        if step_idx >= window_size and step_idx <= last_valid_step:
            step_indices.append(step_idx)
            feat = obs_to_summary_features_one_cnn(obs, cnn_key)
            feature_list.append(feat)
        action = np.array([0.0], dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        steps += 1

    env.close()
    valid_indices = np.array(step_indices, dtype=np.int64)
    X = np.stack(feature_list, axis=0) if feature_list else np.empty((0, 0), dtype=np.float32)
    return valid_indices, X, close_arr, atr_ratio_arr


def _to_lgb_df(X: np.ndarray) -> pd.DataFrame:
    """將 numpy 特徵矩陣轉成具欄位名的 DataFrame，供 LightGBM 使用以消除 feature names 警告。"""
    n_cols = X.shape[1]
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(n_cols)])


def fit_predict_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_lightgbm: bool,
    random_state: int = RANDOM_STATE,
    device: str = "cpu",
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """訓練二分類並回傳 test 預測、AUC、與 precision/recall。device: 'cpu' | 'gpu'（LightGBM 用）。"""
    if use_lightgbm:
        df_train = _to_lgb_df(X_train)
        df_test = _to_lgb_df(X_test)
        lgb_kw: Dict[str, Any] = dict(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=random_state,
            verbosity=-1,
            n_jobs=1,
        )
        if device in ("gpu", "cuda"):
            lgb_kw["device"] = "gpu"
        model = lgb.LGBMClassifier(**lgb_kw)
        model.fit(df_train, y_train)
        proba = model.predict_proba(df_test)[:, 1]
        pred = model.predict(df_test)
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = LogisticRegression(
            penalty="l2", max_iter=3000, random_state=random_state, solver="lbfgs"
        )
        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_test_scaled)[:, 1]
        pred = model.predict(X_test_scaled)

    auc = roc_auc_score(y_test, proba) if np.unique(y_test).size > 1 else 0.5
    acc = accuracy_score(y_test, pred)
    prec, rec, _, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    return pred, auc, {"accuracy": acc, "precision": prec, "recall": rec}


def fit_predict_binary_return_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_lightgbm: bool,
    random_state: int = RANDOM_STATE,
    device: str = "cpu",
) -> Tuple[np.ndarray, float, Dict[str, float], np.ndarray]:
    """訓練二分類並回傳 test 預測、AUC、輔助指標與 test 預測機率（供 Regime 分桶用）。"""
    if use_lightgbm:
        df_train = _to_lgb_df(X_train)
        df_test = _to_lgb_df(X_test)
        lgb_kw = dict(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=random_state,
            verbosity=-1,
            n_jobs=1,
        )
        if device in ("gpu", "cuda"):
            lgb_kw["device"] = "gpu"
        model = lgb.LGBMClassifier(**lgb_kw)
        model.fit(df_train, y_train)
        proba = model.predict_proba(df_test)[:, 1]
        pred = model.predict(df_test)
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = LogisticRegression(
            penalty="l2", max_iter=3000, random_state=random_state, solver="lbfgs"
        )
        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_test_scaled)[:, 1]
        pred = model.predict(X_test_scaled)

    auc = roc_auc_score(y_test, proba) if np.unique(y_test).size > 1 else 0.5
    acc = accuracy_score(y_test, pred)
    prec, rec, _, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    return pred, auc, {"accuracy": acc, "precision": prec, "recall": rec}, proba


def fit_predict_3class(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_lightgbm: bool,
    random_state: int = RANDOM_STATE,
    device: str = "cpu",
) -> Tuple[np.ndarray, float, float, Dict[str, float]]:
    """三分類：y in {-1,0,1}。回傳 test 預測、macro_f1、balanced_accuracy、輔助指標。"""
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    if use_lightgbm:
        df_train = _to_lgb_df(X_train)
        df_test = _to_lgb_df(X_test)
        lgb_kw = dict(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=random_state,
            verbosity=-1,
            n_jobs=1,
        )
        if device in ("gpu", "cuda"):
            lgb_kw["device"] = "gpu"
        model = lgb.LGBMClassifier(**lgb_kw)
        model.fit(df_train, y_train_enc)
        pred_enc = model.predict(df_test)
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = LogisticRegression(
            penalty="l2", max_iter=3000, random_state=random_state, solver="lbfgs"
        )
        model.fit(X_train_scaled, y_train_enc)
        pred_enc = model.predict(X_test_scaled)

    pred = le.inverse_transform(pred_enc)
    macro_f1 = f1_score(y_test_enc, pred_enc, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_test_enc, pred_enc)
    acc = accuracy_score(y_test_enc, pred_enc)
    prec, rec, _, _ = precision_recall_fscore_support(
        y_test_enc, pred_enc, average="macro", zero_division=0
    )
    return pred, macro_f1, bal_acc, {"accuracy": acc, "precision": prec, "recall": rec}


def run_sanity_label_shuffle(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_lightgbm: bool,
    random_state: int = RANDOM_STATE,
    device: str = "cpu",
) -> float:
    """破壞測試 A：打亂 y_train，再訓練二分類，回傳 Test AUC（預期 ~0.5）。"""
    rng = np.random.default_rng(random_state)
    y_train_shuf = rng.permutation(y_train)
    _, auc, _ = fit_predict_binary(
        X_train, y_train_shuf, X_test, y_test,
        use_lightgbm=use_lightgbm, random_state=random_state, device=device,
    )
    return auc


def run_sanity_market_shuffle(
    X_full: np.ndarray,
    y_full: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_features_market: int,
    use_lightgbm: bool,
    random_state: int = RANDOM_STATE,
    device: str = "cpu",
) -> float:
    """
    破壞測試 B：對每個樣本的「市場特徵」（前 n_features_market 維）用隨機 permutation 打亂樣本間對應
    （即：把該維度整列重排），帳戶特徵（後 22 維）不變。再訓練二分類，回傳 Test AUC（預期顯著下降）。

    實作：對 X 的每一行，前 n_features_market 維用「隨機選的另一行的市場部分」替換，後 22 維保留。
    """
    rng = np.random.default_rng(random_state)
    n = X_full.shape[0]
    X_shuf = X_full.copy()
    perm = rng.permutation(n)
    X_shuf[:, :n_features_market] = X_full[perm, :n_features_market]

    X_tr = X_shuf[train_idx]
    y_tr = y_full[train_idx]
    X_te = X_shuf[test_idx]
    y_te = y_full[test_idx]
    _, auc, _ = fit_predict_binary(
        X_tr, y_tr, X_te, y_te,
        use_lightgbm=use_lightgbm, random_state=random_state, device=device,
    )
    return auc


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 direction proxy: LR + LightGBM, sanity tests")
    parser.add_argument("--window_size", type=int, default=288)
    parser.add_argument("--k", type=int, default=HORIZON_K)
    parser.add_argument("--max_steps", type=int, default=None, help="Cap collection steps (default: all)")
    parser.add_argument("--use_3class", action="store_true", help="Use 3-class labels for primary metrics")
    parser.add_argument("--no_sanity", action="store_true", help="Skip sanity tests")
    parser.add_argument("--out", type=str, default="logs/e2_report.txt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print("E2: Loading data and collecting obs...")
    obs_list, valid_indices, close_arr, atr_ratio_arr = collect_obs_and_indices(
        window_size=args.window_size,
        horizon_k=args.k,
        max_steps=args.max_steps,
    )
    n_valid = len(valid_indices)
    print(f"  Collected {n_valid} samples (valid step indices).")

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

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("E2 Direction Proxy Report (LR + LightGBM, per CNN)")
    lines.append("=" * 60)
    lines.append(f"Train/Valid/Test: {len(train_idx)} / {len(valid_idx)} / {len(test_idx)}")
    lines.append("")

    for cnn_key in CNN_KEYS:
        lines.append(f"--- CNN: {cnn_key} ---")
        X_list = [obs_to_summary_features_one_cnn(o, cnn_key) for o in obs_list]
        X = np.stack(X_list, axis=0)
        n_feat = X.shape[1]
        if n_feat == 0:
            lines.append("Skip (no features, F=0).")
            lines.append("")
            continue
        n_features_market = n_feat
        X_train, X_test = X[train_idx], X[test_idx]
        stats_per_ch = STATS_PER_CHANNEL_TEMPORAL
        lines.append(f"Features: {stats_per_ch}*F = {n_feat} (F={n_feat // stats_per_ch})")

        for name, use_lgb in [("LogisticRegression (L2)", False), ("LightGBM", True)]:
            _, auc, extra = fit_predict_binary(
                X_train, y_train_bin, X_test, y_test_bin, use_lightgbm=use_lgb
            )
            lines.append(f"[Binary] {name}: Test AUC = {auc:.4f}, Acc = {extra['accuracy']:.4f}, P = {extra['precision']:.4f}, R = {extra['recall']:.4f}")

        if args.use_3class:
            for name, use_lgb in [("LogisticRegression (L2)", False), ("LightGBM", True)]:
                _, macro_f1, bal_acc, extra = fit_predict_3class(
                    X_train, y_train_3, X_test, y_test_3, use_lightgbm=use_lgb
                )
                lines.append(f"[3-class] {name}: Macro-F1 = {macro_f1:.4f}, BalancedAcc = {bal_acc:.4f}, Acc = {extra['accuracy']:.4f}")

        if not args.no_sanity:
            lines.append("Sanity Tests:")
            auc_label_shuf = run_sanity_label_shuffle(
                X_train, y_train_bin, X_test, y_test_bin, use_lightgbm=True
            )
            lines.append(f"  Label shuffle (LightGBM): Test AUC = {auc_label_shuf:.4f} (expect ~0.5)")
            auc_market_shuf = run_sanity_market_shuffle(
                X, y_binary, train_idx, test_idx, n_features_market, use_lightgbm=True
            )
            lines.append(f"  Market shuffle (LightGBM): Test AUC = {auc_market_shuf:.4f} (expect drop)")
        lines.append("")

    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
