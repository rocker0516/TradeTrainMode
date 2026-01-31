"""
檢查餵入兩個 CNN（5m / 1d）的每個欄位：重複邏輯、高相關、近常數通道。

用法（專案根目錄）：
    python scripts/cnn_channel_redundancy_check.py [--steps 2000] [--corr_threshold 0.85]
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Env.trading_env import TradingEnvironment


def _sample_obs_matrix(
    env: TradingEnvironment,
    *,
    steps: int,
    key: str,
    last_row_only: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """從 env 抽樣 obs[key]，回傳 (N, C) 矩陣與通道名。"""
    obs, _ = env.reset()
    arr = np.asarray(obs[key])
    if arr.ndim == 2 and arr.shape[0] > 0:
        # (L, C) -> 取最後一列
        cols = _channel_names_5m(env) if key == "price_seq" else _channel_names_1d(env)
        if len(cols) != arr.shape[1]:
            cols = [f"ch_{i}" for i in range(arr.shape[1])]
        rows = [arr[-1].astype(np.float64)]
    else:
        rows = [np.asarray(arr).reshape(-1).astype(np.float64)]
        cols = []

    for _ in range(steps - 1):
        action = np.array([np.random.uniform(-0.5, 0.5)], dtype=np.float32)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
        arr = np.asarray(obs[key])
        if arr.ndim == 2 and arr.shape[0] > 0:
            rows.append(arr[-1].astype(np.float64))
        elif arr.ndim == 1:
            rows.append(np.asarray(arr).reshape(-1).astype(np.float64))

    X = np.vstack(rows).astype(np.float64)
    if key == "price_seq" and not cols:
        cols = _channel_names_5m(env)
    if key == "price_seq_1d" and not cols:
        cols = _channel_names_1d(env)
    return X, cols


def _channel_names_5m(env: Any) -> List[str]:
    md = getattr(env, "market_data", None)
    if md is None:
        return []
    return list(getattr(md, "cols_5m", []) or [])


def _channel_names_1d(env: Any) -> List[str]:
    md = getattr(env, "market_data", None)
    if md is None:
        return []
    return list(getattr(md, "cols_1d", []) or [])


def _pairwise_spearman(X: np.ndarray, names: List[str]) -> List[Tuple[str, str, float]]:
    """回傳 (name_i, name_j, rho) 且 |rho| >= threshold，依 |rho| 降序。"""
    n, c = X.shape
    if c != len(names) or n < 10:
        return []
    out: List[Tuple[str, str, float]] = []
    for i in range(c):
        for j in range(i + 1, c):
            xi = X[:, i]
            xj = X[:, j]
            finite = np.isfinite(xi) & np.isfinite(xj)
            if finite.sum() < 10:
                continue
            # Spearman
            from scipy.stats import spearmanr
            r, _ = spearmanr(xi[finite], xj[finite])
            if np.isfinite(r):
                out.append((names[i], names[j], float(r)))
    return sorted(out, key=lambda x: -abs(x[2]))


def _std_per_channel(X: np.ndarray) -> np.ndarray:
    return np.std(X, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="CNN channel redundancy check")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--corr_threshold", type=float, default=0.85)
    parser.add_argument("--std_low", type=float, default=1e-4)
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--window_5m", type=int, default=64)
    parser.add_argument("--window_1d", type=int, default=30)
    args = parser.parse_args()

    env_kwargs: Dict[str, Any] = {
        "target_symbol": args.symbol,
        "window_size": args.window_5m,
        "window_size_1d": args.window_1d,
        "feature_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT"],
    }
    env = TradingEnvironment(env_id=0, random_start=True, **env_kwargs)

    try:
        scipy_available = True
        try:
            from scipy.stats import spearmanr  # noqa: F401
        except ImportError:
            scipy_available = False

        # ---- 5m ----
        X5, cols5 = _sample_obs_matrix(env, steps=args.steps, key="price_seq")
        std5 = _std_per_channel(X5)
        print("\n=== 5m CNN (price_seq) ===")
        print(f"Shape: {X5.shape}, channels: {len(cols5)}")
        low_var_5m = [(cols5[i], float(std5[i])) for i in range(len(cols5)) if std5[i] <= args.std_low]
        if low_var_5m:
            print(f"  Low-variance (std <= {args.std_low}): {low_var_5m[:15]}")
        if scipy_available:
            pairs5 = _pairwise_spearman(X5, cols5)
            high5 = [(a, b, r) for a, b, r in pairs5 if abs(r) >= args.corr_threshold]
            print(f"  High |Spearman| >= {args.corr_threshold}: {len(high5)} pairs")
            for a, b, r in high5[:25]:
                print(f"    {a} <-> {b}: {r:.3f}")

        # ---- 1d ----
        X1, cols1 = _sample_obs_matrix(env, steps=args.steps, key="price_seq_1d")
        std1 = _std_per_channel(X1)
        print("\n=== 1d CNN (price_seq_1d) ===")
        print(f"Shape: {X1.shape}, channels: {len(cols1)}")
        low_var_1d = [(cols1[i], float(std1[i])) for i in range(len(cols1)) if std1[i] <= args.std_low]
        if low_var_1d:
            print(f"  Low-variance (std <= {args.std_low}): {low_var_1d}")
        if scipy_available:
            pairs1 = _pairwise_spearman(X1, cols1)
            high1 = [(a, b, r) for a, b, r in pairs1 if abs(r) >= args.corr_threshold]
            print(f"  High |Spearman| >= {args.corr_threshold}: {len(high1)} pairs")
            for a, b, r in high1[:25]:
                print(f"    {a} <-> {b}: {r:.3f}")
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
