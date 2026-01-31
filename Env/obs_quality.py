"""Observation (obs) quality audit utilities.

本模組目的：
- 針對 Env 的 Dict observation 做「欄位品質」檢查與報告輸出。
- 聚焦可量化問題：shape/dtype、NaN/Inf、範圍(out-of-range)、飽和/overflow 風險、常數/近常數欄位。

設計原則：
- SRP：只負責 obs 審計，不涉入訓練/下單邏輯。
- 可重用：可對任意符合 Gym Dict obs 的環境做 sampling。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from Env.trading_env import TradingEnvironment


@dataclass(frozen=True)
class ObsKeySummary:
    """單一 observation key 的摘要統計。"""

    key: str
    shape: Tuple[int, ...]
    dtype: str
    n: int
    finite_frac: float
    nan_count: int
    inf_count: int
    zero_frac: float
    min: float
    max: float
    mean: float
    std: float
    p01: float
    p50: float
    p99: float
    saturation_frac: float


@dataclass(frozen=True)
class DimSummary:
    """向量/通道維度的摘要（用於找常數欄位、範圍異常欄位）。"""

    name: str
    std: float
    zero_frac: float
    min: float
    max: float
    out_of_range_frac: float


@dataclass(frozen=True)
class ObsQualityReport:
    """完整審計報告。"""

    key_summaries: Tuple[ObsKeySummary, ...]
    price_seq_last_dim_summaries: Tuple[DimSummary, ...]
    price_seq_1d_last_dim_summaries: Tuple[DimSummary, ...]
    account_state_named: Tuple[DimSummary, ...]
    time_state_named: Tuple[DimSummary, ...]
    rhythm_state_named: Tuple[DimSummary, ...]
    cost_state_named: Tuple[DimSummary, ...]


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def _flatten_sample(x: np.ndarray, *, rng: np.random.Generator, max_elems: int) -> np.ndarray:
    """將任意 shape 陣列抽樣展平成 1D（避免記憶體爆炸）。"""
    arr = np.asarray(x)
    if arr.size <= 0:
        return np.asarray([], dtype=np.float64)
    flat = arr.reshape(-1)
    if flat.size <= int(max_elems):
        return flat.astype(np.float64, copy=False)
    idx = rng.integers(0, flat.size, size=int(max_elems), endpoint=False)
    return flat[idx].astype(np.float64, copy=False)


def _summarize_array(x: np.ndarray, *, key: str, rng: np.random.Generator, max_elems: int) -> ObsKeySummary:
    arr = np.asarray(x)
    flat = _flatten_sample(arr, rng=rng, max_elems=int(max_elems))
    if flat.size == 0:
        return ObsKeySummary(
            key=str(key),
            shape=tuple(arr.shape),
            dtype=str(arr.dtype),
            n=0,
            finite_frac=1.0,
            nan_count=0,
            inf_count=0,
            zero_frac=1.0,
            min=0.0,
            max=0.0,
            mean=0.0,
            std=0.0,
            p01=0.0,
            p50=0.0,
            p99=0.0,
            saturation_frac=0.0,
        )

    nan_count = int(np.isnan(flat).sum())
    inf_count = int(np.isinf(flat).sum())
    finite = flat[np.isfinite(flat)]
    finite_frac = float(finite.size / max(1, flat.size))
    if finite.size == 0:
        # 全部是 NaN/Inf：這是最嚴重問題
        return ObsKeySummary(
            key=str(key),
            shape=tuple(arr.shape),
            dtype=str(arr.dtype),
            n=int(flat.size),
            finite_frac=float(finite_frac),
            nan_count=int(nan_count),
            inf_count=int(inf_count),
            zero_frac=0.0,
            min=float("nan"),
            max=float("nan"),
            mean=float("nan"),
            std=float("nan"),
            p01=float("nan"),
            p50=float("nan"),
            p99=float("nan"),
            saturation_frac=0.0,
        )

    zero_frac = float((finite == 0.0).mean())
    # float16 飽和：接近 float16 最大值（65504）
    saturation_frac = 0.0
    if str(arr.dtype) == "float16":
        saturation_frac = float((np.abs(finite) >= 65000.0).mean())

    qs = np.quantile(finite, [0.01, 0.5, 0.99])
    return ObsKeySummary(
        key=str(key),
        shape=tuple(arr.shape),
        dtype=str(arr.dtype),
        n=int(flat.size),
        finite_frac=float(finite_frac),
        nan_count=int(nan_count),
        inf_count=int(inf_count),
        zero_frac=float(zero_frac),
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        mean=float(np.mean(finite)),
        std=float(np.std(finite)),
        p01=float(qs[0]),
        p50=float(qs[1]),
        p99=float(qs[2]),
        saturation_frac=float(saturation_frac),
    )


def _rule_range_for_feature_name(name: str) -> Optional[Tuple[float, float]]:
    """依特徵命名推定合理範圍，用於 out-of-range 偵測。

    備註：範圍是依你 FeatureTransformer/Observer 的 clip 規則推導而來。
    """
    n = str(name)
    if n.endswith("_z") or n.endswith("_zscore"):
        # FeatureTransformer._clip 預設 [-5, 5]
        return (-5.0, 5.0)
    if n in {"bb_pos_48"}:
        return (-2.0, 2.0)
    if n in {"rsi_14"}:
        return (-1.0, 1.0)
    if n in {"price_pos_96", "price_pos_288", "dir_persist_20", "trend_flip_rate_48", "alts_trend_up_ratio"}:
        return (-1.0, 1.0)
    if n.startswith("alts_") and (n.endswith("_mean_z") or n.endswith("_std_z") or n.endswith("_abs_mean_z")):
        return (-5.0, 5.0)
    if n in {"macd_atr", "macd_signal_atr"}:
        return (-10.0, 10.0)
    if n in {"trend_strength_atr"}:
        return (0.0, 10.0)
    if n in {"chop_48"}:
        return (-5.0, 5.0)
    # per-alt expanded features are also z-clipped
    if any(n.endswith(suf) for suf in ("_ret_15m_z", "_rel_ret_15m_z", "_volume_log_z", "_trend_spread_z")):
        return (-5.0, 5.0)
    return None


def _summarize_dims(
    *,
    X: np.ndarray,
    names: Iterable[str],
    default_range: Optional[Tuple[float, float]] = None,
) -> Tuple[DimSummary, ...]:
    """對 (N, D) 矩陣做每個維度摘要。"""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D (N,D), got shape={arr.shape}")
    n, d = int(arr.shape[0]), int(arr.shape[1])
    names_list = [str(x) for x in names]
    if len(names_list) != d:
        # fallback：維度不一致就用 index 命名
        names_list = [f"dim_{i}" for i in range(d)]

    out: List[DimSummary] = []
    for j, nm in enumerate(names_list):
        col = arr[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            out.append(DimSummary(name=nm, std=float("nan"), zero_frac=0.0, min=float("nan"), max=float("nan"), out_of_range_frac=0.0))
            continue

        zero_frac = float((finite == 0.0).mean())
        std = float(np.std(finite))
        mn = float(np.min(finite))
        mx = float(np.max(finite))

        rng = _rule_range_for_feature_name(nm) or default_range
        oor = 0.0
        if rng is not None:
            lo, hi = float(rng[0]), float(rng[1])
            oor = float(((finite < lo) | (finite > hi)).mean())

        out.append(DimSummary(name=nm, std=std, zero_frac=zero_frac, min=mn, max=mx, out_of_range_frac=oor))
    return tuple(out)


def _named_vector_specs() -> Dict[str, List[str]]:
    """向量欄位名稱（與 `LiveTradingRunner.runner_core` 對齊）。"""
    return {
        "account_state": [
            "pos_size_norm",
            "unreal_pnl_ratio",
            "equity_ratio",
            "max_equity_ratio",
            "dd",
            "maint_margin_ratio",
            "profit_rate",
            "long_entry_count_x0p01",
            "short_entry_count_x0p01",
            "episode_stop_loss_count_x0p1",
            "episode_liq_count_x1p0",
            "dist_to_sl_norm",
            "risk_budget",
            "pos_side_long_oh",
            "pos_side_short_oh",
            "pos_side_flat_oh",
            "entry_gap_atr",
            "breakeven_gap_atr",
            "steps_since_trade_norm",
            "holding_time_norm",
            "wallet_balance_ratio",
            "used_margin_ratio",
            "available_balance_ratio",
            "equity_to_position_notional",
            "liq_distance_pct",
            "stop_distance_pct",
            "fee_rate_pct",
            "trend_direction",
            "trend_strength",
        ],
        "time_state": [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "phase8_sin",
            "phase8_cos",
            "is_weekend",
        ],
        "rhythm_state": [
            "atr_ratio",
            "rv_ratio",
        ],
        "cost_state": [
            "step_fee_ratio_stable",
            "rolling_fee_ratio",
            "maint_margin_ratio",
            "dd",
            "leverage_ratio",
            "remaining_fee_budget_ratio",
            "gap_pct",
            "abs_gap_pct",
            "margin_ratio",
            "sl_gap_atr",
            "stop_loss_missing",
            "near_liq",
            "near_margin",
            "near_stop",
            "expected_fee_if_trade_ratio",
            "predicted_used_margin_ratio",
            "predicted_available_balance_ratio",
            "predicted_liq_distance_after_action",
            "predicted_stop_distance_after_action",
            "cooldown_remaining_norm",
            "action_overridden_flag",
            "last_action_raw",
            "last_action_used",
            "last_target_pos_pct",
            "last_final_pos_pct",
            "executed_pos_pct",
            "trade_executed_flag",
        ],
    }


class ObsQualityAuditor:
    """對 TradingEnvironment 進行 obs sampling 並產生品質報告。"""

    def __init__(
        self,
        *,
        rng_seed: int = 0,
        sample_steps: int = 2048,
        max_flat_elems_per_key: int = 4096,
    ) -> None:
        self.rng = np.random.default_rng(int(rng_seed))
        self.sample_steps = int(sample_steps)
        self.max_flat_elems_per_key = int(max_flat_elems_per_key)

    def audit(self, env: TradingEnvironment) -> ObsQualityReport:
        """執行 sampling audit。

        注意：這裡採用「隨機動作」以涵蓋更多狀態分布（但不追求 reward/策略）。
        """
        obs, info = env.reset()
        _ = info  # unused

        # key-level sampling accumulators（subsample flatten）
        flat_samples: Dict[str, List[np.ndarray]] = {}

        # dim-level accumulators（只收最後一列向量：更符合你要看欄位品質的需求）
        ps_last: List[np.ndarray] = []
        ps1d_last: List[np.ndarray] = []
        acc_vec: List[np.ndarray] = []
        time_vec: List[np.ndarray] = []
        rhythm_vec: List[np.ndarray] = []
        cost_vec: List[np.ndarray] = []

        def _ingest_one(o: Dict[str, Any]) -> None:
            for k, v in o.items():
                arr = np.asarray(v)
                flat_samples.setdefault(str(k), []).append(
                    _flatten_sample(arr, rng=self.rng, max_elems=self.max_flat_elems_per_key)
                )

            if "price_seq" in o:
                a = np.asarray(o["price_seq"])
                if a.ndim == 2 and a.shape[0] > 0:
                    ps_last.append(np.asarray(a[-1], dtype=np.float64).reshape(1, -1)[0])
            if "price_seq_1d" in o:
                a = np.asarray(o["price_seq_1d"])
                if a.ndim == 2 and a.shape[0] > 0:
                    ps1d_last.append(np.asarray(a[-1], dtype=np.float64).reshape(1, -1)[0])

            for k_src, bag in (
                ("account_state", acc_vec),
                ("time_state", time_vec),
                ("rhythm_state", rhythm_vec),
                ("cost_state", cost_vec),
            ):
                if k_src in o:
                    bag.append(np.asarray(o[k_src], dtype=np.float64).reshape(1, -1)[0])

        _ingest_one(obs)

        for _i in range(int(self.sample_steps)):
            action = self.rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
            obs, _reward, terminated, truncated, _info = env.step(action)
            _ingest_one(obs)
            if bool(terminated) or bool(truncated):
                obs, _info2 = env.reset()
                _ingest_one(obs)

        # ---- key summaries ----
        key_summaries: List[ObsKeySummary] = []
        for k, chunks in flat_samples.items():
            if not chunks:
                continue
            cat = np.concatenate(chunks, axis=0) if len(chunks) > 1 else np.asarray(chunks[0])
            # reconstruct minimal array for summary (shape 1D sample)
            key_summaries.append(_summarize_array(cat, key=str(k), rng=self.rng, max_elems=int(cat.size)))
        key_summaries = sorted(key_summaries, key=lambda s: s.key)

        # ---- dim summaries ----
        md = getattr(env, "market_data", None)
        cols_5m = list(getattr(md, "cols_5m", []) or [])
        cols_1d = list(getattr(md, "cols_1d", []) or [])

        ps_last_arr = np.asarray(ps_last, dtype=np.float64) if ps_last else np.zeros((0, len(cols_5m)), dtype=np.float64)
        ps1d_last_arr = np.asarray(ps1d_last, dtype=np.float64) if ps1d_last else np.zeros((0, len(cols_1d)), dtype=np.float64)

        # 1d features 都是 z-score clip(-5,5)
        ps_dim = _summarize_dims(X=ps_last_arr, names=cols_5m, default_range=None) if ps_last_arr.size else tuple()
        ps1d_dim = _summarize_dims(X=ps1d_last_arr, names=cols_1d, default_range=(-5.0, 5.0)) if ps1d_last_arr.size else tuple()

        specs = _named_vector_specs()
        acc_dim = _summarize_dims(X=np.asarray(acc_vec, dtype=np.float64), names=specs["account_state"], default_range=None) if acc_vec else tuple()
        time_dim = _summarize_dims(X=np.asarray(time_vec, dtype=np.float64), names=specs["time_state"], default_range=(-1.2, 1.2)) if time_vec else tuple()
        rhythm_dim = _summarize_dims(X=np.asarray(rhythm_vec, dtype=np.float64), names=specs["rhythm_state"], default_range=(0.0, 10.0)) if rhythm_vec else tuple()
        cost_dim = _summarize_dims(X=np.asarray(cost_vec, dtype=np.float64), names=specs["cost_state"], default_range=None) if cost_vec else tuple()

        return ObsQualityReport(
            key_summaries=tuple(key_summaries),
            price_seq_last_dim_summaries=tuple(ps_dim),
            price_seq_1d_last_dim_summaries=tuple(ps1d_dim),
            account_state_named=tuple(acc_dim),
            time_state_named=tuple(time_dim),
            rhythm_state_named=tuple(rhythm_dim),
            cost_state_named=tuple(cost_dim),
        )


def _print_top_flags(
    *,
    title: str,
    dims: Tuple[DimSummary, ...],
    std_low: float = 1e-6,
    zero_high: float = 0.95,
    oor_high: float = 0.001,
    top_k: int = 25,
) -> None:
    print(f"\n== {title} ==")
    if not dims:
        print("(empty)")
        return
    low_var = [d for d in dims if np.isfinite(d.std) and d.std <= float(std_low)]
    high_zero = [d for d in dims if np.isfinite(d.zero_frac) and d.zero_frac >= float(zero_high)]
    high_oor = [d for d in dims if np.isfinite(d.out_of_range_frac) and d.out_of_range_frac >= float(oor_high)]

    def _show(tag: str, xs: List[DimSummary], key_fn) -> None:
        xs2 = sorted(xs, key=key_fn, reverse=True)[: int(top_k)]
        print(f"- {tag}: {len(xs)}")
        for d in xs2:
            print(
                f"  - {d.name}: std={_safe_float(d.std):.6g}, zero={_safe_float(d.zero_frac):.3f}, "
                f"min={_safe_float(d.min):.3g}, max={_safe_float(d.max):.3g}, oor={_safe_float(d.out_of_range_frac):.4f}"
            )

    _show("low_variance", low_var, key_fn=lambda x: -x.std)  # smallest first by flipping sign
    _show("high_zero_frac", high_zero, key_fn=lambda x: x.zero_frac)
    _show("out_of_range", high_oor, key_fn=lambda x: x.out_of_range_frac)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit observation (obs) quality.")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--steps", type=int, default=2048, help="Sampling steps for audit")
    parser.add_argument("--window_5m", type=int, default=32)
    parser.add_argument("--window_1d", type=int, default=30)
    parser.add_argument("--obs_dtype", type=str, default=None, help="Override obs dtype (float16/float32)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature_symbols", type=str, default="", help="Comma-separated symbols for cross-market features")
    parser.add_argument("--data_split_enabled", action="store_true")
    parser.add_argument("--data_mode", type=str, default="full", choices=["full", "train", "eval"])
    parser.add_argument("--holdout_months", type=int, default=3)
    args = parser.parse_args(argv)

    feature_symbols = None
    if str(args.feature_symbols).strip():
        feature_symbols = [s.strip() for s in str(args.feature_symbols).split(",") if s.strip()]

    env_kwargs: Dict[str, Any] = {
        "target_symbol": str(args.symbol),
        "window_size": int(args.window_5m),
        "window_size_1d": int(args.window_1d),
        "feature_symbols": feature_symbols,
        "data_split_enabled": bool(args.data_split_enabled),
        "data_mode": str(args.data_mode),
        "holdout_months": int(args.holdout_months),
        "ensure_filled_obs": bool(str(args.data_mode).lower().strip() == "eval"),
    }
    if args.obs_dtype is not None:
        env_kwargs["obs_dtype"] = str(args.obs_dtype)

    env = TradingEnvironment(env_id=0, random_start=True, **env_kwargs)
    auditor = ObsQualityAuditor(rng_seed=int(args.seed), sample_steps=int(args.steps))
    rep = auditor.audit(env)
    env.close()

    print("\n== Key summaries ==")
    for s in rep.key_summaries:
        print(
            f"- {s.key}: shape={s.shape}, dtype={s.dtype}, finite={s.finite_frac:.5f}, "
            f"nan={s.nan_count}, inf={s.inf_count}, zero={s.zero_frac:.3f}, "
            f"min={s.min:.3g}, max={s.max:.3g}, mean={s.mean:.3g}, std={s.std:.3g}, "
            f"p01={s.p01:.3g}, p50={s.p50:.3g}, p99={s.p99:.3g}, sat16={s.saturation_frac:.4f}"
        )

    _print_top_flags(title="price_seq (last row) channel flags", dims=rep.price_seq_last_dim_summaries)
    _print_top_flags(title="price_seq_1d (last row) channel flags", dims=rep.price_seq_1d_last_dim_summaries)
    _print_top_flags(title="account_state flags", dims=rep.account_state_named, zero_high=0.98, oor_high=0.01)
    _print_top_flags(title="time_state flags", dims=rep.time_state_named, zero_high=0.98, oor_high=0.01)
    _print_top_flags(title="rhythm_state flags", dims=rep.rhythm_state_named, zero_high=0.98, oor_high=0.01)
    _print_top_flags(title="cost_state flags", dims=rep.cost_state_named, zero_high=0.98, oor_high=0.01)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

