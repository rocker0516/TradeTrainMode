from __future__ import annotations

"""
特徵重選共用工具。

用途：
- 綁定 window_size 與 label horizon 候選。
- 解析特徵冗餘群組（同一概念的 z/roc/ema spread 變形）。
- 以靜態掃描方式輸出 normalization / leakage 風險提示。
- 提供簡單統計摘要工具，供離線評分與 mask 評估共用。
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_HORIZON_RATIOS: tuple[float, ...] = (0.5, 1.0, 2.0)
MARKET_CONFIG_FIELDS: tuple[str, ...] = (
    "OBS_PRICE_SEQ_TARGET_COLS",
    "OBS_PRICE_SEQ_OTHERS_COLS",
    "OBS_PRICE_SEQ_1D_TARGET_COLS",
    "OBS_PRICE_SEQ_1D_OTHERS_COLS",
)
OBS_CONFIG_FIELDS: tuple[str, ...] = (
    "OBS_STATE_KEYS",
    "OBS_PRICE_SEQ_TARGET_COLS",
    "OBS_PRICE_SEQ_OTHERS_COLS",
    "OBS_PRICE_SEQ_1D_TARGET_COLS",
    "OBS_PRICE_SEQ_1D_OTHERS_COLS",
    "OBS_ACCOUNT_STATE_COLS",
    "OBS_CONTEXT_STATE_COLS",
)
WINDOW_CONFIG_FIELDS: tuple[str, ...] = ("WINDOW_SIZE", "WINDOW_SIZE_1D")


@dataclass(frozen=True)
class WindowHorizonSpec:
    """單一 observation 視窗與 label horizon 的聯合設定。"""

    window_size: int
    window_size_1d: int
    horizon_k: int
    recent_bars: int


@dataclass(frozen=True)
class LeakageFinding:
    """靜態 normalization / leakage 風險掃描結果。"""

    file_path: str
    line_no: int
    severity: str
    pattern: str
    line_text: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "file_path": self.file_path,
            "line_no": int(self.line_no),
            "severity": self.severity,
            "pattern": self.pattern,
            "line_text": self.line_text,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RedundancyTag:
    """特徵名稱的冗餘分群標籤。"""

    raw_name: str
    normalized_name: str
    concept_key: str
    variant: str
    symbol_scope: str

    def to_dict(self) -> dict[str, str]:
        return {
            "raw_name": self.raw_name,
            "normalized_name": self.normalized_name,
            "concept_key": self.concept_key,
            "variant": self.variant,
            "symbol_scope": self.symbol_scope,
        }


_VARIANT_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_ema_12_48_spread", "ema_12_48_spread"),
    ("_z_short", "z_short"),
    ("_z_long", "z_long"),
    ("_roc_12", "roc_12"),
    ("_roc_1", "roc_1"),
    ("_log_z_short", "log_z_short"),
    ("_log_z_long", "log_z_long"),
    ("_log_z", "log_z"),
    ("_scale", "scale"),
    ("_ratio", "ratio"),
    ("_z", "z"),
)

_SYMBOL_RE = re.compile(r"^[A-Z0-9]+USDT_(.+)$")

_LEAKAGE_RULES: tuple[tuple[str, str, str], ...] = (
    (r"\.bfill\(", "high", "bfill 會把未來值往前補，容易造成 look-ahead leakage。"),
    (
        r"nanpercentile\(",
        "medium",
        "全樣本分位數門檻若直接用在 train/test 比較，可能引入全域統計偏差。",
    ),
    (
        r"\.quantile\(",
        "medium",
        "分位數若未限制只在 train 區間 fit，可能造成全域統計 leakage。",
    ),
)


def build_window_horizon_specs(
    window_sizes: Sequence[int],
    window_sizes_1d: Sequence[int],
    *,
    horizon_ratios: Sequence[float] = DEFAULT_HORIZON_RATIOS,
    min_horizon: int = 6,
    max_horizon: int = 288,
    min_recent_bars: int = 4,
) -> list[WindowHorizonSpec]:
    """建立少量公平比較用的 window / horizon 聯合候選。"""
    specs: list[WindowHorizonSpec] = []
    seen: set[tuple[int, int, int, int]] = set()
    for window_size in window_sizes:
        w = max(1, int(window_size))
        horizons = default_horizon_candidates(
            w,
            ratios=horizon_ratios,
            min_horizon=min_horizon,
            max_horizon=max_horizon,
        )
        for window_size_1d in window_sizes_1d:
            w1d = max(1, int(window_size_1d))
            for horizon_k in horizons:
                recent_bars = min(w, max(int(min_recent_bars), int(horizon_k)))
                key = (w, w1d, horizon_k, recent_bars)
                if key in seen:
                    continue
                seen.add(key)
                specs.append(
                    WindowHorizonSpec(
                        window_size=w,
                        window_size_1d=w1d,
                        horizon_k=int(horizon_k),
                        recent_bars=int(recent_bars),
                    )
                )
    return specs


def default_horizon_candidates(
    window_size: int,
    *,
    ratios: Sequence[float] = DEFAULT_HORIZON_RATIOS,
    min_horizon: int = 6,
    max_horizon: int = 288,
) -> tuple[int, ...]:
    """依 window_size 產生綁定的 label horizon 候選。"""
    w = max(1, int(window_size))
    values = {
        int(np.clip(round(float(w) * float(r)), int(min_horizon), int(max_horizon)))
        for r in ratios
        if float(r) > 0.0
    }
    values.add(int(np.clip(w, int(min_horizon), int(max_horizon))))
    return tuple(sorted(v for v in values if v > 0))


def summarize_metric(values: Sequence[float]) -> dict[str, float]:
    """輸出 mean/std/min/max，空輸入時回傳 0。"""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> tuple[float, float]:
    """對均值做 bootstrap 信賴區間。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        v = float(arr[0])
        return v, v
    rng = np.random.default_rng(random_state)
    boot = np.empty(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boot[i] = np.mean(sample)
    alpha = (1.0 - float(confidence)) / 2.0
    low = float(np.quantile(boot, alpha))
    high = float(np.quantile(boot, 1.0 - alpha))
    return low, high


def tag_redundancy(feature_name: str) -> RedundancyTag:
    """將特徵名稱拆成 symbol 範圍、核心概念與變形型別。"""
    raw_name = str(feature_name)
    normalized_name = _strip_symbol_prefix(raw_name)
    symbol_scope = "symbol" if normalized_name != raw_name else "global"
    concept_key = normalized_name
    variant = "raw"
    for suffix, label in _VARIANT_SUFFIXES:
        if normalized_name.endswith(suffix):
            concept_key = normalized_name[: -len(suffix)]
            variant = label
            break
    concept_key = concept_key.strip("_") or normalized_name
    return RedundancyTag(
        raw_name=raw_name,
        normalized_name=normalized_name,
        concept_key=concept_key,
        variant=variant,
        symbol_scope=symbol_scope,
    )


def build_redundancy_frame(
    feature_names: Iterable[str],
    *,
    metric_by_feature: dict[str, float] | None = None,
) -> pd.DataFrame:
    """將特徵清單轉成可排序的冗餘群組表。"""
    rows: list[dict[str, object]] = []
    metric_by_feature = metric_by_feature or {}
    for name in feature_names:
        tag = tag_redundancy(str(name))
        row = tag.to_dict()
        row["metric"] = float(metric_by_feature.get(str(name), 0.0))
        rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=["raw_name", "normalized_name", "concept_key", "variant", "symbol_scope", "metric"]
        )
    return pd.DataFrame(rows).sort_values(
        by=["concept_key", "metric", "variant", "raw_name"],
        ascending=[True, False, True, True],
    )


def select_representatives(
    redundancy_df: pd.DataFrame,
    *,
    top_k_per_concept: int = 2,
) -> pd.DataFrame:
    """依 concept_key 每群保留前 top_k 個代表欄位。"""
    if redundancy_df.empty:
        return redundancy_df.copy()
    ordered = redundancy_df.sort_values(
        by=["concept_key", "metric", "variant", "raw_name"],
        ascending=[True, False, True, True],
    )
    return ordered.groupby("concept_key", as_index=False, group_keys=False).head(int(top_k_per_concept))


def scan_source_for_leakage(paths: Sequence[str | Path]) -> list[LeakageFinding]:
    """用靜態規則掃描疑似 leakage / 全域統計風險。"""
    findings: list[LeakageFinding] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern, severity, reason in _LEAKAGE_RULES:
                if re.search(pattern, line):
                    findings.append(
                        LeakageFinding(
                            file_path=str(path),
                            line_no=int(line_no),
                            severity=severity,
                            pattern=pattern,
                            line_text=line.strip(),
                            reason=reason,
                        )
                    )
    return findings


def findings_to_frame(findings: Sequence[LeakageFinding]) -> pd.DataFrame:
    """將 leakage findings 轉成 DataFrame 方便輸出 CSV。"""
    if not findings:
        return pd.DataFrame(
            columns=["file_path", "line_no", "severity", "pattern", "line_text", "reason"]
        )
    return pd.DataFrame([item.to_dict() for item in findings]).sort_values(
        by=["severity", "file_path", "line_no"],
        ascending=[True, True, True],
    )


def capture_observation_config_snapshot() -> dict[str, Any]:
    """擷取目前 observation 與視窗設定，供暫時覆寫後還原。"""
    from Env.config import Config
    from Eval.phase_ab_env_config import PhaseABEnvConfig

    snapshot: dict[str, Any] = {}
    for name in OBS_CONFIG_FIELDS:
        snapshot[name] = tuple(getattr(Config, name))
    for name in WINDOW_CONFIG_FIELDS:
        snapshot[name] = int(getattr(PhaseABEnvConfig, name))
    return snapshot


def market_full_universe_overrides() -> dict[str, Any]:
    """建立市場特徵空 tuple 覆寫，表示使用各市場 state 的全部工程特徵。"""
    return {name: tuple() for name in MARKET_CONFIG_FIELDS}


@contextmanager
def temporary_observation_config(overrides: dict[str, Any] | None = None):
    """
    暫時覆寫 Config / PhaseABEnvConfig / FeatureTransformer 的 observation 設定。

    用途：
    - full-feature universe：市場特徵設為空 tuple。
    - candidate retrain：套用候選 observation 配置。
    """
    from Env.config import Config
    from Env.feature_transformer import FeatureTransformer
    from Eval.phase_ab_env_config import PhaseABEnvConfig

    saved = capture_observation_config_snapshot()
    overrides = overrides or {}
    try:
        for name in OBS_CONFIG_FIELDS:
            if name in overrides:
                setattr(Config, name, tuple(overrides[name]))
        for name in WINDOW_CONFIG_FIELDS:
            if name in overrides:
                setattr(PhaseABEnvConfig, name, int(overrides[name]))
        _refresh_feature_transformer_class_attrs(FeatureTransformer)
        yield
    finally:
        for name in OBS_CONFIG_FIELDS:
            setattr(Config, name, tuple(saved[name]))
        for name in WINDOW_CONFIG_FIELDS:
            setattr(PhaseABEnvConfig, name, int(saved[name]))
        _refresh_feature_transformer_class_attrs(FeatureTransformer)


def _strip_symbol_prefix(name: str) -> str:
    match = _SYMBOL_RE.match(str(name))
    if match:
        return str(match.group(1))
    return str(name)


def _refresh_feature_transformer_class_attrs(feature_transformer_cls: Any) -> None:
    """同步 FeatureTransformer 的 class attrs 與目前 Config。"""
    from Env.config import Config

    target_5m = tuple(getattr(Config, "OBS_PRICE_SEQ_TARGET_COLS", ()))
    others_5m = tuple(getattr(Config, "OBS_PRICE_SEQ_OTHERS_COLS", ()))
    target_1d = tuple(getattr(Config, "OBS_PRICE_SEQ_1D_TARGET_COLS", ()))
    others_1d = tuple(getattr(Config, "OBS_PRICE_SEQ_1D_OTHERS_COLS", ()))
    macro_1d = tuple(c for c in others_1d if c in feature_transformer_cls.DEFAULT_1D_MACRO_COLS)
    others_1d_per_symbol = tuple(c for c in others_1d if c not in feature_transformer_cls.DEFAULT_1D_MACRO_COLS)

    feature_transformer_cls.OPTIMIZED_TARGET_5M_COLS = target_5m
    feature_transformer_cls.OTHERS_5M_COLS_PER_SYMBOL = others_5m
    feature_transformer_cls.BASE_5M_COLS = target_5m
    feature_transformer_cls.TARGET_1D_COLS = target_1d
    feature_transformer_cls.OTHERS_1D_COLS_PER_SYMBOL = others_1d_per_symbol
    feature_transformer_cls.PRICE_SEQ_1D_MACRO_COLS = macro_1d
    feature_transformer_cls.PRICE_SEQ_1D_SYMBOL_COLS = target_1d
