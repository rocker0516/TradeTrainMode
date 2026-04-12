from __future__ import annotations

from pathlib import Path

from Eval.feature_selection_utils import (
    bootstrap_mean_ci,
    build_redundancy_frame,
    build_window_horizon_specs,
    default_horizon_candidates,
    scan_source_for_leakage,
    select_representatives,
    tag_redundancy,
)


def test_default_horizon_candidates_ties_to_window_size() -> None:
    values = default_horizon_candidates(24, ratios=(0.5, 1.0, 2.0), min_horizon=6, max_horizon=96)

    assert values == (12, 24, 48)


def test_build_window_horizon_specs_sets_recent_bars_from_horizon() -> None:
    specs = build_window_horizon_specs(
        [12, 24],
        [6],
        horizon_ratios=(1.0,),
        min_horizon=6,
        max_horizon=48,
    )

    assert [s.window_size for s in specs] == [12, 24]
    assert [s.horizon_k for s in specs] == [12, 24]
    assert [s.recent_bars for s in specs] == [12, 24]


def test_tag_redundancy_strips_symbol_prefix_and_groups_variants() -> None:
    tag = tag_redundancy("BTCUSDT_volume_ratio_z_short")

    assert tag.normalized_name == "volume_ratio_z_short"
    assert tag.concept_key == "volume_ratio"
    assert tag.variant == "z_short"
    assert tag.symbol_scope == "symbol"


def test_select_representatives_limits_top_k_per_concept() -> None:
    df = build_redundancy_frame(
        ["ret_1d_z_short", "ret_1d_z_long", "ret_1d_roc_1", "volume_ratio_z_long"],
        metric_by_feature={
            "ret_1d_z_short": 0.4,
            "ret_1d_z_long": 0.8,
            "ret_1d_roc_1": 0.3,
            "volume_ratio_z_long": 0.7,
        },
    )

    reps = select_representatives(df, top_k_per_concept=1)

    assert "ret_1d_z_long" in reps["raw_name"].tolist()
    assert "ret_1d_z_short" not in reps["raw_name"].tolist()
    assert "volume_ratio_z_long" in reps["raw_name"].tolist()


def test_scan_source_for_leakage_flags_bfill_and_quantile(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        "x = s.bfill()\nq = np.nanpercentile(arr, 50)\n",
        encoding="utf-8",
    )

    findings = scan_source_for_leakage([sample])

    patterns = [item.pattern for item in findings]
    assert any("bfill" in p for p in patterns)
    assert any("nanpercentile" in p for p in patterns)


def test_bootstrap_mean_ci_returns_ordered_interval() -> None:
    low, high = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], n_bootstrap=200, random_state=7)

    assert low <= high
    assert low >= 1.0
    assert high <= 4.0
