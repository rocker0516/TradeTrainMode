from __future__ import annotations

import os

from Train.resume_utils import default_best_model_zip_path, normalize_sb3_zip_path, resolve_resume_config


def test_normalize_sb3_zip_path_adds_zip_suffix() -> None:
    assert normalize_sb3_zip_path("models/x/best_model") == "models/x/best_model.zip"
    assert normalize_sb3_zip_path("models/x/best_model.zip") == "models/x/best_model.zip"


def test_default_best_model_zip_path_matches_training_convention() -> None:
    p = default_best_model_zip_path(symbol="BTCUSDT", checkpoint_dir_prefix="models/sac_lag")
    expected = os.path.join("models", "sac_lag_BTCUSDT", "best_model", "best_model.zip")
    assert os.path.normpath(p) == os.path.normpath(expected)


def test_resolve_resume_config_prefers_resume_path(tmp_path) -> None:
    # create a fake zip file
    model_zip = tmp_path / "best_model.zip"
    model_zip.write_bytes(b"fake")

    cfg = resolve_resume_config(resume_path=str(model_zip), resume_best=True, symbol="BTCUSDT", checkpoint_dir_prefix="models/sac_lag")
    assert cfg is not None
    assert cfg.enabled is True
    assert cfg.model_path.endswith("best_model.zip")


def test_resolve_resume_config_returns_none_if_missing(tmp_path) -> None:
    missing = tmp_path / "missing.zip"
    cfg = resolve_resume_config(resume_path=str(missing), resume_best=False, symbol="BTCUSDT", checkpoint_dir_prefix="models/sac_lag")
    assert cfg is None

