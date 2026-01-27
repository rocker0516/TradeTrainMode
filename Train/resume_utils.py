from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ResumeConfig:
    """續訓模型的設定。

    Attributes:
        enabled: 是否啟用續訓。
        model_path: 欲載入的 SB3 模型檔路徑（通常是 .zip）。
    """

    enabled: bool
    model_path: str


def default_best_model_zip_path(*, symbol: str, checkpoint_dir_prefix: str = "models/sac_lag") -> str:
    """回傳預設 best model 的 .zip 路徑（與 `Train/run_sac_lag.py` 的保存規則一致）。"""
    base_dir = f"{checkpoint_dir_prefix}_{symbol}"
    return os.path.join(base_dir, "best_model", "best_model.zip")


def normalize_sb3_zip_path(path: str) -> str:
    """把 SB3 模型路徑正規化成 .zip（若使用者傳入未帶副檔名的路徑）。"""
    p = str(path).strip()
    if not p:
        return p
    if p.lower().endswith(".zip"):
        return p
    # SB3 存檔慣例：model.save(".../best_model") -> 實際檔名 ".../best_model.zip"
    return f"{p}.zip"


def resolve_resume_config(
    *,
    resume_path: Optional[str],
    resume_best: bool,
    symbol: str,
    checkpoint_dir_prefix: str = "models/sac_lag",
) -> Optional[ResumeConfig]:
    """解析 CLI 參數，決定是否要續訓以及載入哪個模型。

    優先順序：
    1) resume_path（若有提供）
    2) resume_best=True -> 使用預設 best_model.zip 路徑

    Returns:
        - 有可用模型檔：回傳 ResumeConfig
        - 否則：回傳 None（表示從頭開始訓練）
    """
    candidate: Optional[str] = None
    if resume_path:
        candidate = normalize_sb3_zip_path(resume_path)
    elif bool(resume_best):
        candidate = default_best_model_zip_path(symbol=str(symbol), checkpoint_dir_prefix=str(checkpoint_dir_prefix))

    if not candidate:
        return None

    candidate = os.path.normpath(candidate)
    if os.path.exists(candidate) and os.path.isfile(candidate):
        return ResumeConfig(enabled=True, model_path=candidate)
    return None

