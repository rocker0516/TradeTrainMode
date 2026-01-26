"""Logging utilities for DataUpdaterService."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logger(*, name: str, log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """建立同時輸出 console + file 的 logger。

    Args:
        name: logger name
        log_dir: log directory（會自動建立）
        level: logging level

    Returns:
        configured Logger
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # avoid duplicated handlers (when service reloads)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    log_path = log_dir / f"{name}.log"
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(log_path) for h in logger.handlers):
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


