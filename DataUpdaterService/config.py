"""Configuration loader for DataUpdaterService.

規則：
- **只有 API 相關**（例如 API Key/Token）允許用環境變數設定
- 其餘設定（symbols/排程/路徑/backfill/log_dir...）一律由 **config 檔案**提供

所有 secret 一律由環境變數提供，禁止硬編碼（符合專案通用安全規則）。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from datetime import time as dt_time
from pathlib import Path
from typing import Any, List, Optional


def _parse_symbols(raw: str) -> List[str]:
    syms = [s.strip() for s in raw.split(",") if s.strip()]
    # 允許使用者傳小寫，但寫檔一律用原樣（Binance/CoinGlass 通常大寫）
    return [s.upper() for s in syms]


def _parse_hhmm(raw: str) -> dt_time:
    parts = raw.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM time string: {raw!r}")
    h = int(parts[0])
    m = int(parts[1])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Invalid HH:MM time string: {raw!r}")
    return dt_time(hour=h, minute=m)


@dataclass(frozen=True)
class DataUpdaterConfig:
    """DataUpdaterService 的設定集合。"""

    data_dir: Path
    symbols: List[str]
    exchange: str

    update_5m_seconds: int
    update_1d_time: dt_time

    initial_backfill_days_5m: int
    initial_backfill_days_1d: int

    coinglass_api_key: Optional[str]

    log_dir: Path

    @staticmethod
    def from_config_file(project_root: Path, *, config_path: Optional[Path] = None) -> "DataUpdaterConfig":
        """由 config 檔案載入設定（非 API），並從環境變數載入 API Key（API 相關）。

        Args:
            project_root: 專案根目錄（用於推導相對路徑）
            config_path: 設定檔路徑；若不提供，預設為 `DataUpdaterService/data_updater_config.json`

        Returns:
            DataUpdaterConfig
        """
        cfg_path = config_path or (project_root / "DataUpdaterService" / "data_updater_config.json")
        data = _load_json_config(cfg_path)

        data_dir = _resolve_path(project_root, _get_str(data, "data_dir", default="Data"))
        log_dir = _resolve_path(project_root, _get_str(data, "log_dir", default="logs/data_updater_service"))

        symbols = _coerce_symbols(data.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT"]))
        exchange = _get_str(data, "exchange", default="Binance")

        update_5m_seconds = _get_int(data, "update_5m_seconds", default=300)
        update_1d_time = _parse_hhmm(_get_str(data, "update_1d_time", default="00:30"))

        initial_backfill_days_5m = _get_int(data, "initial_backfill_days_5m", default=30)
        initial_backfill_days_1d = _get_int(data, "initial_backfill_days_1d", default=3650)

        # 僅 API 相關：允許由環境變數提供
        coinglass_api_key = os.getenv("COINGLASS_API_KEY") or None

        return DataUpdaterConfig(
            data_dir=data_dir,
            symbols=symbols,
            exchange=exchange,
            update_5m_seconds=update_5m_seconds,
            update_1d_time=update_1d_time,
            initial_backfill_days_5m=initial_backfill_days_5m,
            initial_backfill_days_1d=initial_backfill_days_1d,
            coinglass_api_key=coinglass_api_key,
            log_dir=log_dir,
        )

    @staticmethod
    def from_env(project_root: Path) -> "DataUpdaterConfig":
        """相容性入口：舊版以 env 設定，本版改為「config 檔 + API env」。

        注意：此方法 **不再** 讀取任何非 API 的環境變數；非 API 設定請改到
        `DataUpdaterService/data_updater_config.json`。
        """
        return DataUpdaterConfig.from_config_file(project_root=project_root)


def _load_json_config(path: Path) -> dict[str, Any]:
    """讀取 JSON config 檔。"""
    if not path.exists():
        raise FileNotFoundError(
            f"DataUpdaterService config file not found: {path}. "
            "Please create it (see DataUpdaterService/README.md)."
        )
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
    except OSError as e:
        raise OSError(f"Failed to read config file: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON config: {path}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON config root type (expect object): {type(data)}")
    return data


def _resolve_path(project_root: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()


def _get_str(data: dict[str, Any], key: str, *, default: str) -> str:
    v = data.get(key, default)
    if isinstance(v, str):
        return v
    raise ValueError(f"Config key {key!r} must be string, got {type(v)}")


def _get_int(data: dict[str, Any], key: str, *, default: int) -> int:
    v = data.get(key, default)
    if isinstance(v, bool):
        raise ValueError(f"Config key {key!r} must be int, got bool")
    if isinstance(v, int):
        return int(v)
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    raise ValueError(f"Config key {key!r} must be int, got {type(v)}")


def _coerce_symbols(v: Any) -> List[str]:
    if isinstance(v, list):
        if not all(isinstance(x, str) for x in v):
            raise ValueError("Config key 'symbols' must be list[str] or comma-separated string")
        return [str(x).upper() for x in v if str(x).strip()]
    if isinstance(v, str):
        return _parse_symbols(v)
    raise ValueError(f"Config key 'symbols' must be list[str] or string, got {type(v)}")


