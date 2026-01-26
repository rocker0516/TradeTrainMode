"""Configuration loader for DataUpdaterService.

所有 secret 一律由環境變數提供，禁止硬編碼（符合專案通用安全規則）。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from datetime import time as dt_time
from pathlib import Path
from typing import List, Optional


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
    def from_env(project_root: Path) -> "DataUpdaterConfig":
        """由環境變數載入設定。

        Args:
            project_root: 專案根目錄（用於推導預設 data_dir / log_dir）

        Returns:
            DataUpdaterConfig
        """
        data_dir = Path(os.getenv("DATA_DIR", str(project_root / "Data"))).expanduser().resolve()
        log_dir = Path(os.getenv("DATA_UPDATER_LOG_DIR", str(project_root / "logs" / "data_updater_service"))).expanduser().resolve()

        symbols_raw = os.getenv("DATA_UPDATER_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,1000PEPEUSDT")
        symbols = _parse_symbols(symbols_raw)
        exchange = os.getenv("DATA_UPDATER_EXCHANGE", "Binance")

        update_5m_seconds = int(os.getenv("UPDATE_5M_SECONDS", "300"))
        update_1d_time = _parse_hhmm(os.getenv("UPDATE_1D_TIME", "00:30"))

        initial_backfill_days_5m = int(os.getenv("INITIAL_BACKFILL_DAYS_5M", "30"))
        initial_backfill_days_1d = int(os.getenv("INITIAL_BACKFILL_DAYS_1D", "3650"))

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


