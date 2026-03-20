"""
Unified config for standalone data fetch services.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


def _parse_symbols(raw: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    symbols = tuple(part.strip() for part in raw.split(",") if part.strip())
    return symbols or default


@dataclass(frozen=True)
class ServiceConfig:
    """Runtime settings for service loops and fetch jobs."""

    service_enabled: bool
    run_on_startup: bool
    max_cycles: int
    binance_interval_seconds: int
    coinglass_interval_seconds: int
    binance_symbols: Tuple[str, ...]
    coinglass_symbols: Tuple[str, ...]
    binance_kline_interval: str
    coinglass_interval: str
    binance_lookback_days: int
    coinglass_lookback_days: int
    coinglass_exchange: str
    coinglass_api_key: str
    launcher_restart_delay_seconds: int


def load_config() -> ServiceConfig:
    """Load service config from environment variables."""
    default_symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT")
    return ServiceConfig(
        service_enabled=os.getenv("DATA_FETCH_SERVICE_ENABLED", "1") == "1",
        run_on_startup=os.getenv("DATA_FETCH_RUN_ON_STARTUP", "1") == "1",
        max_cycles=int(os.getenv("DATA_FETCH_MAX_CYCLES", "0")),
        binance_interval_seconds=int(os.getenv("BINANCE_FETCH_INTERVAL_SECONDS", "300")), # 5m
        coinglass_interval_seconds=int(os.getenv("COINGLASS_FETCH_INTERVAL_SECONDS", "86400")), # 1d
        binance_symbols=_parse_symbols(
            os.getenv("BINANCE_FETCH_TRADING_PAIRS", ",".join(default_symbols)),
            default_symbols,
        ),
        coinglass_symbols=_parse_symbols(
            os.getenv("COINGLASS_FETCH_TRADING_PAIRS", ",".join(default_symbols)),
            default_symbols,
        ),
        binance_kline_interval=os.getenv("BINANCE_FETCH_INTERVAL", "5m"),
        coinglass_interval=os.getenv("COINGLASS_FETCH_INTERVAL", "1d"),
        binance_lookback_days=int(os.getenv("BINANCE_FETCH_LOOKBACK_DAYS", str(2 * 365))),
        coinglass_lookback_days=int(os.getenv("COINGLASS_FETCH_LOOKBACK_DAYS", str(365 * 6))),
        coinglass_exchange=os.getenv("COINGLASS_EXCHANGE", "Binance"),
        coinglass_api_key=os.getenv("COINGLASS_API_KEY", "1e41abd6360a4d1486b770e83982e33a"),
        launcher_restart_delay_seconds=int(os.getenv("DATA_FETCH_RESTART_DELAY_SECONDS", "10")),
    )

