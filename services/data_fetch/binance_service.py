"""
Standalone Binance fetch service (new module, legacy files untouched).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Dict

from binance.client import Client

from GetTradeData import fetch_futures_data, format_value, upsert_dataframe_to_csv
from services.data_fetch.config import load_config
from services.data_fetch.incremental import interval_to_timedelta, resolve_start_time
from services.data_fetch.logging_utils import init_file_logger


logger = init_file_logger("data_fetch.binance", "logs/data_fetch/binance_service.log")


def _upsert_with_retry(df, filename: str, retries: int = 3, wait_seconds: float = 0.5):
    """
    Retry CSV upsert for transient Windows file handle issues.
    """
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return upsert_dataframe_to_csv(df, filename, key_cols=["timestamp"], parse_dates=["timestamp"])
        except OSError as exc:
            last_error = exc
            if attempt == retries:
                break
            logger.warning(
                "Upsert failed for %s (attempt %s/%s): %s. Retrying in %.1fs...",
                filename,
                attempt,
                retries,
                exc,
                wait_seconds,
            )
            time.sleep(wait_seconds)
    if last_error is not None:
        raise last_error
    return df


def run_once() -> None:
    """Run one Binance backfill/update cycle."""
    cfg = load_config()
    interval_map: Dict[str, str] = {
        "5m": Client.KLINE_INTERVAL_5MINUTE,
        "15m": Client.KLINE_INTERVAL_15MINUTE,
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "1d": Client.KLINE_INTERVAL_1DAY,
    }
    interval = interval_map.get(cfg.binance_kline_interval, Client.KLINE_INTERVAL_5MINUTE)
    step = interval_to_timedelta(cfg.binance_kline_interval, timedelta(minutes=5))
    end_time = datetime.now()
    fallback_start_time = end_time - timedelta(days=cfg.binance_lookback_days)
    logger.info("Target end_time=%s", end_time)

    for symbol in cfg.binance_symbols:
        try:
            filename = f"Data/{symbol}_futures_volume_5years_5min.csv"
            start_time = resolve_start_time(filename, "timestamp", fallback_start_time, step)
            if start_time >= end_time:
                logger.info("%s is up to date, skip.", symbol)
                continue

            logger.info("Fetching %s from %s to %s", symbol, start_time, end_time)
            df = fetch_futures_data(symbol, interval, start_time, end_time)
            if df.empty:
                logger.info("No new rows for %s", symbol)
                continue

            df = _upsert_with_retry(df, filename)
            logger.info("Saved %s, rows=%s", filename, len(df))
            logger.info(
                "[BinanceService] Stats avg_volume=%s avg_buy=%s avg_sell=%s avg_ratio=%s"
                % (
                    format_value(df["volume"].mean()),
                    format_value(df["buy_volume"].mean()),
                    format_value(df["sell_volume"].mean()),
                    format_value(df["volume_ratio"].mean()),
                )
            )
            time.sleep(1)
        except Exception as exc:
            logger.exception("Fetch failed for %s: %s", symbol, exc)


def run_service_loop() -> None:
    """Run periodic Binance service loop."""
    cfg = load_config()
    if not cfg.service_enabled:
        run_once()
        return

    cycle = 0
    if cfg.run_on_startup:
        cycle += 1
        logger.info("Cycle %s started", cycle)
        run_once()
        if cfg.max_cycles > 0 and cycle >= cfg.max_cycles:
            return

    while True:
        logger.info("Sleeping %ss", cfg.binance_interval_seconds)
        time.sleep(cfg.binance_interval_seconds)
        cycle += 1
        logger.info("Cycle %s started", cycle)
        run_once()
        if cfg.max_cycles > 0 and cycle >= cfg.max_cycles:
            return


if __name__ == "__main__":
    run_service_loop()

