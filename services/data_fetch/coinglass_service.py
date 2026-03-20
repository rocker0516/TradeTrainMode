"""
Standalone CoinGlass fetch service (new module, legacy files untouched).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd

from GetTradeData_CoinGlass import CoinGlassClient, run_coinglass_fetch, upsert_dataframe_to_csv
from services.data_fetch.config import load_config
from services.data_fetch.incremental import interval_to_timedelta, resolve_start_time
from services.data_fetch.logging_utils import init_file_logger


logger = init_file_logger("data_fetch.coinglass", "logs/data_fetch/coinglass_service.log")


def _save_macro_indexes(client: CoinGlassClient) -> None:
    fear_greed = client._fetch_fear_greed_history()
    if not fear_greed.empty and "time" in fear_greed.columns:
        fear_greed["time"] = pd.to_datetime(fear_greed["time"], unit="ms")
        upsert_dataframe_to_csv(
            fear_greed,
            "Data/fear_greed_index_history_1d.csv",
            key_cols=["time"],
            parse_dates=["time"],
        )

    endpoint_to_file = {
        "index/altcoin-season": "Data/altcoin_season_index_history_1d.csv",
        "index/bitcoin-sth-sopr": "Data/bitcoin_sth_sopr_index_history_1d.csv",
        "index/bitcoin-lth-sopr": "Data/bitcoin_sth_sopr_index_history_1d.csv",
        "index/bitcoin-macro-oscillator": "Data/bitcoin_macro_oscillator_index_history_1d.csv",
    }
    for endpoint, filename in endpoint_to_file.items():
        df = client._fetch_index_history(endpoint=endpoint)
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            upsert_dataframe_to_csv(df, filename, key_cols=["timestamp"], parse_dates=["timestamp"])


def run_once() -> None:
    """Run one CoinGlass backfill/update cycle."""
    cfg = load_config()
    if not cfg.coinglass_api_key:
        raise ValueError("COINGLASS_API_KEY is required for CoinGlass service.")

    diff = timedelta(days=1)
    end_dt = datetime.now() - diff
    fallback_start_dt = end_dt - timedelta(days=cfg.coinglass_lookback_days)
    step = interval_to_timedelta(cfg.coinglass_interval, timedelta(days=1))
    diff_ms = int(diff.total_seconds() * 1000)

    for symbol in cfg.coinglass_symbols:
        try:
            filename = f"Data/{symbol}_futures_volume_coinglass_5years_{cfg.coinglass_interval}.csv"
            start_dt = resolve_start_time(filename, "time", fallback_start_dt, step)
            if start_dt >= end_dt:
                logger.info("%s is up to date, skip.", symbol)
                continue

            logger.info("Fetching %s from %s to %s", symbol, start_dt, end_dt)
            all_klines = run_coinglass_fetch(
                cfg.coinglass_api_key,
                cfg.coinglass_exchange,
                symbol,
                cfg.coinglass_interval,
                start_dt,
                end_dt,
                diff_ms,
            )
            if all_klines.empty:
                continue
            all_klines = all_klines.sort_values("time").drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)
            all_klines["time"] = pd.to_datetime(all_klines["time"], unit="ms")
            upsert_dataframe_to_csv(all_klines.dropna(), filename, key_cols=["time"], parse_dates=["time"])
            logger.info("Saved %s", filename)
            time.sleep(1)
        except Exception as exc:
            logger.exception("Fetch failed for %s: %s", symbol, exc)

    _save_macro_indexes(CoinGlassClient(api_key=cfg.coinglass_api_key))


def run_service_loop() -> None:
    """Run periodic CoinGlass service loop."""
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
        logger.info("Sleeping %ss", cfg.coinglass_interval_seconds)
        time.sleep(cfg.coinglass_interval_seconds)
        cycle += 1
        logger.info("Cycle %s started", cycle)
        run_once()
        if cfg.max_cycles > 0 and cycle >= cfg.max_cycles:
            return


if __name__ == "__main__":
    run_service_loop()

