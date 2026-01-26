"""CLI helper for updating 1d CoinGlass CSVs.

本檔案保留作為「一次性手動更新」入口。
核心抓取/寫檔邏輯已移至 `DataUpdaterService/`，並移除硬編碼金鑰（符合安全規則）。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from DataUpdaterService.coinglass_client import CoinGlassClient
from DataUpdaterService.coinglass_1d_updater import CoinGlass1dUpdater


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Update Data/*_1d.csv using CoinGlass API.")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,1000PEPEUSDT", help="Comma-separated symbols.")
    parser.add_argument("--exchange", type=str, default="Binance", help="Exchange name for CoinGlass endpoints.")
    parser.add_argument("--data_dir", type=str, default="Data", help="Data directory path.")
    parser.add_argument("--initial_backfill_days", type=int, default=3650, help="Days to backfill if CSV missing/invalid.")
    args = parser.parse_args()

    api_key = os.getenv("COINGLASS_API_KEY")
    if not api_key:
        raise SystemExit("Missing COINGLASS_API_KEY env var.")

    data_dir = Path(args.data_dir).expanduser().resolve()
    symbols = _parse_symbols(args.symbols)

    updater = CoinGlass1dUpdater(client=CoinGlassClient(api_key=api_key))
    fut_res = updater.update_symbols_futures_1d(
        data_dir=data_dir,
        symbols=symbols,
        exchange=str(args.exchange),
        initial_backfill_days=int(args.initial_backfill_days),
    )
    for sym, r in fut_res.items():
        print(f"[1d coinglass] {sym}: rows={r.rows_fetched} -> {r.csv_path}")

    macro_res = updater.update_macro_indexes_1d(data_dir=data_dir)
    for name, r in macro_res.items():
        print(f"[1d macro] {name}: rows={r.rows_fetched} -> {r.csv_path}")


if __name__ == "__main__":
    main()