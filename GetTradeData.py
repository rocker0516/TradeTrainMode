"""CLI helper for updating 5m CSVs.

本檔案保留作為「一次性手動更新」入口。
核心抓取/寫檔邏輯已移至 `DataUpdaterService/`，並移除硬編碼金鑰（符合安全規則）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from DataUpdaterService.binance_5m_updater import Binance5mUpdater


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Update Data/*_futures_volume_5years_5min.csv using public Binance endpoints.")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,1000PEPEUSDT", help="Comma-separated symbols.")
    parser.add_argument("--data_dir", type=str, default="Data", help="Data directory path.")
    parser.add_argument("--initial_backfill_days", type=int, default=30, help="Days to backfill if CSV missing/invalid.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    symbols = _parse_symbols(args.symbols)

    updater = Binance5mUpdater()
    for sym in symbols:
        r = updater.update_symbol(data_dir=data_dir, symbol=sym, initial_backfill_days=int(args.initial_backfill_days))
        print(f"[5m] {sym}: rows={r.rows_fetched} -> {r.csv_path}")


if __name__ == "__main__":
    main()
