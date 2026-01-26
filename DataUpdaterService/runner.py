"""Orchestration runner for DataUpdaterService.

此模組把「5m 更新」與「1d 更新」組合在一起，讓：
- Windows Service 可以呼叫
- CLI/手動執行也可以呼叫
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from DataUpdaterService.binance_5m_updater import Binance5mUpdater
from DataUpdaterService.coinglass_1d_updater import CoinGlass1dUpdater
from DataUpdaterService.coinglass_client import CoinGlassClient
from DataUpdaterService.config import DataUpdaterConfig
from DataUpdaterService.logging_utils import setup_logger


@dataclass(frozen=True)
class RunSummary:
    """一次 run 的摘要（方便 log/監控）。"""

    updated_5m_symbols: int
    updated_1d_symbols: int
    updated_1d_macro_files: int


class DataUpdaterRunner:
    """高層 orchestrator（依賴抽象 updater）。"""

    def __init__(self, *, config: DataUpdaterConfig, project_root: Path) -> None:
        self._config = config
        self._project_root = project_root
        self._logger = setup_logger(name="data_updater", log_dir=config.log_dir)

        self._binance_5m = Binance5mUpdater()
        if config.coinglass_api_key:
            cg_client = CoinGlassClient(api_key=config.coinglass_api_key)
            self._coinglass_1d = CoinGlass1dUpdater(client=cg_client)
        else:
            self._coinglass_1d = None

    def run_5m_once(self) -> RunSummary:
        """執行一次 5m 更新。"""
        cnt = 0
        for sym in self._config.symbols:
            r = self._binance_5m.update_symbol(
                data_dir=self._config.data_dir,
                symbol=sym,
                initial_backfill_days=self._config.initial_backfill_days_5m,
            )
            if r.rows_fetched > 0:
                cnt += 1
            self._logger.info("5m update %s rows=%s file=%s", r.symbol, r.rows_fetched, r.csv_path)
        return RunSummary(updated_5m_symbols=cnt, updated_1d_symbols=0, updated_1d_macro_files=0)

    def run_1d_once(self) -> RunSummary:
        """執行一次 1d 更新（coinglass futures + macro）。"""
        if self._coinglass_1d is None:
            self._logger.warning("Skip 1d update: COINGLASS_API_KEY not set.")
            return RunSummary(updated_5m_symbols=0, updated_1d_symbols=0, updated_1d_macro_files=0)

        fut_res = self._coinglass_1d.update_symbols_futures_1d(
            data_dir=self._config.data_dir,
            symbols=self._config.symbols,
            exchange=self._config.exchange,
            initial_backfill_days=self._config.initial_backfill_days_1d,
        )
        updated_1d_symbols = sum(1 for _sym, r in fut_res.items() if r.rows_fetched > 0)
        for _sym, r in fut_res.items():
            self._logger.info("1d coinglass %s rows=%s file=%s", r.name, r.rows_fetched, r.csv_path)

        macro_res = self._coinglass_1d.update_macro_indexes_1d(data_dir=self._config.data_dir)
        updated_macro = sum(1 for _n, r in macro_res.items() if r.rows_fetched > 0)
        for _n, r in macro_res.items():
            self._logger.info("1d macro %s rows=%s file=%s", r.name, r.rows_fetched, r.csv_path)

        return RunSummary(updated_5m_symbols=0, updated_1d_symbols=updated_1d_symbols, updated_1d_macro_files=updated_macro)


