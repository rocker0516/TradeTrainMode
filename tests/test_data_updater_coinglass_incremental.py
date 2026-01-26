from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from DataUpdaterService.coinglass_1d_updater import CoinGlass1dUpdater


class _FakeCoinGlassClient:
    """最小 fake CoinGlass client：回傳固定 DataFrame。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Optional[Dict[str, Any]]]] = []

    def fetch_dataframe(self, *, endpoint: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:  # noqa: D401
        self.calls.append((endpoint, params))
        if endpoint == "futures/price/history":
            # 需要 time 欄位
            start = int(params.get("start_time", 0)) if params else 0
            return pd.DataFrame(
                {
                    "time": [start + 1000, start + 2000],
                    "close": [1.0, 2.0],
                    "high": [1.1, 2.1],
                    "low": [0.9, 1.9],
                }
            )
        # 其他 futures endpoint：回空（仍然能跑完整流程）
        if endpoint.startswith("futures/"):
            return pd.DataFrame()
        # index endpoints：回 timestamp
        if endpoint.startswith("index/"):
            return pd.DataFrame({"timestamp": [1_000, 2_000], "altcoin_index": [10, 20], "lth_sopr": [0.9, 1.1], "bmo_value": [0.1, 0.2]})
        return pd.DataFrame()

    def fetch_fear_greed_history(self) -> pd.DataFrame:
        return pd.DataFrame({"time": [1_000, 2_000], "fear_greed_index": [50, 60], "price": [30000, 31000]})


def test_coinglass_writes_lth_sopr_to_separate_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)

    updater = CoinGlass1dUpdater(client=_FakeCoinGlassClient())

    macro_res = updater.update_macro_indexes_1d(data_dir=data_dir)
    # 確認有產出 LTH SOPR 的獨立檔案（修正舊版覆蓋 bug）
    assert (data_dir / "bitcoin_lth_sopr_index_history_1d.csv").exists()
    assert "bitcoin_lth_sopr" in macro_res


def test_coinglass_futures_1d_creates_symbol_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)

    updater = CoinGlass1dUpdater(client=_FakeCoinGlassClient())
    r = updater.update_symbol_futures_1d(data_dir=data_dir, symbol="BTCUSDT", exchange="Binance", initial_backfill_days=1)
    assert r.csv_path.exists()
    out = pd.read_csv(r.csv_path)
    assert "time" in out.columns


