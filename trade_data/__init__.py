"""
trade_data: 抓取交易資料（Binance 等）並匯入 CSV / Supabase。
"""

from trade_data.csv_io import upsert_dataframe_to_csv
from trade_data.supabase_client import upsert_klines

__all__ = [
    "main",
    "upsert_dataframe_to_csv",
    "upsert_klines",
]


def main() -> None:
    """執行 K 線 pipeline（fetch → CSV → Supabase）。由 trade_data.run.main 實作。"""
    from trade_data.run import main as _main
    _main()
