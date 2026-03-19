"""統一入口：執行 Binance / CoinGlass 資料抓取。"""
import os
import sys
import argparse

# 從專案根目錄（此檔所在目錄）載入 .env，確保 SUPABASE_* / BINANCE_* 被讀到
_root = os.path.dirname(os.path.abspath(__file__))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

from trade_data.run import main
from trade_data.supabase_client import verify_supabase
from GetTradeData_CoinGlass import (
    main as coinglass_main,
    verify_coinglass_supabase_tables,
)


def parse_args() -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description="Unified data fetch entrypoint")
    parser.add_argument(
        "--source",
        choices=["binance", "coinglass", "both"],
        default="binance",
        help="Choose data source to run.",
    )
    parser.add_argument(
        "--verify-supabase",
        action="store_true",
        help="Only verify Supabase connection and exit.",
    )
    parser.add_argument(
        "--verify-supabase-coinglass",
        action="store_true",
        help="Verify CoinGlass Supabase tables and exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.verify_supabase:
        ok = verify_supabase()
        sys.exit(0 if ok else 1)

    if args.verify_supabase_coinglass:
        ok = verify_coinglass_supabase_tables()
        sys.exit(0 if ok else 1)

    if args.source in ("binance", "both"):
        main()

    if args.source in ("coinglass", "both"):
        coinglass_main()
