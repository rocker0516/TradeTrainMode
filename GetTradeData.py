"""入口：取得 Binance 期貨 K 線並寫入 CSV 與 Supabase。"""
import os
import sys

# 從專案根目錄（此檔所在目錄）載入 .env，確保 SUPABASE_* / BINANCE_* 被讀到
_root = os.path.dirname(os.path.abspath(__file__))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

from trade_data.run import main
from trade_data.supabase_client import verify_supabase


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--verify-supabase":
        ok = verify_supabase()
        sys.exit(0 if ok else 1)
    main()
