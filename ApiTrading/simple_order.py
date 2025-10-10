"""簡單下單腳本（使用 ApiTrading.Trading）。

說明:
- 透過 `build_trading_client` 建立交易客戶端並下市價單（差量下單）。
- 需要在環境變數提供 API 金鑰（例如 `BINANCE_TRADE_API_KEY/SECRET` 或 `BINANCE_API_KEY/SECRET`）。
- 預設為 dry-run（不實際下單）；加入 `--live` 才會真實下單。

使用範例:
    python ApiTrading/simple_order.py --symbol BTCUSDT --side BUY --qty 0.002 --leverage 10 --testnet
    python ApiTrading/simple_order.py --symbol ETHUSDT --side SELL --qty 0.01 --live
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional
import os
import sys

# 支援從專案根目錄直接執行：python ApiTrading/simple_order.py
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ApiTrading import ITradingClient, build_trading_client


def _fetch_last_price(client: ITradingClient, symbol: str) -> float:
    """取得標的最新價格（優先使用 Mark Price）。

    參數:
        client: 交易客戶端（需包含底層 `client`）。
        symbol: 交易對，例如 'BTCUSDT'。

    回傳:
        最新價格（float）。

    例外:
        RuntimeError: 若無法取得價格。
    """
    low: Any = getattr(client, "client", None)
    if low is None:
        raise RuntimeError("Low-level client is unavailable; cannot fetch price.")

    price: Optional[float] = None
    try:
        mp: Dict[str, Any] = low.futures_mark_price(symbol=symbol)
        if mp and "markPrice" in mp:
            price = float(mp["markPrice"])
    except Exception:
        # 標記價格端點不可用時，改用 ticker 價格
        price = None

    if price is None:
        try:
            tk: Dict[str, Any] = low.futures_symbol_ticker(symbol=symbol)
            if tk and "price" in tk:
                price = float(tk["price"])
        except Exception:
            price = None

    if price is None:
        raise RuntimeError(f"Unable to fetch price for {symbol}")
    return price


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple market order via Binance UM-Futures")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol, e.g., BTCUSDT")
    parser.add_argument("--side", type=str, choices=["BUY", "SELL"], default="BUY", help="Order side")
    parser.add_argument("--qty", type=float, default=0.001, help="Order base quantity (absolute value)")
    parser.add_argument("--leverage", type=int, default=10, help="Leverage to set before ordering (1-125)")
    parser.add_argument("--testnet", action="store_true", help="Use Binance Futures testnet endpoint")
    parser.add_argument("--live", action="store_true", help="Place a real order (disable dry-run)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # 建立交易客戶端（憑證由環境變數提供）
    client: ITradingClient = build_trading_client(testnet=args.testnet)

    symbol = args.symbol.upper()
    side = args.side.upper()
    qty = float(args.qty)
    leverage = int(args.leverage)
    dry_run = not bool(args.live)

    # 設定槓桿
    client.set_leverage(symbol, leverage)

    # 取得交易限制與價格
    filters = client.get_symbol_filters(symbol, default_min_notional=5.0)
    last_price = _fetch_last_price(client, symbol)

    # 計算差量（買入為正、賣出為負）
    delta = qty if side == "BUY" else -qty

    result = client.place_delta_order(
        symbol=symbol,
        delta=delta,
        step_size=filters.step_size,
        min_notional=filters.min_notional,
        last_price=last_price,
        dry_run=dry_run,
    )

    info: Dict[str, Any] = {
        "environment": "testnet" if args.testnet else "mainnet",
        "dry_run": dry_run,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "last_price": last_price,
        "result": result if result is not None else "skipped (too small or below min_notional)",
    }
    print(info)


if __name__ == "__main__":
    main()


