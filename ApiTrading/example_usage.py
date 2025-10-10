"""幣安合約交易 API 使用範例。

此腳本示範如何使用 Trading.py 模組進行各種交易操作。
"""

import os
from Trading import build_trading_client, ITradingClient


def example_account_info() -> None:
    """範例：查詢帳戶資訊。"""
    print("=== Example: Account Information ===")
    
    # 建立交易客戶端（從環境變數讀取憑證）
    client: ITradingClient = build_trading_client(testnet=True)
    
    # 取得帳戶摘要
    summary = client.get_account_summary()
    print(f"Wallet Balance: {summary['wallet_balance']} USDT") # 錢包餘額
    print(f"Available Balance: {summary['available_balance']} USDT") # 可用餘額
    print(f"Unrealized PnL: {summary['unrealized_pnl']} USDT") # 未實現損益
    print(f"Margin Balance: {summary['margin_balance']} USDT") # 保證金餘額
    print()


def example_check_positions() -> None:
    """範例：檢查所有未平倉部位。"""
    print("=== Example: Open Positions ===")
    
    client: ITradingClient = build_trading_client(testnet=True)
    
    # 取得所有非零持倉
    positions = client.get_open_positions()
    
    if not positions:
        print("No open positions.")
    else:
        for pos in positions:
            print(f"Symbol: {pos['symbol']}")
            print(f"  Position Size: {pos['position_amt']}")
            print(f"  Entry Price: {pos['entry_price']}")
            print(f"  Unrealized PnL: {pos['unrealized_pnl']}")
            print(f"  Leverage: {pos['leverage']}x")
            print()


def example_place_order() -> None:
    """Example: Place a delta order (DRY RUN)."""
    print("=== Example: Place Delta Order (DRY RUN) ===")
    
    client: ITradingClient = build_trading_client(testnet=True)
    symbol = "BTCUSDT"
    
    # Get symbol filters
    filters = client.get_symbol_filters(symbol=symbol, default_min_notional=10.0)
    print(f"Symbol Filters for {symbol}:")
    print(f"  Step Size: {filters.step_size}")
    print(f"  Min Quantity: {filters.min_qty}")
    print(f"  Min Notional: {filters.min_notional} USDT")
    print()
    
    # 目前持倉
    current_pos = client.get_current_position_size(symbol)
    print(f"Current Position: {current_pos}")
    
    # 模擬下單：將持倉增加 0.001 BTC
    delta = 0.001
    last_price = 50000.0  # 範例價格
    
    result = client.place_delta_order(
        symbol=symbol,
        delta=delta,
        step_size=filters.step_size,
        min_notional=filters.min_notional,
        last_price=last_price,
        dry_run=True,  # 模擬模式 - 不會實際下單
    )
    
    if result:
        print(f"Order (DRY RUN): {result}")
    else:
        print("Order skipped (quantity or notional too small)")
    print()


def example_set_leverage() -> None:
    """範例：設定交易對槓桿。"""
    print("=== Example: Set Leverage ===")
    
    client: ITradingClient = build_trading_client(testnet=True)
    symbol = "BTCUSDT"
    leverage = 10
    
    print(f"Setting leverage for {symbol} to {leverage}x...")
    client.set_leverage(symbol=symbol, leverage=leverage)
    print("Leverage set successfully!")
    print()


def main() -> None:
    """執行所有範例。"""
    # 確認環境變數已設定
    if not os.getenv("BINANCE_TRADE_API_KEY") and not os.getenv("BINANCE_API_KEY"):
        print("ERROR: Please set BINANCE_TRADE_API_KEY or BINANCE_API_KEY environment variable.")
        print("       Also set BINANCE_TRADE_API_SECRET or BINANCE_API_SECRET.")
        print("       Optional: Set BINANCE_TESTNET=1 to use testnet.")
        return
    
    try:
        # 執行範例
        example_account_info()
        example_check_positions()
        example_set_leverage()
        example_place_order()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

