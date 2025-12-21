"""Binance Futures Trading API Module.

This module provides a clean, object-oriented interface for Binance UM-Futures trading operations.
Follows SOLID principles with abstract base classes for extensibility and testability.

Main Components:
    - ITradingClient: Abstract base class defining trading operations interface
    - BinanceFuturesClient: Concrete implementation for Binance Futures API
    - SymbolFilters: Dataclass for trading filters (step size, min quantity, etc.)
    - build_trading_client: Factory function to create trading clients

Example:
    >>> from ApiTrading import build_trading_client
    >>> client = build_trading_client(testnet=True)
    >>> summary = client.get_account_summary()
    >>> print(summary['wallet_balance'])
"""

from .Trading import (
    BinanceFuturesClient,
    ITradingClient,
    SymbolFilters,
    build_trading_client,
)

__all__ = [
    "ITradingClient",
    "BinanceFuturesClient",
    "SymbolFilters",
    "build_trading_client",
]

__version__ = "1.0.0"

