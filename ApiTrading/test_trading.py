"""Unit tests for Trading.py module.

Tests the BinanceFuturesClient implementation using mock objects.
"""

import pytest
from unittest import mock
from Trading import (
    BinanceFuturesClient,
    ITradingClient,
    SymbolFilters,
    build_trading_client,
)


class TestSymbolFilters:
    """Test SymbolFilters dataclass."""

    def test_symbol_filters_creation(self) -> None:
        """Test creating SymbolFilters instance."""
        filters = SymbolFilters(step_size=0.001, min_qty=0.0001, min_notional=10.0)
        assert filters.step_size == 0.001
        assert filters.min_qty == 0.0001
        assert filters.min_notional == 10.0


class TestBinanceFuturesClient:
    """Test BinanceFuturesClient implementation."""

    def setup_method(self) -> None:
        """Setup test fixtures before each test."""
        self.mock_client = mock.MagicMock()
        self.trading_client = BinanceFuturesClient.__new__(BinanceFuturesClient)
        self.trading_client.client = self.mock_client

    def test_get_equity_usdt_success(self) -> None:
        """Test getting USDT balance successfully."""
        self.mock_client.futures_account_balance.return_value = [
            {"asset": "USDT", "balance": "1000.5"},
            {"asset": "BNB", "balance": "10.0"},
        ]
        
        equity = self.trading_client.get_equity_usdt()
        assert equity == 1000.5

    def test_get_equity_usdt_not_found(self) -> None:
        """Test getting USDT balance when USDT not in balance list."""
        self.mock_client.futures_account_balance.return_value = [
            {"asset": "BNB", "balance": "10.0"},
        ]
        
        equity = self.trading_client.get_equity_usdt()
        assert equity == 0.0

    def test_get_current_position_size_long(self) -> None:
        """Test getting long position size."""
        self.mock_client.futures_position_information.return_value = [
            {"positionAmt": "0.5"},
        ]
        
        position = self.trading_client.get_current_position_size("BTCUSDT")
        assert position == 0.5

    def test_get_current_position_size_short(self) -> None:
        """Test getting short position size."""
        self.mock_client.futures_position_information.return_value = [
            {"positionAmt": "-0.3"},
        ]
        
        position = self.trading_client.get_current_position_size("BTCUSDT")
        assert position == -0.3

    def test_get_current_position_size_no_position(self) -> None:
        """Test getting position when no position exists."""
        self.mock_client.futures_position_information.return_value = []
        
        position = self.trading_client.get_current_position_size("BTCUSDT")
        assert position == 0.0

    def test_get_account_summary(self) -> None:
        """Test getting account summary."""
        self.mock_client.futures_account.return_value = {
            "availableBalance": "500.25",
            "assets": [
                {
                    "asset": "USDT",
                    "walletBalance": "1000.50",
                    "unrealizedProfit": "50.75",
                    "marginBalance": "1051.25",
                }
            ],
        }
        
        summary = self.trading_client.get_account_summary()
        assert summary["wallet_balance"] == 1000.50
        assert summary["available_balance"] == 500.25
        assert summary["unrealized_pnl"] == 50.75
        assert summary["margin_balance"] == 1051.25

    def test_get_open_positions(self) -> None:
        """Test getting open positions."""
        self.mock_client.futures_position_information.return_value = [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.5",
                "entryPrice": "50000.0",
                "unRealizedProfit": "100.0",
                "leverage": "10",
            },
            {
                "symbol": "ETHUSDT",
                "positionAmt": "0.0",
                "entryPrice": "3000.0",
                "unRealizedProfit": "0.0",
                "leverage": "5",
            },
            {
                "symbol": "SOLUSDT",
                "positionAmt": "-2.0",
                "entryPrice": "100.0",
                "unRealizedProfit": "-5.0",
                "leverage": "3",
            },
        ]
        
        positions = self.trading_client.get_open_positions()
        assert len(positions) == 2  # Only non-zero positions
        assert positions[0]["symbol"] == "BTCUSDT"
        assert positions[0]["position_amt"] == 0.5
        assert positions[1]["symbol"] == "SOLUSDT"
        assert positions[1]["position_amt"] == -2.0

    def test_get_symbol_filters_success(self) -> None:
        """Test getting symbol filters successfully."""
        self.mock_client.futures_exchange_info.return_value = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "filters": [
                        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5.0"},
                    ],
                }
            ]
        }
        
        filters = self.trading_client.get_symbol_filters("BTCUSDT", default_min_notional=10.0)
        assert filters.step_size == 0.001
        assert filters.min_qty == 0.001
        assert filters.min_notional == 5.0

    def test_get_symbol_filters_not_found(self) -> None:
        """Test getting symbol filters when symbol not found."""
        self.mock_client.futures_exchange_info.return_value = {"symbols": []}
        
        filters = self.trading_client.get_symbol_filters("UNKNOWN", default_min_notional=10.0)
        assert filters.step_size == 0.0001  # Default value
        assert filters.min_qty == 0.0001    # Default value
        assert filters.min_notional == 10.0  # Default value

    def test_place_delta_order_buy(self) -> None:
        """Test placing a BUY order."""
        self.mock_client.futures_create_order.return_value = {
            "orderId": "12345",
            "symbol": "BTCUSDT",
        }
        
        result = self.trading_client.place_delta_order(
            symbol="BTCUSDT",
            delta=0.5,  # Positive = BUY
            step_size=0.001,
            min_notional=10.0,
            last_price=50000.0,
            dry_run=False,
        )
        
        assert result is not None
        assert result["orderId"] == "12345"
        self.mock_client.futures_create_order.assert_called_once()

    def test_place_delta_order_sell(self) -> None:
        """Test placing a SELL order."""
        self.mock_client.futures_create_order.return_value = {
            "orderId": "67890",
            "symbol": "BTCUSDT",
        }
        
        result = self.trading_client.place_delta_order(
            symbol="BTCUSDT",
            delta=-0.3,  # Negative = SELL
            step_size=0.001,
            min_notional=10.0,
            last_price=50000.0,
            dry_run=False,
        )
        
        assert result is not None
        self.mock_client.futures_create_order.assert_called_once()

    def test_place_delta_order_dry_run(self) -> None:
        """Test placing order in dry run mode."""
        result = self.trading_client.place_delta_order(
            symbol="BTCUSDT",
            delta=0.5,
            step_size=0.001,
            min_notional=10.0,
            last_price=50000.0,
            dry_run=True,
        )
        
        assert result is not None
        assert result["dry_run"] is True
        assert result["side"] == "BUY"
        assert result["quantity"] == 0.5
        self.mock_client.futures_create_order.assert_not_called()

    def test_place_delta_order_too_small_quantity(self) -> None:
        """Test order is skipped when quantity rounds to zero."""
        result = self.trading_client.place_delta_order(
            symbol="BTCUSDT",
            delta=0.0001,  # Too small
            step_size=0.001,
            min_notional=10.0,
            last_price=50000.0,
            dry_run=False,
        )
        
        assert result is None
        self.mock_client.futures_create_order.assert_not_called()

    def test_place_delta_order_below_min_notional(self) -> None:
        """Test order is skipped when notional value is too small."""
        result = self.trading_client.place_delta_order(
            symbol="BTCUSDT",
            delta=0.001,
            step_size=0.001,
            min_notional=100.0,  # High min notional
            last_price=10.0,     # Low price -> low notional
            dry_run=False,
        )
        
        assert result is None  # 0.001 * 10 = 0.01 < 100.0
        self.mock_client.futures_create_order.assert_not_called()

    def test_set_leverage(self) -> None:
        """Test setting leverage."""
        self.trading_client.set_leverage("BTCUSDT", 10)
        self.mock_client.futures_change_leverage.assert_called_once_with(
            symbol="BTCUSDT", leverage=10
        )

    def test_set_leverage_clamps_to_valid_range(self) -> None:
        """Test leverage is clamped to valid range [1, 125]."""
        # Test upper bound
        self.trading_client.set_leverage("BTCUSDT", 200)
        self.mock_client.futures_change_leverage.assert_called_with(
            symbol="BTCUSDT", leverage=125
        )
        
        # Test lower bound
        self.trading_client.set_leverage("BTCUSDT", 0)
        self.mock_client.futures_change_leverage.assert_called_with(
            symbol="BTCUSDT", leverage=1
        )

    def test_floor_to_step(self) -> None:
        """Test _floor_to_step static method."""
        assert self.trading_client._floor_to_step(0.12345, 0.001) == 0.123
        assert self.trading_client._floor_to_step(1.567, 0.01) == 1.56
        assert self.trading_client._floor_to_step(10.999, 1.0) == 10.0
        assert self.trading_client._floor_to_step(5.5, 0.0) == 5.5  # step=0 returns original


class TestBuildTradingClient:
    """Test build_trading_client factory function."""

    @mock.patch.dict("os.environ", {"BINANCE_TRADE_API_KEY": "test_key", "BINANCE_TRADE_API_SECRET": "test_secret"})
    @mock.patch("Trading.Client")
    def test_build_trading_client(self, mock_client_class: mock.MagicMock) -> None:
        """Test building trading client."""
        client = build_trading_client(testnet=False)
        assert isinstance(client, ITradingClient)
        assert isinstance(client, BinanceFuturesClient)

    def test_interface_compliance(self) -> None:
        """Test that BinanceFuturesClient implements ITradingClient."""
        # This test verifies that all abstract methods are implemented
        assert issubclass(BinanceFuturesClient, ITradingClient)
        
        # Verify all methods exist
        mock_client = BinanceFuturesClient.__new__(BinanceFuturesClient)
        mock_client.client = mock.MagicMock()
        
        assert callable(getattr(mock_client, "get_equity_usdt"))
        assert callable(getattr(mock_client, "get_current_position_size"))
        assert callable(getattr(mock_client, "get_account_summary"))
        assert callable(getattr(mock_client, "get_open_positions"))
        assert callable(getattr(mock_client, "get_symbol_filters"))
        assert callable(getattr(mock_client, "place_delta_order"))
        assert callable(getattr(mock_client, "set_leverage"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

