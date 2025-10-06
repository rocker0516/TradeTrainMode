import os
import types
import builtins
import importlib
from unittest import mock
import pytest


@pytest.mark.skipif(not os.path.exists('RealTrade.py'), reason='RealTrade module not present')
def test_build_trader_from_env_price(monkeypatch):
    # Arrange environment
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.setenv("BINANCE_TESTNET", "1")

    # Mock python-binance Client and exceptions
    fake_client_cls = mock.MagicMock()
    fake_client = mock.MagicMock()
    fake_client.futures_symbol_ticker.return_value = {"price": "12345.6"}
    fake_client_cls.return_value = fake_client

    fake_excs = types.SimpleNamespace(
        BinanceAPIException=Exception,
        BinanceOrderException=Exception,
    )

    with mock.patch.dict("sys.modules", {
        "binance": types.ModuleType("binance"),
        "binance.client": types.SimpleNamespace(Client=fake_client_cls),
        "binance.exceptions": fake_excs,
    }):
        # Import module fresh
        if "RealTrade" in list(globals()):
            del globals()["RealTrade"]
        import RealTrade as rt
        importlib.reload(rt)

        # Act
        trader = rt.build_trader_from_env()
        price = trader.get_price("BTCUSDT")

        # Assert
        assert price == 12345.6
        fake_client.futures_symbol_ticker.assert_called_once_with(symbol="BTCUSDT")


@pytest.mark.skipif(not os.path.exists('RealTrade.py'), reason='RealTrade module not present')
def test_trader_market_buy_and_cancel(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")

    fake_client_cls = mock.MagicMock()
    fake_client = mock.MagicMock()
    fake_client.futures_create_order.return_value = {"orderId": 1}
    fake_client.futures_cancel_all_open_orders.return_value = {"code": 200}
    fake_client.futures_change_leverage.return_value = {"leverage": 10}
    fake_client.futures_position_information.return_value = [{"symbol": "BTCUSDT"}]
    fake_client_cls.return_value = fake_client

    fake_excs = types.SimpleNamespace(
        BinanceAPIException=Exception,
        BinanceOrderException=Exception,
    )

    with mock.patch.dict("sys.modules", {
        "binance": types.ModuleType("binance"),
        "binance.client": types.SimpleNamespace(Client=fake_client_cls),
        "binance.exceptions": fake_excs,
    }):
        import RealTrade as rt
        importlib.reload(rt)

        trader = rt.build_trader_from_env()

        res_buy = trader.market_buy("BTCUSDT", 0.001)
        assert res_buy["orderId"] == 1
        fake_client.futures_create_order.assert_called_with(symbol="BTCUSDT", side="BUY", type="MARKET", quantity=0.001)

        res_cancel = trader.cancel_all_orders("BTCUSDT")
        assert res_cancel["code"] == 200
        fake_client.futures_cancel_all_open_orders.assert_called_with(symbol="BTCUSDT")

        res_lev = trader.set_leverage("BTCUSDT", 999)
        assert res_lev["leverage"] == 10
        fake_client.futures_change_leverage.assert_called_with(symbol="BTCUSDT", leverage=125)

        pos = trader.get_position_information("BTCUSDT")
        assert isinstance(pos, list)


