"""單元測試：`ApiTrading.Trading` 的 Binance 期貨交易客戶端（unittest 版）。

無需實際 API 金鑰與網路，透過注入的 mock 低階 client 驗證行為。
使用方式：`python ApiTrading/test.py`。
"""

from __future__ import annotations

from typing import Any, Dict, List
import os
import sys

# 支援從專案根目錄直接執行：python ApiTrading/test.py
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import unittest

from ApiTrading import BinanceFuturesClient


class FakeClient:
    """簡單可編程的假 client，模擬 python-binance 的必要方法。"""

    def __init__(self) -> None:
        self.timestamp_offset = 0
        self.last_order: Dict[str, Any] | None = None
        self.last_leverage: Dict[str, Any] | None = None

    def get_server_time(self) -> Dict[str, int]:
        return {"serverTime": 1_700_000_000_000}

    def futures_account_balance(self, recvWindow: int) -> List[Dict[str, Any]]:  # noqa: N803
        return [
            {"asset": "BNB", "balance": "3"},
            {"asset": "USDT", "balance": "123.45"},
        ]

    def futures_account(self, recvWindow: int) -> Dict[str, Any]:  # noqa: N803
        return {
            "availableBalance": "100.0",
            "assets": [
                {"asset": "USDT", "walletBalance": "150.5", "unrealizedProfit": "-2.5", "marginBalance": "148.0"}
            ],
        }

    def futures_position_information(self, symbol: str | None = None, recvWindow: int | None = None) -> List[Dict[str, Any]]:  # noqa: N803,E501
        if symbol:
            return [{"symbol": symbol, "positionAmt": "0.012"}]
        return [
            {"symbol": "BTCUSDT", "positionAmt": "0"},
            {"symbol": "ETHUSDT", "positionAmt": "-1.5", "entryPrice": "2000", "unRealizedProfit": "10", "leverage": "10"},  # noqa: E501
        ]

    def futures_exchange_info(self) -> Dict[str, Any]:  # noqa: N803
        return {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "filters": [
                        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                        {"filterType": "NOTIONAL", "notional": "5"},
                    ],
                }
            ]
        }

    def futures_create_order(self, symbol: str, side: str, type: str, quantity: float, recvWindow: int) -> Dict[str, Any]:  # noqa: A002,E501
        self.last_order = {"symbol": symbol, "side": side, "type": type, "quantity": quantity}
        return {"orderId": 1, **self.last_order}

    def futures_change_leverage(self, symbol: str, leverage: int, recvWindow: int) -> Dict[str, Any]:  # noqa: N803
        self.last_leverage = {"symbol": symbol, "leverage": leverage}
        return {"leverage": leverage}


class TestBinanceFuturesClient(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeClient()
        self.client = BinanceFuturesClient(client=self.fake)

    def assertFloatAlmostEqual(self, a: float, b: float, tol: float = 1e-9) -> None:  # type: ignore[override]
        self.assertTrue(abs(a - b) <= tol, msg=f"{a} != {b}")

    def test_get_equity_usdt(self) -> None:
        self.assertFloatAlmostEqual(self.client.get_equity_usdt(), 123.45)

    def test_get_current_position_size(self) -> None:
        self.assertFloatAlmostEqual(self.client.get_current_position_size("BTCUSDT"), 0.012)

    def test_get_account_summary(self) -> None:
        summary = self.client.get_account_summary()
        self.assertFloatAlmostEqual(summary["wallet_balance"], 150.5)
        self.assertFloatAlmostEqual(summary["available_balance"], 100.0)
        self.assertFloatAlmostEqual(summary["unrealized_pnl"], -2.5)
        self.assertFloatAlmostEqual(summary["margin_balance"], 148.0)

    def test_get_open_positions(self) -> None:
        positions = self.client.get_open_positions()
        self.assertEqual(len(positions), 1)
        p0 = positions[0]
        self.assertEqual(p0["symbol"], "ETHUSDT")
        self.assertFloatAlmostEqual(p0["position_amt"], -1.5)
        self.assertFloatAlmostEqual(p0["entry_price"], 2000.0)
        self.assertFloatAlmostEqual(p0["unrealized_pnl"], 10.0)
        self.assertFloatAlmostEqual(p0["leverage"], 10.0)

    def test_get_symbol_filters(self) -> None:
        f = self.client.get_symbol_filters("BTCUSDT", default_min_notional=10.0)
        self.assertFloatAlmostEqual(f.step_size, 0.001)
        self.assertFloatAlmostEqual(f.min_qty, 0.001)
        self.assertFloatAlmostEqual(f.min_notional, 5.0)

    def test_place_delta_order_rules(self) -> None:
        f = self.client.get_symbol_filters("BTCUSDT", default_min_notional=5.0)
        # 向下取整後為 0，略過
        res0 = self.client.place_delta_order("BTCUSDT", delta=0.00049, step_size=f.step_size, min_notional=f.min_notional, last_price=50000.0, dry_run=False)  # noqa: E501
        self.assertIsNone(res0)
        # 不足 notional，略過
        res1 = self.client.place_delta_order("BTCUSDT", delta=0.001, step_size=f.step_size, min_notional=f.min_notional, last_price=1000.0, dry_run=False)  # noqa: E501
        self.assertIsNone(res1)
        # 合法下單
        res2 = self.client.place_delta_order("BTCUSDT", delta=0.002, step_size=f.step_size, min_notional=f.min_notional, last_price=5000.0, dry_run=False)  # noqa: E501
        self.assertIsNotNone(res2)
        assert res2 is not None
        self.assertFloatAlmostEqual(float(res2.get("quantity", 0.0)), 0.002)
        self.assertEqual(res2.get("type"), "MARKET")

    def test_place_delta_order_dry_run(self) -> None:
        f = self.client.get_symbol_filters("BTCUSDT", default_min_notional=5.0)
        res = self.client.place_delta_order("BTCUSDT", delta=0.005, step_size=f.step_size, min_notional=f.min_notional, last_price=30000.0, dry_run=True)  # noqa: E501
        self.assertEqual(res, {"dry_run": True, "side": "BUY", "quantity": 0.005})

    def test_set_leverage(self) -> None:
        self.client.set_leverage("BTCUSDT", 200)
        self.assertEqual(self.fake.last_leverage, {"symbol": "BTCUSDT", "leverage": 125})


if __name__ == "__main__":
    unittest.main(verbosity=2)