"""Realtime fee-rate provider for live loop.

優先嘗試 Binance USDT-M Futures 簽名端點 ``futures_commission_rate``（需有效 API Key，
與 `build_trading_client` 相同）；失敗時回退到上次成功快取或 ``default_fee_pct``（通常對齊
``Config.TRANSACTION_FEE``）。

回傳單位採「百分比」，例如 0.04 代表 0.04%（與 ``TradeExecutor`` / 訓練 Config 同口徑）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ApiTrading import build_trading_client


@dataclass
class BinanceFeeRateProvider:
    """Get realtime taker fee rate (percent) with safe fallback."""

    default_fee_pct: float
    _cached_fee_pct: Optional[float] = None

    def get_fee_rate_percent(self, *, symbol: str) -> float:
        """Return taker fee rate in percent for the symbol.

        Args:
            symbol: Futures symbol, e.g. ``BTCUSDT``.

        Returns:
            Fee rate in percent. Falls back to cached/default value on failures.
        """
        try:
            client = build_trading_client(testnet=False)
            raw_client = getattr(client, "client", None)
            if raw_client is None:
                raise RuntimeError("Underlying Binance client is unavailable.")

            payload = raw_client.futures_commission_rate(symbol=str(symbol))
            raw_rate = payload.get("takerCommissionRate")
            if raw_rate is None:
                raise RuntimeError("Missing takerCommissionRate in response.")

            # Binance API usually returns decimal (e.g. 0.0004 = 0.04%)
            rate_percent = float(raw_rate) * 100.0
            if rate_percent <= 0:
                raise RuntimeError(f"Invalid fee rate: {rate_percent}")

            self._cached_fee_pct = float(rate_percent)
            return float(rate_percent)
        except Exception:
            if self._cached_fee_pct is not None:
                return float(self._cached_fee_pct)
            return float(self.default_fee_pct)

