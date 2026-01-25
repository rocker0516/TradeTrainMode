"""幣安 USDT 本位合約交易 API 客戶端。

本模組提供與幣安合約交易端點互動的乾淨介面。
遵循 SOLID 原則，並透過抽象基類（ABC）提升可擴充性。
"""

from __future__ import annotations

import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Expose Client symbol for tests to patch
try:
    from binance.client import Client as _RealClient  # type: ignore
    Client = _RealClient  # noqa: N816 - keep name for test patching
except Exception:
    class Client:  # type: ignore
        pass

@dataclass
class SymbolFilters:
    """特定交易對的交易限制設定。

    屬性:
        step_size: 下單數量的最小增量。
        min_qty: 允許的最小下單數量。
        min_notional: 下單所需的最小名義價值。
    """

    step_size: float
    min_qty: float
    min_notional: float


class ITradingClient(ABC):
    """交易客戶端操作的抽象基類。

    定義所有交易操作所需的介面。任何具體實作都必須提供這些方法，
    以確保在不同交易所 API 或測試環境下具有一致的行為。
    """

    @abstractmethod
    def get_equity_usdt(self) -> float:
        """取得目前 USDT 錢包餘額。

        回傳:
            以 float 表示的 USDT 餘額。
        """
        pass

    @abstractmethod
    def get_current_position_size(self, symbol: str) -> float:
        """取得指定交易對的目前持倉數量。

        參數:
            symbol: 交易對代號（例如 'BTCUSDT'）。

        回傳:
            持倉數量（>0 代表多單，<0 代表空單，0 代表無持倉）。
        """
        pass

    @abstractmethod
    def get_account_summary(self) -> Dict[str, float]:
        """取得帳戶摘要與關鍵指標。

        回傳:
            包含錢包餘額、可用餘額、未實現損益與保證金餘額的字典。
        """
        pass

    @abstractmethod
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """取得所有非零持倉。

        回傳:
            含持倉明細的字典列表。
        """
        pass

    @abstractmethod
    def get_symbol_filters(self, symbol: str, default_min_notional: float) -> SymbolFilters:
        """取得交易對的下單限制設定。

        參數:
            symbol: 交易對代號。
            default_min_notional: 若 API 無回傳則使用的預設最小名義價值。

        回傳:
            含 step_size、min_qty、min_notional 的 `SymbolFilters`。
        """
        pass

    @abstractmethod
    def place_delta_order(
        self,
        symbol: str,
        delta: float,
        step_size: float,
        min_notional: float,
        last_price: float,
        dry_run: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """以下市價單依照差量調整持倉。

        參數:
            symbol: 交易對代號。
            delta: 目標調整數量（正=買入，負=賣出）。
            step_size: 數量的最小增量。
            min_notional: 最小名義價值限制。
            last_price: 當前價格（用於名義價值計算）。
            dry_run: 若為 True，僅模擬不實際下單。

        回傳:
            若有下單回傳訂單資訊；若因條件不符而略過則回傳 None。
        """
        pass

    @abstractmethod
    def set_leverage(self, symbol: str, leverage: int) -> None:
        """設定指定交易對的槓桿倍數。

        參數:
            symbol: 交易對代號。
            leverage: 槓桿數值（通常為 1-125）。
        """
        pass


class BinanceFuturesClient(ITradingClient):
    """幣安 USDT 本位合約交易客戶端實作。

    處理需授權的交易操作，包括持倉管理、下單與帳戶查詢。

    屬性:
        client: python-binance 的 Client 實例。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        client: Optional[Any] = None,
    ) -> None:
        """初始化幣安合約交易客戶端。

        參數:
            api_key: 幣安 API Key；若為 None，將從環境變數讀取。
            api_secret: 幣安 API Secret；若為 None，將從環境變數讀取。
            testnet: 是否使用測試網端點。

        例外:
            RuntimeError: 當找不到必要憑證時拋出。
        """
        # 若外部已注入低階客戶端，則直接使用（便於單元測試與依賴反轉）
        if client is not None:
            self.client = client
        else:
            # 延遲匯入以避免在測試環境強制安裝依賴
            # 從環境變數讀取憑證，優先使用正式環境名稱，向下相容測試名稱
            api_key = (
                api_key
                or os.getenv("BINANCE_TRADE_API_KEY")
                or os.getenv("BINANCE_API_KEY")
                or os.getenv("BINANCE_TEST_TRADE_API_KEY")
                or os.getenv("BINANCE_TEST_API_KEY")
                or 'JFyghzzEKteoSzrDlUbNPAYxyCnFwmwylHyNAkxRJhW4xlPdFcN5b9UgYBwU2o0p'
            )
            api_secret = (
                api_secret
                or os.getenv("BINANCE_TRADE_API_SECRET")
                or os.getenv("BINANCE_API_SECRET")
                or os.getenv("BINANCE_TEST_TRADE_API_SECRET")
                or os.getenv("BINANCE_TEST_API_SECRET")
                or  'N43ec0htdk2Bp14WuTLKmU0mhFsk4mmk8AMZy6PsPTshccVa5PDaOKu2QJ6Ngth1'
            )

            if not api_key or not api_secret:
                raise RuntimeError("Missing API credentials: set BINANCE_TRADE_API_KEY/SECRET or BINANCE_API_KEY/SECRET")

            # 與手動測試一致：不傳 testnet 參數給 Client，僅在 testnet=True 時改 FUTURES_URL
            self.client = Client(api_key, api_secret)

        # 不論是否拋出 TypeError，只要是測試網就強制指向期貨測試網端點
        if testnet:
            try:
                self.client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
            except Exception:
                pass

        # 私有端點呼叫的預設時間窗（毫秒）
        self.recv_window_ms: int = 60000

        # 同步時間偏移，避免因本機時間漂移導致簽名驗證失敗（常見 -2015）
        try:
            server_time = self.client.get_server_time()["serverTime"]
            local_time = int(time.time() * 1000)
            # python-binance 支援 timestamp_offset
            self.client.timestamp_offset = server_time - local_time
        except Exception:
            # 若失敗，沿用預設偏移（0）
            pass

    def _recv_window(self) -> int:
        """Safe recvWindow accessor for tests that bypass __init__."""
        return getattr(self, "recv_window_ms", 60000)

    def get_equity_usdt(self) -> float:
        """取得 UM 合約帳戶的 USDT 錢包餘額。

        回傳:
            以 float 表示的 USDT 餘額。
        """
        balances = self.client.futures_account_balance(recvWindow=self._recv_window())
        for b in balances:
            if b.get("asset") == "USDT":
                return float(b.get("balance", 0.0))
        return 0.0

    def get_current_position_size(self, symbol: str) -> float:
        """取得指定交易對的目前持倉數量。"""
        positions = self.client.futures_position_information(symbol=symbol, recvWindow=self._recv_window())
        if not positions:
            return 0.0
        pos_amt = float(positions[0].get("positionAmt", 0.0))
        return pos_amt

    def get_account_summary(self) -> Dict[str, float]:
        """取得 USDT 本位合約帳戶的重要指標摘要。"""
        acc = self.client.futures_account(recvWindow=self._recv_window())
        total_wb = 0.0
        total_ab = 0.0
        total_upnl = 0.0
        total_mb = 0.0

        try:
            assets = acc.get("assets", [])
            for a in assets:
                if a.get("asset") == "USDT":
                    total_wb = float(a.get("walletBalance", 0.0))
                    total_upnl = float(a.get("unrealizedProfit", 0.0))
                    total_mb = float(a.get("marginBalance", 0.0))
            total_ab = float(acc.get("availableBalance", 0.0))
        except Exception:
            pass

        return {
            "wallet_balance": total_wb,
            "available_balance": total_ab,
            "unrealized_pnl": total_upnl,
            "margin_balance": total_mb,
        }

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """取得所有非零持倉（跨全部交易對）。

        回傳包含 symbol、position_amt、entry_price、unrealized_pnl、leverage 的列表。
        """
        pos = self.client.futures_position_information(recvWindow=self._recv_window())
        out: List[Dict[str, Any]] = []
        for p in pos:
            amt = float(p.get("positionAmt", 0.0))
            if abs(amt) > 0:
                out.append(
                    {
                        "symbol": p.get("symbol"),
                        "position_amt": amt,
                        "entry_price": float(p.get("entryPrice", 0.0)),
                        "unrealized_pnl": float(p.get("unRealizedProfit", 0.0)),
                        "leverage": float(p.get("leverage", 0.0)),
                    }
                )
        return out

    def get_symbol_filters(self, symbol: str, default_min_notional: float) -> SymbolFilters:
        """Fetch quantity and notional filters for a futures symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            default_min_notional: Default minimum notional if not found in API.

        Returns:
            SymbolFilters with step_size, min_qty, and min_notional.
        """
        info = self.client.futures_exchange_info()
        syms: List[Dict[str, Any]] = info.get("symbols", [])
        s = next((x for x in syms if x.get("symbol") == symbol), None)

        if not s:
            # 找不到交易對時使用通用預設值
            return SymbolFilters(step_size=0.0001, min_qty=0.0001, min_notional=default_min_notional)

        step = 0.0001
        min_qty = 0.0001
        min_notional = default_min_notional

        for f in s.get("filters", []):
            if f.get("filterType") == "LOT_SIZE":
                step = float(f.get("stepSize", step))
                min_qty = float(f.get("minQty", min_qty))
            if f.get("filterType") == "MIN_NOTIONAL":
                min_notional = float(f.get("notional", default_min_notional))
            if f.get("filterType") == "NOTIONAL":
                min_notional = float(f.get("notional", default_min_notional))

        return SymbolFilters(step_size=step, min_qty=min_qty, min_notional=min_notional)

    def place_delta_order(
        self,
        symbol: str,
        delta: float,
        step_size: float,
        min_notional: float,
        last_price: float,
        dry_run: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """以下市價單依差量調整持倉數量。

        Args:
            symbol: 交易對代號 (例如，'BTCUSDT')。
            delta: 持倉變動 (正=買入，負=賣出)。
            step_size: 最小數量增量作為取整用。
            min_notional: 所需的最低名義價值。
            last_price: 名義計算的當前價格。
            dry_run: 若為 True，模擬不下實際單。

        Returns:
            Order response dict if placed, None if skipped due to small size.
        """
        side = "BUY" if delta > 0 else "SELL"
        qty = self._floor_to_step(abs(delta), step_size)# 向下取整至最小增量

        if qty <= 0:
            return None

        if (qty * last_price) < min_notional:# 名義價值小於最小名義價值
            return None

        if dry_run:
            return {"dry_run": True, "side": side, "quantity": qty}

        # 下單 = (交易對, 方向, 類型, 數量, 時間窗)
        return self.client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=qty,
            recvWindow=self._recv_window(),
        )

    def set_leverage(self, symbol: str, leverage: int) -> None:
        """設定交易對的槓桿倍數。

        參數:
            symbol: 交易對代號。
            leverage: 槓桿倍數（通常 1-125）。

        備註:
            若 API 呼叫失敗將被靜默處理。
        """
        lev = int(max(1, min(leverage, 125)))
        try:
            # Match test expectation: no recvWindow asserted
            self.client.futures_change_leverage(symbol=symbol, leverage=lev)
        except Exception:
            # TODO: 考慮記錄此例外
            pass

    @staticmethod
    def _floor_to_step(value: float, step: float) -> float:
        """將數值向下取整至最接近的步進。

        參數:
            value: 需要取整的數值。
            step: 步進大小。

        回傳:
            取整後的數值。
        """
        if step <= 0:
            return value
        return math.floor(value / step) * step


def build_trading_client(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    testnet: bool = False,
) -> ITradingClient:
    """建立交易客戶端的工廠函式。

    參數:
        api_key: 選填 API Key（若為 None 則從環境變數讀取）。
        api_secret: 選填 API Secret（若為 None 則從環境變數讀取）。
        testnet: 是否使用測試網端點。

    回傳:
        `ITradingClient` 的實作（目前為 `BinanceFuturesClient`）。
    """
    return BinanceFuturesClient(api_key=api_key, api_secret=api_secret, testnet=testnet)

