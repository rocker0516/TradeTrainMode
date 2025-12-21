# Binance Futures Trading API

這是一個用於 Binance UM-Futures 交易的 Python API 模組，遵循 SOLID 原則設計，提供清晰的介面和良好的可擴展性。

## 功能特點

- ✅ **抽象介面設計**：使用 ABC 定義 `ITradingClient` 介面，方便擴展和測試
- ✅ **完整的交易功能**：帳戶查詢、持倉管理、下單操作
- ✅ **型別提示**：所有 public 函式都包含完整的 type hints
- ✅ **詳細文件**：每個類別和方法都有 docstring
- ✅ **環境變數支援**：不會硬編碼 API 憑證
- ✅ **測試網支援**：可切換至 Binance 測試網環境

## 架構設計

### 核心類別

```
ITradingClient (抽象基類)
    ├── get_equity_usdt()          # 取得 USDT 餘額
    ├── get_current_position_size() # 取得當前持倉
    ├── get_account_summary()       # 取得帳戶摘要
    ├── get_open_positions()        # 取得所有開倉
    ├── get_symbol_filters()        # 取得交易規則
    ├── place_delta_order()         # 下單調整持倉
    └── set_leverage()              # 設定槓桿

BinanceFuturesClient (實作類別)
    └── 實作所有 ITradingClient 定義的方法
```

### SOLID 原則實踐

1. **單一職責原則 (SRP)**：`BinanceFuturesClient` 只處理交易相關操作
2. **開放封閉原則 (OCP)**：透過繼承 `ITradingClient` 來擴充功能
3. **里氏替換原則 (LSP)**：`BinanceFuturesClient` 可完全替代 `ITradingClient`
4. **介面隔離原則 (ISP)**：介面專注於交易操作，不包含無關方法
5. **依賴反轉原則 (DIP)**：高層模組依賴 `ITradingClient` 抽象，而非具體實作

## 安裝

```bash
pip install python-binance
```

## 環境設定

設定以下環境變數（**請勿在程式碼中硬編碼**）：

```bash
# API 憑證
export BINANCE_TRADE_API_KEY="your_api_key"
export BINANCE_TRADE_API_SECRET="your_api_secret"

# 可選：使用測試網
export BINANCE_TESTNET=1
```

## 使用範例

### 1. 基本使用

```python
from ApiTrading import build_trading_client

# 建立客戶端（從環境變數讀取憑證）
client = build_trading_client(testnet=True)

# 取得帳戶摘要
summary = client.get_account_summary()
print(f"Wallet Balance: {summary['wallet_balance']} USDT")
print(f"Available Balance: {summary['available_balance']} USDT")
```

### 2. 查詢持倉

```python
# 取得所有開倉
positions = client.get_open_positions()
for pos in positions:
    print(f"{pos['symbol']}: {pos['position_amt']}")

# 取得特定交易對的持倉
position_size = client.get_current_position_size("BTCUSDT")
print(f"BTCUSDT Position: {position_size}")
```

### 3. 設定槓桿

```python
# 設定 BTCUSDT 的槓桿為 10x
client.set_leverage(symbol="BTCUSDT", leverage=10)
```

### 4. 下單調整持倉

```python
# 取得交易規則
filters = client.get_symbol_filters("BTCUSDT", default_min_notional=10.0)

# 模擬下單（dry_run=True）
result = client.place_delta_order(
    symbol="BTCUSDT",
    delta=0.001,  # 增加 0.001 BTC 持倉
    step_size=filters.step_size,
    min_notional=filters.min_notional,
    last_price=50000.0,
    dry_run=True,  # 模擬模式，不會真實下單
)

if result:
    print(f"Order placed: {result}")
else:
    print("Order skipped (too small)")
```

### 5. 實際下單

```python
# 實際下單（dry_run=False）
result = client.place_delta_order(
    symbol="BTCUSDT",
    delta=0.001,
    step_size=filters.step_size,
    min_notional=filters.min_notional,
    last_price=50000.0,
    dry_run=False,  # 實際下單
)
```

### 6. 使用介面進行依賴注入

```python
from ApiTrading import ITradingClient, build_trading_client

def trading_strategy(client: ITradingClient, symbol: str) -> None:
    """交易策略函式，依賴抽象介面而非具體實作。"""
    equity = client.get_equity_usdt()
    position = client.get_current_position_size(symbol)
    print(f"Equity: {equity}, Position: {position}")

# 可以注入任何實作 ITradingClient 的類別
client = build_trading_client()
trading_strategy(client, "BTCUSDT")
```

## API 參考

### ITradingClient

所有交易客戶端都必須實作的抽象介面。

#### 方法

- `get_equity_usdt() -> float`
  - 取得 USDT 錢包餘額

- `get_current_position_size(symbol: str) -> float`
  - 取得特定交易對的當前持倉大小
  - 返回值：>0 為多頭，<0 為空頭，0 為無持倉

- `get_account_summary() -> Dict[str, float]`
  - 取得帳戶摘要
  - 返回字典包含：wallet_balance, available_balance, unrealized_pnl, margin_balance

- `get_open_positions() -> List[Dict[str, Any]]`
  - 取得所有非零持倉
  - 返回包含 symbol, position_amt, entry_price, unrealized_pnl, leverage 的字典列表

- `get_symbol_filters(symbol: str, default_min_notional: float) -> SymbolFilters`
  - 取得交易對的規則限制
  - 返回 SymbolFilters dataclass (step_size, min_qty, min_notional)

- `place_delta_order(symbol, delta, step_size, min_notional, last_price, dry_run) -> Optional[Dict]`
  - 下市價單調整持倉
  - delta > 0：買入，delta < 0：賣出
  - dry_run=True：模擬模式，不會實際下單

- `set_leverage(symbol: str, leverage: int) -> None`
  - 設定交易對的槓桿倍數

### BinanceFuturesClient

`ITradingClient` 的 Binance Futures 實作。

#### 初始化

```python
client = BinanceFuturesClient(
    api_key=None,      # None 則從環境變數讀取
    api_secret=None,   # None 則從環境變數讀取
    testnet=False      # True 使用測試網
)
```

### SymbolFilters

交易對規則的 dataclass。

```python
@dataclass
class SymbolFilters:
    step_size: float      # 最小數量增量
    min_qty: float        # 最小下單數量
    min_notional: float   # 最小名義價值
```

## 測試

執行單元測試：

```bash
pytest test_realtrading.py -v
```

## 注意事項

⚠️ **安全性**
- 絕對不要在程式碼中硬編碼 API Key 和 Secret
- 使用環境變數或安全的配置管理系統
- 建議先在測試網進行測試

⚠️ **錯誤處理**
- 所有 API 呼叫都可能拋出異常
- 建議使用 try-except 捕捉特定例外
- 網路問題、API 限制、權限不足都可能導致錯誤

⚠️ **交易風險**
- 使用實盤 API 前請充分測試
- 建議設定合理的風控參數
- 交易有風險，請謹慎操作

## 與 RealTrading.py 的整合

`RealTrading.py` 已經整合了這個 Trading API 模組：

```python
from ApiTrading import ITradingClient, build_trading_client

# 在 trading_loop 中使用
trade_client = build_trading_client(testnet=True)
equity = trade_client.get_equity_usdt()
position = trade_client.get_current_position_size(symbol)
```

## 擴展性

如果需要支援其他交易所，只需：

1. 繼承 `ITradingClient`
2. 實作所有抽象方法
3. 更新 `build_trading_client` 工廠函式

範例：

```python
class OtherExchangeClient(ITradingClient):
    def get_equity_usdt(self) -> float:
        # 實作其他交易所的邏輯
        pass
    
    # 實作其他方法...
```

## 授權

請參考專案根目錄的授權文件。

## 貢獻

歡迎提交 Issue 和 Pull Request！

---

**更新日期**: 2025-10-10  
**版本**: 1.0.0

