# LiveTradingRunner

## 目的
用訓練好的模型 `models/sac_lag_BTCUSDT/best_model/best_model.zip` 做每 5 分鐘推論，並可選擇是否呼叫「交易 API」進行真實下單。

## 安全開關（重要）
- 預設 **不會呼叫交易 API / 不會下單**。
- 只有加上 `--enable_trade_api` 才會建立 `ApiTrading` client 並可能送出訂單。
- 即使開了 `--enable_trade_api`，若加 `--dry_run` 則 **不送單**（只做查詢/計算/輸出）。

## 執行方式（範例）
- 單次 tick（建議先用這個驗證）

```bash
python LiveTradingRunner/live_trading_loop.py --once --symbol BTCUSDT --model models/sac_lag_BTCUSDT/best_model/best_model.zip
```

- 持續 loop（每根新 5m K 線觸發一次）

```bash
python LiveTradingRunner/live_trading_loop.py --loop --symbol BTCUSDT --model models/sac_lag_BTCUSDT/best_model/best_model.zip
```

- 真實下單（mainnet，風險自負）

```bash
export BINANCE_TRADE_API_KEY="..."
export BINANCE_TRADE_API_SECRET="..."
python LiveTradingRunner/live_trading_loop.py --loop --enable_trade_api --symbol BTCUSDT --model models/sac_lag_BTCUSDT/best_model/best_model.zip
```

## 依賴資料
- 5m 市場資料：即時從 Binance 公開端點抓取（不需要金鑰）。
- 1d 資料：從本機 `Data/*_1d.csv` 讀取（用於 1d 特徵；建議你後續用排程每日更新 CSV）。


