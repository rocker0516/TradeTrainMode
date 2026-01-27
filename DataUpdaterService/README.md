# DataUpdaterService

此資料夾提供 **自動更新 Data/ CSV** 的程式碼，並可安裝成 **Windows Service（pywin32）**。

## 更新策略（預設）
- 5m：每 5 分鐘增量更新 `Data/{SYMBOL}_futures_volume_5years_5min.csv`
- 1d：每日固定時間增量更新：
  - `Data/{SYMBOL}_futures_volume_coinglass_5years_1d.csv`
  - `Data/fear_greed_index_history_1d.csv`
  - `Data/altcoin_season_index_history_1d.csv`
  - `Data/bitcoin_sth_sopr_index_history_1d.csv`
  - `Data/bitcoin_lth_sopr_index_history_1d.csv`
  - `Data/bitcoin_macro_oscillator_index_history_1d.csv`

## 環境變數
- `COINGLASS_API_KEY`（API 相關、必填：更新 1d coinglass/macro）

> 依專案規則：**環境變數只用於 API 相關設定**；其餘設定請改用 config 檔案。

## Config 檔案（非 API 設定）
預設路徑：`DataUpdaterService/data_updater_config.json`

範例（已附在 repo，可直接修改）：

```json
{
  "data_dir": "Data",
  "log_dir": "logs/data_updater_service",
  "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT"],
  "exchange": "Binance",
  "update_5m_seconds": 300,
  "update_1d_time": "00:30",
  "initial_backfill_days_5m": 30,
  "initial_backfill_days_1d": 3650
}
```

說明：
- `data_dir`: Data 目錄（可相對於專案根目錄或使用絕對路徑）
- `log_dir`: log 目錄（可相對或絕對）
- `symbols`: 交易對清單（也可用逗號字串，但建議用 array）
- `exchange`: CoinGlass 的 exchange 參數（預設 `Binance`）
- `update_5m_seconds`: 5m 更新間隔（秒）
- `update_1d_time`: 1d 更新時間（HH:MM，local time）
- `initial_backfill_days_5m` / `initial_backfill_days_1d`: CSV 不存在或讀不到最後時間時的回補天數

## Windows Service（pywin32）
> 注意：只有在 Windows 安裝 `pywin32` 後可用。

```bash
python -m DataUpdaterService.service --install
python -m DataUpdaterService.service --start

python -m DataUpdaterService.service --stop
python -m DataUpdaterService.service --remove
```

### 雙擊 .bat 快速安裝/移除（Windows）
`DataUpdaterService/windows_service_scripts/` 內提供：
- `01_install_start_service.bat`：安裝並啟動（請右鍵「以系統管理員身分執行」）
- `02_stop_remove_service.bat`：停止並移除（同上）

## 先在本機手動跑一次（除錯用）
（之後會提供 `python -m DataUpdaterService.runner --once` 類似入口）


