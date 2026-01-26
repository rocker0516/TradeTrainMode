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
- `COINGLASS_API_KEY`（必填：更新 1d coinglass/macro）
- `DATA_UPDATER_SYMBOLS`（選填，預設：BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,1000PEPEUSDT）
- `DATA_UPDATER_EXCHANGE`（選填，預設：Binance）
- `DATA_DIR`（選填，預設：專案根目錄的 `Data/`）
- `UPDATE_5M_SECONDS`（選填，預設：300）
- `UPDATE_1D_TIME`（選填，預設：00:30）
- `INITIAL_BACKFILL_DAYS_5M`（選填，預設：30；當 CSV 不存在或無法讀到最後時間時使用）
- `INITIAL_BACKFILL_DAYS_1D`（選填，預設：3650；同上）
- `DATA_UPDATER_LOG_DIR`（選填，預設：`logs/data_updater_service/`）

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


