# Data Fetch Service Module

這是新的整合模組，舊檔案 `GetTradeData.py` / `GetTradeData_CoinGlass.py` 不需修改。

## 執行方式

- 手動啟動整合服務：
  - `python -m services.data_fetch.launcher`
- 只跑 Binance：
  - `python -m services.data_fetch.binance_service`
- 只跑 CoinGlass：
  - `python -m services.data_fetch.coinglass_service`

## 必要環境變數

- `COINGLASS_API_KEY`（CoinGlass 必填）

## 常用可調參數

- `DATA_FETCH_SERVICE_ENABLED`：`1`/`0`
- `DATA_FETCH_RUN_ON_STARTUP`：`1`/`0`
- `DATA_FETCH_MAX_CYCLES`：`0` 代表無限循環
- `BINANCE_FETCH_INTERVAL_SECONDS` 
- `COINGLASS_FETCH_INTERVAL_SECONDS`
- `BINANCE_FETCH_TRADING_PAIRS`（逗號分隔）
- `COINGLASS_FETCH_TRADING_PAIRS`（逗號分隔）

## Windows 服務（NSSM）

1. 下載並放置 `nssm.exe`（預設路徑 `C:\tools\nssm\nssm.exe`）
2. 一鍵管理（會自動跳 UAC 提權）：
   - 直接雙擊：`services/data_fetch/windows/manage_windows_service.bat`
   - 或 PowerShell：
     - `powershell -ExecutionPolicy Bypass -File services/data_fetch/windows/manage_windows_service.ps1`
3. 也可直接安裝（未提權時會自動重開管理員視窗）：
   - `powershell -ExecutionPolicy Bypass -File services/data_fetch/windows/install_windows_service.ps1`
   - 安裝腳本會優先使用 `.\.conda\python.exe`，找不到才退回系統 `python`
   - 安裝完成後預設會自動 `Start-Service` 並回報 launcher/binance/coinglass 行程數
4. 檢查服務：
   - `Get-Service TradeTrainDataFetchService`
5. 解除安裝（未提權時會自動重開管理員視窗）：
   - `powershell -ExecutionPolicy Bypass -File services/data_fetch/windows/uninstall_windows_service.ps1`

## Log 機制（每次啟動刷新）

- `logs/data_fetch/launcher.log`
- `logs/data_fetch/binance_service.log`
- `logs/data_fetch/coinglass_service.log`

每個行程啟動時都會以覆寫模式（`mode="w"`）建立 log，所以不會保留舊資料。

