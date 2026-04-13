# Live 圖表視窗偏離 — 計畫（含：執行時清空舊資料）

## 根因（簡述）

- `kline_session_rows` 從 state 還原後，若與「現在 API 最新收盤」之間隔了很久，buffer **只 append 新 bar、不補中間**，時間軸上會出現 **大空洞**。
- `_apply_shared_xlim_from_ohlc` 若仍用 **整表最早～最晚 timestamp** 設 `xlim`，會把空洞也畫進視窗 → 中間全空、兩端有資料。

## 你補充的策略（優先納入）

**認為問題點可能是：執行時要把原本的資料清空。**

此作法與根因一致：避免把「上次執行殘留、已與市場不連續」的 OHLC 繼續當繪圖來源。

### 建議實作方向（擇一或併用）

1. **程序啟動時清空繪圖用 buffer（推薦與 CLI 並存）**  
   - 在 [`LiveTradingRunner/live_trading_loop.py`](../../LiveTradingRunner/live_trading_loop.py) `main()`：在 `_load_state` 之後（或之前，依你是否要保留其他欄位而定），將 `state.kline_session_rows = []`（或等價清空）。  
   - 下一輪 `update_kline_session_buffer` 會走「緩衝為空」分支，用當次 API 的 `closed_df` 尾端 **重新種滿** `window_size_5m`，時間連續。

2. **可選開關（避免每次都清）**  
   - 例如 `--keep-kline-buffer`：預設 **清空**；若使用者要接續同一視窗再開 `--keep-kline-buffer`。  
   - 或反向：`--reset-kline-buffer` 預設 false，需要時顯式清空。

3. **與「斷層偵測重種」並用（runner_core）**  
   - 在 [`LiveTradingRunner/runner_core.py`](../../LiveTradingRunner/runner_core.py) `update_kline_session_buffer`：若 `buf[-1]` 與本輪 `last_closed` 時間差 **超過門檻**（例如 > 1～2 根 5m 以上且明顯為「停機」），則 **清空並重種**（與你說的「清空」同精神，但更省手動 flag）。

## 顯示層（仍建議保留）

- [`LiveTradingRunner/live_render.py`](../../LiveTradingRunner/live_render.py)：x 軸改為 **錨在最後一根 K**、寬度約 **`min(列數, _max_visible) × 5m`**，避免未來再出現「左端殘渣 + 右端新資料」時又把整段空白撐滿視窗。

## 驗證

- 啟動前 state 內故意留舊 `kline_session_rows`，**預設清空** 後第一屏應為連續近期 K 線。  
- 若使用 `--keep-kline-buffer` 且人為製造斷層，應靠 **斷層重種** 或 **trailing xlim** 至少一項恢復可讀性。

## 待辦（實作時）

| ID | 內容 |
|----|------|
| clear-on-start | `live_trading_loop`：啟動時清空 `kline_session_rows`（+ 可選 CLI） |
| buffer-resync | `runner_core`：大時間斷層則清空 buffer 並重種 |
| xlim-trailing | `live_render`：trailing 時間視窗 xlim |
| tests | 測試清空／斷層／xlim 行為 |
