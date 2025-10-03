## AI 訓練影響評估報告（TradingEnvironment / Reward / Execution）

### 1) 架構總覽與現況
- 環境 `TradingEnvironment` 已將兩大責任外移：
  - 獎勵：`Env/reward.py`（`RewardCalculator` + `RewardNormalizer` + `RewardContext`）
  - 交易：`Env/execution.py`（`TradingExecutor` + `TradeContext`）
- 好處：責任單一、可測性提升、可替換性提升（支援策略化實驗）。

### 2) 觀察空間與前處理（對收斂與泛化的影響）
- 現況：
  - 使用滑動視窗內的 min–max 正規化（每步重新計算）。
  - 帳戶相關特徵（倉位、資產、餘額等）用「重複鋪滿整個窗口」。
- 風險：
  - 非平穩（non-stationary）：隨視窗移動導致尺度漂移，演算法學不到穩定映射。
  - 洩漏（leakage）與分佈漂移使價值網路擬合困難，提升樣本需求與方差。
- 建議（優先級：高）：
  - 用對數收益或 z-score（以訓練集統計）替代每步 min–max。
  - 帳戶特徵僅在當步通道填入（不需鋪滿窗口），或只在最後一列提供。
  - 視窗長度與步長要與策略決策頻率匹配，避免過度冗餘訊息。

### 3) 動作空間與執行映射（對策略穩定性的影響）
- 現況：
  - 動作為 `[position_percent ∈ [-1,1], take_profit% ∈ [0, 50], stop_loss% ∈ [0, 20]]`。
  - `TradingExecutor` 目標倉位：`target_position = total_value * position_percent * leverage / price`。
  - 名目金額 notional = size * price；手續費按 notional 計算；保證金 = notional / leverage。
  - 止盈/止損依倉位方向更新；先判斷 TP/SL，再執行調整與強平檢查。
- 影響：
  - 後兩維動作尺度（%）較大，對連續控制（例如 SAC/tanh）不友善，易造成梯度不穩。
- 建議（優先級：中-高）：
  - 將 TP/SL 也壓到 [-1,1] 再於環境線性映射（外部策略輸出保持一致尺度）。
  - 合理收斂 TP/SL 範圍（例如 0–5% 或 0–10%）以減少極端行為與數值爆炸。

### 4) 獎勵設計（對 credit assignment 的影響）
- 現況：
  - 權重預設：`pnl 0.4 / risk_management 0.45 / 其他 0.15`。
  - `penalty` 非對稱：負值懲罰按比例裁剪，正值給小額獎勵且封頂。
  - 風險管理包含資產比、槓桿使用、TP/SL 設置、交易頻率等訊號。
- 影響：
  - 塑形較重，且部分訊號跨期或全域（如交易頻率），加大 credit assignment 難度。
  - 可能壓過 PnL 主訊號，使策略學習偏向「規則」而非可獲利行為。
- 建議（優先級：高）：
  - 以 PnL/回撤為主，`risk_management` 權重下調；跨期/全域訊號改為 `info` 監控，不進獎勵。
  - 確保獎勵分佈穩定：縮小多重裁剪與非線性堆疊，必要時做 reward scaling。

### 5) 終止條件與強平（對回報方差與穩定性的影響）
- 現況：
  - 強平：未實現虧損超過已用保證金的 80% 觸發；`min_balance` 也會終止。
  - `data_exhausted` 可能成為主要終止來源。
- 影響：
  - 初期策略容易被早期截斷，導致高方差回報與不穩定訓練曲線。
- 建議（優先級：中）：
  - 固定 episode 長度，`reset` 隨機起點以覆蓋不同區段。
  - 適度提高強平容忍或設置 warm-up，不讓探索過早終止。

### 6) 交易執行正確性與市場擬真（對學到的策略有效性的影響）
- 現況（已改善）：
  - 手續費按名目金額計算；每次開/平倉分別計提；保證金統一口徑；先判斷 TP/SL 再執行調整。
- 影響：
  - 較貼近真實合約交易成本曲線，回報形狀更可信，有利學到可遷移的行為。
- 後續建議（優先級：中）：
  - 加入滑點模型（觸發時以鄰近價加點 spread），避免過度樂觀。

### 7) 測試覆蓋（對可維護性與實驗節奏的影響）
- 現況：
  - 整合測試：`tests/test_trading_env_integration.py` 覆蓋 reset/觀察、開倉/TP/SL、槓桿與費用、終止條件。
  - 模組測試：建議保留 `reward`、`execution` 的單元測試（便於快速定位與回歸）。
- 推薦補強：
  - `RewardCalculator`：各 component 的邊界值、權重/scale 改動後的穩健性。
  - `TradingExecutor`：多次加減倉平均成本、反向開倉流程、極端手續費、極端槓桿與最小下單。

### 8) 對 RL（特別是 SAC/TD3/PPO）訓練的影響與建議配置
- 對 SAC：
  - 動作輸出：tanh → [-1,1]，建議環境映射 TP/SL；減少尺度不一致。
  - Reward Scale：若分佈過寬，做 reward scaling（例如 ×5 或 ×10）有助 critic 擬合。
  - 觀察正規化：z-score/robust-scaler；確保訓練/驗證統計一致。
  - 超參考值：`gamma=0.98~0.995`、`tau=0.005`、`batch=256~1024`、`lr=3e-4`、自動溫度（entropy tuning）。
  - Replay：warm-up steps（≥1e4）、優先回放可選（但需注意偏置）。
- 對 PPO：
  - 動作分佈更敏感，務必讓 TP/SL 映射平滑且範圍合理；clip range 設計需配合獎勵尺度。

### 9) 監控指標（訓練時至少追蹤）
- Episode 指標：最終資產、max drawdown、勝率、平均持倉時長、每日/每百步交易次數。
- Reward 組件拆分與貢獻：`pnl`、`risk_management`、`penalty` 等分量的均值/方差/極值。
- 市場成本：累計手續費、平均有效杠桿使用率、強平次數。
- 分佈穩定性：觀察均值/方差隨時間的漂移、reward 分佈的 IQR/長尾情況。

### 10) 建議落地步驟（優先順序）
1. 調整觀察正規化：改用 z-score 或 log-returns，固定訓練統計。
2. 對齊動作尺度：TP/SL 經由 [-1,1] 線性映射；合理收斂範圍。
3. 精簡獎勵：降低 `risk_management` 權重；頻率類訊號移至 `info`。
4. Episode 設計：固定長度 + 隨機起點；放寬早期強平或設 warm-up。
5. 增加滑點模型；使回報更貼近真實。
6. 恢復/擴充 `reward` 與 `execution` 的單元測試，確保快速回歸。

---

如需，我可以：
- 直接提交觀察/動作/獎勵的改版實作（保留參數化開關），
- 加入滑點與更細的交易成本模型，
- 提供預設的 SAC 訓練腳本與超參數模板，
- 產生隨訓練自動輸出的訓練儀表板（包括上述監控指標）。



