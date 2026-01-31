# Observation 優化討論：讓 Agent 學會觀察市場趨勢

## 一、問題描述

目前現象：
- Agent **沒有學會觀察市場趨勢**，行為像隨機交易
- **沒有逐漸穩定方向**：持倉方向不隨趨勢收斂，容易來回翻倉

這表示 policy 沒有有效利用「趨勢」資訊來做「順勢／逆勢」決策，可能原因在 **obs 設計**、**reward 設計** 與 **架構** 三方面。

---

## 二、現況簡述

### 2.1 Observation 結構

| 輸入 | Shape | 內容概要 |
|------|--------|----------|
| `price_seq` | (288, F_5m) | 5m 序列：ret、range、volume、EMA/MACD、price_pos、chop、trend_flip 等 |
| `price_seq_1d` | (30, F_1d) | 1d 序列：OI、funding、多空比、ret_1d、ema_20_60_spread、fear_greed 等 |
| `account_state` | (27,) | 倉位、權益、浮盈、止損距離、entry_gap_atr、pos_side one-hot 等 |
| `time_state` | (7,) | 小時/星期/phase 的 sin/cos、是否週末 |
| `rhythm_state` | (2,) | atr_ratio、rv_ratio |
| `cost_state` | (27,) | 手續費/風險/止損/預測效果、行為偏差揭露 |

- **趨勢相關資訊**：幾乎全部塞在 **5m / 1d 序列** 裡（ret_1h、ema_12_48_spread、macd_atr、dir_persist_20、1d 的 ret_1d、ema_20_60_spread 等）。
- **向量分支**（account / time / rhythm / cost）**沒有**「當前市場趨勢方向」或「regime」的**顯式純量**。
- 因此「該做多還是做空」完全要靠 CNN 從長序列裡自己學，**沒有在 MLP 分支給一個明確的 trend 信號**。

### 2.2 Reward

- 主線是 **純 log-return**（`RewardCalculator`），沒有對「順勢」額外加分。
- **Conviction trend bonus** 已實作（強趨勢 + 大倉 + 同向才加分），但 **預設關閉**（`conviction_trend_bonus_weight=0`），所以目前沒有任何 reward 鼓勵「看對趨勢並持倉」。

### 2.3 架構

- 雙 CNN（5m + 1d）→ 融合 → 再與向量 concat → policy。
- 所有 5m 通道一視同仁，**沒有**把「趨勢類」通道特別標記或加權；若冗餘通道多、雜訊大，CNN 可能難以突出趨勢。

---

## 三、根因歸納

1. **趨勢沒有「直接可見」**  
   趨勢只藏在 5m/1d 的很多通道裡，沒有在向量 obs 裡提供「當前趨勢方向／強度」的純量，MLP 分支無法直接用到「市場往上/往下/震盪」。

2. **Reward 不鼓勵順勢**  
   純 log-return 對「來回小賺」和「順勢持倉」一視同仁，甚至高頻小賺可能更常發生，導致 agent 傾向亂動而非穩住方向。

3. **趨勢訊號可能被稀釋**  
   5m 通道多、部分冗餘（見 `CNN_channel_audit.md`），若再混入大量量能/波動，CNN 要從中學出「穩定方向」難度較高。

4. **無顯式「regime」**  
   沒有「趨勢市 / 震盪市」的標籤或強度，agent 難以學「震盪少做、趨勢市才放大倉位」。

---

## 四、Obs 優化建議

### 4.1 在向量分支加入「趨勢方向／強度」純量（優先）

**目的**：讓 MLP 分支直接看到「當前市場偏多/偏空/中性」，不必只靠 CNN 從序列推論。

**做法**（擇一或組合）：

- **方案 A：用既有 `trend_score` 放進 obs**  
  - `MarketData` 已有 `trend_score_arr = (ma_50 - ma_200) / (ma_200 + ε)`（長週期趨勢）。  
  - 在 `observer` 的 **account_state 或 cost_state** 多 1～2 維：  
    - 例如 `trend_direction`: 將 `trend_score` clip 後壓到 [-1, 1]（或 tanh）；  
    - 可選：`trend_strength`: abs(trend_score) 的 clip，表示趨勢強度。  
  - 需同步：`observation_space` 中對應的 Box shape 加 1～2。

- **方案 B：用 5m 即時趨勢代理**  
  - 從 `MarketData` 或 `FeatureTransformer` 在當前 step 的 5m 特徵裡，取「最後一根」的：  
    - `ema_12_48_spread`（或 close_over_ema_12）→ 當作短期趨勢方向；  
    - 或 `ret_1h` / `ret_4h` 的當前值。  
  - 正規化到 [-1, 1] 後寫入 **account_state** 或 **cost_state** 的一維。  
  - 這樣「該做多還是做空」在向量裡就有直接輸入。

**實作注意**：

- 若放在 `cost_state`，目前為 27 維，需改成 28/29 並在 `observer` 與 `sb3_cnn_policy` 的向量維度一併更新。
- 若放在 `account_state`，同理 27 → 28/29。
- 保持數值範圍可控（clip/tanh），避免極端值影響訓練。

### 4.2 5m 通道順序：趨勢類放前

**目的**：讓 CNN 靠前的 channel 更容易學到「方向性」特徵（若沒有做 channel attention）。

**做法**：  
在 `FeatureTransformer.BASE_5M_COLS` 中，把「明確趨勢相關」的通道集中放在前面，例如：

- 現有：ret_1_z, ret_15m_z, ret_1h_z, range_z, volume_log_z, ...
- 可調整為：ret_1_z, ret_1h_z, ret_4h_z, close_over_ema_12_z, ema_12_48_spread_z, macd_atr, trend_strength_atr, dir_persist_20, ... 再接 range/volume/波動等。

不改變通道總數與語意，只調整順序；若已有 checkpoint，需確認與舊版相容或重訓。

### 4.3 啟用 Conviction Trend Bonus（Reward 面）

**目的**：顯式獎勵「強趨勢 + 大倉 + 同向」，促使 agent 學「先看趨勢再下大方向」。

**做法**：  
在建立 env 或 train config 中，將 `conviction_trend_bonus_weight` 設為小正數（例如 0.1～0.3），並保留：

- `conviction_trend_min_strength`：例如 0.8（趨勢夠強才加分）  
- `conviction_min_abs_pos`：例如 0.15（倉位夠大才加分，避免小倉刷分）

**注意**：  
- `trend_score` 目前來自 `market_data.trend_score_arr`（ma_50 - ma_200），屬長週期；若希望更貼近 5m 決策，可考慮改為 5m 的 macd/ema_spread 等，並在 env 傳入 reward 的 `kwargs`。  
- 先小權重試跑，觀察是否減少「胡亂翻倉」、增加「持倉方向穩定度」。

### 4.4 維持 Obs 品質、減少冗餘

**目的**：避免無效或高度重複通道干擾學習。

**做法**：  
- 定期跑 `python -m Env.obs_quality` 與 `scripts/cnn_channel_redundancy_check.py`（見 `CNN_channel_audit.md`）。  
- 若發現某通道常數/近常數或與他通道 Spearman 極高，依審計建議刪除或合併，讓趨勢相關通道更突出。

### 4.5 可選：1d「regime」純量

**目的**：讓 agent 區分「趨勢市 vs 震盪市」，學「震盪少動、趨勢市才放大」。

**做法**：  
- 在 1d 特徵中取「當前對應日」的：  
  - `chop` 或類似震盪指標、或  
  - `ema_20_60_spread` 的絕對值（趨勢強度）  
- 正規化後寫入 **cost_state** 或 **account_state** 一維（例如 `regime_trend_strength`）。  
- 實作時需在 `MarketData` 或 observer 取得「當前 step 對應的 1d 列」並傳入 observer。

---

## 五、實作優先順序建議

| 優先級 | 項目 | 預期效果 | 改動範圍 |
|--------|------|----------|----------|
| 1 | 向量分支加入 trend_direction（及可選 trend_strength） | 讓 policy 直接看到「多/空/中性」，減少胡亂交易、易於穩定方向 | Observer + observation_space + sb3 向量維度 |
| 2 | 啟用 conviction_trend_bonus（小權重） | 獎勵順勢持倉，與 obs 的趨勢訊號配合 | Env/train config |
| 3 | 5m 通道順序：趨勢類前置 | 有助 CNN 優先學趨勢，可選 | FeatureTransformer |
| 4 | 跑 obs_quality + 冗餘檢查 | 避免死通道與冗餘稀釋趨勢 | 審計腳本 + 必要時刪通道 |
| 5 | 1d regime 純量 | 區分趨勢/震盪，進階 | Observer + MarketData/1d 對齊 |

建議先做 **1 + 2**，觀察「是否開始會看趨勢、持倉方向是否較穩定」；若有效再考慮 3～5。

---

## 六、小結

- **核心問題**：趨勢資訊只藏在 5m/1d 序列裡，向量分支沒有「市場方向」的顯式輸入，且 reward 未鼓勵順勢。  
- **最關鍵的 obs 優化**：在 **account_state 或 cost_state** 加入 **trend_direction（及可選 trend_strength）**，並 **啟用 conviction trend bonus**。  
- 在此基礎上再視需要調整通道順序、清理冗餘、加入 regime，可望讓 agent 逐漸學會觀察市場趨勢並穩定持倉方向。
