# 雙 CNN 輸入欄位審計：重複邏輯與無效欄位

本文件對照 `Env/feature_transformer.py` 與 `Train/sb3_cnn_policy.py`，檢查餵入 **5m CNN**（`price_seq`）與 **1d CNN**（`price_seq_1d`）的每個通道：公式來源、是否與其他通道重複、是否可視為無效/冗餘，並給出建議（保留／刪除／合併）。

**實施狀態**：下表「建議刪除／合併」已於 `FeatureTransformer` 實施（5m 刪除 9 個冗餘通道、1d 刪除 `close_over_ema_20_z`）；程式內註解標註刪除理由。

---

## 建議動作摘要（實測 Spearman 後）

| 分支 | 建議刪除／合併 | 理由（公式或實測 ρ≈1） |
|------|----------------|------------------------|
| 5m | **long_short_ratio_z** 或 **volume_ratio_z** 二擇一刪 | 實測 ρ=1.000，完全重複 |
| 5m | **ema_12_slope_z** 或 **close_over_ema_12_z** 二擇一刪 | 實測 ρ=1.000（斜率與位置等價） |
| 5m | **ema_48_slope_z** 或 **close_over_ema_48_z** 二擇一刪 | 實測 ρ=1.000 |
| 5m | **body_z** | 實測與 ret_1_z ρ=1.000（單根 body ≈ ret） |
| 5m | **vol_imbalance_z** 與 volume_ratio/long_short 三擇一 | 實測 ρ≈0.998 |
| 5m | **macd_signal_atr** | 與 macd_atr ρ=0.94，留 macd_atr + trend_strength_atr |
| 5m | **close_over_ema_48_z** | 與 ema_12_48_spread、bb_pos_48 高相關，可由他欄推得 |
| 5m | **abs_ret_1_z** | 與 ret_1_z 重複（只差符號） |
| 5m | **quote_volume_log_z** 或 **volume_log_z** 二擇一 | 量能重複 |
| 5m | **log_close_z** | 與 price_pos / close_over_ema 重疊 |
| 1d | **close_over_ema_20_z** | 實測與 ret_1d_z ρ=1.000，日報酬即價格相對均線變化 |

實施時需同步修改 `FeatureTransformer` 的 `BASE_5M_COLS` / `PRICE_SEQ_1D_*` 與 `build_*_features` 的輸出順序，並重跑 observation_space 與訓練。

---

## 一、5m CNN（price_seq）通道清單與邏輯

| 通道名 | 公式/來源 | 重複/冗餘說明 | 建議 |
|--------|-----------|----------------|------|
| **ret_1_z** | 單根 log return，rolling z | 與 ret_15m/1h/4h 為「不同 horizon 的同一變數」，高度相關 | 保留（最短 horizon，訊號最即時） |
| **ret_15m_z** | ret_1 的 3 根和，z | 與 ret_1_z、ret_1h_z 重疊（ret_15m ⊂ ret_1h 若 1h=12 根） | 可考慮刪除，或保留 1+1h+4h 三檔即可 |
| **ret_1h_z** | ret_1 的 12 根和，z | 同上，與 ret_15m/ret_4h 階層重疊 | 保留 |
| **range_z** | (H-L)/prev_close，z | 與 body_z 重疊：body = (C-O)，range 含 body | 保留（range 含振幅資訊） |
| **body_z** | (C-O)/prev_close，z | 與 range 相關但語意不同（實體 vs 全幅） | 保留 |
| **volume_log_z** | log(V/V_ma20)，z | 與 quote_volume_log_z 高度相關（量級） | 二擇一或保留其一即可 |
| **quote_volume_log_z** | log1p(quote_v)，z | 同上 | 建議保留 volume_log_z，刪除本欄或合併為「量能」單一通道 |
| **trades_z** | 筆數，z | 與 volume 相關但不同維度 | 保留 |
| **vol_imbalance_z** | (buy_v-sell_v)/(buy_v+sell_v)，z | 獨立 | 保留 |
| **amihud_z** | \|ret_1\|/(quote_v+ε)，z | 流動性；與 ret_1 相關 | 保留 |
| **volume_ratio_z** | 外部 volume_ratio，z | 若與 volume_log 重複可刪 | 視資料來源決定 |
| **long_short_ratio_z** | 多空比，z | 獨立 | 保留 |
| **atr_ratio_z** | ATR/close，z | 與 rv_ratio_z 同為波動率，高度相關 | 二擇一或保留（ATR 價位、RV 報酬） |
| **rv_ratio_z** | RV_20/RV_288，z | 同上 | 建議保留 rv_ratio，atr 可保留（止損/風險用） |
| **log_close_z** | log(close)，z | 與趨勢/EMA 類重疊（長期 level） | 可刪：已有 close_over_ema、price_pos |
| **close_over_ema_12_z** | log(c/ema12)，z | 與 ema_12_48_spread、close_over_ema_48 線性相關（spread = ema12/ema48 - 1） | 保留其一組即可：建議保留 close_over_ema_12 + ema_12_48_spread |
| **close_over_ema_48_z** | log(c/ema48)，z | 可由 close_over_ema_12 與 ema_12_48_spread 推得近似 | **建議刪除**，減少冗餘 |
| **ema_12_48_spread_z** | (ema12-ema48)/ema48，z | 趨勢強度；與上兩欄重複資訊 | 保留（單一通道表徵短期 vs 長期） |
| **ema_12_slope_z** | diff(log(ema12))，z | 與 close_over_ema_12 變化率相關 | 可保留（動量） |
| **ema_48_slope_z** | diff(log(ema48))，z | 與 ema_12_slope 相關 | 二擇一或保留 12 即可 |
| **price_pos_96** | 價格在 96 根 high-low 區間位置，[-1,1] | 與 price_pos_288 重複邏輯（不同窗長） | 保留 288 即可，或兩者保留（多尺度） |
| **price_pos_288** | 同上，288 根 | 同上 | 保留 |
| **bb_width_48_z** | BB 寬度，z | 與 range/波動類相關 | 保留 |
| **bb_pos_48** | (c-mid)/(2*sd)，[-2,2] | 與 RSI、close_over_ema 相關 | 保留 |
| **rsi_14** | RSI 正規化到 [-1,1] | 獨立震盪指標 | 保留 |
| **macd_atr** | (ema12-ema26)/ATR | 與 macd_signal_atr、trend_strength_atr 同源（MACD） | 三擇二或保留 macd_atr + trend_strength_atr |
| **macd_signal_atr** | EMA(macd,9)/ATR | MACD 平滑版 | **建議刪除**（macd_atr 已具代表性） |
| **trend_strength_atr** | \|MACD\|/ATR | MACD 強度無向 | 保留 |
| **dir_persist_20** | 近 20 根正報酬比例 → [-1,1] | 與 ret 類、ema spread 相關 | 保留（行為型態） |
| **abs_ret_1_z** | \|ret_1\|，z | 與 ret_1_z 重複（只差符號資訊） | **建議刪除**（ret_1_z 已含幅度） |
| **ret_4h_z** | ret_1 的 48 根和，z | 與 ret_1h、ret_15m 同系列 | 保留（最長 horizon） |
| **hl_range_20_z** | (H_roll20-L_roll20)/prev_c，z | 與 range_z 重複邏輯（不同窗長） | 可刪或保留（多尺度） |
| **chop_48** | Choppiness 指標，z | 與 trend_flip_rate_48 同為「regime」 | 保留 |
| **trend_flip_rate_48** | EMA12-EMA48 翻轉率 | 同上 | 保留 |
| **alts_ret_15m_mean_z** 等 6 條 | 跨市場摘要 | 彼此可能相關（mean/std/ratio） | 保留；若 alt 數多可審視 alts_* 是否過多 |
| **{sym}_ret_15m_z** 等 4 條/幣 | 各 alt 的 ret/rel_ret/vol/trend | 與 alts_* 摘要重疊（摘要即從此聚合） | 可考慮只保留「摘要 6 條」或只保留 per-symbol，二擇一減冗餘 |

---

## 二、1d CNN（price_seq_1d）通道清單與邏輯

| 通道名 | 公式/來源 | 重複/冗餘說明 | 建議 |
|--------|-----------|----------------|------|
| **oi_close_z** | 未平倉，z | 獨立 | 保留 |
| **funding_close_z** | 資金費率，z | 獨立 | 保留 |
| **ls_account_ratio_z** | 多空帳戶比，z | 與 ls_position_ratio 相關 | 保留 |
| **ls_position_ratio_z** | 多空持倉比，z | 同上 | 保留 |
| **liq_long_log_z** | 多頭清算 log，z | 與 liq_short 可合併為淨/比 | 保留或合併為單一「清算強度」 |
| **liq_short_log_z** | 空頭清算 log，z | 同上 | 同上 |
| **ob_imbalance_z** | 訂單簿失衡，z | 獨立 | 保留 |
| **ret_1d_z** | 日報酬，z | 與 range_1d_z 相關（高 ret 常高 range） | 保留 |
| **range_1d_z** | 日振幅，z | 同上 | 保留 |
| **close_over_ema_20_z** | c/ema20-1，z | 與 ema_20_60_spread 重複邏輯（價格 vs 均線 vs 均線差） | 保留其一即可；建議保留 close_over_ema_20 + ema_20_60_spread |
| **ema_20_60_spread_z** | (ema20-ema60)/ema60，z | 日線趨勢強度 | 保留 |
| **fear_greed_z** | 恐懼貪婪，z | 獨立 | 保留 |
| **altcoin_season_z** | 山寨季指數，z | 與 fear_greed 同為情緒/regime | 保留 |
| **bmo_z** | 比特幣宏觀振盪器，z | 獨立 | 保留 |
| **sopr_z** | LTH SOPR，z | 獨立 | 保留 |

1d 端重複較少；唯一可選優化：**close_over_ema_20** 與 **ema_20_60_spread** 留兩者或只留 spread（與 5m 的 ema_12_48_spread 對齊）。

---

## 三、跨 CNN 重複（5m vs 1d）

- 5m 與 1d **沒有同一欄位名**，但語意上：
  - 5m 的 **atr_ratio_z / rv_ratio_z**（波動）與 1d 的 **range_1d_z** 都是「波動/regime」。
  - 5m 的 **ret_* / close_over_ema_*** 與 1d 的 **ret_1d_z / close_over_ema_20_z** 都是「報酬/趨勢」。
- 設計上 5m 為「高頻型態」、1d 為「背景 regime」，**跨分支重複可接受**，不建議為去冗餘而刪 1d。

---

## 四、建議刪除或合併的欄位（摘要）

| 分支 | 通道 | 理由 |
|------|------|------|
| 5m | **close_over_ema_48_z** | 可由 close_over_ema_12 + ema_12_48_spread 推得 |
| 5m | **macd_signal_atr** | 與 macd_atr 同源，留 macd_atr + trend_strength_atr 即可 |
| 5m | **abs_ret_1_z** | 與 ret_1_z 重複（只差符號） |
| 5m | **quote_volume_log_z** 或 **volume_log_z** | 二擇一（量能重複） |
| 5m | **log_close_z** | 與 price_pos / close_over_ema 重疊 |
| 5m | **ema_48_slope_z** | 與 ema_12_slope 擇一即可 |
| 1d | （可選）**close_over_ema_20_z** 或只留 **ema_20_60_spread_z** | 減少一條 EMA 相關冗餘 |

實施時需同步修改 `FeatureTransformer.BASE_5M_COLS` / `PRICE_SEQ_1D_*` 與相關 `build_*m_features` 的輸出順序，並重新訓練或至少重跑 observation_space 與 checkpoint 維度。

---

## 五、無效欄位（常數或近常數）

由 `Env/obs_quality.py` 審計結果決定。若某通道在抽樣後 **std < 1e-4** 或 **zero_frac > 0.95**，可視為無效：

- 先前 1d 的 **ret_1d_z**、**fear_greed_z** 曾因資料對齊與缺漏而幾乎全 0；經 **1d timestamp 正規化 + 5m 推導日 OHLC 補齊** 已改善。
- 若未來新增欄位，建議用 `python -m Env.obs_quality` 與 `scripts/cnn_channel_redundancy_check.py` 定期檢查高相關與低變異通道。

---

## 六、實測高相關配對（Spearman ≥ 0.82，抽樣約 1500 步）

以下為 `scripts/cnn_channel_redundancy_check.py` 在真實資料上的結果，供對照公式分析做刪減決策。

### 5m CNN（|ρ| ≥ 0.82）

| 通道 A | 通道 B | ρ | 建議 |
|--------|--------|---|------|
| volume_ratio_z | long_short_ratio_z | 1.000 | **刪其一**（完全重複） |
| close_over_ema_12_z | ema_12_slope_z | 1.000 | **刪其一**（斜率與位置高度等價） |
| close_over_ema_48_z | ema_48_slope_z | 1.000 | **刪其一**（同上） |
| ret_1_z | body_z | 1.000 | **刪 body_z**（單根 body ≈ ret_1） |
| vol_imbalance_z | volume_ratio_z / long_short_ratio_z | 0.998 | **三擇一**（量能/多空方向重複） |
| quote_volume_log_z | trades_z | 0.968 | 二擇一或保留 |
| macd_atr | macd_signal_atr | 0.941 | **刪 macd_signal_atr** |
| bb_pos_48 | close_over_ema_48_z / ema_48_slope_z / macd_atr | 0.91~0.92 | 保留 bb_pos_48，刪 close_over_ema_48_z + ema_48_slope_z |
| ema_12_48_spread_z | macd_atr / macd_signal_atr / ret_4h_z | 0.89~0.91 | 保留 spread，刪 macd_signal、可審視 ret_4h |
| alts_ret_15m_mean_z | 各 alt 的 ret_15m_z | 0.89~0.93 | **摘要與 per-symbol 重疊**：可只保留摘要 6 條或只保留 per-symbol |
| atr_ratio_z | hl_range_20_z | 0.875 | 二擇一（波動類） |

### 1d CNN（|ρ| ≥ 0.82）

| 通道 A | 通道 B | ρ | 建議 |
|--------|--------|---|------|
| ret_1d_z | close_over_ema_20_z | 1.000 | **刪 close_over_ema_20_z**（日報酬與價格相對均線完全同步） |
| liq_long_log_z | sopr_z / ret_1d_z / close_over_ema_20_z / range_1d_z | 0.88~0.96 | 保留 liq_long + liq_short，sopr 獨立；ret_1d 與 close_over_ema_20 刪其一 |
| range_1d_z | fear_greed_z / sopr_z | 0.90~0.90 | 保留（語意不同） |
| oi_close_z | bmo_z / ema_20_60_spread_z | 0.83~0.91 | 保留（OI 與 macro 可並存） |

---

## 七、如何跑相關性檢查（可選）

依專案依賴是否含 `scipy`，可執行：

```bash
python scripts/cnn_channel_redundancy_check.py --steps 2000 --corr_threshold 0.85
```

會輸出 5m / 1d 各通道兩兩 Spearman 相關 ≥ 0.85 的配對與低變異通道，供對照本報告做最終刪減決策。
