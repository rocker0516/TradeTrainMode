# 當前 Observation (obs) 規格說明

本文說明交易環境的 **觀察空間 (Observation)** 結構，每個欄位的用途與對策略/訓練的影響。obs 由 `TradingObserver` 產生，分為三大塊：**市場 (market)**、**帳戶 (account)**、**情境 (context)**。

---

## 一、整體結構

| 區塊 | Key | Shape | 說明 |
|------|-----|--------|------|
| 市場 | `price_seq_target` | `(window_size, F_target_5m)` | 5m 目標交易對特徵序列 |
| 市場 | `price_seq_others` | `(window_size, F_others_5m)` | 5m 其他幣種/跨市場特徵序列 |
| 市場 | `price_seq_1d_target` | `(window_size_1d, F_target_1d)` | 1d 目標交易對特徵序列 |
| 市場 | `price_seq_1d_others` | `(window_size_1d, F_others_1d)` | 1d 其他幣種 + 總經特徵序列 |
| 帳戶 | `account_state` | `(21,)` | 帳戶與風險相關純量 |
| 情境 | `context_state` | `(8,)` | 上一動執行結果與預測風險 |

- **window_size**：預設 432（約 1.5 天 5m K 線）。
- **window_size_1d**：預設 30（30 根日線）。
- **dtype**：由 `Config.OBS_DTYPE` 決定，通常為 `float16` 或 `float32`。
- 所有陣列會做 `nan_to_num` 並保證 `c_contiguous`，以利訓練穩定性與 GPU 傳輸。

---

## 二、市場觀察 (Market Obs)

### 2.1 `price_seq_target` — 5m 目標交易對特徵

**Shape**: `(window_size, price_seq_target_features_dim)`  
**用途**: 提供「當前交易對」在 5 分鐘週期上的價格、量能、技術指標、結構與風險特徵，供策略判斷進出場與倉位。

特徵由 `FeatureTransformer.OPTIMIZED_TARGET_5M_COLS` 定義，大致分類如下：

| 分類 | 特徵名 | 用途與影響 |
|------|--------|------------|
| **基礎回報與價格** | `ret_15m_z` | 15 分鐘（3 根）累積報酬 z-score。影響：短線趨勢強度。（已移除 ret_1_z 單 bar 高噪音） |
| | `range_z` | (high-low)/prev_close 的 z-score。影響：當根 K 波動幅度。 |
| | `body_z` | (close-open)/prev_close 的 z-score。影響：實體方向與大小。 |
| | `log_close_z` | log(close) 的 z-score。影響：價格水位相對近期常態。 |
| **成交量與流動性** | `volume_log_z` | 成交量相對 20 期均量的 log 的 z-score。影響：放量/縮量。 |
| | `quote_volume_log_z` | 成交額的 log z-score。影響：真實參與度。 |
| | `trades_z` | 成交筆數 z-score。影響：活躍度。 |
| | `vol_imbalance_z` | (buy_vol - sell_vol)/(buy_vol + sell_vol) 的 z-score。影響：買賣力道。 |
| | `amihud_z` | \|ret\|/(quote_vol) 的 z-score（Amihud 非流動性）。影響：流動性風險。 |
| | `volume_ratio_z` | 量比類指標 z-score。影響：相對常態的放量程度。 |
| **技術指標** | `close_over_ema_12_z` | log(close/EMA12) 的 z-score。影響：短均線多空。 |
| | `close_over_ema_48_z` | log(close/EMA48) 的 z-score。影響：中均線多空。 |
| | `ema_12_48_spread_z` | (EMA12-EMA48)/EMA48 的 z-score。影響：均線金叉/死叉語意。 |
| | `ema_12_slope_z` | EMA12 對數斜率 z-score。影響：短趨勢加速度。 |
| | `ema_48_slope_z` | EMA48 對數斜率 z-score。影響：中趨勢加速度。 |
| | `rsi_14` | RSI(14)。影響：超買超賣與動量延續。 |
| | `macd_atr` | MACD 以 ATR 正規化。影響：趨勢與動能強度。 |
| **市場結構** | `price_pos_96` | 96 根內 (close-low)/(high-low) 線性映射到 [-1,1]。影響：區間內相對高低。 |
| | `bb_width_48_z` | 48 期布林寬度 z-score。影響：波動擴張/收斂。 |
| | `bb_pos_48` | 48 期布林帶內位置。影響：超買超賣與回歸。 |
| | `trend_strength_atr` | 趨勢強度（以 ATR 正規化）。影響：趨勢可信度。 |
| | `chop_48` | 48 期震盪指標。影響：區間盤整 vs 趨勢。 |
| **波動率與風險** | `atr_ratio_z` | ATR/close 的 z-score。影響：波動環境，與止損/倉位設計相關。 |
| | `rv_ratio_z` | 短期/長期 realized vol 比。影響：波動 regime 切換。 |
| **關鍵價位** | `dist_to_support_96_atr` | 當前價到 96 期低點的距離（ATR 為單位）。影響：支撐距離、風控。 |
| | `dist_to_resistance_96_atr` | 當前價到 96 期高點的距離（ATR 為單位）。影響：阻力距離。 |
| | `price_jump_z` | 價格跳躍（ret 相對近期 std）的 z-score。影響：異常波動偵測。 |
| **訂單簿（若有資料）** | `ob_depth_imbalance_z` | 買賣深度不平衡 z-score。影響：短期壓力方向。 |
| | `ob_slope_bid_z` | 買盤深度斜率 z-score。影響：買盤韌性。 |
| | `ob_slope_ask_z` | 賣盤深度斜率 z-score。影響：賣盤韌性。 |
| **跨市場摘要** | `alts_ret_15m_mean_z` | 其他幣 15m 報酬均值的 z-score。影響：山寨整體動能。 |
| | `alts_ret_15m_std_z` | 其他幣 15m 報酬標準差 z-score。影響：山寨波動/風險偏好。 |
| | `alts_rel_ret_15m_abs_mean_z` | 其他幣相對目標幣報酬的絕對均值 z。影響：輪動/背離。 |
| | `alts_trend_up_ratio` | 其他幣中趨勢向上的比例。影響：大盤多空氛圍。 |
| | `alts_volume_log_mean_z` | 其他幣成交量 log 均值 z。影響：整體市場熱度。 |
| | `alts_trend_spread_std_z` | 其他幣趨勢 spread 的 std z。影響：分化程度。 |

**影響總結**: 若無 `feature_symbols`，others 維度可能為 0，target 仍提供完整 5m 資訊；CNN/注意力模型可從此序列學到時序型態與進出場時機。

---

### 2.2 `price_seq_others` — 5m 其他幣種特徵

**Shape**: `(window_size, price_seq_others_features_dim)`  
**用途**: 每個「非目標」幣種一組固定通道（見 `OTHERS_5M_COLS_PER_SYMBOL`），用於跨市場、相關性與輪動資訊。

每幣約 10 通道：`ret_1_z`, `ret_15m_z`, `ret_1h_z`, `volume_log_z`, `quote_volume_log_z`, `close_over_ema_12_z`, `rsi_14`, `atr_ratio_z`, `trend_strength_atr`, `vol_imbalance_z`。  
若 `feature_symbols` 為 `None`，此矩陣可能為空（0 個 others 特徵），維度由 `MarketData` 依資料決定。

**影響**: 有設定其他幣時，可學習「大盤/山寨」聯動與資金輪動，有助於過濾假突破或順勢加碼。

---

### 2.3 `price_seq_1d_target` — 1d 目標交易對特徵

**Shape**: `(window_size_1d, price_seq_1d_target_features_dim)`  
**用途**: 日線級別的持倉量、資金費率、多空結構、價格 regime 與時間特徵，提供較長週期情境。

對齊策略：每個 5m 時間點對應「上一根已收盤」日線，避免 look-ahead。

| 分類 | 特徵名 | 用途與影響 |
|------|--------|------------|
| **Coinglass** | `oi_close_z` | 持倉量 z-score。影響：籌碼與趨勢延續性。 |
| | `funding_close_z` | 資金費率 z-score。影響：多空成本與極端情緒。 |
| | `ls_account_ratio_z` | 多空帳戶比 z。影響：散戶多空結構。 |
| | `ls_position_ratio_z` | 多空持倉比 z。影響：大戶方向。 |
| | `liq_long_log_z` | 多單強平量 log z。影響：多頭槓桿風險。 |
| | `liq_short_log_z` | 空單強平量 log z。影響：空頭槓桿風險。 |
| | `ob_imbalance_z` | 訂單簿失衡 z。影響：日線級壓力。 |
| **Price regime** | `ret_1d_z` | 日報酬 z-score。影響：日線動量。 |
| | `range_1d_z` | 日線 range z。影響：日波動。 |
| | `close_over_ema_20_z` | 收盤相對 EMA20 的 z。影響：日線趨勢。 |
| | `ema_20_60_spread_z` | 日線 EMA20–60 利差 z。影響：日線均線結構。 |
| **關鍵價位** | `dist_to_support_1d_atr` | 距日線支撐的 ATR 距離。影響：日線支撐/風控。 |
| | `dist_to_resistance_1d_atr` | 距日線阻力的 ATR 距離。影響：日線目標/壓力。 |
| **VWAP** | `vwap_distance_1d_atr` | 與日線 VWAP 的距離（ATR 單位）。影響：成本區與均值回歸。 |
| **時間** | `hour_sin`, `hour_cos` | 小時的週期編碼。影響：日內模式（若 1d 為當日則較弱）。 |
| | `day_of_week_sin`, `day_of_week_cos` | 星期編碼。影響：週內效應。 |

**影響總結**: 提供「日線 regime」與籌碼資訊，避免 5m 過度反應噪音，並可與 5m 特徵一起做多尺度決策。

---

### 2.4 `price_seq_1d_others` — 1d 其他幣種 + 總經

**Shape**: `(window_size_1d, price_seq_1d_others_features_dim)`  
**用途**: 其他幣種的 1d 特徵（每幣約 6 通道：oi、funding、ls_account、ls_position、ob、ret_1d、close_over_ema_20）＋ 總經 4 通道。

總經通道（`PRICE_SEQ_1D_MACRO_COLS`）：
- `fear_greed_z`：恐懼貪婪類指數 z。
- `altcoin_season_z`：山寨季指標 z。
- `bmo_z`：比特幣主導/市值相關 z。
- `sopr_z`：SOPR 等鏈上/行為 z。

**影響**: 提供跨幣與宏觀情境，有助於過濾「僅目標幣」的局部噪音、辨識牛熊階段。

---

## 三、帳戶觀察 (account_state) — 22 維

**Shape**: `(22,)`  
**用途**: 讓 agent 知道當前持倉、權益、風險距離、手續費等，以控制槓桿、止損與交易頻率。已移除常數、與序列重複、易誘發不良行為或回合依賴的欄位。

| 索引 | 名稱 | 範圍/計算 | 用途與影響 |
|------|------|-----------|------------|
| 0 | **position_side** | -1 / 0 / 1 | 空 / 平 / 多。影響：對稱化策略與 action 解讀。 |
| 1 | **position_size_norm** | [0, 1] | 持倉名目/（權益×槓桿）。影響：倉位與槓桿使用，避免過度曝險。 |
| 2 | **equity_ratio** | [0, 5] | 權益/初始資金。影響：總績效與存活，>1 為獲利。 |
| 3 | **realized_pnl_ratio** | [-1, 5] | 已實現損益/初始資金。影響：平倉累計盈虧。 |
| 4 | **unrealized_pnl_atr** | [-10, 10] | 未實現損益/(初始×atr_ratio)。影響：持倉浮盈虧以「波動單位」表示。 |
| 5 | **drawdown** | [0, 1] | (max_equity - equity)/max_equity。影響：回撤控制與保守行為。 |
| 6 | **liq_distance_atr** | [0, 10] | 現價到強平價距離（ATR 倍數），無倉時=10。影響：避免逼近強平。 |
| 7 | **stop_loss_distance_atr** | [-10, 10] | 現價到止損價距離（ATR），有符號。影響：止損緩衝與 stop-buffer 成本對齊。 |
| 8 | **margin_usage_ratio** | [0, 1.1] | 維持保證金/權益。影響：保證金壓力，接近 1 危險。 |
| 9 | **cooldown_remaining_norm** | [0, 1] | 止損冷卻剩餘步數/STOP_LOSS_COOLDOWN_STEPS。影響：止損後是否可立刻再進場。 |
| 10 | **fee_rate** | [0, 0.05] | 手續費率（小數）。影響：交易成本意識。 |
| 11 | **rolling_fee_ratio** | [0, 1] | 滾動手續費/權益。影響：近期成本負擔。 |
| 12 | **trade_count_log** | [0, ∞) | log1p(進場次數)。影響：交易頻率與 over-trading 懲罰。 |
| 13 | **stop_loss_count_log** | [0, ∞) | log1p(本回合止損次數)。影響：止損頻率與風控品質。 |
| 14 | **holding_time_log** | [0, ∞) | log1p(持倉步數)。影響：持倉時間與成本/報酬取捨。 |
| 15 | **buffer_to_min_balance_ratio** | [0, 1] | (equity - min_balance)/initial_balance，0=觸及底線。影響：避免 balance_insufficient 終止。 |
| 16 | **steps_since_trade_norm** | [0, 1] | log1p(距上次成交步數)/log1p(episode_max_steps)。影響：交易頻率與 flat cost 學習。 |
| 17 | **trade_freq_remaining_ratio** | [0, 1] | 交易頻率硬限制剩餘額度。影響：與 cost_trade_freq 對齊。 |
| 18 | **trade_freq_blocked_last** | 0/1 | 上一步是否因額度滿被擋。影響：區分「未下單」與「被擋」。 |
| 19 | **entry_price_ratio** | [0.5, 1.5] | 進場價/當前價，無倉=1。影響：持倉成本與盈虧。 |
| 20 | **stop_loss_price_ratio** | [0.5, 1.5] | 止損價/當前價，無止損=1。影響：止損距離感。 |
| 21 | **recent_flat_ratio** | [0, 1] | 最近 N 步空倉比例（與 cost_flat 同口徑）。影響：flat cost 學習。 |

**已移除欄位**: stop_loss_rate（與 trade_count_log/stop_loss_count_log 冗餘、回合內高方差）、fee_budget_remaining（常數 1.0）、realized_pnl_per_close_norm（易誘發只平贏單）、episode_progress（回合依賴）、trend_strength_last / chop_last（與 price_seq_target 最後一筆重複）。

**影響總結**: 此 22 維直接綁定獎勵/成本設計（強平、止損、手續費、drawdown、flat），agent 需依此在「進攻」與「風控」之間取得平衡。平均統計中的「止損率」(stop_loss_rate_pct) 仍由 Callback 從 episode info 計算並顯示，與本觀察向量無關。

---

## 四、情境觀察 (context_state) — 8 維

**Shape**: `(8,)`  
**用途**: 回饋「上一步動作的執行結果」，讓 agent 知道指令是否被改動、實際倉位與預測風險，有助於 credit assignment 與探索。

| 索引 | 名稱 | 範圍/來源 | 用途與影響 |
|------|------|-----------|------------|
| 0 | **action_overridden** | 0/1 | 上一步動作是否被風控/限制覆寫。影響：區分「沒做」與「被系統改掉」。 |
| 1 | **last_action_raw** | [-1, 1] | 上一步原始 action（未 clip）。影響：意圖與實際差異。 |
| 2 | **last_action_used** | [-1, 1] | 上一步實際採用的 action（含 clip/override）。影響：真實執行動作。 |
| 3 | **last_target_pos_pct** | [-1, 1] | 上一步目標持倉比例。影響：與 final 對比可知執行落差。 |
| 4 | **last_final_pos_pct** | [-1, 1] | 上一步執行後實際持倉比例。影響：當前倉位的前一步來源。 |
| 5 | **trade_executed_flag** | 0/1 | 上一步是否發生成交。影響：區分「掛單/無成交」與「有成交」，利於手續費與滑價學習。 |
| 6 | **predicted_liq_distance_after** | [0, 5] | 若執行上一步後預測的強平距離（ATR 或類似尺度）。影響：避免連續動作導致快速逼近強平。 |
| 7 | **available_balance_after_norm** | [0, 2] | 執行上一步後預測可用餘額/初始資金。影響：餘額不足與槓桿使用。 |

**影響總結**: 減少「做了什麼」與「環境實際發生什麼」之間的資訊不對稱，有助於學習何時被限制、何時真的成交，以及風險預測是否合理。

---

## 五、風險訊號 (Risk Signals) — 僅內部使用

`TradingObserver.compute_risk_signals()` 會計算強平價、強平距離、保證金率、止損距離（含 ATR 正規化）、near_liq/near_margin/near_stop 等，這些**不直接進 obs**，而是：
- 供 **account_state** 使用（如 liq_distance_atr、stop_loss_distance_atr、margin_usage_ratio 等）；
- 供 reward/cost 計算使用（例如 stop-buffer、強平懲罰）。

因此 obs 中的風險相關資訊已經是以「正規化純量」形式呈現在 `account_state` 與 `context_state` 中。

---

## 六、與訓練的關係（簡要）

- **CNN/注意力**：主要吃 `price_seq_target`、`price_seq_1d_target`（及可選的 others），學習時序與多尺度型態。
- **MLP/共享層**：通常把 `account_state`、`context_state` 接在特徵後，讓策略同時依賴「市場」與「狀態」。
- **獎勵/成本**：強平、止損、手續費、drawdown、flat、stop-buffer 等設計，都對應到上述欄位；obs 設計與 reward/cost 一致時，Lagrangian 與 SAC 較易收斂並滿足約束。

若你之後要改 obs（例如增減欄位、改 1d 對齊、或分離 target/others 維度），建議同步檢查：
1. `MarketData` 與 `FeatureTransformer` 的維度與欄位列表；
2. `TradingObserver._build_*_space()` 與 `_get_*_obs()` 的 shape 與數值範圍；
3. `sb3_cnn_policy`（或你使用的 policy）的 input 處理是否仍符合新 shape。
