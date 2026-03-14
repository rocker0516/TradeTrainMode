# Phase B lambda_risk (cost_risk) 調參分析

固定：`lambda_buffer=0.001`、`reward_scale=10`、`min_balance=0.6*initial`。變動：`--lambda-risk` = 1, 2, 5, 10。

---

## 一、實驗與資料來源

| Run | lambda_risk | 訓練步數 | 評估 JSON |
|-----|-------------|----------|-----------|
| lr1 | 1.0 | 6M | phase_b_..._lr1_lb001_min06_eval.json |
| lr2 | 2.0 | 6M | phase_b_..._lr2_lb001_min06_eval.json |
| lr5 | 5.0 | 6M | phase_b_..._lr5_lb001_min06_eval.json |
| lr10 | 10.0 | ~4.75M | 無 |

---

## 二、評估結果摘要（100 episodes，holdout，deterministic）

| 指標 | lr1 | lr2 | lr5 |
|------|-----|-----|-----|
| **log_return_mean** | -0.152 | -0.520 | -0.523 |
| **final_balance_mean** | 1017.1 | 594.4 | 592.5 |
| **final_balance 範圍** | 560 ~ 2424 | 574 ~ 600 | 558 ~ 600 |
| **cost_risk_sum_mean** | 1.11 | 1.63 | 1.66 |
| **balance_insufficient 次數** | 65 | 100 | 100 |
| **max_steps_reached 次數** | 35 | 0 | 0 |
| **0 死亡回合數** | 35 | 0 | 0 |

- **lr1**：35 回合撐到 max_steps（0 死亡）、65 回合 balance_insufficient；平均 balance 1017，有正報酬潛力。
- **lr2 / lr5**：100 回合全部 balance_insufficient，平均 balance ~594（貼 min_balance 600），評估面「全死」。

---

## 三、訓練曲線尾段（最後約 1.5M 步）

### lr1（lambda_risk=1）
- **log_return_mean**：常為正（0.2 ~ 2.0），偶爾負。
- **final_balance_mean**：2k ~ 22k 常見，最高約 22k。
- **cost_risk_mean**：0 ~ 1.5，常見 0.3~0.9，有明顯「0 死亡」窗格。
- **cost_risk_dense_mean**：約 450 ~ 1777。
- 解讀：訓練後期能學到既賺又避險，死亡懲罰適中，行為多樣、有存活窗格。

### lr2（lambda_risk=2）
- **log_return_mean**：0.1 ~ 2.14，多數正。
- **final_balance_mean**：1.6k ~ 15k，尾段有 0 死亡窗格（cost_risk_mean 0、0.19 等）。
- **cost_risk_mean**：常見 0.29~0.99，多次出現 0。
- **cost_risk_dense_mean**：約 225 ~ 1540。
- 解讀：訓練曲線看起來不錯，但評估 100% 死亡 → 可能過擬合訓練分佈，或 holdout 更難。

### lr5（lambda_risk=5）
- **log_return_mean**：-0.53 ~ 0.94，方差大，負值不少。
- **final_balance_mean**：832 ~ 5670，整體較 lr1/lr2 低。
- **cost_risk_mean**：0.33 ~ 1.6，偶爾 0，多數 >0.5。
- **cost_risk_dense_mean**：約 463 ~ 2271。
- 解讀：死亡懲罰變大，策略更保守，訓練面「賺得少、仍常死」，評估也全死。

### lr10（lambda_risk=10）
- **log_return_mean**：全程約 -0.29 ~ -0.52，幾乎無正報酬。
- **final_balance_mean**：596 ~ 875，多數貼 600~650。
- **cost_risk_mean**：1.1 ~ 1.67，幾乎每窗格都死。
- **cost_risk_dense_mean**：1065 ~ 2304，長期貼線。
- 解讀：死亡懲罰過強，reward 被 cost_risk 主導，策略只學到「極度保守」卻仍常觸線，未學到「邊賺邊遠離線」。

---

## 四、綜合分析

### 1. lambda_risk 與行為
- **過小（lr1）**：訓練與評估都有存活與正報酬，但評估仍有 65% 死亡；若目標是「評估 0 死亡」，需再微調或加長訓練。
- **適中偏小（lr2）**：訓練曲線佳、有 0 死亡窗格，但評估全死 → 泛化不足或 holdout 難度較高。
- **偏大（lr5）**：訓練就更保守、報酬與存活都不如 lr1/lr2，評估全死。
- **過大（lr10）**：策略幾乎不學賺錢，只學到貼線＋常死，cost_risk_dense 高、log_return 負。

### 2. 訓練 vs 評估
- lr2 的「訓練好看、評估全死」顯示：**在訓練資料上能 0 死亡不代表在 holdout 上也能**；可能需更多步數、不同 seed、或搭配 lb/資料切分再驗證。
- lr1 是唯一在評估中有 35 回合 0 死亡、且平均 balance >1000 的設定，可當作**目前最佳候選**。

### 3. cost_risk 與 cost_risk_dense
- 四組的 **cost_risk_dense_mean** 在訓練尾段多落在數百～兩千，代表策略常有一段時間貼近 min_balance。
- lr10 的 cost_risk_dense 最高且 log_return 最差，印證「只壓死亡懲罰、不給足夠報酬信號」會導致貼線且無法學到獲利。

---

## 五、結論與建議

| 結論 | 說明 |
|------|------|
| **lr1 在評估上最佳** | 35 回合 0 死亡、mean balance 1017；若目標是「盡量少死且能賺」，優先沿用 lr1 並微調。 |
| **lr2 訓練好、評估差** | 建議同設定加跑不同 seed 或延長訓練，看評估是否穩定出現存活；或略調高 lambda_risk（如 1.2~1.5）試折衷。 |
| **lr5 過於保守** | 訓練與評估都無優勢，不建議再加大。 |
| **lr10 懲罰過重** | 策略學不到正報酬、貼線嚴重，不建議使用。 |

**建議下一步**
- 以 **lr1** 為基準，試 **lr=1.2、1.5** 與 **lambda_buffer=0.001～0.005**，看評估 0 死亡比例是否提升。
- 對 lr2 做 **多 seed 或更長訓練**，確認評估是否偶爾能出現存活，再決定是否納入候選。
