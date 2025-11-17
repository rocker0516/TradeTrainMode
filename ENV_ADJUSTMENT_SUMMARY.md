# 交易環境調整總結報告

## 📋 調整需求

根據用戶需求，對交易環境進行以下調整：

1. **簡化結束條件**：只保留資料耗盡、資金不足
2. **允許繼續交易**：止損/強平後不終止環境
3. **新增統計功能**：完整記錄止損、強平次數
4. **獎勵方案調整**：採用方案 B（平衡版獎勵）
5. **不限制次數**：不限制單回合止損/強平次數

---

## ✅ 完成項目

### 1. 修改 `trading_env.py`

#### 變更內容：
- **移除環境層統計變數**：
  - 刪除 `self.episode_stop_loss_count`
  - 刪除 `self.episode_liq_count`
  
- **簡化終止條件**：
  ```python
  def _check_termination(self, equity: float) -> Tuple[bool, str]:
      # 成功條件：資料耗盡
      if self.current_step >= len(self.df) - 1:
          return True, 'data_exhausted'
      
      # 失敗條件：資金不足
      if equity <= self.min_balance:
          return True, 'balance_insufficient'
      
      return False, ''
  ```

- **統一統計來源**：所有統計改從 `executor` 獲取

#### 影響範圍：
- `__init__()`: 移除環境層統計變數
- `reset()`: 移除環境層統計重置
- `step()`: 移除環境層統計累加
- `_check_termination()`: 簡化終止邏輯
- `collect_episode_info()`: 移除統計參數傳遞

---

### 2. 修改 `trade_executor.py`

#### 變更內容：
- **新增統計累加邏輯**：
  ```python
  # 止損觸發時累加
  if 觸發止損:
      self.stop_loss_triggered = True
      self.total_stop_loss_count += 1
  
  # 強平觸發時累加
  if 觸發強平:
      self.liq_triggered = True
      self.total_liq_count += 1
  ```

- **每步重置觸發標記**：
  ```python
  def execute(...):
      self.stop_loss_triggered = False
      self.liq_triggered = False
      # ... 執行邏輯
  ```

#### 影響範圍：
- `execute()`: 新增統計累加與標記重置
- 保留原有的 `total_stop_loss_count` 和 `total_liq_count` 變數

---

### 3. 修改 `info_collector.py`

#### 變更內容：
- **更新介面**：移除 `stop_loss_count` 和 `liq_count` 參數
- **改從 executor 獲取統計**：
  ```python
  'stop_loss_count': int(executor.total_stop_loss_count),
  'liq_count': int(executor.total_liq_count),
  ```

#### 影響範圍：
- `InfoCollector` 抽象基類
- `DefaultInfoCollector` 實現
- `ExtendedInfoCollector` 實現

---

### 4. 新增 `RewardCalculatorBalanced`（方案 B）

#### 設計原則：
1. **存活獎勵**：每步 +0.1（鼓勵長期存活）
2. **收益獎勵**：log return 標準化（±2% → ±10 分）
3. **風險懲罰**：margin buffer < 0.5 時二次懲罰（最高 -10 分）
4. **止損懲罰**：觸發時 -20 分（但不終止）
5. **強平懲罰**：觸發時 -30 分（但不終止）
6. **終局獎勵**：
   - 成功：+50 基礎分 + 報酬率 × 200 加成
   - 失敗：-100 分

#### 獎勵配置：
```python
w_survival = 0.1           # 存活獎勵
w_return = 10.0            # 收益獎勵
w_risk = 10.0              # 風險懲罰
w_stop_loss = 20.0         # 止損懲罰
w_liquidation = 30.0       # 強平懲罰
w_terminal_success = 50.0  # 終局成功
w_terminal_fail = 100.0    # 終局失敗
w_terminal_return_bonus = 200.0  # 報酬率加成
```

#### 使用方式：
```python
# 方案 B（推薦，已設為默認）
from Env.trading_env import TradingEnvironment
env = TradingEnvironment(df=data)  # 自動使用 RewardCalculatorBalanced

# 或手動指定
from Env.reward import RewardCalculatorBalanced
calculator = RewardCalculatorBalanced()
env = TradingEnvironment(df=data, reward_calculator=calculator)

# 進階版（原方案）
from Env.reward import create_advanced_calculator
env = TradingEnvironment(df=data, reward_calculator=create_advanced_calculator())
```

---

## 🧪 測試結果

### 測試1：驗證結束條件
- ✅ 止損觸發 10 次，環境未終止
- ✅ 統計正確累加
- ✅ 止損後允許繼續交易

### 測試2：驗證統計邏輯
- ✅ 多回合統計獨立正確
- ✅ executor 統計持久累加

### 測試3：驗證方案 B 獎勵
```
情境1（小幅盈利）: +2.59 分
情境2（觸發止損）: -29.90 分
情境3（接近危險區）: -1.50 分
情境4（終局成功，盈利20%）: +100.00 分
情境5（終局失敗）: -120.00 分
```
✅ 所有情境計算正確

### 測試4：完整回合測試
- ✅ 成功執行到 data_exhausted（949 步）
- ✅ 止損 52 次，環境持續運行
- ✅ 總交易 688 筆，統計完整

---

## 📊 調整前後對比

| 項目 | 調整前 | 調整後 |
|------|--------|--------|
| **結束條件** | 資料耗盡 / 資金不足 / 強平達上限 | 資料耗盡 / 資金不足 |
| **止損/強平行為** | ❌ 可能終止環境 | ✅ 只扣分，繼續交易 |
| **統計位置** | 環境層 + executor 層（重複） | 統一在 executor 層 |
| **止損/強平限制** | 有上限限制 | ❌ 無限制 |
| **獎勵計算** | 複雜多層（5 個子項） | 平衡版（6 個子項，權重優化） |
| **獎勵訊號** | 止損 -50，終局 -120×steps | 止損 -20，強平 -30，終局±100 |

---

## 🎯 獎勵設計建議

### 當前方案 B 的特點：

#### ✅ 優點：
1. **訊號清晰**：每個懲罰項獨立明確
2. **平衡性好**：風險控制與收益追求並重
3. **鼓勵長存**：存活獎勵防止過早放棄
4. **漸進懲罰**：風險懲罰採用二次函數，接近危險區懲罰加重
5. **終局激勵**：報酬率加成鼓勵盈利策略

#### ⚠️ 潛在問題與建議：

1. **存活獎勵可能導致「躺平」**
   - **現象**：Agent 學會不交易，只拿存活獎勵
   - **解決方案**：
     ```python
     # 方案 A：降低存活獎勵
     w_survival = 0.01  # 從 0.1 降到 0.01
     
     # 方案 B：加入持倉鼓勵
     if has_position:
         total += 0.05  # 持倉時額外獎勵
     ```

2. **止損懲罰可能過輕**
   - **現象**：Agent 不怕止損，頻繁冒險
   - **解決方案**：
     ```python
     w_stop_loss = 30.0  # 從 20 提升到 30
     # 或加入連續止損額外懲罰
     if 連續止損 > 3:
         total -= 10.0 * 連續次數
     ```

3. **風險閾值可能過低**
   - **現象**：margin buffer < 0.5 才懲罰，可能過於寬鬆
   - **解決方案**：
     ```python
     risk_threshold = 0.7  # 從 0.5 提升到 0.7
     ```

### 進階調整方向：

#### 1. 動態權重（根據訓練階段調整）
```python
# 初期：重視存活
if training_step < 100000:
    w_survival = 0.2
    w_risk = 15.0
# 中期：平衡
elif training_step < 500000:
    w_survival = 0.1
    w_risk = 10.0
# 後期：重視收益
else:
    w_survival = 0.05
    w_return = 15.0
```

#### 2. 多目標獎勵（Lagrange 方法）
```python
# 主目標：最大化收益
primary_reward = return_reward + terminal_reward

# 約束：控制風險
risk_constraint = risk_penalty + stop_loss_penalty

# 合成
total_reward = primary_reward - lambda * max(0, risk_constraint - threshold)
```

#### 3. 情境感知獎勵
```python
# 趨勢市：鼓勵持倉
if 檢測到趨勢:
    w_return *= 1.5
    
# 震盪市：鼓勵保守
if 檢測到震盪:
    w_risk *= 2.0
```

---

## 📝 使用範例

### 基本使用（默認方案 B）
```python
import pandas as pd
from Env.trading_env import TradingEnvironment

# 載入數據
df = pd.read_csv('data.csv')

# 創建環境（自動使用 RewardCalculatorBalanced）
env = TradingEnvironment(
    df=df,
    initial_balance=10000,
    window_size=288,  # 24 小時
    leverage=10,
    margin_mode='isolated'
)

# 訓練循環
obs, info = env.reset()
for step in range(1000):
    action = agent.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        print(f"Episode 結束：{info['termination_reason']}")
        print(f"止損次數：{info['stop_loss_count']}")
        print(f"強平次數：{info['liq_count']}")
        break
```

### 自定義權重
```python
from Env.reward import RewardCalculatorBalanced

# 自定義權重
calculator = RewardCalculatorBalanced(
    w_survival=0.05,      # 降低存活獎勵
    w_stop_loss=30.0,     # 提高止損懲罰
    risk_threshold=0.7    # 提高風險閾值
)

env = TradingEnvironment(
    df=df,
    reward_calculator=calculator
)
```

### 使用進階版（原方案）
```python
from Env.reward import create_advanced_calculator

env = TradingEnvironment(
    df=df,
    reward_calculator=create_advanced_calculator()
)
```

---

## 🔍 後續監控建議

### 訓練時需觀察的指標：

1. **獎勵分解**
   - 存活獎勵佔比（若 >50% 可能躺平）
   - 收益獎勵趨勢（應逐步上升）
   - 懲罰頻率（止損/強平應逐步下降）

2. **行為指標**
   - 持倉時間比例（若 <10% 可能不敢交易）
   - 平均持倉槓桿（若接近上限過於冒險）
   - 止損觸發率（應 <5%）

3. **終局統計**
   - data_exhausted 佔比（應逐步提升到 >80%）
   - 平均報酬率（應逐步為正）
   - 最大回撤（應逐步降低）

### 調參建議：

| 現象 | 可能原因 | 調整方案 |
|------|----------|----------|
| Agent 不交易 | 存活獎勵過高 | 降低 `w_survival` 到 0.01 |
| 頻繁止損 | 止損懲罰過輕 | 提高 `w_stop_loss` 到 30-50 |
| 過度冒險 | 風險懲罰過輕 | 提高 `w_risk` 或 `risk_threshold` |
| 收益不佳 | 收益獎勵過低 | 提高 `w_return` 或 `w_terminal_return_bonus` |

---

## 📚 相關文件

- `Env/trading_env.py` - 交易環境主邏輯
- `Env/trade_executor.py` - 交易執行與統計
- `Env/reward.py` - 獎勵計算器（方案 B + 原方案）
- `Env/info_collector.py` - 資訊收集器
- `test_env_adjustment.py` - 完整測試腳本

---

## ✨ 總結

本次調整成功實現了：
1. ✅ 簡化終止條件，只保留資料耗盡和資金不足
2. ✅ 止損/強平後允許繼續交易
3. ✅ 完整的止損/強平統計功能
4. ✅ 平衡版獎勵方案（方案 B），適合生產環境
5. ✅ 所有測試通過，邏輯正確

**下一步建議**：
1. 開始訓練並監控上述指標
2. 根據訓練表現微調權重
3. 若發現問題，參考「進階調整方向」進行優化

祝訓練順利！🚀

