# TradingEnvironment 重構總結

## 重構目標

根據用戶需求，將 `TradingEnvironment` 重構為：
1. ✅ 符合 SAC 拉格朗日 Agent 訓練需求
2. ✅ 只處理帳戶狀態，技術特徵由外部傳入
3. ✅ 明確的成功/失敗條件
4. ✅ 方便擴展的 Info、Feature、Reward 設計

## 重構內容

### 新增文件

#### 1. `account_features.py` - 帳戶特徵構建器
**職責**：定義 Agent 觀察到的帳戶狀態特徵

**類別結構**：
```
AccountFeatureBuilder (抽象基類)
├── DefaultAccountFeatureBuilder (4 個基礎特徵)
└── ExtendedAccountFeatureBuilder (8 個擴展特徵)
```

**設計原則**：
- 單一職責：只負責帳戶特徵計算
- 開放封閉：透過繼承擴展
- 依賴反轉：環境依賴抽象介面

**基礎特徵**（DefaultAccountFeatureBuilder）：
1. `position_ratio` - 持倉方向與強度 [-1, 1]
2. `equity_ratio` - 權益比例 (equity / initial_balance)
3. `wallet_ratio` - 錢包比例 (wallet_balance / initial_balance)
4. `unrealized_ratio` - 未實現損益比例

**擴展特徵**（ExtendedAccountFeatureBuilder）：
5. `leverage_usage` - 槓桿使用率
6. `profit_loss_ratio` - 盈虧比
7. `position_side` - 持倉方向 (-1/0/1)
8. `equity_drawdown` - 權益回撤

#### 2. `info_collector.py` - 資訊收集器
**職責**：收集訓練/評估過程中的統計資訊

**類別結構**：
```
InfoCollector (抽象基類)
├── DefaultInfoCollector (基礎統計)
└── ExtendedInfoCollector (詳細統計)
```

**基礎統計**（DefaultInfoCollector）：
- 每步：價格、持倉、權益、止損觸發
- 回合：終止原因、成功/失敗、報酬率、交易次數、費用

**擴展統計**（ExtendedInfoCollector）：
- 額外統計：最大回撤、夏普比率、勝率、平均交易損益

#### 3. `CUSTOMIZATION_GUIDE.md` - 自定義指南
詳細說明如何擴展三個核心組件，包含：
- 設計原則說明
- 自定義步驟範例
- 使用範例
- 與 SAC Agent 整合
- 調參建議
- 常見問題

#### 4. `test_customization.py` - 測試腳本
驗證重構後的環境功能：
- 測試 1：默認組件
- 測試 2：擴展帳戶特徵
- 測試 3：自定義獎勵函數
- 測試 4：擴展資訊收集器
- 測試 5：終止條件驗證

### 更新文件

#### `trading_env.py` - 核心環境
**主要改進**：
1. 依賴注入三個可擴展組件（feature_builder、reward_calculator、info_collector）
2. 傳遞 `leverage` 參數給特徵構建器
3. 修正終端輸出字符（避免編碼問題）

**觀察空間設計**：
```
shape = (n_features, window_size)
where:
  n_features = data_feature_count + account_feature_count
  data_feature_count = df 欄位數量（由外部傳入的技術特徵）
  account_feature_count = feature_builder.feature_count()
```

**動作空間設計**：
```
連續動作 [-1.0, 1.0]
表示目標持倉比例（-1=全倉空，0=平倉，1=全倉多）
```

**終止條件**：
- ✅ **成功**：`current_step >= len(df) - 1` → `'data_exhausted'`
- ❌ **失敗1**：`equity <= min_balance` → `'balance_insufficient'`
- ❌ **失敗2**：`episode_liq_count >= max_liq_count` → `'liq_limit'`

#### `__init__.py` - 套件初始化
正確導出所有組件：
```python
from Env import (
    TradingEnvironment,
    TradeExecutor,
    RewardCalculator,
    create_default_calculator,
    AccountFeatureBuilder,
    DefaultAccountFeatureBuilder,
    ExtendedAccountFeatureBuilder,
    InfoCollector,
    DefaultInfoCollector,
    ExtendedInfoCollector,
)
```

## 使用方式

### 基礎使用（默認組件）
```python
from Env import TradingEnvironment

env = TradingEnvironment(
    df=df,
    initial_balance=10_000,
    leverage=10,
    window_size=288,
    max_liq_count=3,
)
```

### 進階使用（自定義組件）
```python
from Env import (
    TradingEnvironment,
    ExtendedAccountFeatureBuilder,
    RewardCalculator,
    ExtendedInfoCollector,
)

env = TradingEnvironment(
    df=df,
    initial_balance=10_000,
    leverage=10,
    window_size=288,
    max_liq_count=3,
    # 注入自定義組件
    account_feature_builder=ExtendedAccountFeatureBuilder(),
    reward_calculator=RewardCalculator(
        w_stop_loss=100.0,
        w_terminal=200.0,
        w_return=20.0,
    ),
    info_collector=ExtendedInfoCollector(),
)
```

### 與 SAC Agent 整合
```python
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=100_000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
)

model.learn(total_timesteps=100_000)
```

## 設計優勢

### 1. 符合 SOLID 原則
- **單一職責**：每個類別只負責一件事
  - `TradingEnvironment`：環境步進與狀態管理
  - `AccountFeatureBuilder`：帳戶特徵計算
  - `RewardCalculator`：獎勵計算
  - `InfoCollector`：資訊收集
  - `TradeExecutor`：交易執行與風控

- **開放封閉**：透過繼承擴展，不修改基類
  ```python
  class MyFeatureBuilder(AccountFeatureBuilder):
      # 自定義實現
  ```

- **依賴反轉**：環境依賴抽象介面，不依賴具體實現
  ```python
  def __init__(self, account_feature_builder: Optional[AccountFeatureBuilder] = None):
      self.account_feature_builder = account_feature_builder or DefaultAccountFeatureBuilder()
  ```

### 2. 高度可擴展
只需調整三個組件，不需修改核心環境：
- 調整帳戶特徵 → 繼承 `AccountFeatureBuilder`
- 調整獎勵函數 → 繼承或配置 `RewardCalculator`
- 調整統計資訊 → 繼承 `InfoCollector`

### 3. 技術特徵與帳戶狀態分離
- 技術特徵：由外部傳入（df 的欄位）
- 帳戶特徵：由 `AccountFeatureBuilder` 計算
- 優勢：可以在訓練前預處理技術特徵，環境只專注於交易邏輯

### 4. 明確的終止條件
- 成功：存活至資料耗盡
- 失敗：強平次數達上限 或 資金耗盡
- 符合實際交易風險管理原則

## 測試結果

所有測試通過 ✅：
1. ✅ 默認組件運行正常
2. ✅ 擴展特徵（8特徵）運行正常
3. ✅ 自定義獎勵函數運行正常
4. ✅ 擴展資訊收集器運行正常
5. ✅ 終止條件驗證通過

測試命令：
```bash
python Env/test_customization.py
```

## 後續優化建議

### 1. 帳戶特徵優化
可根據訓練效果添加：
- 持倉時長
- 平均進場價距離
- 動態槓桿調整指標
- 資金利用率

### 2. 獎勵函數優化
根據訓練目標調整權重：
- 追求穩定：提高 `w_risk`、`w_terminal`
- 追求收益：提高 `w_return`
- 避免頻繁止損：提高 `w_stop_loss`

### 3. 資訊收集優化
可添加更多統計：
- 最大連續虧損
- 盈虧比（平均盈利/平均虧損）
- 交易頻率統計
- 各時段表現分析

### 4. 向量化環境
配合 `stable_baselines3` 使用：
```python
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# 創建多個環境並行訓練
vec_env = SubprocVecEnv([make_env(i) for i in range(8)])
model = SAC("MlpPolicy", vec_env)
```

## 相關文件

- `account_features.py` - 帳戶特徵構建器
- `info_collector.py` - 資訊收集器
- `trading_env.py` - 核心環境
- `reward.py` - 獎勵計算器
- `trade_executor.py` - 交易執行器
- `CUSTOMIZATION_GUIDE.md` - 詳細使用指南
- `test_customization.py` - 測試腳本

## 版本資訊

- **當前版本**：v2.0
- **重構日期**：2025-11-16
- **Python 版本**：3.8+
- **依賴套件**：gymnasium, numpy, pandas

## 聯絡與反饋

如有問題或建議，請參考：
1. `CUSTOMIZATION_GUIDE.md` - 詳細使用指南
2. `test_customization.py` - 測試範例
3. 各模組的 docstring - 函數/類別說明

---

**重構完成！環境已準備好用於 SAC 拉格朗日 Agent 訓練。** 🎉

