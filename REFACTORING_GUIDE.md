# 交易環境重構指南

## 概述

本次重構將 `TradingEnvironment` 改造為高度模組化、符合 SOLID 原則的架構，專為 **SAC 拉格朗日 agent** 訓練優化。

---

## 重構目標

### 1. **單一職責原則 (SRP)**
- 環境類專注於步進邏輯與狀態管理
- 技術特徵計算移除（改由外部數據傳入）
- 帳戶特徵、獎勵、資訊收集各自獨立模組

### 2. **開放封閉原則 (OCP)**
- 透過抽象基類 (ABC) 定義擴展點
- 新增功能只需繼承，不修改核心程式碼

### 3. **依賴反轉原則 (DIP)**
- 環境依賴抽象介面，不依賴具體實現
- 透過依賴注入 (Dependency Injection) 傳入自定義模組

### 4. **易於調整的訓練優化點**
- **帳戶特徵** (`AccountFeatureBuilder`)
- **獎勵函數** (`RewardCalculator`)
- **資訊收集** (`InfoCollector`)

---

## 架構設計

### 核心模組

```
Env/
├── trading_env.py              # 核心環境（專注步進邏輯）
├── trade_executor.py           # 交易執行器（已存在）
├── account_features.py         # 帳戶特徵構建器（新）
├── info_collector.py           # 資訊收集器（新）
├── reward.py                   # 獎勵計算器（已存在）
├── features.py                 # 技術特徵（舊版，可選）
└── __init__.py                 # 套件導出
```

### 類別關係圖

```
TradingEnvironment
    ├── TradeExecutor              (交易執行)
    ├── AccountFeatureBuilder      (帳戶特徵構建，依賴注入)
    ├── RewardCalculator           (獎勵計算，依賴注入)
    └── InfoCollector              (資訊收集，依賴注入)
```

---

## 主要改動

### 1. **移除技術特徵計算**

#### 舊版
```python
# 環境內部計算 MACD、ATR、流動性等技術指標
log_ret_1 = log_close.diff().fillna(0.0)
atr = pd.Series(true_range, ...).rolling(14).mean()
extra_features = build_all_features(df_num, lookback=...)
```

#### 新版
```python
# 直接使用傳入的數據（技術特徵由外部預處理）
self.df = df[numeric_columns].copy()
data_window = self.df.iloc[start_idx:end_idx].values.T
obs[:self.data_feature_count] = data_window
```

**優點：**
- 環境專注於交易邏輯，不處理特徵工程
- 特徵計算可在資料預處理階段完成（更高效）
- 不同策略可使用不同特徵集

---

### 2. **帳戶特徵模組化**

#### 抽象基類
```python
class AccountFeatureBuilder(ABC):
    @abstractmethod
    def feature_names(self) -> List[str]:
        pass
    
    @abstractmethod
    def compute_features(
        self, position_size, position_value, equity, 
        wallet_balance, current_price, initial_balance
    ) -> np.ndarray:
        pass
```

#### 預設實現
```python
class DefaultAccountFeatureBuilder(AccountFeatureBuilder):
    def feature_names(self):
        return ['position', 'position_value', 'equity', 'wallet']
    
    def compute_features(self, ...):
        # 計算 4 個基礎特徵
        return np.array([...])
```

#### 擴展實現
```python
class ExtendedAccountFeatureBuilder(AccountFeatureBuilder):
    def feature_names(self):
        return ['position', 'position_value', 'equity', 
                'wallet', 'leverage_ratio', 'available_margin']
    
    def compute_features(self, ...):
        # 計算 6 個擴展特徵
        return np.array([...])
```

---

### 3. **資訊收集模組化**

#### 抽象基類
```python
class InfoCollector(ABC):
    @abstractmethod
    def collect_step_info(
        self, current_step, current_price, position_size, equity, **kwargs
    ) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def collect_episode_info(
        self, termination_reason, final_equity, initial_balance, 
        episode_steps, **kwargs
    ) -> Dict[str, Any]:
        pass
```

#### 預設實現
```python
class DefaultInfoCollector(InfoCollector):
    def collect_step_info(self, ...):
        # 收集止損標記
        return {'stop_loss_triggered': ...}
    
    def collect_episode_info(self, ...):
        # 收集終止原因、損益、交易次數
        return {...}
```

#### 擴展實現
```python
class ExtendedInfoCollector(InfoCollector):
    def collect_episode_info(self, ...):
        # 額外收集最大回撤、夏普比率
        info['max_drawdown'] = ...
        info['sharpe_ratio'] = ...
        return info
```

---

### 4. **終止條件重新定義**

#### 成功條件
- **資料耗盡** (`data_exhausted`)：成功存活至交易數據用盡

#### 失敗條件
1. **強平次數達上限** (`liq_limit`)：累積強平次數 >= `max_liq_count`
2. **資金耗盡** (`balance_insufficient`)：權益 <= `min_balance`

#### 程式碼
```python
def _check_termination(self, equity: float) -> Tuple[bool, str]:
    # 成功：資料耗盡
    if self.current_step >= len(self.df) - 1:
        return True, 'data_exhausted'
    
    # 失敗1：資金耗盡
    if equity <= self.min_balance:
        return True, 'balance_insufficient'
    
    # 失敗2：強平次數達上限
    if self.executor.liq_triggered:
        self.episode_liq_count += 1
        if self.episode_liq_count >= self.max_liq_count:
            return True, 'liq_limit'
    
    return False, ''
```

---

### 5. **依賴注入設計**

#### 環境初始化
```python
env = TradingEnvironment(
    df=df,
    initial_balance=10_000,
    max_liq_count=3,  # 新參數：最大強平次數
    # 依賴注入：可自定義以下模組
    account_feature_builder=ExtendedAccountFeatureBuilder(),
    reward_calculator=CustomRewardCalculator(),
    info_collector=ExtendedInfoCollector()
)
```

#### 預設值
如果不傳入自定義模組，使用預設實現：
- `DefaultAccountFeatureBuilder`
- `create_default_calculator()` (獎勵)
- `DefaultInfoCollector`

---

## 擴展指南

### 情境 1: 新增帳戶特徵

**需求：** 增加「風險指標」特徵（如 VaR、夏普比率）

**步驟：**

1. 繼承 `AccountFeatureBuilder`
2. 實現 `feature_names()` 和 `compute_features()`
3. 注入環境

```python
class RiskAccountFeatureBuilder(AccountFeatureBuilder):
    def feature_names(self):
        return ['position', 'position_value', 'equity', 'wallet', 'var_95']
    
    def compute_features(self, ...):
        # 計算 VaR
        var_95 = self._compute_var(...)
        return np.array([..., var_95])
    
    def _compute_var(self, ...):
        # 自定義 VaR 計算邏輯
        pass

# 使用
env = TradingEnvironment(
    df=df,
    account_feature_builder=RiskAccountFeatureBuilder()
)
```

---

### 情境 2: 調整獎勵函數

**需求：** 強調長期持倉，懲罰頻繁交易

**步驟：**

1. 繼承 `RewardCalculator`
2. 實現 `compute()`
3. 注入環境

```python
class LongTermRewardCalculator(RewardCalculator):
    def compute(self, **kwargs):
        # 基礎獎勵：權益變化
        reward = (kwargs['new_equity'] - kwargs['last_equity']) / kwargs['last_equity']
        
        # 持倉獎勵
        if kwargs.get('has_position', False):
            reward += 0.01  # 每步持倉獎勵
        
        # 交易懲罰
        if kwargs.get('traded', False):
            reward -= 0.05  # 交易手續費懲罰
        
        return float(reward)

# 使用
env = TradingEnvironment(
    df=df,
    reward_calculator=LongTermRewardCalculator()
)
```

---

### 情境 3: 自定義資訊收集

**需求：** 追蹤每筆交易的持倉時長與盈虧

**步驟：**

1. 繼承 `InfoCollector`
2. 實現 `collect_step_info()` 和 `collect_episode_info()`
3. 注入環境

```python
class TradeTrackerInfoCollector(InfoCollector):
    def __init__(self):
        self.trade_durations = []
        self.trade_pnls = []
    
    def reset(self):
        self.trade_durations = []
        self.trade_pnls = []
    
    def collect_step_info(self, ...):
        # 追蹤交易
        if kwargs.get('trade_closed', False):
            self.trade_durations.append(kwargs['duration'])
            self.trade_pnls.append(kwargs['pnl'])
        return {}
    
    def collect_episode_info(self, ...):
        info = {...}
        info['avg_trade_duration'] = np.mean(self.trade_durations)
        info['avg_trade_pnl'] = np.mean(self.trade_pnls)
        return info

# 使用
env = TradingEnvironment(
    df=df,
    info_collector=TradeTrackerInfoCollector()
)
```

---

## 遷移指南

### 從舊版環境遷移

#### 舊版程式碼
```python
from Env.trading_env import TradingEnvironment

env = TradingEnvironment(
    df=df,
    initial_balance=10_000,
    reward_weights={'equity': 1.0, 'turnover': -0.1}
)
```

#### 新版程式碼
```python
from Env import TradingEnvironment

env = TradingEnvironment(
    df=df,  # 直接傳入數據（含技術特徵）
    initial_balance=10_000,
    max_liq_count=3,  # 新參數
    # reward_weights 已移除，改用 reward_calculator
)
```

#### 數據準備
**新版需要外部預處理技術特徵：**

```python
import pandas as pd
import ta  # 技術分析庫

# 載入原始數據
df = pd.read_csv('BTCUSDT.csv')

# 外部計算技術特徵
df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
df['macd'] = ta.trend.MACD(df['close']).macd()
df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

# 傳入環境
env = TradingEnvironment(df=df, ...)
```

---

## SAC 訓練配置範例

### 環境配置
```python
from Env import TradingEnvironment, ExtendedAccountFeatureBuilder, ExtendedInfoCollector

env = TradingEnvironment(
    df=df,
    initial_balance=10_000,
    transaction_fee=0.001,
    window_size=288,  # 24小時
    leverage=10,
    max_liq_count=3,  # 最多強平 3 次
    random_start=True,  # 訓練時隨機起點
    account_feature_builder=ExtendedAccountFeatureBuilder(max_leverage=20.0),
    info_collector=ExtendedInfoCollector()
)
```

### SAC 訓練
```python
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

# 創建 SAC agent
model = SAC(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    buffer_size=100_000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    verbose=1
)

# 訓練
model.learn(
    total_timesteps=1_000_000,
    callback=EvalCallback(eval_env, eval_freq=10_000)
)

# 保存模型
model.save('sac_trading_agent')
```

---

## 效能考量

### 1. **特徵預計算**
將技術特徵計算移至數據預處理階段，避免每次 reset 重複計算。

```python
# 預處理管線
df = load_raw_data('BTCUSDT.csv')
df = add_technical_features(df)  # 一次性計算
df.to_feather('BTCUSDT_with_features.feather')  # 快速序列化

# 訓練時直接載入
df = pd.read_feather('BTCUSDT_with_features.feather')
env = TradingEnvironment(df=df, ...)
```

### 2. **向量化操作**
帳戶狀態時間序列使用 NumPy array，提升觀測提取速度。

### 3. **最小化複製**
觀測提取使用切片而非複製，減少記憶體開銷。

---

## 測試建議

### 單元測試
```python
# test_account_features.py
def test_default_feature_builder():
    builder = DefaultAccountFeatureBuilder()
    features = builder.compute_features(
        position_size=0.1,
        position_value=5000,
        equity=10000,
        wallet_balance=5000,
        current_price=50000,
        initial_balance=10000
    )
    assert features.shape == (4,)
    assert 0 <= features[2] <= 2  # equity_norm 在合理範圍

# test_info_collector.py
def test_episode_info():
    collector = DefaultInfoCollector()
    info = collector.collect_episode_info(
        termination_reason='data_exhausted',
        final_equity=12000,
        initial_balance=10000,
        episode_steps=1000
    )
    assert info['profit'] == 2000
    assert info['profit_rate'] == 20.0
```

### 整合測試
```python
# test_trading_env.py
def test_environment_termination():
    df = load_test_data()
    env = TradingEnvironment(df=df, max_liq_count=3)
    
    obs, info = env.reset()
    done = False
    
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
    
    assert info['termination_reason'] in ['data_exhausted', 'liq_limit', 'balance_insufficient']
```

---

## 常見問題 (FAQ)

### Q1: 為什麼移除技術特徵計算？
**A:** 遵循單一職責原則，環境專注於交易邏輯。特徵工程應在資料預處理階段完成，提升效能與彈性。

### Q2: 如何在不修改核心程式碼的情況下調整獎勵？
**A:** 繼承 `RewardCalculator`，實現自定義 `compute()` 方法，並透過依賴注入傳入環境。

### Q3: `max_liq_count` 如何影響訓練？
**A:** 設定過低（如 1）會導致 episode 過早終止，agent 缺乏學習機會；設定過高（如 10）會讓 agent 學會過度冒險。建議從 3-5 開始調整。

### Q4: 如何驗證自定義模組正確性？
**A:** 編寫單元測試，確保輸出形狀與數值範圍符合預期。使用 `pytest` 框架。

### Q5: 舊版訓練的模型能否用於新版環境？
**A:** 不能直接使用，因為觀測空間形狀可能不同。需要重新訓練或進行模型遷移。

---

## 總結

### 重構優勢
✅ **模組化**：各模組職責清晰，易於維護  
✅ **可擴展**：透過繼承與依賴注入輕鬆擴展  
✅ **高效能**：特徵預計算 + 向量化操作  
✅ **SAC 友善**：連續動作空間 + 清晰終止條件  
✅ **符合 SOLID**：遵循物件導向最佳實踐  

### 擴展點
- **帳戶特徵** (`AccountFeatureBuilder`)
- **獎勵函數** (`RewardCalculator`)
- **資訊收集** (`InfoCollector`)

### 下一步
1. 根據策略需求自定義獎勵函數
2. 設計合適的帳戶特徵集
3. 配置 SAC 超參數並開始訓練
4. 使用 `ExtendedInfoCollector` 追蹤訓練指標

---

**文件版本:** 1.0  
**最後更新:** 2025-11-16  
**作者:** AI Assistant

