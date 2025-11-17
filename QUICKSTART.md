# 快速入門指南

## 安裝依賴

```bash
pip install -r requirements.txt
```

---

## 基礎使用

### 1. 準備數據

```python
import pandas as pd

# 載入數據（必須包含 OHLCV 列）
df = pd.read_csv('Data/BTCUSDT_futures_volume_5years_5min.csv')

# 檢查必要列
required_columns = ['open', 'high', 'low', 'close', 'volume']
print(f"數據列: {df.columns.tolist()}")
print(f"數據長度: {len(df)}")
```

### 2. 創建環境

```python
from Env import TradingEnvironment

# 使用預設配置
env = TradingEnvironment(
    df=df,
    initial_balance=10_000,      # 初始資金
    transaction_fee=0.001,        # 手續費 0.1%
    window_size=288,              # 觀測窗口（24小時，5分K）
    leverage=10,                  # 槓桿倍數
    max_liq_count=3,              # 最大強平次數
    random_start=True             # 隨機起點（訓練時建議開啟）
)
```

### 3. 執行交易

```python
# 重置環境
obs, info = env.reset()
print(f"觀測形狀: {obs.shape}")  # (n_features, window_size)

# 執行一步
action = env.action_space.sample()  # 隨機動作 [-1.0, 1.0]
obs, reward, done, truncated, info = env.step(action)

print(f"獎勵: {reward:.4f}")
print(f"是否結束: {done}")

if done:
    print(f"終止原因: {info['termination_reason']}")
    print(f"最終權益: {info['final_balance']:.2f}")
    print(f"收益率: {info['profit_rate']:.2f}%")
```

---

## 進階配置

### 1. 使用擴展帳戶特徵

```python
from Env import TradingEnvironment, ExtendedAccountFeatureBuilder

# 擴展特徵包含：槓桿比率、可用保證金
feature_builder = ExtendedAccountFeatureBuilder(max_leverage=20.0)

env = TradingEnvironment(
    df=df,
    account_feature_builder=feature_builder,  # 注入自定義特徵構建器
    max_liq_count=5
)

print(f"帳戶特徵: {feature_builder.feature_names()}")
# ['position', 'position_value', 'equity', 'wallet', 'leverage_ratio', 'available_margin']
```

### 2. 自定義獎勵函數

```python
from Env import TradingEnvironment
from Env.reward import RewardCalculator

class MyRewardCalculator(RewardCalculator):
    def compute(self, **kwargs):
        # 簡化獎勵：只看權益變化
        last_equity = kwargs.get('last_equity', 0.0)
        new_equity = kwargs.get('new_equity', 0.0)
        
        if last_equity > 0:
            return (new_equity - last_equity) / last_equity * 100
        return 0.0

env = TradingEnvironment(
    df=df,
    reward_calculator=MyRewardCalculator()  # 注入自定義獎勵計算器
)
```

### 3. 擴展資訊收集

```python
from Env import TradingEnvironment, ExtendedInfoCollector

# 擴展資訊包含：最大回撤、夏普比率
info_collector = ExtendedInfoCollector()

env = TradingEnvironment(
    df=df,
    info_collector=info_collector  # 注入自定義資訊收集器
)

# 執行回合
obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

# 查看擴展資訊
if done:
    print(f"最大回撤: {info.get('max_drawdown', 0):.4f}")
    print(f"夏普比率: {info.get('sharpe_ratio', 0):.4f}")
```

---

## SAC 訓練

### 1. 安裝 Stable-Baselines3

```bash
pip install stable-baselines3
```

### 2. 訓練腳本

```python
import pandas as pd
from Env import TradingEnvironment, ExtendedAccountFeatureBuilder, ExtendedInfoCollector
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

# 載入數據
df = pd.read_csv('Data/BTCUSDT_futures_volume_5years_5min.csv')

# 創建訓練環境
train_env = TradingEnvironment(
    df=df,
    initial_balance=10_000,
    window_size=288,
    leverage=10,
    max_liq_count=3,
    random_start=True,  # 訓練時隨機起點
    account_feature_builder=ExtendedAccountFeatureBuilder(),
    info_collector=ExtendedInfoCollector()
)

# 創建評估環境
eval_env = TradingEnvironment(
    df=df,
    initial_balance=10_000,
    window_size=288,
    leverage=10,
    max_liq_count=3,
    random_start=False,  # 評估時固定起點
    account_feature_builder=ExtendedAccountFeatureBuilder(),
    info_collector=ExtendedInfoCollector()
)

# 創建 SAC agent
model = SAC(
    'MlpPolicy',
    train_env,
    learning_rate=3e-4,
    buffer_size=100_000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    verbose=1,
    tensorboard_log='./logs/sac_trading/'
)

# 設定評估回調
eval_callback = EvalCallback(
    eval_env,
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
    verbose=1
)

# 訓練
model.learn(
    total_timesteps=1_000_000,
    callback=eval_callback
)

# 保存模型
model.save('models/sac_trading_agent')

# 測試訓練好的模型
obs, info = eval_env.reset()
done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = eval_env.step(action)
    total_reward += reward

print(f"\n測試結果:")
print(f"  總獎勵: {total_reward:.4f}")
print(f"  終止原因: {info['termination_reason']}")
print(f"  最終權益: {info['final_balance']:.2f}")
print(f"  收益率: {info['profit_rate']:.2f}%")
print(f"  最大回撤: {info.get('max_drawdown', 0):.4f}")
```

---

## 終止條件說明

### 成功條件
- ✅ **資料耗盡** (`data_exhausted`)：成功存活至交易數據用盡

### 失敗條件
- ❌ **強平次數達上限** (`liq_limit`)：累積強平次數 >= `max_liq_count`
- ❌ **資金耗盡** (`balance_insufficient`)：權益 <= `min_balance`

---

## 環境參數說明

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `df` | DataFrame | 必填 | 交易數據（必須包含 OHLCV） |
| `initial_balance` | float | 10000 | 初始資金 |
| `transaction_fee` | float | 0.001 | 交易手續費比例（0.1%） |
| `window_size` | int | 288 | 觀測窗口大小（時間步數） |
| `leverage` | float | 10 | 槓桿倍數 |
| `min_balance` | float | 100 | 最小資金（低於此值視為資金耗盡） |
| `min_trade_qty` | float | 0.001 | 最低交易數量 |
| `margin_mode` | str | 'isolated' | 保證金模式（'cross' 或 'isolated'） |
| `max_liq_count` | int | 3 | 最大強平次數（達到此值則失敗） |
| `random_start` | bool | False | 是否隨機起始位置 |
| `account_feature_builder` | AccountFeatureBuilder | None | 帳戶特徵構建器（可自定義） |
| `reward_calculator` | RewardCalculator | None | 獎勵計算器（可自定義） |
| `info_collector` | InfoCollector | None | 資訊收集器（可自定義） |

---

## 觀測空間

### 形狀
```python
observation_space.shape = (n_features, window_size)
```

### 組成
1. **數據特徵**：df 中的所有數值列（如 OHLCV、技術指標）
2. **帳戶特徵**：由 `AccountFeatureBuilder` 構建
   - 預設 4 個：`position`, `position_value`, `equity`, `wallet`
   - 擴展 6 個：上述 + `leverage_ratio`, `available_margin`

### 範例
```python
env = TradingEnvironment(df=df, window_size=288)
obs, info = env.reset()

print(f"觀測形狀: {obs.shape}")  # (n_features, 288)
print(f"數據特徵數: {env.data_feature_count}")
print(f"帳戶特徵數: {env.account_feature_count}")
print(f"總特徵數: {env.n_features}")
```

---

## 動作空間

### 形狀
```python
action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=float32)
```

### 語義
- **[-1.0, 0.0)**：做空（負值越大，空倉比例越高）
- **0.0**：平倉
- **(0.0, 1.0]**：做多（正值越大，多倉比例越高）

### 範例
```python
action = np.array([0.5])   # 目標持倉：50% 多倉
action = np.array([-0.8])  # 目標持倉：80% 空倉
action = np.array([0.0])   # 目標持倉：平倉
```

---

## 常見問題

### Q: 如何添加技術指標？
**A:** 在數據預處理階段添加，然後傳入環境。

```python
import ta

df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
df['macd'] = ta.trend.MACD(df['close']).macd()

env = TradingEnvironment(df=df)  # 技術指標會自動包含在觀測中
```

### Q: 如何調整獎勵函數？
**A:** 繼承 `RewardCalculator` 並注入環境。參考[進階配置 #2](#2-自定義獎勵函數)。

### Q: `max_liq_count` 設多少合適？
**A:** 建議從 3-5 開始，根據訓練結果調整：
- 太低（1-2）：episode 過早終止
- 太高（10+）：agent 學會過度冒險

### Q: 訓練時應該開啟 `random_start` 嗎？
**A:** 建議開啟，可增加樣本多樣性，避免過擬合特定起點。

### Q: 如何監控訓練進度？
**A:** 使用 TensorBoard：

```bash
tensorboard --logdir ./logs/sac_trading/
```

---

## 相關文檔

- [重構指南](REFACTORING_GUIDE.md)：詳細架構設計與擴展方法
- [訓練架構](TRAIN_ARCHITECTURE.md)：訓練系統整體設計
- [範例程式](Env/example_usage_refactored.py)：完整使用範例

---

## 支援

如遇問題，請檢查：
1. 數據是否包含必要列（open, high, low, close, volume）
2. 依賴套件是否正確安裝（`pip install -r requirements.txt`）
3. Python 版本是否 >= 3.8

---

**文件版本:** 1.0  
**最後更新:** 2025-11-16

