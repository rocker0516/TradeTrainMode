# TradingEnvironment 快速開始

## 5 分鐘上手

### 1. 基礎使用
```python
import pandas as pd
from Env import TradingEnvironment

# 載入數據（必須包含 OHLCV）
df = pd.read_csv('your_data.csv')

# 創建環境
env = TradingEnvironment(
    df=df,
    initial_balance=10_000,
    leverage=10,
    window_size=288,  # 24小時（5分K）
    max_liq_count=3,  # 最大強平次數
)

# 訓練循環
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # 替換為你的 agent 輸出
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"最終權益: {info['final_equity']:.2f}")
print(f"報酬率: {info['profit_rate']:.2f}%")
```

### 2. 自定義配置
```python
from Env import (
    TradingEnvironment,
    ExtendedAccountFeatureBuilder,
    RewardCalculator,
    ExtendedInfoCollector,
)

# 使用擴展特徵（8個帳戶特徵）
feature_builder = ExtendedAccountFeatureBuilder()

# 自定義獎勵權重
reward_calculator = RewardCalculator(
    w_stop_loss=100.0,   # 止損懲罰
    w_terminal=200.0,    # 終局懲罰
    w_return=20.0,       # 收益獎勵
    w_risk=10.0,         # 風險懲罰
    w_struct=3.0,        # 結構性風險
)

# 使用擴展統計
info_collector = ExtendedInfoCollector()

# 創建環境
env = TradingEnvironment(
    df=df,
    initial_balance=10_000,
    leverage=10,
    window_size=288,
    max_liq_count=3,
    # 注入自定義組件
    account_feature_builder=feature_builder,
    reward_calculator=reward_calculator,
    info_collector=info_collector,
)

print(f"觀測空間: {env.observation_space.shape}")
# 輸出: (data_features + 8, 288)
```

### 3. 使用 SAC Agent
```python
from stable_baselines3 import SAC

# 創建 SAC agent
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=100_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
)

# 訓練
model.learn(total_timesteps=100_000)

# 保存模型
model.save("sac_trading_model")

# 評估
obs, info = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"評估結果: {info['profit_rate']:.2f}%")
```

## 三大擴展點

### 1. 帳戶特徵（AccountFeatureBuilder）
控制 Agent 觀察到的帳戶狀態：
- `DefaultAccountFeatureBuilder` - 4 個基礎特徵
- `ExtendedAccountFeatureBuilder` - 8 個擴展特徵
- 自定義：繼承 `AccountFeatureBuilder`

### 2. 獎勵函數（RewardCalculator）
引導 Agent 學習策略：
- 調整權重：`w_stop_loss`, `w_terminal`, `w_return`, `w_risk`
- 調整尺度：`return_clip`, `risk_threshold`

### 3. 資訊收集（InfoCollector）
收集訓練/評估統計：
- `DefaultInfoCollector` - 基礎統計
- `ExtendedInfoCollector` - 詳細統計（回撤、夏普、勝率）
- 自定義：繼承 `InfoCollector`

## 終止條件

### ✅ 成功
- 資料耗盡（走完所有數據）

### ❌ 失敗
- 強平次數達上限（`max_liq_count`）
- 資金耗盡（`equity <= min_balance`）

## 參數說明

### 環境參數
| 參數 | 說明 | 默認值 | 範圍 |
|-----|------|-------|-----|
| `initial_balance` | 初始資金 | 10,000 | > 0 |
| `leverage` | 槓桿倍數 | 10 | 1-100 |
| `window_size` | 觀測窗口 | 288 | > 0 |
| `max_liq_count` | 最大強平次數 | 3 | > 0 |
| `min_balance` | 最小資金 | 100 | > 0 |
| `transaction_fee` | 交易手續費 | 0.001 | 0-1 |

### 獎勵權重（保守 vs 激進）
```python
# 保守策略（重視風險控制）
RewardCalculator(
    w_stop_loss=100.0,
    w_terminal=200.0,
    w_return=10.0,
    w_risk=20.0,
)

# 激進策略（追求收益）
RewardCalculator(
    w_stop_loss=50.0,
    w_terminal=100.0,
    w_return=30.0,
    w_risk=5.0,
)
```

## 常見問題

**Q: 如何準備數據？**
```python
# 必須包含 OHLCV 列
df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...],
    # 可選：額外的技術指標
    'rsi': [...],
    'macd': [...],
})
```

**Q: 觀測空間是什麼？**
```
shape = (n_features, window_size)
n_features = df 欄位數 + 帳戶特徵數
```

**Q: 動作空間是什麼？**
```
連續動作 [-1.0, 1.0]
-1.0 = 全倉空
 0.0 = 平倉
 1.0 = 全倉多
```

**Q: 如何調參？**
1. 先用默認組件訓練基礎模型
2. 觀察 info 統計，找出問題
3. 調整 reward 權重或使用擴展特徵
4. 重新訓練並比較效果

## 詳細文檔

- 📖 **完整指南**：`CUSTOMIZATION_GUIDE.md`
- 📝 **重構總結**：`REFACTORING_SUMMARY.md`
- 🧪 **測試範例**：`test_customization.py`

## 快速測試

```bash
# 運行測試驗證環境
python Env/test_customization.py

# 輸出：所有測試通過 ✓
```

---

**準備好開始訓練了嗎？** 🚀

