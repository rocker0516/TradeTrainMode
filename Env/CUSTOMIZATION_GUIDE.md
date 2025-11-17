# TradingEnvironment 自定義指南

本指南說明如何擴展 `TradingEnvironment` 的三個核心組件：帳戶特徵、獎勵函數、資訊收集器。

## 設計原則

遵循 **SOLID 原則**，特別是：
- **開放封閉原則（OCP）**：透過繼承擴展，不修改核心類別
- **依賴反轉原則（DIP）**：環境依賴抽象介面，不依賴具體實現

## 三大擴展點

### 1. 帳戶特徵構建器（AccountFeatureBuilder）

**用途**：定義 Agent 觀察到的帳戶狀態特徵

**內建實現**：
- `DefaultAccountFeatureBuilder`：4個基礎特徵（持倉比例、權益比例等）
- `ExtendedAccountFeatureBuilder`：8個擴展特徵（含槓桿使用率、回撤等）

**自定義步驟**：

```python
from Env.account_features import AccountFeatureBuilder
import numpy as np
from typing import List

class MyCustomFeatureBuilder(AccountFeatureBuilder):
    """自定義帳戶特徵構建器"""
    
    def feature_names(self) -> List[str]:
        """返回特徵名稱列表"""
        return [
            'my_feature_1',
            'my_feature_2',
            # ... 更多特徵
        ]
    
    def feature_count(self) -> int:
        """返回特徵數量"""
        return len(self.feature_names())
    
    def compute_features(
        self,
        *,
        position_size: float,
        position_value: float,
        equity: float,
        wallet_balance: float,
        current_price: float,
        initial_balance: float,
        leverage: float = 10.0,  # 可選參數
        **kwargs
    ) -> np.ndarray:
        """計算特徵值"""
        # 你的自定義邏輯
        feature_1 = equity / initial_balance
        feature_2 = position_value / equity if equity > 0 else 0.0
        
        return np.array([feature_1, feature_2], dtype=np.float32)
```

### 2. 獎勵計算器（RewardCalculator）

**用途**：定義每步的獎勵函數，引導 Agent 學習策略

**內建實現**：
- `RewardCalculator`：分層獎勵（止損 > 終局 > 風險 > 收益）

**自定義步驟**：

```python
from dataclasses import dataclass

@dataclass
class MyCustomRewardCalculator:
    """自定義獎勵計算器"""
    
    # 定義權重參數
    w_profit: float = 10.0
    w_risk: float = 5.0
    
    def compute(
        self,
        *,
        last_equity: float,
        new_equity: float,
        done: bool = False,
        termination_reason: str = None,
        margin_buffer: float = None,
        stop_loss_triggered: bool = False,
        **kwargs
    ) -> float:
        """計算獎勵"""
        reward = 0.0
        
        # 1. 利潤獎勵
        if last_equity > 1e-8:
            profit_rate = (new_equity - last_equity) / last_equity
            reward += profit_rate * self.w_profit
        
        # 2. 風險懲罰
        if margin_buffer is not None and margin_buffer < 0.5:
            risk_penalty = (0.5 - margin_buffer) * self.w_risk
            reward -= risk_penalty
        
        # 3. 終局懲罰/獎勵
        if done:
            if termination_reason == 'data_exhausted':
                reward += 100.0  # 成功獎勵
            else:
                reward -= 100.0  # 失敗懲罰
        
        return float(reward)
    
    def get_info(self) -> dict:
        """返回配置資訊"""
        return {
            'type': 'my_custom_reward',
            'weights': {
                'profit': self.w_profit,
                'risk': self.w_risk,
            }
        }
```

### 3. 資訊收集器（InfoCollector）

**用途**：收集訓練/評估過程中的統計資訊

**內建實現**：
- `DefaultInfoCollector`：基礎資訊（價格、持倉、報酬率等）
- `ExtendedInfoCollector`：擴展統計（最大回撤、夏普比率、勝率等）

**自定義步驟**：

```python
from Env.info_collector import InfoCollector
from typing import Dict, Any

class MyCustomInfoCollector(InfoCollector):
    """自定義資訊收集器"""
    
    def __init__(self):
        """初始化統計變數"""
        self.step_count = 0
        self.total_reward = 0.0
    
    def reset(self) -> None:
        """重置統計（可選）"""
        self.step_count = 0
        self.total_reward = 0.0
    
    def collect_step_info(
        self,
        *,
        current_step: int,
        current_price: float,
        position_size: float,
        equity: float,
        stop_loss_triggered: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """收集每步資訊"""
        self.step_count += 1
        
        return {
            'step': int(current_step),
            'price': float(current_price),
            'equity': float(equity),
            'my_custom_metric': self.step_count * 2,
        }
    
    def collect_episode_info(
        self,
        *,
        termination_reason: str,
        final_equity: float,
        initial_balance: float,
        episode_steps: int,
        executor: Any,
        stop_loss_count: int,
        liq_count: int,
        **kwargs
    ) -> Dict[str, Any]:
        """收集回合結束資訊"""
        profit_rate = ((final_equity - initial_balance) / initial_balance) * 100.0
        is_success = (termination_reason == 'data_exhausted')
        
        return {
            'termination_reason': str(termination_reason),
            'is_success': bool(is_success),
            'final_equity': float(final_equity),
            'profit_rate': float(profit_rate),
            'episode_steps': int(episode_steps),
            'my_custom_summary': f"Steps: {self.step_count}",
        }
```

## 使用範例

### 使用默認組件

```python
import pandas as pd
from Env import TradingEnvironment

# 載入數據
df = pd.read_csv('Data/BTCUSDT_futures_volume_5years_5min.csv')

# 創建環境（使用默認組件）
env = TradingEnvironment(
    df=df,
    initial_balance=10_000,
    leverage=10,
    window_size=288,  # 24小時
    max_liq_count=3,
)

# 訓練循環
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # 隨機動作
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

### 使用自定義組件

```python
from Env import (
    TradingEnvironment,
    ExtendedAccountFeatureBuilder,
    RewardCalculator,
    ExtendedInfoCollector,
)

# 創建自定義組件
feature_builder = ExtendedAccountFeatureBuilder()  # 8個特徵
reward_calculator = RewardCalculator(
    w_stop_loss=100.0,      # 調高止損懲罰
    w_terminal=150.0,       # 調高終局懲罰
    w_return=20.0,          # 調高收益獎勵
)
info_collector = ExtendedInfoCollector()  # 收集更多統計

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

print(f"觀測空間形狀: {env.observation_space.shape}")
# 輸出: 觀測空間形狀: (數據特徵數 + 8, 288)
```

### 使用完全自定義的組件

```python
from Env import TradingEnvironment

# 使用你自己實現的類別
env = TradingEnvironment(
    df=df,
    initial_balance=10_000,
    account_feature_builder=MyCustomFeatureBuilder(),
    reward_calculator=MyCustomRewardCalculator(),
    info_collector=MyCustomInfoCollector(),
)
```

## 終止條件

### 成功條件（返回 True, 'data_exhausted'）
- 資料耗盡（走完所有數據）

### 失敗條件
- **強平次數達上限**（返回 True, 'liq_limit'）：`episode_liq_count >= max_liq_count`
- **資金耗盡**（返回 True, 'balance_insufficient'）：`equity <= min_balance`

## 與 SAC 拉格朗日 Agent 的整合

```python
from stable_baselines3 import SAC

# 創建環境
env = TradingEnvironment(
    df=df,
    initial_balance=10_000,
    leverage=10,
    window_size=288,
    max_liq_count=3,
    # 自定義組件
    reward_calculator=RewardCalculator(
        w_stop_loss=50.0,
        w_terminal=120.0,
        w_return=12.0,
        w_risk=15.0,
    ),
)

# 訓練 SAC agent
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

## 調參建議

### 帳戶特徵
- **基礎訓練**：使用 `DefaultAccountFeatureBuilder`（4特徵）
- **進階訓練**：使用 `ExtendedAccountFeatureBuilder`（8特徵）
- **特殊需求**：繼承並添加領域特定特徵

### 獎勵函數
- **保守策略**：提高 `w_risk`、`w_terminal`
- **激進策略**：提高 `w_return`，降低 `w_risk`
- **避免止損**：提高 `w_stop_loss`

### 資訊收集
- **快速實驗**：使用 `DefaultInfoCollector`
- **詳細分析**：使用 `ExtendedInfoCollector`
- **自定義指標**：繼承並添加特定統計

## 常見問題

### Q: 如何調整觀測窗口大小？
A: 設定 `window_size` 參數。例如 24小時（5分K）= 288 步。

### Q: 如何修改強平次數上限？
A: 設定 `max_liq_count` 參數（默認 3 次）。

### Q: 如何添加自定義帳戶特徵？
A: 繼承 `AccountFeatureBuilder`，實現 `feature_names()` 和 `compute_features()` 方法。

### Q: 獎勵函數應該如何設計？
A: 建議分層設計：致命錯誤（止損、清算）> 風險控制 > 收益優化。

### Q: 環境是否支持多 GPU 訓練？
A: 是的，可配合 `stable_baselines3` 的 `VecEnv` 使用。

## 參考文件

- `account_features.py`：帳戶特徵構建器實現
- `reward.py`：獎勵計算器實現
- `info_collector.py`：資訊收集器實現
- `trading_env.py`：核心環境實現
- `trade_executor.py`：交易執行器（處理持倉、保證金、強平邏輯）

## 版本歷史

- **v2.0**（當前版本）：完全重構，支援依賴注入與可擴展設計
- **v1.0**：初始版本，功能耦合於單一類別

