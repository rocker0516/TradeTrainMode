# Quick Start: SAC-Lagrangian + RUDDER Reward System

## 快速開始指南

這份指南將帶你快速上手新的 SAC-Lagrangian + RUDDER 獎勵系統。

---

## 步驟 1：驗證系統完整性

首先運行測試腳本，確保所有組件正常工作：

```bash
python test_rudder_reward_system.py
```

**預期輸出**：
```
[PASS] 測試 1 通過：所有獎勵組件正常工作
[PASS] 測試 2 通過：環境 info 協議完整
[PASS] 測試 3 通過：RUDDER 回填邏輯正常
[PASS] 測試 4 通過：Lagrangian 控制器更新正常
[PASS] 測試 5 通過：Wrapper 正常工作
[SUCCESS] 所有測試通過！
```

---

## 步驟 2：基礎使用（不使用 Wrapper）

### 創建環境

```python
import pandas as pd
from Env.trading_env import TradingEnvironment
from Train.utils.rudder_replay_buffer import RUDDERReplayBuffer

# 加載數據
df = pd.read_csv("Data/BTCUSDT_futures_volume_5years_5min.csv")

# 創建環境（啟用 RUDDER）
env = TradingEnvironment(
    df=df,
    initial_balance=10000,
    transaction_fee=0.001,
    leverage=10,
    use_rudder=True,      # 啟用 RUDDER 模式
    random_start=True,
)

# 創建 RUDDER Replay Buffer
buffer = RUDDERReplayBuffer(
    capacity=100000,
    observation_shape=env.observation_space.shape,
    action_dim=env.action_space.shape[0],
    device='cuda',  # or 'cpu'
)
```

### 收集經驗

```python
import numpy as np

# 收集一個 episode
obs, info = env.reset()
done = False

while not done:
    # 隨機動作（實際應使用訓練好的 agent）
    action = np.array([np.random.uniform(-1, 1)])
    
    next_obs, reward, done, truncated, info = env.step(action)
    
    # 添加到 buffer（RUDDER 回填自動處理）
    buffer.add(obs, action, reward, next_obs, done, info)
    
    # 檢查進場/出場事件
    if info.get('is_entry', False):
        print(f"進場：trade_id={info['entered_trade_id']}")
    
    if info.get('is_exit', False):
        print(f"出場：trade_id={info['exited_trade_id']}, "
              f"exit_reason={info['exit_reason']}, "
              f"outcome={info['outcome_delta_to_entry']:.2f}")
    
    obs = next_obs

# 查看 buffer 統計
stats = buffer.get_stats()
print(f"Buffer size: {stats['size']}")
print(f"Mean outcome: {stats['mean_outcome']:.4f}")
print(f"Total outcomes backfilled: {stats['total_outcomes_backfilled']}")
```

### 採樣訓練

```python
if buffer.is_ready(batch_size=256):
    batch = buffer.sample(256)
    
    # batch 包含：
    # - 'state', 'action', 'next_state', 'done'
    # - 'reward': shaping reward only（小額）
    # - 'outcome_delta': 回填的 outcome（進場步非零）
    # - 'step_cost': 手續費等成本
    # - 'cost_prob': 機率違規（0/1）
    # - 'loss_cvar_sample': CVaR 尾損樣本
    
    # 組合 total reward（用於訓練 Q-network）
    total_reward = batch['reward'] + batch['outcome_delta']
    
    # 使用 total_reward 訓練你的 SAC agent
    # agent.update(batch, total_reward)
```

---

## 步驟 3：進階使用（使用 Wrapper + Lagrangian）

### 添加 Wrapper

```python
from Train.utils.wrappers import RiskPenaltyWrapper, InfoLoggerWrapper
from Train.utils.lagrangian import LagrangianController

# 創建環境
env = TradingEnvironment(df, use_rudder=True)

# 添加 Wrapper
env = RiskPenaltyWrapper(
    env,
    lambda_prob=1.0,   # 初始 λ（機率違規）
    lambda_cvar=1.0,   # 初始 λ（CVaR 尾損）
)
env = InfoLoggerWrapper(env)  # 自動記錄 episode 統計

# 創建 Lagrangian 控制器
lagrangian = LagrangianController(
    lambda_prob_init=1.0,
    lambda_cvar_init=1.0,
    lr_prob=0.01,          # λ 學習率
    lr_cvar=0.01,
    target_prob=0.03,      # 目標：3% 止損/強平率
    target_cvar=0.01,      # 目標：1% CVaR 上界
    lambda_min=0.001,
    lambda_max=100.0,
)
```

### 訓練循環

```python
num_episodes = 1000
batch_size = 256
update_freq = 10  # 每 10 個 episode 更新一次 λ

for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        # 選擇動作
        action = agent.select_action(obs)  # 你的 SAC agent
        
        # 執行步驟
        next_obs, reward, done, truncated, info = env.step(action)
        # reward 已經包含 Lagrangian 懲罰
        
        # 添加到 buffer
        buffer.add(obs, action, reward, next_obs, done, info)
        
        # 訓練 agent
        if buffer.is_ready(batch_size):
            batch = buffer.sample(batch_size)
            agent.update(batch)
        
        obs = next_obs
        episode_reward += reward
    
    # 每 N 個 episode 更新 λ
    if (episode + 1) % update_freq == 0 and buffer.is_ready(batch_size):
        # 採樣批次成本
        batch = buffer.sample(batch_size)
        
        # 更新 Lagrangian 乘子
        stats = lagrangian.update(
            cost_prob_batch=batch['cost_prob'].cpu().numpy(),
            loss_cvar_batch=batch['loss_cvar_sample'].cpu().numpy(),
        )
        
        # 更新 wrapper 的 λ
        env.update_lambdas(
            lambda_prob=stats['lambda_prob'],
            lambda_cvar=stats['lambda_cvar'],
        )
        
        print(f"[Episode {episode+1}] "
              f"λ_prob={stats['lambda_prob']:.4f}, "
              f"λ_cvar={stats['lambda_cvar']:.4f}, "
              f"violation_prob={stats['violation_prob']:.4f}, "
              f"violation_cvar={stats['violation_cvar']:.4f}")
    
    # 記錄 episode 統計
    if 'episode_stats' in info:
        stats = info['episode_stats']
        print(f"[Episode {episode+1}] "
              f"reward={episode_reward:.2f}, "
              f"entries={stats['entry_count']}, "
              f"exits={stats['exit_count']}, "
              f"stop_losses={stats['stop_loss_count']}, "
              f"total_outcome={stats['total_outcome']:.2f}")
```

---

## 步驟 4：監控與日誌

### 使用 TrainingLogger

```python
from Train.utils.logger import TrainingLogger

# 創建日誌記錄器
logger = TrainingLogger(
    log_dir='logs',
    experiment_name='sac_lagrangian_rudder',
    enable_tensorboard=True,
    enable_csv=True,
)

# 訓練循環中記錄指標
for episode in range(num_episodes):
    # ... 訓練代碼 ...
    
    # 記錄 episode 指標
    logger.log_scalars({
        'episode_reward': episode_reward,
        'episode_length': episode_length,
        'lambda_prob': lagrangian.lambda_prob,
        'lambda_cvar': lagrangian.lambda_cvar,
        'mean_outcome': stats['total_outcome'] / max(1, stats['exit_count']),
        'stop_loss_rate': stats['stop_loss_count'] / max(1, stats['exit_count']),
    }, step=episode, prefix='train/')

# 訓練結束後關閉日誌
logger.close()
```

### 查看 TensorBoard

```bash
tensorboard --logdir=logs/sac_lagrangian_rudder/tensorboard
```

---

## 步驟 5：調整參數

### 調整 PBRS 權重（如果訓練不穩定）

```python
from Env.reward import PBRSPotentialCalculator, ShapingRewardCalculator

# 創建自訂的 shaping calculator
potential_calc = PBRSPotentialCalculator(
    w_phi_survival=0.6,        # 增加存活權重
    w_phi_struct=0.2,
    w_phi_extreme_entry=0.2,   # 降低極值進場權重
    gamma=0.99,
)

shaping_calc = ShapingRewardCalculator(
    potential_calc=potential_calc,
    w_risk=15.0,               # 降低風險懲罰（如果過於保守）
    w_struct=5.0,
    w_entry=0.8,               # 增加進場懲罰（如果過度交易）
    w_entry_streak=0.5,        # 增加連續進場懲罰
)

# 手動構建 reward system
from Env.reward import OutcomeCalculator, CostCalculator
outcome_calc = OutcomeCalculator(w_outcome=1.5)  # 增加 outcome 強度
cost_calc = CostCalculator()

# 替換環境中的 calculator
env.shaping_calc = shaping_calc
env.outcome_calc = outcome_calc
env.cost_calc = cost_calc
```

### 調整 Lagrangian 目標

```python
# 如果止損率過高，降低目標
lagrangian = LagrangianController(
    target_prob=0.02,      # 2% 止損/強平率（更嚴格）
    target_cvar=0.005,     # 0.5% CVaR 上界（更嚴格）
    lr_prob=0.02,          # 增加學習率（更快收斂）
    lr_cvar=0.02,
)

# 如果止損率過低（過於保守），提高目標
lagrangian = LagrangianController(
    target_prob=0.05,      # 5% 止損/強平率（更寬鬆）
    target_cvar=0.02,      # 2% CVaR 上界（更寬鬆）
    lr_prob=0.005,         # 降低學習率（更穩定）
    lr_cvar=0.005,
)
```

---

## 步驟 6：常見問題排查

### 1. 訓練不穩定 / Reward 波動過大

**原因**：outcome 權重過大或 PBRS 權重不平衡。

**解決方案**：
- 降低 `w_outcome`（例如從 1.0 降到 0.5）
- 增加 `w_phi_survival`（提高存活勢能）
- 使用 `NormalizeRewardWrapper` 正規化 reward

```python
from Train.utils.wrappers import NormalizeRewardWrapper
env = NormalizeRewardWrapper(env, gamma=0.99)
```

### 2. Agent 過度保守 / 很少交易

**原因**：風險懲罰過大或進場懲罰過大。

**解決方案**：
- 降低 `w_risk`（例如從 20.0 降到 10.0）
- 降低 `w_entry`（例如從 0.6 降到 0.3）
- 提高 Lagrangian 目標（允許更高止損率）

### 3. Agent 過度交易

**原因**：進場懲罰太小或 outcome 權重過大（追求收益）。

**解決方案**：
- 增加 `w_entry`（例如從 0.6 增加到 1.0）
- 增加 `w_entry_streak`（例如從 0.3 增加到 0.5）
- 降低 `w_outcome`（降低收益誘因）

### 4. 止損率無法收斂到目標

**原因**：Lagrangian 學習率不當或目標設定不合理。

**解決方案**：
- 檢查 buffer 中的 `cost_prob_rate`（實際止損率）
- 調整 `lr_prob`（如果收斂太慢，增加學習率）
- 調整 `target_prob`（如果目標過於嚴格，放寬目標）
- 檢查 `lambda_prob` 是否達到上界（`lambda_max`）

### 5. RUDDER 回填不正確

**原因**：trade_id 追蹤錯誤或出場信息未傳遞。

**解決方案**：
- 檢查 `info` 中的 `entered_trade_id` 和 `exited_trade_id`
- 檢查 `buffer.get_stats()` 中的 `total_outcomes_backfilled`
- 確保 `forced_close_on_done` 在 episode 結束時被調用

---

## 步驟 7：與舊版對比

如果你想對比新舊版本的效果：

```python
# 舊版（use_rudder=False）
env_old = TradingEnvironment(df, use_rudder=False)

# 新版（use_rudder=True）
env_new = TradingEnvironment(df, use_rudder=True)

# 使用相同的 agent 分別訓練，對比：
# 1. 收斂速度
# 2. 最終收益
# 3. 止損率
# 4. 訓練穩定性
```

---

## 下一步

1. **閱讀完整文檔**：查看 `SAC_LAGRANGIAN_RUDDER_REWARD_SYSTEM.md` 了解詳細設計。

2. **整合到訓練腳本**：將 RUDDER buffer 和 Lagrangian 控制器整合到 `Train/train_sac.py`。

3. **超參數搜索**：使用 Optuna 或 Ray Tune 自動搜索最佳參數。

4. **回測與評估**：使用訓練好的 agent 在測試集上回測，評估真實表現。

---

## 聯繫與支持

如有問題或建議，請查看：
- 完整文檔：`SAC_LAGRANGIAN_RUDDER_REWARD_SYSTEM.md`
- 測試腳本：`test_rudder_reward_system.py`
- 範例代碼：`example_usage.py`（即將添加）

---

**Happy Trading! 🚀**

