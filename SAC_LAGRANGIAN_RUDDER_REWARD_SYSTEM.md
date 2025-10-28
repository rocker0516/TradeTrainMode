# SAC-Lagrangian + RUDDER Reward System

## 系統概述

本系統實現了基於 **SAC-Lagrangian（Constrained RL）** 和 **RUDDER（Return Decomposition for Delayed Rewards）** 的完整獎勵設計，專為槓桿交易環境設計。

### 核心目標

1. **存活到最後**（不得強平/資金不足）
2. **收益越大越好**（以可比較尺度）
3. **止損越小越好**（在同等收益下偏好較小回撤）
4. **鼓勵在相對極值進場**（低點做多/高點做空）

---

## 設計原則

### 1. 每步 Reward = Shaping Only（小額引導）

環境返回的 `reward` 僅包含小額的 **PBRS（Potential-Based Reward Shaping）** 和結構性懲罰：

- **PBRS 勢能差**：`r' = γΦ(s') - Φ(s)`
  - `Φ_survival`：margin_buffer 越高越好（遠離危險區）
  - `Φ_struct`：MAE/ATR 越小、距近期極值越遠越好
  - `Φ_extreme_entry`：持倉且接近相對極值時較高

- **風險懲罰（小額）**：margin_buffer 凸性懲罰（越危險越負）
- **結構性懲罰（小額）**：MAE/ATR 與距極值指標的合成
- **交易行為抑制**：每次進場小額負獎勵，同向連續進場遞增

**注意**：止損/強平的大額懲罰「不在 shaping」，只在 outcome（防止訊號混雜且與 RUDDER 一致）

### 2. Outcome 回填（RUDDER 機制）

出場事件（reduce/close/stop_loss/liq/forced_close）的大額 outcome 透過 `info['outcome_delta_to_entry']` 傳遞，由訓練端回填到該筆交易的「進場 transition」。

**Outcome 度量**（方案 B）：
```python
outcome_delta = w_outcome × clip(realized_pnl / notional, -1, 1)
```

**止損/強平額外懲罰**合併進 outcome（一起回填到進場步）：
- 止損：`-50.0`
- 強平：`-90.0`

**強制收斂**：若回合結束仍有持倉（`data_exhausted`），以最後價 `forced_close_on_done` 結算 outcome 並回填。

### 3. 成本/約束（Lagrangian）

每步成本 `step_cost`（手續費/滑點/資金費用）：不回溯，直接記在當步（供成本頭使用）。

**機率違規（chance constraint）**：
```python
cost_prob = 1 若本步止損或強平，否則 0
目標：E[cost_prob] ≤ δ (例如 3%)
```

**CVaR 尾損**：
```python
loss_cvar_sample = max(0, −outcome_delta)
目標：CVaR_α ≤ b_cvar (例如 1%)
```

---

## 訓練端兩種路徑

### 路徑 1：HER 開啟（使用 Wrapper）

使用外層 `RiskPenaltyWrapper` 施加懲罰：
```python
reward' = reward - λ_prob * cost_prob - λ_cvar * loss_cvar_sample
```

λ 由 `LagrangianController` 在線更新（梯度上升）：
```python
λ ← λ + lr * (constraint_violation)
λ ← clip(λ, λ_min, λ_max)
```

### 路徑 2：HER 關閉（SACLagrangian 版）

使用 `SACLagrangian` 演算法（成本 Q-head + 演員懲罰），更完整地把成本進入 actor 損失。

---

## Env → Agent 的 Info 協議

### 關鍵欄位

| 欄位 | 類型 | 說明 |
|------|------|------|
| `is_entry` | bool | 是否為進場事件（開倉/加倉） |
| `is_reduce` | bool | 是否為減倉事件（部分平倉） |
| `is_exit` | bool | 是否為出場事件（完全平倉） |
| `trade_id` | int | 當前持倉的交易 ID（-1 表示無倉位） |
| `entered_trade_id` | int | 本步進場的交易 ID（若 is_entry=True） |
| `exited_trade_id` | int | 本步出場的交易 ID（若 is_exit/is_reduce=True） |
| `exit_reason` | str | 出場原因：'close' / 'reduce' / 'stop_loss' / 'liq' / 'forced_close_on_done' |
| `outcome_delta_to_entry` | float | 本步出場事件產生的 outcome 增量（回填到進場步） |
| `step_cost` | float | 本步總成本（手續費 + 滑點 + 資金費用） |
| `cost_prob` | float | 機率違規成本（1 若本步止損/強平，否則 0） |
| `loss_cvar_sample` | float | CVaR 尾損樣本（max(0, -outcome_delta)） |
| `stop_loss_triggered` | bool | 本步是否觸發止損 |

---

## 使用範例

### 1. 創建環境（啟用 RUDDER）

```python
from Env.trading_env import TradingEnvironment
import pandas as pd

# 加載數據
df = pd.read_csv("Data/BTCUSDT_futures_volume_5years_5min.csv")

# 創建環境（啟用 RUDDER）
env = TradingEnvironment(
    df=df,
    initial_balance=10000,
    leverage=10,
    use_rudder=True,  # 啟用 RUDDER 模式
    random_start=True,
)

# 環境會自動使用新的 reward system
obs, info = env.reset()
```

### 2. 使用 RUDDER Replay Buffer

```python
from Train.utils.rudder_replay_buffer import RUDDERReplayBuffer

# 創建 buffer
buffer = RUDDERReplayBuffer(
    capacity=100000,
    observation_shape=env.observation_space.shape,
    action_dim=env.action_space.shape[0],
    device='cuda',
)

# 收集經驗（自動處理 RUDDER 回填）
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        
        # 添加到 buffer（RUDDER 回填會自動處理）
        buffer.add(obs, action, reward, next_obs, done, info)
        
        obs = next_obs

# 採樣訓練
if buffer.is_ready(batch_size=256):
    batch = buffer.sample(256)
    # batch 包含：
    # - 'reward': shaping reward only
    # - 'outcome_delta': 回填的 outcome（進場步非零）
    # - 'step_cost', 'cost_prob', 'loss_cvar_sample': 成本約束
```

### 3. 使用 Risk Penalty Wrapper（路徑 1）

```python
from Train.utils.wrappers import RiskPenaltyWrapper, InfoLoggerWrapper
from Train.utils.lagrangian import LagrangianController

# Wrap 環境
env = TradingEnvironment(df, use_rudder=True)
env = RiskPenaltyWrapper(env, lambda_prob=1.0, lambda_cvar=1.0)
env = InfoLoggerWrapper(env)

# 創建 Lagrangian 控制器
lagrangian = LagrangianController(
    lambda_prob_init=1.0,
    lambda_cvar_init=1.0,
    lr_prob=0.01,
    lr_cvar=0.01,
    target_prob=0.03,  # 3% 止損/強平率
    target_cvar=0.01,  # 1% CVaR 上界
)

# 訓練循環
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        # reward 已經包含 Lagrangian 懲罰
        
        buffer.add(obs, action, reward, next_obs, done, info)
        obs = next_obs
    
    # 每個 episode 結束後更新 λ
    if buffer.is_ready(256):
        batch = buffer.sample(256)
        stats = lagrangian.update(
            cost_prob_batch=batch['cost_prob'].cpu().numpy(),
            loss_cvar_batch=batch['loss_cvar_sample'].cpu().numpy(),
        )
        
        # 更新 wrapper 的 λ
        env.update_lambdas(
            lambda_prob=stats['lambda_prob'],
            lambda_cvar=stats['lambda_cvar'],
        )
```

### 4. 向後兼容（舊版 Reward）

```python
# 若不想使用 RUDDER，設定 use_rudder=False
env = TradingEnvironment(
    df=df,
    use_rudder=False,  # 使用舊版整合 reward
    reward_calculator=None,  # 使用預設
)

# 環境會使用舊版 RewardCalculator（整合所有獎勵）
```

---

## 參數建議（可微調）

### PBRS 勢能
- `γ = 0.99`
- `w_phi_survival = 0.5`
- `w_phi_struct = 0.2`
- `w_phi_extreme_entry = 0.3`

### Shaping 權重
- `w_risk = 20.0`（風險懲罰）
- `w_struct = 6.0`（結構性懲罰）
- `w_entry = 0.6`（進場懲罰）
- `w_entry_streak = 0.3`（連續進場遞增）

### Outcome 權重
- `w_outcome = 1.0`（需根據學習穩定性微調）
- `stop_loss_penalty = 50.0`
- `liq_penalty = 90.0`

### 風險尺度
- `risk_threshold = 0.85`（安全區上界）
- `risk_danger = 0.25`（危險區下界）
- `mae_atr_scale = 3.0`（MAE 達 3 ATR → -1）
- `extreme_safe_atr = 3.0`（距極值 3 ATR 視為安全）

### Lagrangian 目標
- `target_prob (δ) ∈ [1%, 5%]`（機率違規目標）
- `target_cvar (b_cvar) ∈ [0.5%, 1%]`（CVaR 目標）
- `lr_prob = 0.01`（λ_prob 學習率）
- `lr_cvar = 0.01`（λ_cvar 學習率）

---

## 邊界行為

### 反手（同步平舊開新）
先對舊倉 `exit` 事件回填 outcome，再註冊新倉 `entry` 索引（當步兩者皆有）。

### 回合結束仍持倉
`forced_close_on_done`，回填 outcome，避免跨回合外洩。

---

## 測試

運行測試腳本以驗證系統完整性：

```bash
python test_rudder_reward_system.py
```

測試涵蓋：
1. PBRS 勢能計算與 shaping reward
2. Outcome 計算與回填機制
3. 成本計算（step_cost, cost_prob, CVaR）
4. RUDDER Replay Buffer 的回填邏輯
5. Lagrangian 控制器更新
6. 環境 info 協議完整性
7. Wrapper 功能

---

## 檔案結構

```
├── Env/
│   ├── reward.py                    # 新版 reward system（PBRS + Outcome + Cost）
│   ├── trading_env.py               # 環境（支援 RUDDER info 協議）
│   ├── trade_executor.py            # 執行器（追蹤 trade_id 與 exit_reason）
│   └── features.py                  # 特徵工程
│
├── Train/utils/
│   ├── rudder_replay_buffer.py      # RUDDER Replay Buffer（支援回填）
│   ├── wrappers.py                  # 環境 Wrapper（RiskPenalty, InfoLogger）
│   ├── lagrangian.py                # Lagrangian 控制器（在線更新 λ）
│   ├── logger.py                    # 訓練日誌記錄器
│   └── replay_buffer.py             # 基礎 Replay Buffer（向後兼容）
│
└── test_rudder_reward_system.py     # 完整測試腳本
```

---

## 架構圖

```
┌─────────────────────────────────────────────────────────────┐
│                     TradingEnvironment                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  TradeExecutor (追蹤 trade_id, exit_reason)         │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Reward System (use_rudder=True)                     │   │
│  │  • ShapingCalculator (PBRS + 小額懲罰)              │   │
│  │  • OutcomeCalculator (大額回填)                     │   │
│  │  • CostCalculator (step_cost, cost_prob, CVaR)      │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  info = {is_entry, is_exit, outcome_delta, step_cost, ...}  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  RiskPenaltyWrapper (可選)                   │
│  reward' = reward - λ_prob * cost_prob - λ_cvar * tail_loss │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  RUDDERReplayBuffer                          │
│  • 追蹤 trade_id → entry_step_idx                           │
│  • 回填 outcome_delta 到進場步                              │
│  • 儲存 step_cost, cost_prob, loss_cvar_sample             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  SAC-Lagrangian Agent                        │
│  • 使用 shaping + outcome_delta 訓練價值網路               │
│  • 使用 step_cost / cost_prob 訓練成本頭                   │
│  • LagrangianController 動態調整 λ                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 總結

本系統實現了完整的 **SAC-Lagrangian + RUDDER** 獎勵設計，具備以下優勢：

1. **延遲獎勵問題解決**：透過 RUDDER 回填機制，大額 outcome 精確回填到進場步，避免信用分配問題。

2. **約束滿足**：透過 Lagrangian 方法，在線調整懲罰強度，確保止損率與尾損控制在目標範圍內。

3. **訓練穩定性**：每步只使用小額 shaping reward，避免大幅度 reward 波動影響訓練。

4. **可擴展性**：支援兩種訓練路徑（Wrapper 版與 SACLagrangian 版），可根據需求選擇。

5. **向後兼容**：保留舊版 RewardCalculator，現有訓練腳本可無痛遷移。

6. **完整測試**：提供完整的測試腳本，驗證所有組件功能正常。

---

## 下一步

1. **整合到訓練腳本**：將 RUDDER replay buffer 和 Lagrangian 控制器整合到 `Train/train_sac.py`。

2. **超參數調優**：根據實際訓練效果，調整 PBRS 權重、outcome 權重、Lagrangian 學習率等。

3. **監控與記錄**：使用 `TrainingLogger` 記錄 λ 變化、約束違規率、outcome 分佈等關鍵指標。

4. **對比實驗**：對比 RUDDER 版與舊版的訓練效果，驗證改進效果。

---

**作者**：SAC-Lagrangian + RUDDER Reward System Team  
**版本**：v1.0.0  
**日期**：2025-11-01

