# Train.py 執行步驟與原理詳解

## 📋 目錄
1. [整體架構](#整體架構)
2. [執行流程](#執行流程)
3. [核心組件原理](#核心組件原理)
4. [SAC 算法原理](#sac-算法原理)
5. [技術細節](#技術細節)

---

## 🏗️ 整體架構

```
train.py 執行流程
│
├── 1. 數據準備
│   ├── 讀取 CSV 歷史交易數據
│   ├── 時間切片（取最近 N 個月）
│   └── 數據分片（多環境並行）
│
├── 2. 環境構建
│   ├── 創建 TradingEnvironment（交易模擬器）
│   ├── 包裝多層 Wrapper
│   │   ├── TimeLimit（時間限制）
│   │   ├── ActionTransformWrapper（動作映射）
│   │   ├── EpisodeStatsWrapper（統計收集）
│   │   ├── RewardLogWrapper（逐步日志）
│   │   └── Monitor（Stable-Baselines3 監控）
│   └── 向量化環境（並行訓練）
│
├── 3. SAC 模型訓練
│   ├── 初始化 SAC 算法
│   ├── 經驗回放緩衝區
│   ├── 策略網絡 + Q 網絡訓練
│   └── 自動調整熵係數
│
├── 4. 結果輸出
│   ├── 保存訓練好的模型
│   ├── 生成訓練曲線
│   ├── 生成統計圖表
│   └── 輸出 CSV 日誌
│
└── 5. 可視化分析
    ├── 回合收益率趨勢
    ├── 資金曲線
    ├── 回撤分析
    └── 交易次數統計
```

---

## 🔄 執行流程（步驟詳解）

### 階段 1：命令行參數解析與初始化

```python
# 第 478-518 行
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', ...)           # 數據文件路徑
    parser.add_argument('--months', default=3)  # 使用最近 N 個月數據
    parser.add_argument('--vec_envs', default=16)  # 並行環境數量
    parser.add_argument('--total_timesteps', default=100000)  # 總訓練步數
    ...
```

**關鍵參數**：
- `--csv`: 歷史交易數據（OHLCV + 成交量指標）
- `--vec_envs`: 並行環境數（提升訓練效率）
- `--initial_balance`: 初始資金（默認 10000 USDT）
- `--leverage`: 槓桿倍數（默認 10x）
- `--position_scale`: 倉位限制（默認 0.5，即最大 50% 倉位）

---

### 階段 2：數據準備

#### 2.1 讀取並切片數據（第 28-51 行）

```python
def read_csv_last_months(csv_path: str, months: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').set_index('timestamp')
    
    # 取最近 N 個月的數據
    end_time = df.index.max()
    start_time = end_time - pd.DateOffset(months=months)
    df_slice = df.loc[df.index >= start_time].copy()
    return df_slice
```

**原理**：
- 只使用最近的數據進行訓練（避免過早的市場特徵）
- 時間索引排序確保數據連續性
- 檢查必要欄位：`open, high, low, close, volume, buy_volume, sell_volume, ...`

#### 2.2 數據分片（第 54-75 行）

```python
def split_dataframe_into_shards(df: pd.DataFrame, num_shards: int, min_len: int):
    # 將完整數據切分為 N 份，每份供一個並行環境使用
    shard_size = n // num_shards
    shards = []
    for i in range(num_shards):
        start = i * shard_size
        end = (i + 1) * shard_size if i < num_shards - 1 else n
        shards.append(df.iloc[start:end])
    return shards
```

**原理**：
- **數據多樣性**：每個環境看到不同時間段的數據
- **並行訓練**：16 個環境同時運行，提升樣本採集效率
- **最小長度保護**：確保每個分片至少包含 `window_size + 500` 筆數據

---

### 階段 3：環境構建（多層 Wrapper 設計）

#### 3.1 環境包裝順序（第 232-256 行）

```python
def make_env_fn(...):
    # 從內到外的包裝順序（重要！）
    env = TradingEnvironment(...)           # 核心交易環境
    env = TimeLimit(env, ...)               # 時間限制
    env = ActionTransformWrapper(env, ...)  # 動作映射
    env = EpisodeStatsWrapper(env, ...)     # 統計收集
    env = RewardLogWrapper(env, ...)        # 日誌記錄
    env = Monitor(env)                      # SB3 監控
    return env
```

**Wrapper 原理（洋蔥模型）**：
```
┌─────────────────────────────┐
│      Monitor (最外層)        │
│  ┌──────────────────────┐   │
│  │   RewardLogWrapper   │   │
│  │ ┌─────────────────┐  │   │
│  │ │ EpisodeStats... │  │   │
│  │ │ ┌────────────┐  │  │   │
│  │ │ │ ActionTran │  │  │   │
│  │ │ │ ┌────────┐ │  │  │   │
│  │ │ │ │TimeLimit│ │  │  │   │
│  │ │ │ │┌──────┐│ │  │  │   │
│  │ │ │ ││Trading││ │  │  │   │  ← 核心環境
│  │ │ │ ││  Env ││ │  │  │   │
│  │ │ │ │└──────┘│ │  │  │   │
└──┴─┴─┴─┴────────┴─┴──┴──┴───┘
```

#### 3.2 關鍵 Wrapper 原理

##### **ActionTransformWrapper（動作映射）**
第 111-136 行

```python
class ActionTransformWrapper(gym.Wrapper):
    """將 SAC 輸出的連續動作 [-1, 1] 映射到環境需求"""
    
    def step(self, action):
        # SAC 輸出 3 個連續值，每個 ∈ [-1, 1]
        a0, a1, a2 = action[0], action[1], action[2]
        
        # 映射規則：
        # a0: 倉位比例 [-1,1] → [-0.5, 0.5] (position_scale=0.5)
        #     -0.5 = 50% 空倉, 0 = 空倉, +0.5 = 50% 多倉
        a0_scaled = np.clip(a0, -1, 1) * self.position_scale
        
        # a1: 止盈百分比 [-1,1] → [0, 50]
        #     -1 → 0%, +1 → 50%
        a1_mapped = (np.clip(a1, -1, 1) + 1) * 0.5 * 50.0
        
        # a2: 止損百分比 [-1,1] → [0, 20]
        #     -1 → 0%, +1 → 20%
        a2_mapped = (np.clip(a2, -1, 1) + 1) * 0.5 * 20.0
        
        return self.env.step([a0_scaled, a1_mapped, a2_mapped])
```

**為什麼需要映射？**
- SAC 輸出範圍固定為 `[-1, 1]`
- 交易環境需要具體數值（倉位比例、止盈/止損百分比）
- 映射確保動作在合理範圍內（風險控制）

##### **EpisodeStatsWrapper（統計收集）**
第 138-227 行

```python
class EpisodeStatsWrapper(gym.Wrapper):
    """收集每回合的關鍵指標"""
    
    def reset(self, **kwargs):
        # 每回合開始時初始化統計
        self._episode_reward = 0.0
        self._min_total_value = current_tv
        self._peak_total_value = current_tv
        self._max_drawdown = 0.0
        self._trade_count = 0
        ...
    
    def step(self, action):
        # 每步更新統計
        self._episode_reward += reward
        
        # 實時追蹤最大回撤
        dd = (peak - current) / peak
        if dd > self._max_drawdown:
            self._max_drawdown = dd
        
        # 計算交易次數（倉位變化 = 交易）
        if current_position != last_position:
            self._trade_count += 1
        
        # 回合結束時寫入 CSV
        if done or truncated:
            self.log_file.write(f"{env_id},{episode},{steps},...")
```

**收集的指標**：
- `end_total_value`: 期末總資產
- `min_total_value`: 期間最低資產
- `max_drawdown`: 最大回撤百分比
- `episode_reward`: 回合累積獎勵
- `return_rate`: 收益率 (end - initial) / initial
- `trade_count`: 交易次數
- `aggressive_step_ratio`: 激進操作比例
- `done_reason`: 結束原因（time_limit, balance_insufficient, ...）

##### **RewardLogWrapper（逐步日誌）**
第 78-109 行

```python
class RewardLogWrapper(gym.Wrapper):
    """記錄每一步的 reward 和帳戶狀態"""
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # 從最底層環境讀取狀態（unwrapped 跳過所有 wrapper）
        base_env = self.env.unwrapped
        total_value = base_env.total_value
        balance = base_env.balance
        btc_held = base_env.btc_held
        
        # 寫入 CSV：step, reward, total_value, balance, btc_held, current_step
        self.log_file.write(f"{step},{reward},{total_value},...")
```

**為什麼需要 `env.unwrapped`？**
```
action → Monitor → RewardLog → EpisodeStats → ActionTransform → TimeLimit → TradingEnv
                      ↑                                                        ↑
                  想讀取狀態                                            實際狀態在這裡
```
- 多層 Wrapper 嵌套導致 `self.env` 只是下一層 Wrapper
- `unwrapped` 直接訪問最底層的 `TradingEnvironment`

---

### 階段 4：向量化環境（並行訓練）

第 533-544 行

```python
# 創建 16 個環境函數
env_fns = []
for i in range(args.vec_envs):
    env_fns.append(make_env_fn(shards[i], ...))

# 並行化環境
vec_env = SubprocVecEnv(env_fns)  # 多進程並行
# 或
vec_env = DummyVecEnv(env_fns)    # 單進程順序執行

vec_env = VecMonitor(vec_env, ...)  # 向量化監控
```

**SubprocVecEnv vs DummyVecEnv**：

| 特性 | SubprocVecEnv | DummyVecEnv |
|------|--------------|-------------|
| 並行方式 | 多進程（真並行） | 單進程（順序執行） |
| 訓練速度 | 快（CPU 多核利用） | 慢 |
| 內存佔用 | 高（每個進程獨立） | 低 |
| 調試難度 | 高（進程間通信） | 低 |
| 適用場景 | 生產訓練 | 調試測試 |

**並行原理**：
```
主進程
├── 子進程 1 → Env 0 → 數據分片 0 → 採集樣本
├── 子進程 2 → Env 1 → 數據分片 1 → 採集樣本
├── 子進程 3 → Env 2 → 數據分片 2 → 採集樣本
...
└── 子進程 16 → Env 15 → 數據分片 15 → 採集樣本
          ↓
    合併所有樣本到經驗回放緩衝區
          ↓
    SAC 算法統一訓練神經網絡
```

---

## 🧠 SAC 算法原理

### SAC (Soft Actor-Critic) 核心概念

第 553-569 行

```python
model = SAC(
    policy='MlpPolicy',              # 多層感知機策略
    env=vec_env,                     # 向量化環境
    learning_rate=3e-4,              # 學習率
    buffer_size=1_000_000,           # 經驗回放緩衝區大小
    batch_size=256,                  # 每次訓練批次大小
    tau=0.005,                       # 目標網絡軟更新係數
    gamma=0.99,                      # 折扣因子
    train_freq=(1, 'step'),          # 每步訓練一次
    gradient_steps=1,                # 每次訓練的梯度步數
    target_entropy='auto',           # 自動調整目標熵
    use_sde=True,                    # 使用狀態依賴探索
    policy_kwargs=dict(
        net_arch=dict(pi=[256,256], qf=[256,256])  # 網絡架構
    )
)
```

### SAC 三大組件

```
┌─────────────────────────────────────────┐
│           SAC 算法架構                   │
├─────────────────────────────────────────┤
│                                         │
│  1. Actor (策略網絡 π)                  │
│     ┌──────────────────────────┐       │
│     │ 輸入: 觀察 (obs)          │       │
│     │ 輸出: 動作分佈 (μ, σ)     │       │
│     │ 架構: [256, 256] → 3維動作 │       │
│     └──────────────────────────┘       │
│              ↓                          │
│     採樣動作 a ~ N(μ, σ)                │
│                                         │
│  2. Critic (價值網絡 Q)                 │
│     ┌──────────────────────────┐       │
│     │ Q1(s,a): [256, 256] → 1  │       │
│     │ Q2(s,a): [256, 256] → 1  │       │
│     │ (雙 Q 網絡減少過估計)     │       │
│     └──────────────────────────┘       │
│                                         │
│  3. Temperature (熵係數 α)              │
│     ┌──────────────────────────┐       │
│     │ 自動調整探索/利用平衡     │       │
│     │ 高 α → 更多探索           │       │
│     │ 低 α → 更多利用           │       │
│     └──────────────────────────┘       │
└─────────────────────────────────────────┘
```

### SAC 訓練循環

```python
for step in range(total_timesteps):
    # 1. 環境交互（16 個並行環境）
    actions = model.predict(observations)  # Actor 輸出動作
    next_obs, rewards, dones, infos = vec_env.step(actions)
    
    # 2. 存入經驗回放緩衝區
    replay_buffer.add(obs, actions, rewards, next_obs, dones)
    
    # 3. 從緩衝區採樣訓練（每步執行）
    if step >= learning_starts:
        batch = replay_buffer.sample(batch_size=256)
        
        # 更新 Critic（Q 網絡）
        q_loss = compute_q_loss(batch)
        q_optimizer.step()
        
        # 更新 Actor（策略網絡）
        policy_loss = compute_policy_loss(batch)
        policy_optimizer.step()
        
        # 更新熵係數 α
        alpha_loss = compute_alpha_loss(batch)
        alpha_optimizer.step()
        
        # 軟更新目標網絡
        for param, target_param in zip(q_net, target_q_net):
            target_param = tau * param + (1 - tau) * target_param
```

### 關鍵參數解釋

#### **gamma=0.99（折扣因子）**
```python
# 未來獎勵的重要性
Q(s,a) = r + γ * max Q(s',a')
```
- γ = 0.99：非常重視未來（長期策略）
- γ = 0.0：只看當前（短視策略）
- 交易場景適合高 γ（長期收益優先）

#### **tau=0.005（軟更新係數）**
```python
# 目標網絡更新速度
target = 0.005 * current + 0.995 * target
```
- 防止 Q 值過度波動
- 訓練更穩定

#### **use_sde=True（狀態依賴探索）**
```python
# 傳統探索：a = μ(s) + ε (ε 固定噪聲)
# SDE 探索：a = μ(s) + σ(s) * ε (σ 依賴狀態)
```
- 不同市場狀態使用不同探索策略
- 波動大時更謹慎，波動小時更激進

#### **buffer_size=1_000_000**
- 存儲 100 萬筆經驗 (s, a, r, s', done)
- 打破時間相關性
- 提升樣本利用率

---

## 🔧 技術細節

### 1. 隨機種子設定（第 520-521 行）

```python
np.random.seed(42)
torch.manual_seed(42)
```

**目的**：確保訓練可重現
- 固定環境初始化
- 固定神經網絡初始權重
- 固定數據分片順序

### 2. 設備選擇（第 260-263 行）

```python
def determine_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
```

**GPU 加速**：
- CUDA 可用 → GPU 訓練（快 10-100 倍）
- 無 GPU → CPU 訓練

### 3. 停止條件（第 571-585 行）

```python
# 方式 1：按步數停止
--total_timesteps 100000

# 方式 2：按回合數停止
--episodes 1000  # 每個環境 1000 回合，總計 16000 回合
```

**估算公式**：
```
總步數 ≈ 回合數 × 平均回合長度 × 環境數
```

### 4. 訓練主循環（第 587-590 行）

```python
start = time.time()
model.learn(total_timesteps=estimated_steps, 
           progress_bar=True, 
           callback=callback)
elapsed = time.time() - start
```

**learn() 內部流程**：
```
for step in range(total_timesteps):
    1. 並行採集 16 個環境的樣本
    2. 存入經驗回放緩衝區
    3. 每步從緩衝區採樣訓練
    4. 更新 Actor、Critic、Alpha
    5. 記錄日誌和統計
    6. 檢查停止條件
```

---

## 📊 結果輸出與可視化

### 1. 模型保存（第 592-596 行）

```python
model.save(model_path)  # 保存為 .zip 文件
```

**包含內容**：
- Actor 網絡參數
- Critic 網絡參數
- 熵係數 α
- 優化器狀態
- 訓練超參數

**使用方式**：
```python
model = SAC.load("sac_btc_3m_model.zip")
obs = env.reset()
action, _states = model.predict(obs, deterministic=True)
```

### 2. 訓練曲線（第 266-322 行）

```python
plot_training_curve(args.logdir, curve_path)
```

**讀取數據**：
- 從 Monitor 生成的 CSV 讀取
- 提取每回合的 reward
- 平滑處理（移動平均）

**輸出圖表**：
- X 軸：總步數（timesteps）
- Y 軸：回合獎勵（episode reward）
- 藍線：原始數據
- 橙線：平滑曲線

### 3. 逐步獎勵曲線（第 324-376 行）

```python
plot_step_reward_curve(step_logs_dir, step_curve_path)
```

**作用**：
- 觀察獎勵函數在訓練過程中的變化
- 診斷獎勵設計是否合理
- 檢測異常獎勵峰值

### 4. 回合統計圖表（第 378-476 行）

```python
plot_episode_stats(ep_logs_dir, ep_figure_path)
```

**6 個子圖**：
1. **收益率趨勢**：每回合的 return_rate
2. **期末資金**：每回合結束時的總資產
3. **最大回撤**：風險指標
4. **交易次數**：策略活躍度
5. **累積收益**：複利效果
6. **總獎勵**：學習進度

**統計摘要**：
```
=== 訓練統計摘要 ===
總回合數: 165
勝率: 7.14%
平均收益率: -84.28%
平均交易次數/回合: 17.0
最終累積收益: -100.00%
```

---

## 🎯 完整執行範例

### 基礎訓練
```bash
python Train/train.py \
  --csv Data/BTCUSDT_futures_volume_5years_5min.csv \
  --months 3 \
  --vec_envs 16 \
  --total_timesteps 100000 \
  --initial_balance 10000 \
  --leverage 3 \
  --position_scale 0.3
```

### 進階訓練（按回合數）
```bash
python Train/train.py \
  --csv Data/ETHUSDT_futures_volume_5years_5min.csv \
  --months 6 \
  --vec_envs 32 \
  --episodes 2000 \
  --episode_steps 500 \
  --window_size 240 \
  --leverage 5 \
  --position_scale 0.5 \
  --logdir training_results/eth_6m_sac
```

---

## 🔍 常見問題解答

### Q1: 為什麼使用 SAC 而不是 PPO？
**A**: SAC 優勢：
- **連續動作空間**：倉位比例、止盈/止損百分比
- **樣本效率高**：Off-policy，經驗回放
- **自動探索調整**：熵正則化

### Q2: 為什麼需要多個並行環境？
**A**: 
- **提升採集效率**：16 個環境 = 16 倍樣本速度
- **數據多樣性**：不同時間段數據
- **打破相關性**：避免連續樣本過度相關

### Q3: ActionTransformWrapper 的作用？
**A**:
- SAC 輸出固定範圍 `[-1, 1]`
- 交易環境需要具體數值
- 映射並限制倉位（風險控制）

### Q4: 為什麼會快速爆倉？
**A**: 環境風險控制問題（非訓練腳本問題）：
- 槓桿過高（10x）
- 強制平倉比例過於激進
- 建議降低槓桿至 3-5x

### Q5: 如何判斷訓練是否成功？
**A**: 觀察指標：
- 訓練曲線是否上升
- 勝率是否提升
- 最大回撤是否降低
- 交易次數是否合理

---

## 📚 總結

### 核心流程回顧
```
數據準備 → 環境構建 → SAC訓練 → 結果輸出
   ↓          ↓          ↓         ↓
 切片分片   多層Wrapper  經驗回放   可視化分析
```

### 關鍵技術點
1. **並行訓練**：SubprocVecEnv 提升效率
2. **動作映射**：連續空間 → 交易指令
3. **經驗回放**：Off-policy 高樣本利用率
4. **熵正則化**：自動探索/利用平衡
5. **多層 Wrapper**：統計、日誌、限制分離

### 優化建議
- 降低槓桿至 3-5x
- 增加 window_size 至 240-480
- 調整 position_scale 至 0.2-0.3
- 使用更多訓練步數（500k-1M）
- 調整獎勵函數權重

---

**完整技術文檔完成！** 🎉

