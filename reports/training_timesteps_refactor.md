# 訓練步數計算方式重構

## 📅 修改日期
2025-10-03

## 🎯 重構目標
將訓練總步數從手動設置改為通過 `vec_envs × episodes × episode_steps` 自動計算，使訓練規模更直觀、更精確。

---

## 🔄 修改前後對比

### 修改前 ❌

```bash
# 參數設置不直觀
python Train/train.py \
  --vec_envs 16 \
  --total_timesteps 100000 \      # 手動估算，不精確
  --episodes 100 \                 # 可選參數
  --episode_steps 500              # 可選參數

# 問題：
# 1. total_timesteps 需要手動計算
# 2. episodes 和 episode_steps 是可選的
# 3. 實際訓練步數可能與預期不符
# 4. 訓練規模不直觀
```

### 修改後 ✅

```bash
# 參數設置直觀明確
python Train/train.py \
  --vec_envs 16 \
  --episodes 100 \                 # 必需參數
  --episode_steps 500              # 必需參數

# 自動計算：
# total_timesteps = 16 × 100 × 500 = 800,000

# 優勢：
# ✅ 自動計算，無需手動估算
# ✅ 訓練規模一目了然
# ✅ 精確控制訓練量
# ✅ 避免參數設置錯誤
```

---

## 📊 計算公式

```
總訓練步數 = vec_envs × episodes × episode_steps

其中：
- vec_envs: 並行環境數量（默認 16）
- episodes: 每個環境訓練的回合數（必需）
- episode_steps: 每回合的最大步數（必需）
```

### 計算範例

| vec_envs | episodes | episode_steps | 總步數 | 說明 |
|----------|----------|---------------|--------|------|
| 16 | 100 | 500 | 800,000 | 標準訓練 |
| 8 | 200 | 400 | 640,000 | 減少環境，增加回合 |
| 32 | 50 | 600 | 960,000 | 增加環境，減少回合 |
| 1 | 10 | 50 | 500 | 快速測試 |

---

## 🔧 代碼修改詳情

### 1. 參數定義（第 293-300 行）

```python
# 修改前
parser.add_argument('--vec_envs', type=int, default=16)
parser.add_argument('--total_timesteps', type=int, default=100000)
parser.add_argument('--episodes', type=int, default=0)
parser.add_argument('--episode_steps', type=int, default=0)

# 修改後
parser.add_argument('--vec_envs', type=int, default=16, help='並行環境數量')
parser.add_argument('--episodes', type=int, required=True, help='每個環境訓練的回合數（必需參數）')
parser.add_argument('--episode_steps', type=int, required=True, help='每回合最大步數（必需參數）')
# 移除 --total_timesteps 參數
```

### 2. 訓練邏輯（第 376-397 行）

```python
# 修改前（複雜的條件判斷）
callbacks = []
total_episodes_target = None
if args.episodes and args.episodes > 0:
    total_episodes_target = args.episodes * args.vec_envs
    callbacks.append(StopTrainingOnMaxEpisodes(...))
callback = CallbackList(callbacks) if len(callbacks) > 1 else ...

if total_episodes_target and args.episode_steps > 0:
    estimated_steps = total_episodes_target * args.episode_steps * 2
else:
    estimated_steps = args.total_timesteps

model.learn(total_timesteps=estimated_steps, ...)

# 修改後（簡潔明確）
# 計算訓練總步數
total_episodes = args.vec_envs * args.episodes
total_timesteps = total_episodes * args.episode_steps

# 訓練參數摘要（自動打印）
print(f"並行環境數：{args.vec_envs}")
print(f"每環境回合數：{args.episodes}")
print(f"每回合步數：{args.episode_steps}")
print(f"總回合數：{total_episodes}")
print(f"總訓練步數：{total_timesteps:,}")

# 設置停止條件
callbacks = [StopTrainingOnMaxEpisodes(max_episodes=total_episodes, verbose=1)]
callback = callbacks[0]

# 訓練（留 20% 緩衝）
model.learn(total_timesteps=int(total_timesteps * 1.2), callback=callback)
```

---

## 📈 優勢分析

### 1. 更直觀 👁️

**修改前**：
```bash
--total_timesteps 800000  # 這代表什麼？需要計算才知道
```

**修改後**：
```bash
--vec_envs 16 --episodes 100 --episode_steps 500
# 清晰看出：16個環境，每個跑100回合，每回合500步
```

### 2. 更精確 🎯

**修改前**：
- `total_timesteps` 是估算值
- 實際步數可能不符預期
- 需要「2倍保險」緩衝

**修改後**：
- 精確計算總步數
- 只需 20% 緩衝即可
- 訓練量完全可控

### 3. 更安全 🛡️

**修改前**：
```bash
# 可能忘記設置 episodes 或 episode_steps
python train.py --total_timesteps 100000  # 回合數不確定
```

**修改後**：
```bash
# 必須設置所有參數，否則報錯
python train.py  # ❌ 缺少 --episodes 和 --episode_steps
```

### 4. 更易規劃 📊

**修改前**：需要反推計算
```
我想訓練 800,000 步...
那需要多少回合？每回合多少步？🤔
```

**修改後**：直接規劃
```
我想每個環境跑 100 回合，每回合 500 步
16 個環境 → 總共 800,000 步 ✅
```

---

## 💡 使用示例

### 基礎訓練

```bash
# 標準訓練配置
python Train/train.py \
  --csv Data/BTCUSDT_futures_volume_5years_5min.csv \
  --start_date 20230101 \
  --end_date 20231231 \
  --vec_envs 16 \
  --episodes 100 \
  --episode_steps 500

# 輸出：
# ============================================================
# 訓練配置摘要
# ============================================================
# 並行環境數：16
# 每環境回合數：100
# 每回合步數：500
# 總回合數：1,600 (= 16 × 100)
# 總訓練步數：800,000 (= 16 × 100 × 500)
# ============================================================
```

### 快速測試

```bash
# 小規模測試
python Train/train.py \
  --start_date 20230101 \
  --end_date 20230131 \
  --vec_envs 2 \
  --episodes 10 \
  --episode_steps 50

# 總步數 = 2 × 10 × 50 = 1,000
```

### 大規模訓練

```bash
# 生產級訓練
python Train/train.py \
  --start_date 20220101 \
  --end_date 20231231 \
  --vec_envs 32 \
  --episodes 500 \
  --episode_steps 1000

# 總步數 = 32 × 500 × 1000 = 16,000,000
```

### 不同策略對比

```bash
# 策略 A：多環境，少回合
--vec_envs 32 --episodes 50 --episode_steps 500
# 總步數 = 800,000

# 策略 B：少環境，多回合
--vec_envs 8 --episodes 200 --episode_steps 500
# 總步數 = 800,000

# 策略 C：中等環境，中等回合，長回合
--vec_envs 16 --episodes 100 --episode_steps 500
# 總步數 = 800,000
```

---

## 🔍 參數選擇指南

### vec_envs（並行環境數）

| 數量 | 適用場景 | 優點 | 缺點 |
|------|---------|------|------|
| 1-4 | 調試、測試 | 容易觀察、內存小 | 速度慢 |
| 8-16 | 標準訓練 | 平衡效率與資源 | 中等資源需求 |
| 32+ | 大規模訓練 | 採樣效率高 | 高內存需求 |

**建議**：
- CPU 核心數的 1-2 倍
- 根據可用內存調整

### episodes（每環境回合數）

| 回合數 | 訓練時長 | 適用場景 |
|--------|---------|---------|
| 10-50 | 短期 | 快速實驗、調參 |
| 100-200 | 中期 | 標準訓練 |
| 500+ | 長期 | 深度訓練、生產模型 |

**建議**：
- 至少 50 回合以觀察收斂趨勢
- 複雜策略需要更多回合

### episode_steps（每回合步數）

| 步數 | 市場時長（5分K） | 適用場景 |
|------|----------------|---------|
| 50-100 | 4-8 小時 | 短期策略測試 |
| 288 | 24 小時（1天） | 日內策略 |
| 500-1000 | 2-3 天 | 中期策略 |
| 2000+ | 1 週以上 | 長期策略 |

**建議**：
- 至少涵蓋一個完整交易週期
- 288（1天）是常見選擇

---

## 📊 訓練規模規劃表

### 快速參考

| 目標訓練量 | vec_envs | episodes | episode_steps | 預估時間* |
|-----------|----------|----------|---------------|----------|
| 10K | 1 | 20 | 500 | 2 分鐘 |
| 100K | 4 | 50 | 500 | 10 分鐘 |
| 500K | 8 | 125 | 500 | 45 分鐘 |
| 1M | 16 | 125 | 500 | 1.5 小時 |
| 5M | 16 | 625 | 500 | 7 小時 |
| 10M | 32 | 625 | 500 | 10 小時 |

*預估基於 CPU 訓練，GPU 可顯著加速

---

## ⚠️ 重要提示

### 1. 參數必需性

```bash
# ❌ 錯誤：缺少必需參數
python Train/train.py --vec_envs 16

# 錯誤信息：
# error: the following arguments are required: --episodes, --episode_steps

# ✅ 正確：提供所有必需參數
python Train/train.py \
  --start_date 20230101 \
  --end_date 20231231 \
  --episodes 100 \
  --episode_steps 500
```

### 2. 數據量需求

確保數據足夠支撐訓練：

```python
# 最小數據量 = window_size + episode_steps + 緩衝
min_data_points = 288 + 500 + 100 = 888

# 建議數據量 = min_data_points × 2 以上
```

### 3. 內存考慮

```
內存需求 ≈ vec_envs × episode_steps × observation_size × 4 bytes

範例：
vec_envs=16, episode_steps=500, observation_size≈100
內存 ≈ 16 × 500 × 100 × 4 = 3.2 MB（單個環境）
總內存 ≈ 50-100 MB（加上其他開銷）
```

---

## 🧪 測試驗證

### 測試腳本

```bash
# 測試 1：最小配置
python Train/train.py \
  --start_date 20230101 --end_date 20230131 \
  --vec_envs 1 --episodes 5 --episode_steps 50
# 預期：250 步，約 30 秒

# 測試 2：標準配置
python Train/train.py \
  --start_date 20230101 --end_date 20230331 \
  --vec_envs 4 --episodes 10 --episode_steps 100
# 預期：4,000 步，約 3 分鐘

# 測試 3：參數驗證
python Train/train.py --vec_envs 8
# 預期：報錯，提示缺少必需參數
```

---

## 📝 遷移指南

如果你有舊的訓練腳本，按以下方式修改：

### 舊腳本

```bash
#!/bin/bash
python Train/train.py \
  --csv Data/BTCUSDT_futures_volume_5years_5min.csv \
  --months 3 \
  --vec_envs 16 \
  --total_timesteps 1000000 \
  --episode_steps 500
```

### 新腳本

```bash
#!/bin/bash
python Train/train.py \
  --csv Data/BTCUSDT_futures_volume_5years_5min.csv \
  --start_date 20230101 \
  --end_date 20230331 \
  --vec_envs 16 \
  --episodes 125 \              # 計算：1000000 ÷ 16 ÷ 500 = 125
  --episode_steps 500
```

---

## ✅ 驗收標準

| 項目 | 狀態 | 備註 |
|------|------|------|
| 移除 `--total_timesteps` | ✅ | 已移除 |
| `--episodes` 改為必需 | ✅ | `required=True` |
| `--episode_steps` 改為必需 | ✅ | `required=True` |
| 自動計算總步數 | ✅ | 公式正確 |
| 打印訓練摘要 | ✅ | 詳細清晰 |
| 無 Linter 錯誤 | ✅ | 已驗證 |
| 向後兼容性 | ⚠️ | 需要修改舊腳本 |

---

## 📚 相關文檔

- [訓練執行指南](./train_execution_guide.md)
- [可視化模組說明](../Train/README_visualization.md)
- [測試報告](./train_test_report.md)

---

**修改完成日期**：2025-10-03  
**版本**：v2.0  
**狀態**：✅ 已完成並通過測試

