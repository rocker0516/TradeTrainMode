# SAC 交易訓練系統（Stable-Baselines3）

使用 **Stable-Baselines3** 內建的 **SAC (Soft Actor-Critic)** 算法進行交易模型訓練。

## 📋 特性

- ✅ 使用成熟穩定的 SB3 框架
- ✅ 內建 SAC 算法，經過充分測試
- ✅ 自動保存最佳模型和檢查點
- ✅ TensorBoard 日誌支持
- ✅ 完整的評估和回調系統
- ✅ 簡單易用的 API

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install stable-baselines3
```

或

```bash
pip install -r requirements.txt
```

### 2. 快速測試（10K 步）

```bash
python Train/train_sac.py --mode quick_test
```

### 3. 完整訓練（100K 步）

```bash
python Train/train_sac.py --timesteps 100000
```

### 4. 評估模型

```bash
python Train/evaluate_sac.py --model ./models/best_model.zip --episodes 10
```

## 📚 使用指南

### 基本訓練

```bash
# 默認配置
python Train/train_sac.py

# 指定訓練步數
python Train/train_sac.py --timesteps 200000

# 使用不同數據
python Train/train_sac.py --data ./Data/ETHUSDT_futures_volume_5years_5min.csv

# 指定設備
python Train/train_sac.py --device cuda

# 調整批次大小
python Train/train_sac.py --batch_size 512
```

### 繼續訓練

```bash
# 加載已有模型繼續訓練
python Train/train_sac.py --load_model ./models/best_model.zip --timesteps 50000
```

### 自定義環境參數

```bash
python Train/train_sac.py \
  --initial_balance 20000 \
  --leverage 20 \
  --window_size 200 \
  --reward_mode pct
```

### 評估模型

```bash
# 基本評估
python Train/evaluate_sac.py --model ./models/best_model.zip

# 評估多個回合
python Train/evaluate_sac.py --model ./models/best_model.zip --episodes 20

# 使用不同數據評估
python Train/evaluate_sac.py \
  --model ./models/best_model.zip \
  --data ./Data/ETHUSDT_futures_volume_5years_5min.csv \
  --episodes 20

# 安靜模式（不輸出詳細信息）
python Train/evaluate_sac.py --model ./models/best_model.zip --quiet
```

## ⚙️ 參數說明

### 訓練參數

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `--timesteps` | int | - | 總訓練步數（若省略則使用回合模式） |
| `--episodes` | int | 50 | 總訓練回合數（回合模式時生效） |
| `--mode` | str | 'train' | 運行模式（train/quick_test） |
| `--lr` | float | 3e-4 | 學習率 |
| `--batch_size` | int | 256 | 批次大小 |
| `--buffer_size` | int | 100000 | 經驗回放緩衝區大小 |
| `--device` | str | 'auto' | 訓練設備（auto/cuda/cpu） |

### 環境參數

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `--initial_balance` | float | 10000.0 | 初始資金 |
| `--leverage` | float | 10.0 | 槓桿倍數 |
| `--window_size` | int | 288 | 觀察窗口大小 |
| `--reward_mode` | str | 'delta_equity' | 獎勵模式（delta_equity/pct/log） |

### 路徑參數

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `--data` | str | './Data/BTCUSDT_...' | 訓練數據路徑 |
| `--start_date` | str | - | 訓練資料開始日期（YYYY-MM-DD） |
| `--end_date` | str | - | 訓練資料結束日期（YYYY-MM-DD） |
| `--model_dir` | str | './models' | 模型保存目錄 |
| `--log_dir` | str | './logs' | 日誌目錄 |
| `--load_model` | str | None | 加載已有模型路徑 |

## 📊 訓練輸出

訓練過程中會生成以下文件：

```
models/
├── best_model.zip              # 評估性能最佳的模型
├── final_model.zip             # 最終模型
├── sac_trading_10000_steps.zip # 檢查點（每 10K 步）
└── sac_trading_20000_steps.zip

logs/
├── SAC_1/                      # TensorBoard 日誌
└── evaluations.npz             # 評估結果
```

### 查看 TensorBoard

```bash
tensorboard --logdir ./logs
```

然後在瀏覽器打開 http://localhost:6006

## 🎯 訓練技巧

### 快速測試

用於驗證系統是否正常工作：

```bash
python Train/train_sac.py --mode quick_test
```

這將使用較少的步數和較小的緩衝區。

### 標準訓練

適合大多數情況：

```bash
python Train/train_sac.py --timesteps 100000
```

### 長期訓練

追求最佳性能：

```bash
python Train/train_sac.py --timesteps 500000 --buffer_size 500000
```

### 調整參數

**如果訓練不穩定**:
- 降低學習率: `--lr 1e-4`
- 增加批次大小: `--batch_size 512`

**如果 GPU 記憶體不足**:
- 減小批次大小: `--batch_size 128`
- 使用 CPU: `--device cpu`

**如果想加快訓練**:
- 減小觀察窗口: `--window_size 100`
- 增大批次大小: `--batch_size 512`

## 🔧 使用自定義環境

如果您想使用不同的環境配置：

```python
# 在 train_sac.py 中修改 create_environment 函數
env = create_environment(
    data_path='./Data/ETHUSDT_futures_volume_5years_5min.csv',
    initial_balance=20000.0,
    leverage=20.0,
    window_size=200,
    reward_mode='pct'
)
```

## 📖 SB3 文檔

Stable-Baselines3 提供豐富的功能和文檔：

- [官方文檔](https://stable-baselines3.readthedocs.io/)
- [SAC 算法說明](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [自定義回調](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html)

## 🐛 故障排除

### 問題 1: 找不到數據文件

**錯誤**: `FileNotFoundError: 數據文件不存在`

**解決**:
```bash
# 確認數據文件存在
ls Data/

# 使用正確的路徑
python Train/train_sac.py --data ./Data/BTCUSDT_futures_volume_5years_5min.csv
```

### 問題 2: CUDA 記憶體不足

**錯誤**: `RuntimeError: CUDA out of memory`

**解決**:
```bash
# 使用 CPU
python Train/train_sac.py --device cpu

# 或減小批次大小
python Train/train_sac.py --batch_size 128
```

### 問題 3: 獎勵始終為負

這是正常現象。SAC 需要時間學習。建議：
- 繼續訓練更多步數（例如 200K+）
- 檢查評估結果是否有改善
- 嘗試不同的獎勵模式（pct 或 log）

## 📝 程式碼範例

### 基本使用

```python
from stable_baselines3 import SAC
from Env.trading_env import TradingEnvironment
from Env.reward import RewardCalculator
import pandas as pd

# 創建環境
df = pd.read_csv('./Data/BTCUSDT_futures_volume_5years_5min.csv')
reward_calculator = RewardCalculator(mode='delta_equity', scale=1.0)
env = TradingEnvironment(
    df=df,
    initial_balance=10000,
    leverage=10,
    reward_calculator=reward_calculator
)

# 創建模型
model = SAC('MlpPolicy', env, verbose=1)

# 訓練
model.learn(total_timesteps=100000)

# 保存
model.save('sac_trading')

# 加載
model = SAC.load('sac_trading')

# 使用
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    if done:
        break
```

### 自定義回調

```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def _on_step(self) -> bool:
        # 每步後執行的邏輯
        if self.n_calls % 1000 == 0:
            print(f"Step: {self.n_calls}")
        return True

# 使用回調
model.learn(total_timesteps=100000, callback=CustomCallback())
```

## 🎓 優勢

使用 Stable-Baselines3 的 SAC 有以下優勢：

1. **成熟穩定**: 經過大量測試和實際應用
2. **開箱即用**: 無需自己實作複雜的算法細節
3. **活躍維護**: 持續更新和改進
4. **豐富功能**: 內建評估、回調、日誌等功能
5. **社群支持**: 大量教程和社群幫助

## 📄 授權

本項目使用 Stable-Baselines3（MIT License）。

---

**開始訓練您的交易模型吧！** 🚀

```bash
python Train/train_sac.py --mode quick_test
```

