# 可視化模組說明文檔

## 📁 模組結構

```
Train/
├── train.py              # 主訓練腳本
├── visualization.py      # 可視化模組（新增）
└── README_visualization.md  # 本文檔
```

## 🎯 重構目的

將 `train.py` 中的繪圖功能提取為獨立模組，遵循以下原則：

### SOLID 原則
- ✅ **單一職責原則 (SRP)**：`visualization.py` 專注於可視化，`train.py` 專注於訓練
- ✅ **開放封閉原則 (OCP)**：易於擴展新的圖表類型而無需修改現有代碼
- ✅ **依賴反轉原則 (DIP)**：通過類別抽象提供統一接口

### 優勢
1. **代碼組織更清晰**：訓練邏輯與可視化邏輯分離
2. **易於維護**：修改圖表樣式只需修改 `visualization.py`
3. **可重用性高**：其他腳本也可導入使用
4. **易於測試**：可以獨立測試可視化功能
5. **易於擴展**：添加新圖表只需繼承 `TrainingVisualizer`

---

## 📊 可視化模組 API

### 1. 類別架構

```python
TrainingVisualizer (基類)
├── TrainingCurvePlotter      # 訓練曲線繪製器
├── StepRewardPlotter          # 逐步獎勵繪製器
└── EpisodeStatsPlotter        # 回合統計繪製器
```

### 2. 使用方式

#### 方式一：使用便利函數（向後兼容）

```python
from Train.visualization import (
    plot_training_curve, 
    plot_step_reward_curve, 
    plot_episode_stats
)

# 繪製訓練曲線
plot_training_curve(
    monitor_logs_dir='training_results/sac_btc_3m/monitor',
    output_path='training_results/sac_btc_3m/training_curve.png'
)

# 繪製逐步獎勵曲線
plot_step_reward_curve(
    step_logs_dir='training_results/sac_btc_3m/step_rewards',
    output_path='training_results/sac_btc_3m/step_reward_curve.png',
    smooth_window=500
)

# 繪製回合統計圖表
plot_episode_stats(
    ep_logs_dir='training_results/sac_btc_3m/episode_stats',
    output_path='training_results/sac_btc_3m/episode_stats.png',
    combined_csv_out='training_results/sac_btc_3m/episode_stats_summary.csv'
)
```

#### 方式二：使用類別（OOP 方式）

```python
from Train.visualization import (
    TrainingCurvePlotter,
    StepRewardPlotter,
    EpisodeStatsPlotter
)

# 創建繪圖器實例（可自定義 DPI）
curve_plotter = TrainingCurvePlotter(dpi=200)
step_plotter = StepRewardPlotter(dpi=200)
stats_plotter = EpisodeStatsPlotter(dpi=200)

# 繪製圖表
curve_plotter.plot(monitor_logs_dir, output_path)
step_plotter.plot(step_logs_dir, output_path, smooth_window=500)
stats_plotter.plot(ep_logs_dir, output_path, combined_csv_out)
```

---

## 📈 各繪圖器詳細說明

### 1. TrainingCurvePlotter - 訓練曲線繪製器

**功能**：繪製 Episode Reward 隨訓練步數的變化趨勢

**輸入數據來源**：
- `Monitor` 生成的 CSV 文件
- 格式：`monitor.csv`, `monitor_*.csv`, `*.monitor.csv`

**輸出圖表**：
- 原始 Episode Reward（半透明藍線）
- 平滑 Episode Reward（橙色實線）
- 自動平滑窗口大小 = 數據長度 / 20

**方法**：
```python
plot(monitor_logs_dir: str, output_path: str) -> None
```

**示例**：
```python
plotter = TrainingCurvePlotter(dpi=150)
plotter.plot(
    monitor_logs_dir='training_results/sac_btc_3m/monitor',
    output_path='training_results/sac_btc_3m/training_curve.png'
)
```

---

### 2. StepRewardPlotter - 逐步獎勵繪製器

**功能**：繪製每個環境每步的 Reward 及平均線

**輸入數據來源**：
- `RewardLogWrapper` 生成的 CSV 文件
- 格式：`env_0.csv`, `env_1.csv`, ...
- 列：`step, reward, total_value, balance, btc_held, current_step`

**輸出圖表**：
- 各環境的平滑 Step Reward（半透明彩色線）
- 所有環境的平均 Step Reward（橙色粗線）

**方法**：
```python
plot(step_logs_dir: str, output_path: str, smooth_window: int = 500) -> None
```

**參數**：
- `smooth_window`: 平滑窗口大小，默認 500

**示例**：
```python
plotter = StepRewardPlotter()
plotter.plot(
    step_logs_dir='training_results/sac_btc_3m/step_rewards',
    output_path='training_results/sac_btc_3m/step_reward_curve.png',
    smooth_window=300  # 自定義平滑窗口
)
```

---

### 3. EpisodeStatsPlotter - 回合統計繪製器

**功能**：繪製 2x3 子圖的訓練統計面板

**輸入數據來源**：
- `EpisodeStatsWrapper` 生成的 CSV 文件
- 格式：`env_0.csv`, `env_1.csv`, ...
- 列：`env_id, episode, steps, end_total_value, min_total_value, max_drawdown, episode_reward, return_rate, trade_count, aggressive_step_ratio, done_reason`

**輸出圖表（2x3 子圖）**：

1. **收益率趨勢** (Return Rate per Episode)
   - 每回合的收益率百分比
   - 顯示勝率（Win Rate）

2. **期末資金** (End Total Value per Episode)
   - 每回合結束時的總資產
   - 紅色虛線標示初始資金

3. **最大回撤** (Max Drawdown per Episode)
   - 每回合的最大回撤比例
   - 風險評估指標

4. **交易次數** (Trade Count per Episode)
   - 每回合的交易次數
   - 顯示平均交易次數

5. **累積收益率** (Cumulative Return Rate)
   - 複利累積收益曲線
   - 包含溢出保護

6. **回合總獎勵** (Episode Total Reward)
   - 每回合的累積 reward
   - 學習進度參考

**方法**：
```python
plot(ep_logs_dir: str, output_path: str, combined_csv_out: Optional[str] = None) -> None
```

**參數**：
- `combined_csv_out`: 可選的彙總 CSV 輸出路徑

**輸出統計摘要**：
```
=== 訓練統計摘要 ===
總回合數: 165
勝率: 7.14%
平均收益率: -84.28%
平均交易次數/回合: 17.0
最終累積收益: -100.00%
```

**示例**：
```python
plotter = EpisodeStatsPlotter(dpi=200)
plotter.plot(
    ep_logs_dir='training_results/sac_btc_3m/episode_stats',
    output_path='training_results/sac_btc_3m/episode_stats.png',
    combined_csv_out='training_results/sac_btc_3m/summary.csv'  # 可選
)
```

---

## 🔧 擴展新圖表

若需要添加新的圖表類型，只需繼承 `TrainingVisualizer` 基類：

```python
from Train.visualization import TrainingVisualizer
import matplotlib.pyplot as plt

class CustomPlotter(TrainingVisualizer):
    """自定義繪圖器"""
    
    def plot(self, data_dir: str, output_path: str) -> None:
        """繪製自定義圖表"""
        # 1. 載入數據
        data = self._load_data(data_dir)
        
        # 2. 繪製圖表
        plt.figure(figsize=(10, 6))
        plt.plot(data)
        plt.title('My Custom Plot')
        
        # 3. 保存圖片
        self._ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
    
    def _load_data(self, data_dir: str):
        """載入數據（私有方法）"""
        # 實現數據載入邏輯
        pass
```

---

## 📝 在 train.py 中的使用

```python
# 第 27 行：導入可視化函數
from Train.visualization import (
    plot_training_curve, 
    plot_step_reward_curve, 
    plot_episode_stats
)

# 第 407-419 行：訓練結束後繪製圖表
def main():
    # ... 訓練代碼 ...
    
    # 繪製訓練曲線
    curve_path = os.path.join(args.logdir, 'training_curve.png')
    plot_training_curve(args.logdir, curve_path)
    print(f"Training curve saved to {curve_path}")

    # 繪製逐步 reward 曲線（平滑）
    step_curve_path = os.path.join(args.logdir, 'step_reward_curve.png')
    plot_step_reward_curve(
        os.path.join(args.logdir, 'step_rewards'), 
        step_curve_path
    )
    print(f"Step reward curve saved to {step_curve_path}")

    # 繪製每回合統計圖表並匯出彙總 CSV
    ep_stats_dir = os.path.join(args.logdir, 'episode_stats')
    ep_summary_csv = os.path.join(args.logdir, 'episode_stats_summary.csv')
    ep_figure_path = os.path.join(args.logdir, 'episode_stats.png')
    plot_episode_stats(ep_stats_dir, ep_figure_path, combined_csv_out=ep_summary_csv)
    print(f"Episode stats saved to {ep_figure_path} and {ep_summary_csv}")
```

---

## ✅ 測試建議

### 單元測試範例

```python
# tests/test_visualization.py
import pytest
from Train.visualization import (
    TrainingCurvePlotter,
    StepRewardPlotter,
    EpisodeStatsPlotter
)

def test_training_curve_plotter():
    """測試訓練曲線繪製器"""
    plotter = TrainingCurvePlotter(dpi=100)
    plotter.plot(
        monitor_logs_dir='test_data/monitor',
        output_path='test_output/training_curve.png'
    )
    assert os.path.exists('test_output/training_curve.png')

def test_step_reward_plotter():
    """測試逐步獎勵繪製器"""
    plotter = StepRewardPlotter(dpi=100)
    plotter.plot(
        step_logs_dir='test_data/step_rewards',
        output_path='test_output/step_reward_curve.png',
        smooth_window=100
    )
    assert os.path.exists('test_output/step_reward_curve.png')

def test_episode_stats_plotter():
    """測試回合統計繪製器"""
    plotter = EpisodeStatsPlotter(dpi=100)
    plotter.plot(
        ep_logs_dir='test_data/episode_stats',
        output_path='test_output/episode_stats.png',
        combined_csv_out='test_output/summary.csv'
    )
    assert os.path.exists('test_output/episode_stats.png')
    assert os.path.exists('test_output/summary.csv')
```

---

## 📊 輸出示例

### 1. 訓練曲線圖
![Training Curve](training_curve_example.png)
- 顯示訓練進度
- 平滑處理後更容易觀察趨勢

### 2. 逐步獎勵曲線
![Step Reward Curve](step_reward_curve_example.png)
- 各環境的 reward 變化
- 平均線顯示整體趨勢

### 3. 回合統計面板
![Episode Stats](episode_stats_example.png)
- 6 個子圖全面展示訓練效果
- 包含收益率、回撤、交易次數等關鍵指標

---

## 🔄 遷移指南

如果你的舊代碼使用了 `train.py` 中的繪圖函數，無需修改，保持向後兼容：

```python
# 舊代碼（仍然可用）
from Train.train import plot_training_curve

# 新代碼（推薦）
from Train.visualization import plot_training_curve
```

---

## 📚 相關文檔

- [train.py 執行步驟與原理詳解](../reports/train_execution_guide.md)
- [訓練測試報告](../reports/train_test_report.md)
- [修正摘要](../reports/train_fix_summary.md)

---

**最後更新**：2025-10-03  
**版本**：v1.0  
**作者**：AI Assistant

