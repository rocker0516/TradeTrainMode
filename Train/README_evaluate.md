# 模型評估模組說明

## 📋 概述

`evaluate.py` 提供簡潔的命令行接口，用於評估訓練好的 SAC 模型。

---

## 🚀 快速開始

### 最簡單的使用方式

```bash
python Train/evaluate.py \
  --model models/sac_btc.zip \
  --start_date 20240101 \
  --end_date 20240331 \
  --episodes 10 \
  --deterministic
```

這將：
1. 載入模型 `models/sac_btc.zip`
2. 在 2024年1月1日 至 3月31日 的數據上測試
3. 運行 10 個回合
4. 使用確定性策略（推薦用於評估）
5. 自動生成評估報告和圖表

---

## 📊 輸出內容

### 1. 命令行輸出

```
載入模型：models/sac_btc.zip
載入測試數據：Data/BTCUSDT_futures_volume_5years_5min.csv
數據載入成功：26,000 筆資料

============================================================
評估配置
============================================================
模型：models/sac_btc.zip
測試回合數：10
每回合步數：無限制
策略模式：確定性
初始資金：$10000.0
槓桿倍數：5x
倉位上限：50.0%
輸出目錄：evaluation_results/20241003_143025
============================================================

開始評估：10 回合...
回合 1/10: 收益率=2.34%, 最終資產=$10234.00, 交易次數=15
回合 2/10: 收益率=-1.23%, 最終資產=$9877.00, 交易次數=12
...
回合 10/10: 收益率=3.45%, 最終資產=$10345.00, 交易次數=18

結果已保存：evaluation_results/20241003_143025/evaluation_results.csv

============================================================
評估摘要
============================================================
總回合數：10
平均收益率：1.25%
勝率：60.00%
平均最終資產：$10125.50
平均最大回撤：5.23%
平均交易次數：14.5
平均回合步數：450.3
============================================================

收益率分佈：
  最小值：-2.34%
  25分位：-0.50%
  中位數：1.12%
  75分位：2.85%
  最大值：4.56%

圖表已保存：evaluation_results/20241003_143025/evaluation_stats.png

✅ 評估完成！
```

### 2. 文件輸出

```
evaluation_results/
└── 20241003_143025/              # 時間戳命名的評估目錄
    ├── evaluation_results.csv    # 詳細結果（CSV格式）
    ├── evaluation_stats.png      # 統計圖表（6個子圖）
    └── episode_stats/            # 繪圖用的數據
        └── env_0.csv
```

#### evaluation_results.csv 內容
```csv
episode,final_balance,return_rate,episode_reward,steps,trade_count,max_drawdown,done_reason
0,10234.00,0.0234,125.3,458,15,0.0345,completed
1,9877.00,-0.0123,-45.2,432,12,0.0567,completed
...
```

---

## 🎯 參數詳解

### 必需參數

```bash
--model PATH              # 模型路徑（.zip 文件）
--start_date YYYYMMDD     # 測試起始日期
--end_date YYYYMMDD       # 測試結束日期
```

### 評估控制參數

```bash
--episodes N              # 測試回合數（默認：10）
--episode_steps N         # 每回合最大步數（默認：0=無限制）
--deterministic           # 使用確定性策略（推薦用於評估）
```

### 環境參數（應與訓練時一致）

```bash
--csv PATH                # 測試數據路徑
--window_size N           # 觀察窗口大小（默認：288）
--initial_balance FLOAT   # 初始資金（默認：10000.0）
--leverage N              # 槓桿倍數（默認：5）
--position_scale FLOAT    # 倉位比例上限（默認：0.5）
--transaction_fee FLOAT   # 手續費率（默認：0.001）
--min_trade_amount FLOAT  # 最小交易金額（默認：10.0）
```

### 輸出參數

```bash
--output PATH             # 輸出目錄（默認：evaluation_results）
```

---

## 📈 評估指標說明

### 核心指標

| 指標 | 計算公式 | 說明 |
|------|---------|------|
| **收益率** | (最終資產 - 初始資產) / 初始資產 | 回合盈虧百分比 |
| **勝率** | 盈利回合數 / 總回合數 × 100% | 獲利回合占比 |
| **最大回撤** | (峰值 - 谷值) / 峰值 | 最大資產回撤幅度 |
| **交易次數** | 倉位變化次數 | 策略活躍度 |
| **平均步數** | 總步數 / 回合數 | 回合持續時間 |

### 分佈指標

- **最小值/最大值**：極端情況表現
- **25/50/75分位**：收益率分佈情況
- **中位數**：中位數收益（受極端值影響較小）

---

## 🔬 使用場景

### 1. 模型驗證

```bash
# 在未見過的數據上測試模型泛化能力
python Train/evaluate.py \
  --model training_results/sac_btc/sac_btc_model.zip \
  --start_date 20240101 \
  --end_date 20240331 \
  --episodes 50 \
  --deterministic
```

### 2. 不同時間段測試

```bash
# 牛市測試
python Train/evaluate.py \
  --model models/sac_btc.zip \
  --start_date 20230101 \
  --end_date 20230331 \
  --episodes 30 \
  --deterministic \
  --output eval/bull_market

# 熊市測試
python Train/evaluate.py \
  --model models/sac_btc.zip \
  --start_date 20220501 \
  --end_date 20220731 \
  --episodes 30 \
  --deterministic \
  --output eval/bear_market
```

### 3. 多幣種評估

```bash
# 評估同一模型在不同幣種的表現
for coin in BTC ETH SOL; do
    python Train/evaluate.py \
        --model models/multi_coin.zip \
        --csv Data/${coin}USDT_futures_volume_5years_5min.csv \
        --start_date 20240101 \
        --end_date 20240331 \
        --episodes 30 \
        --deterministic \
        --output eval/${coin}
done
```

### 4. 參數敏感性測試

```bash
# 測試不同槓桿的影響
for lev in 3 5 10; do
    python Train/evaluate.py \
        --model models/sac_btc.zip \
        --start_date 20240101 \
        --end_date 20240331 \
        --episodes 30 \
        --leverage $lev \
        --deterministic \
        --output eval/leverage_${lev}
done
```

---

## 🎓 最佳實踐

### 1. 確定性策略 vs 隨機策略

```bash
# 評估時推薦使用確定性策略
--deterministic  # 使用動作均值，結果可重現

# 訓練時自動使用隨機策略（探索）
# 不需要在訓練腳本中設置
```

### 2. 測試數據選擇

```
✅ 好的做法：
- 使用訓練後的時間段數據
- 選擇不同市場狀態的數據（牛/熊/震盪）
- 確保數據質量與訓練數據一致

❌ 避免：
- 使用訓練期內的數據（過擬合風險）
- 測試數據太少（< 5 回合）
- 時間段跨度太短
```

### 3. 回合數選擇

| 回合數 | 適用場景 | 預估時間 |
|--------|---------|---------|
| 10-20 | 快速驗證 | 2-5分鐘 |
| 50-100 | 標準評估 | 10-20分鐘 |
| 200+ | 統計顯著性 | 30分鐘+ |

### 4. 環境參數一致性

```bash
# ⚠️ 重要：評估時的環境參數必須與訓練時一致

# 訓練時使用
python Train/train.py \
  --leverage 5 \
  --position_scale 0.3 \
  --window_size 288

# 評估時也要使用相同參數
python Train/evaluate.py \
  --leverage 5 \          # ✅ 與訓練一致
  --position_scale 0.3 \  # ✅ 與訓練一致
  --window_size 288       # ✅ 與訓練一致
```

---

## 🔍 結果分析

### 查看詳細數據

```bash
# 使用 pandas 分析
python -c "
import pandas as pd
df = pd.read_csv('evaluation_results/20241003_143025/evaluation_results.csv')
print(df.describe())
print(f'\n勝率：{(df.return_rate > 0).mean() * 100:.2f}%')
print(f'最佳回合：{df.return_rate.max() * 100:.2f}%')
print(f'最差回合：{df.return_rate.min() * 100:.2f}%')
"
```

### 對比多次評估

```python
import pandas as pd
import matplotlib.pyplot as plt

# 載入多次評估結果
eval1 = pd.read_csv('eval/test1/evaluation_results.csv')
eval2 = pd.read_csv('eval/test2/evaluation_results.csv')

# 對比收益率分佈
plt.figure(figsize=(10, 5))
plt.hist(eval1['return_rate'], alpha=0.5, label='Test 1')
plt.hist(eval2['return_rate'], alpha=0.5, label='Test 2')
plt.legend()
plt.xlabel('Return Rate')
plt.ylabel('Frequency')
plt.title('Return Rate Distribution Comparison')
plt.savefig('comparison.png')
```

---

## ⚠️ 常見問題

### Q1: 評估結果與訓練差異大？
**A**: 檢查以下幾點：
1. 環境參數是否一致（leverage, position_scale等）
2. 測試數據是否與訓練數據分佈相似
3. 是否使用了 `--deterministic` 標誌
4. 模型是否過擬合訓練數據

### Q2: 如何選擇測試時間段？
**A**: 建議策略：
- 時間上：選擇訓練後的數據
- 市場狀態：涵蓋牛市、熊市、震盪市
- 長度：至少 1-3 個月的數據

### Q3: 評估回合數多少合適？
**A**: 
- 快速驗證：10-20 回合
- 標準評估：50-100 回合
- 統計分析：200+ 回合

### Q4: 評估很慢怎麼辦？
**A**: 
- 減少 `--episodes` 數量
- 設置 `--episode_steps` 限制每回合長度
- 使用較短的測試時間段

---

## 📚 相關文檔

- 📄 [評估快速參考](../EVALUATION_QUICK_REFERENCE.md)
- 📁 [評估範例腳本](../examples/evaluate_example.sh)
- 📄 [訓練快速參考](../TRAINING_QUICK_REFERENCE.md)
- 📄 [可視化模組說明](README_visualization.md)

---

**文檔版本**：v1.0  
**最後更新**：2025-10-03

