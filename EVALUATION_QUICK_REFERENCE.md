# 🔬 模型評估快速參考

## 💡 最簡潔用法

```bash
python Train/evaluate.py \
  --model models/sac_btc.zip \
  --start_date 20240101 \
  --end_date 20240331 \
  --episodes 10 \
  --deterministic
```

---

## 📋 必需參數

| 參數 | 說明 | 範例 |
|------|------|------|
| `--model` | 模型路徑 | `models/sac_btc.zip` |
| `--start_date` | 測試起始日期 | `20240101` |
| `--end_date` | 測試結束日期 | `20240331` |

---

## ⚙️ 常用可選參數

| 參數 | 默認值 | 說明 |
|------|--------|------|
| `--episodes` | 10 | 測試回合數 |
| `--deterministic` | False | 使用確定性策略（推薦） |
| `--episode_steps` | 0 | 每回合步數（0=不限） |
| `--output` | evaluation_results | 輸出目錄 |

---

## 🎯 典型使用場景

### 1. 快速測試（10回合）
```bash
python Train/evaluate.py \
  --model models/sac_btc.zip \
  --start_date 20240101 \
  --end_date 20240131 \
  --episodes 10 \
  --deterministic
```

### 2. 標準評估（50回合）
```bash
python Train/evaluate.py \
  --model models/sac_btc.zip \
  --start_date 20240101 \
  --end_date 20240331 \
  --episodes 50 \
  --deterministic \
  --output eval/q1_2024
```

### 3. 深度評估（100回合，限制步數）
```bash
python Train/evaluate.py \
  --model models/sac_btc.zip \
  --start_date 20240101 \
  --end_date 20240630 \
  --episodes 100 \
  --episode_steps 500 \
  --deterministic \
  --output eval/deep_test
```

---

## 📊 輸出內容

### 命令行輸出
```
回合 1/10: 收益率=2.34%, 最終資產=$10234.00, 交易次數=15
回合 2/10: 收益率=-1.23%, 最終資產=$9877.00, 交易次數=12
...

============================================================
評估摘要
============================================================
總回合數：10
平均收益率：1.25%
勝率：60.00%
平均最終資產：$10125.50
平均最大回撤：5.23%
平均交易次數：14.5
============================================================
```

### 文件輸出
```
evaluation_results/
├── 20241003_143025/
│   ├── evaluation_results.csv      # 詳細數據
│   ├── evaluation_stats.png        # 統計圖表
│   └── episode_stats/
│       └── env_0.csv
```

---

## 🔧 環境參數（應與訓練時一致）

```bash
python Train/evaluate.py \
  --model models/sac_btc.zip \
  --start_date 20240101 \
  --end_date 20240331 \
  --episodes 50 \
  --deterministic \
  --leverage 5 \              # 槓桿倍數
  --position_scale 0.5 \      # 倉位上限
  --initial_balance 10000 \   # 初始資金
  --transaction_fee 0.001     # 手續費率
```

---

## 📈 評估指標

| 指標 | 說明 |
|------|------|
| **收益率** | (最終資產 - 初始資產) / 初始資產 |
| **勝率** | 盈利回合數 / 總回合數 |
| **最大回撤** | (峰值 - 谷值) / 峰值 |
| **交易次數** | 倉位變化次數 |
| **平均步數** | 每回合平均執行步數 |

---

## 🎯 評估策略建議

### 時間段選擇

| 測試時段 | 目的 |
|---------|------|
| **近期數據** | 驗證模型當前表現 |
| **牛市數據** | 測試上漲行情適應性 |
| **熊市數據** | 測試下跌行情表現 |
| **震盪數據** | 測試橫盤市場適應性 |

### 回合數選擇

| 回合數 | 適用場景 |
|--------|---------|
| 10-20 | 快速驗證 |
| 50-100 | 標準評估 |
| 200+ | 統計顯著性測試 |

---

## 🔄 評估流程

```
1. 準備測試數據
   └── 選擇未見過的時間段

2. 執行評估
   └── python Train/evaluate.py --model ... --episodes 50 --deterministic

3. 分析結果
   ├── 查看 evaluation_results.csv
   ├── 查看 evaluation_stats.png
   └── 對比訓練期表現

4. 調整策略（如需要）
   ├── 分析失敗案例
   ├── 調整參數
   └── 重新訓練
```

---

## 💡 進階技巧

### 1. 批次評估多個模型

```bash
# 創建評估腳本
for model in models/*.zip; do
    echo "評估: $model"
    python Train/evaluate.py \
        --model "$model" \
        --start_date 20240101 \
        --end_date 20240331 \
        --episodes 50 \
        --deterministic \
        --output "eval/$(basename $model .zip)"
done
```

### 2. 不同參數對比

```bash
# 測試不同槓桿
for leverage in 3 5 10; do
    python Train/evaluate.py \
        --model models/sac_btc.zip \
        --start_date 20240101 \
        --end_date 20240331 \
        --episodes 30 \
        --leverage $leverage \
        --deterministic \
        --output "eval/leverage_$leverage"
done
```

### 3. 月度表現分析

```bash
# 評估每個月的表現
for month in 01 02 03; do
    python Train/evaluate.py \
        --model models/sac_btc.zip \
        --start_date "2024${month}01" \
        --end_date "2024${month}28" \
        --episodes 20 \
        --deterministic \
        --output "eval/2024_month_$month"
done
```

---

## ⚠️ 注意事項

### 1. 環境參數一致性
確保評估時的環境參數與訓練時相同：
- `--leverage`
- `--position_scale`
- `--window_size`
- `--transaction_fee`

### 2. 確定性策略
生產環境建議使用 `--deterministic` 標誌：
```bash
--deterministic  # 使用均值而非採樣
```

### 3. 測試數據獨立性
確保測試數據未在訓練中使用：
```
訓練期：2023-01-01 ~ 2023-12-31
測試期：2024-01-01 ~ 2024-03-31  ✅
```

---

## 📚 更多資訊

- 📄 [評估腳本源碼](Train/evaluate.py)
- 📁 [評估範例](examples/evaluate_example.sh)
- 📄 [訓練快速參考](TRAINING_QUICK_REFERENCE.md)

---

**更新日期**：2025-10-03  
**版本**：v1.0

