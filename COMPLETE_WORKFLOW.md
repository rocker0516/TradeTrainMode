# 🔄 完整工作流程指南

從訓練到評估的完整流程

---

## 📋 工作流程概覽

```
1. 準備數據 → 2. 訓練模型 → 3. 評估模型 → 4. 部署使用
```

---

## 1️⃣ 訓練模型

### 最簡潔語法
```bash
python Train/train.py \
  --start_date 20230101 \
  --end_date 20231231 \
  --vec_envs 16 \
  --episodes 100 \
  --episode_steps 500
```

### 關鍵參數
- `--vec_envs`: 並行環境數
- `--episodes`: 每環境回合數
- `--episode_steps`: 每回合步數

### 總步數計算
```
總步數 = vec_envs × episodes × episode_steps
      = 16 × 100 × 500
      = 800,000 步
```

### 輸出
- 訓練好的模型：`training_results/*/sac_btc_3m_model.zip`
- 訓練曲線：`training_results/*/training_curve.png`
- 統計圖表：`training_results/*/episode_stats.png`

📚 **詳細文檔**: [TRAINING_QUICK_REFERENCE.md](TRAINING_QUICK_REFERENCE.md)

---

## 2️⃣ 評估模型

### 最簡潔語法
```bash
python Train/evaluate.py \
  --model training_results/sac_btc_3m/sac_btc_3m_model.zip \
  --start_date 20240101 \
  --end_date 20240331 \
  --episodes 10 \
  --deterministic
```

### 關鍵參數
- `--model`: 訓練好的模型路徑
- `--start_date/--end_date`: 測試時間段（應該是訓練後的數據）
- `--episodes`: 測試回合數
- `--deterministic`: 使用確定性策略（推薦）

### 輸出
```
============================================================
評估摘要
============================================================
總回合數：10
平均收益率：1.25%
勝率：60.00%
平均最終資產：$10125.50
平均最大回撤：5.23%
============================================================
```

📚 **詳細文檔**: [EVALUATION_QUICK_REFERENCE.md](EVALUATION_QUICK_REFERENCE.md)

---

## 🎯 完整範例

### 步驟 1：訓練模型（2023年數據）

```bash
# 使用 2023 年數據訓練
python Train/train.py \
  --csv Data/BTCUSDT_futures_volume_5years_5min.csv \
  --start_date 20230101 \
  --end_date 20231231 \
  --vec_envs 16 \
  --episodes 100 \
  --episode_steps 500 \
  --leverage 5 \
  --position_scale 0.3 \
  --logdir training_results/btc_2023

# 預期時間：約 2-3 小時
# 輸出：training_results/btc_2023/sac_btc_3m_model.zip
```

### 步驟 2：評估模型（2024年數據）

```bash
# 使用 2024 年數據測試
python Train/evaluate.py \
  --model training_results/btc_2023/sac_btc_3m_model.zip \
  --csv Data/BTCUSDT_futures_volume_5years_5min.csv \
  --start_date 20240101 \
  --end_date 20240331 \
  --episodes 50 \
  --deterministic \
  --leverage 5 \
  --position_scale 0.3 \
  --output evaluation_results/btc_2024_q1

# 預期時間：約 10-15 分鐘
# 輸出：evaluation_results/btc_2024_q1/evaluation_results.csv
```

---

## 📊 參數對照表

### 訓練 vs 評估參數對照

| 參數類型 | 訓練參數 | 評估參數 | 說明 |
|---------|---------|---------|------|
| **時間段** | `--start_date 20230101`<br>`--end_date 20231231` | `--start_date 20240101`<br>`--end_date 20240331` | 評估用新數據 |
| **規模控制** | `--vec_envs 16`<br>`--episodes 100`<br>`--episode_steps 500` | `--episodes 50` | 評估回合數較少 |
| **環境參數** | `--leverage 5`<br>`--position_scale 0.3`<br>`--initial_balance 10000` | `--leverage 5`<br>`--position_scale 0.3`<br>`--initial_balance 10000` | ⚠️ 必須一致 |
| **輸出** | `--logdir training_results/...` | `--output evaluation_results/...` | 不同的輸出目錄 |

---

## ⚠️ 重要注意事項

### 1. 環境參數必須一致

```bash
# ✅ 正確：評估時使用與訓練相同的參數
# 訓練
--leverage 5 --position_scale 0.3 --window_size 288

# 評估
--leverage 5 --position_scale 0.3 --window_size 288
```

### 2. 測試數據應該是未見過的

```bash
# ✅ 正確
訓練：2023-01-01 ~ 2023-12-31
評估：2024-01-01 ~ 2024-03-31

# ❌ 錯誤：測試數據與訓練數據重疊
訓練：2023-01-01 ~ 2023-12-31
評估：2023-10-01 ~ 2023-12-31  # 重疊！
```

### 3. 評估時使用確定性策略

```bash
# ✅ 推薦：評估時加上 --deterministic
python Train/evaluate.py --model ... --deterministic

# ⚠️ 不推薦：不加 --deterministic（結果會有隨機性）
python Train/evaluate.py --model ...
```

---

## 🔄 迭代優化流程

```
1. 訓練初始模型
   ↓
2. 評估模型表現
   ↓
3. 分析評估結果
   ├─ 表現良好 → 部署使用 ✅
   └─ 表現不佳 → 調整參數 ↓
                  │
4. 調整訓練參數 ←─┘
   ├─ 調整槓桿（--leverage）
   ├─ 調整倉位上限（--position_scale）
   ├─ 調整訓練數據時間段
   ├─ 增加訓練回合數
   └─ 調整獎勵權重
   ↓
5. 重新訓練
   ↓
返回步驟 2
```

---

## 📈 效能基準

### 訓練時間參考（CPU）

| 配置 | 總步數 | 預估時間 |
|------|--------|---------|
| `--vec_envs 4 --episodes 50 --episode_steps 500` | 100K | 30分鐘 |
| `--vec_envs 8 --episodes 100 --episode_steps 500` | 400K | 1.5小時 |
| `--vec_envs 16 --episodes 100 --episode_steps 500` | 800K | 2.5小時 |
| `--vec_envs 16 --episodes 500 --episode_steps 1000` | 8M | 24小時 |

### 評估時間參考

| 回合數 | 預估時間 |
|--------|---------|
| 10 | 2-5分鐘 |
| 50 | 10-15分鐘 |
| 100 | 20-30分鐘 |

---

## 🎓 最佳實踐

### 訓練階段

1. **數據選擇**
   - 使用至少 6-12 個月的歷史數據
   - 確保數據包含不同市場狀態

2. **參數設置**
   - 初學者：`--vec_envs 8 --episodes 100 --episode_steps 500`
   - 進階：`--vec_envs 16 --episodes 200 --episode_steps 500`
   - 專業：`--vec_envs 32 --episodes 500 --episode_steps 1000`

3. **風險控制**
   - 建議槓桿 3-5x（`--leverage 5`）
   - 建議倉位上限 20-30%（`--position_scale 0.3`）

### 評估階段

1. **測試數據**
   - 使用訓練後的時間段
   - 涵蓋不同市場狀態（牛/熊/震盪）
   - 至少 1-3 個月的數據

2. **回合數**
   - 快速驗證：10-20 回合
   - 標準評估：50-100 回合
   - 深度測試：200+ 回合

3. **分析重點**
   - 勝率 > 50%
   - 平均收益率 > 0
   - 最大回撤 < 20%
   - 交易次數合理（不過度交易）

---

## 📁 文件組織建議

```
TradeTrainMode/
├── Data/                           # 數據文件
│   ├── BTCUSDT_*.csv
│   └── ETHUSDT_*.csv
│
├── training_results/               # 訓練輸出
│   ├── btc_2023/
│   │   ├── sac_btc_3m_model.zip   # 模型文件
│   │   ├── training_curve.png
│   │   └── episode_stats.png
│   └── eth_2023/
│
├── evaluation_results/             # 評估輸出
│   ├── btc_2024_q1/
│   │   ├── evaluation_results.csv
│   │   └── evaluation_stats.png
│   └── eth_2024_q1/
│
└── models/                         # 生產模型
    ├── sac_btc_prod_v1.zip
    └── sac_eth_prod_v1.zip
```

---

## 🔍 故障排查

### 訓練問題

**Q: 訓練很慢？**
- 減少 `--vec_envs` 數量
- 減少 `--episodes` 或 `--episode_steps`
- 考慮使用 GPU

**Q: 訓練結果不好？**
- 檢查勝率和收益率統計
- 嘗試調整 `--leverage` 和 `--position_scale`
- 增加訓練回合數

**Q: 內存不足？**
- 減少 `--vec_envs` 數量
- 減少 `--episode_steps`

### 評估問題

**Q: 評估結果與訓練差異大？**
- 確認環境參數與訓練時一致
- 檢查測試數據是否合適
- 確認使用了 `--deterministic`

**Q: 評估很慢？**
- 減少 `--episodes` 數量
- 設置 `--episode_steps` 限制回合長度

---

## 📚 相關文檔

| 文檔 | 用途 |
|------|------|
| [TRAINING_QUICK_REFERENCE.md](TRAINING_QUICK_REFERENCE.md) | 訓練快速參考 |
| [EVALUATION_QUICK_REFERENCE.md](EVALUATION_QUICK_REFERENCE.md) | 評估快速參考 |
| [Train/README_evaluate.md](Train/README_evaluate.md) | 評估詳細文檔 |
| [examples/train_example.sh](examples/train_example.sh) | 訓練範例 |
| [examples/evaluate_example.sh](examples/evaluate_example.sh) | 評估範例 |

---

## ✅ 快速檢查清單

### 訓練前
- [ ] 數據文件準備好
- [ ] 確認時間段範圍
- [ ] 設置合適的訓練規模
- [ ] 確認環境參數

### 訓練後
- [ ] 模型文件已生成
- [ ] 查看訓練曲線
- [ ] 查看統計圖表
- [ ] 記錄訓練參數

### 評估前
- [ ] 準備測試數據（未見過的時間段）
- [ ] 確認環境參數與訓練一致
- [ ] 決定測試回合數

### 評估後
- [ ] 查看評估摘要
- [ ] 分析 CSV 詳細數據
- [ ] 查看圖表
- [ ] 決定是否部署或重新訓練

---

**最後更新**：2025-10-03  
**版本**：v1.0

🎉 **開始您的 AI 交易之旅！**

