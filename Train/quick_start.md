# 快速入門指南（SB3 SAC）

## 🚀 3 分鐘開始訓練

### 步驟 1: 安裝依賴

```bash
pip install stable-baselines3
```

### 步驟 2: 快速測試

```bash
python Train/train_sac.py --mode quick_test
```

這將訓練 10,000 步（約 2-5 分鐘），驗證整個系統。

### 步驟 3: 完整訓練

```bash
python Train/train_sac.py --timesteps 100000
```

### 步驟 4: 評估模型

```bash
python Train/evaluate_sac.py --model ./models/best_model.zip --episodes 10
```

## 📋 常用命令

### 訓練

```bash
# 基本訓練（100K 步）
python Train/train_sac.py

# 長期訓練（500K 步）
python Train/train_sac.py --timesteps 500000

# 使用不同數據
python Train/train_sac.py --data ./Data/ETHUSDT_futures_volume_5years_5min.csv

# 自定義參數
python Train/train_sac.py --initial_balance 20000 --leverage 20 --batch_size 512

# 使用 CPU
python Train/train_sac.py --device cpu

# 繼續訓練
python Train/train_sac.py --load_model ./models/best_model.zip --timesteps 50000
```

### 評估

```bash
# 基本評估
python Train/evaluate_sac.py --model ./models/best_model.zip

# 多回合評估
python Train/evaluate_sac.py --model ./models/best_model.zip --episodes 20

# 安靜模式
python Train/evaluate_sac.py --model ./models/best_model.zip --quiet
```

## 📊 查看訓練進度

### TensorBoard

```bash
tensorboard --logdir ./logs
```

然後在瀏覽器打開 http://localhost:6006

### 訓練輸出

訓練過程中會實時顯示：
- Episode 編號
- 步數
- 獎勵
- 最終資金
- 盈虧百分比

## ⚙️ 調整參數

### 快速實驗（測試想法）

```bash
python Train/train_sac.py --mode quick_test
```

### 標準訓練（平衡）

```bash
python Train/train_sac.py --timesteps 100000
```

### 高性能訓練（追求最佳）

```bash
python Train/train_sac.py --timesteps 500000 --batch_size 512 --buffer_size 500000
```

## 🔧 常見問題

### Q: 訓練很慢怎麼辦？

A: 減小觀察窗口或使用 GPU：
```bash
python Train/train_sac.py --window_size 100 --device cuda
```

### Q: GPU 記憶體不足？

A: 減小批次大小或使用 CPU：
```bash
python Train/train_sac.py --batch_size 128 --device cpu
```

### Q: 獎勵一直是負數？

A: 這是正常的，繼續訓練。SAC 需要時間學習：
```bash
python Train/train_sac.py --timesteps 200000
```

### Q: 如何知道模型訓練好了？

A: 查看評估結果：
```bash
python Train/evaluate_sac.py --model ./models/best_model.zip --episodes 20
```

觀察：
- 平均盈虧是否為正
- 勝率是否 > 50%
- 最佳盈虧和最差盈虧的差距

## 📖 下一步

1. ✅ 運行快速測試確認系統正常
2. 📊 查看 TensorBoard 了解訓練進度
3. 🎯 調整參數優化性能
4. 💰 評估模型並分析結果
5. 🚀 部署到實盤交易（RealTrading）

## 💡 提示

- **最佳模型**: 系統會自動保存評估性能最好的模型到 `models/best_model.zip`
- **檢查點**: 每 10K 步保存一次檢查點，可用於繼續訓練
- **日誌**: TensorBoard 日誌保存在 `logs/` 目錄，可視化訓練過程
- **評估**: 定期評估模型，確保訓練方向正確

---

**開始您的第一次訓練！** 🚀

```bash
python Train/train_sac.py --mode quick_test
```

