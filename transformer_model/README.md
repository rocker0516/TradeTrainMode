# Multi-Scale Transformer + Actor-Critic 交易模型

這是一個基於 **Multi-Scale Transformer** 和 **Actor-Critic** 架構的自動交易訓練系統，專門針對加密貨幣期貨交易設計，目標是實現損益最大化。

## 🎯 模型特點

### 1. Multi-Scale Transformer 架構
- **多尺度時間窗口**: 短期(32步)、中期(96步)、長期(288步)
- **自注意力機制**: 捕捉長期依賴關係
- **位置編碼**: 處理時序信息
- **特徵融合**: 整合不同時間尺度的信息

### 2. 豐富的特徵工程
- **價格特徵**: OHLC數據
- **成交量特徵**: volume, buy_volume, sell_volume, quote_volume
- **微觀結構特徵**: volume_ratio, long_short_ratio, trades
- **技術指標**: RSI, MACD, 布林帶, ROC, ATR, Williams%R, Stochastic
- **時間特徵**: 小時、分鐘、星期、月份等週期性特徵
- **賬戶狀態**: 持倉、資金、總資產等

### 3. Actor-Critic 策略網絡
- **連續動作空間**: [交易方向(-1~1), 止盈比例(0.2~10), 止損比例(0.1~0.3)]
- **多頭輸出**: 分別處理不同類型的動作
- **分佈建模**: Normal分佈(方向) + Beta分佈(止盈止損)

### 4. PPO 訓練算法
- **穩定訓練**: Proximal Policy Optimization
- **GAE**: 廣義優勢估計
- **梯度裁剪**: 防止梯度爆炸
- **學習率調度**: 自適應學習率

## 📁 項目結構

```
transformer_model/
├── configs/
│   └── model_config.py          # 模型配置
├── models/
│   ├── feature_extractor.py     # 特徵提取器
│   ├── multi_scale_transformer.py # Multi-Scale Transformer
│   └── actor_critic.py          # Actor-Critic網絡
├── utils/
│   └── data_preprocessing.py    # 數據預處理
├── transformer_env_adapter.py   # 環境適配器
├── train_transformer.py        # 訓練腳本
└── README.md                   # 說明文檔
```

## 🚀 快速開始

### 1. 環境準備

```bash
pip install torch pandas numpy matplotlib talib gymnasium
```

### 2. 數據準備

確保你的CSV數據包含以下欄位：
```
timestamp,open,high,low,close,volume,buy_volume,sell_volume,volume_ratio,long_short_ratio,trades,quote_volume
```

### 3. 開始訓練

```bash
cd transformer_model
python train_transformer.py --data_path ../Data/BTCUSDT_futures_volume_5years_5min.csv --iterations 1000
```

### 4. 訓練參數

```bash
python train_transformer.py \
    --data_path ../Data/BTCUSDT_futures_volume_5years_5min.csv \
    --iterations 2000 \
    --rollout_steps 4096 \
    --save_interval 100
```

## ⚙️ 配置參數

### 模型配置 (configs/model_config.py)

```python
# 數據配置
WINDOW_SIZE = 288  # 24小時 * 60分鐘 / 5分鐘

# Transformer配置
TRANSFORMER_CONFIG = {
    'embed_dim': 256,
    'num_heads': 8,
    'num_layers': 6,
    'dropout': 0.1,
    'ff_dim': 1024,
}

# 多尺度配置
MULTI_SCALE_CONFIG = {
    'short_term': 32,
    'medium_term': 96,
    'long_term': 288,
    'scale_weights': [0.3, 0.3, 0.4],
}

# Actor-Critic配置
ACTOR_CRITIC_CONFIG = {
    'hidden_dim': 512,
    'num_layers': 3,
    'action_dim': 3,
    'action_bounds': [(-1.0, 1.0), (0.2, 10.0), (0.1, 0.3)],
}

# 訓練配置
TRAINING_CONFIG = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.01,
    'value_loss_coef': 0.5,
    'batch_size': 64,
    'epochs_per_update': 10,
}
```

## 📊 模型架構

### 1. 特徵處理流程

```
原始數據 → 特徵提取 → 歸一化 → 嵌入投影 → 特徵融合
    ↓
價格特徵 (OHLC) → Price Embedding
成交量特徵 → Volume Embedding  
微觀結構特徵 → Microstructure Embedding
技術指標 → Technical Embedding
時間特徵 → Time Embedding
賬戶狀態 → Account Embedding
    ↓
特徵融合層 → [batch, seq_len, embed_dim]
```

### 2. Multi-Scale Transformer

```
輸入特徵 [batch, 288, embed_dim]
    ↓
├─ 短期分支 (最近32步) → Transformer Layers → 短期特徵
├─ 中期分支 (最近96步) → Transformer Layers → 中期特徵  
└─ 長期分支 (全部288步) → Transformer Layers → 長期特徵
    ↓
多尺度融合 → 全局注意力 → 輸出 [batch, 1, embed_dim]
```

### 3. Actor-Critic 網絡

```
狀態特徵 [batch, embed_dim]
    ↓
共享層 (MLP)
    ↓
├─ Actor分支:
│   ├─ 交易方向頭 → Normal(μ, σ) → [-1, 1]
│   ├─ 止盈比例頭 → Beta(α, β) → [0.2, 10]
│   └─ 止損比例頭 → Beta(α, β) → [0.1, 0.3]
└─ Critic分支:
    └─ 價值估計頭 → V(s)
```

## 📈 訓練監控

### 訓練指標
- **Episode Reward**: 每個episode的累計獎勵
- **Episode Length**: 每個episode的長度
- **Policy Loss**: 策略網絡損失
- **Value Loss**: 價值網絡損失
- **Entropy**: 策略熵（探索程度）

### 輸出文件
- `best_model.pth`: 最佳模型檢查點
- `checkpoint_*.pth`: 定期保存的檢查點
- `training_progress.png`: 訓練進度圖表
- `training_stats_*.npz`: 訓練統計數據

## 🔧 自定義配置

### 修改網絡架構
```python
# 在 model_config.py 中調整
TRANSFORMER_CONFIG = {
    'embed_dim': 512,      # 增加嵌入維度
    'num_heads': 16,       # 增加注意力頭數
    'num_layers': 12,      # 增加層數
}
```

### 調整訓練參數
```python
TRAINING_CONFIG = {
    'learning_rate': 1e-4,    # 降低學習率
    'batch_size': 128,        # 增加批次大小
    'rollout_steps': 4096,    # 增加rollout步數
}
```

### 修改動作空間
```python
ACTOR_CRITIC_CONFIG = {
    'action_bounds': [
        (-2.0, 2.0),      # 擴大交易方向範圍
        (0.1, 20.0),      # 擴大止盈範圍
        (0.05, 0.5),      # 擴大止損範圍
    ],
}
```

## 💡 使用建議

### 1. 訓練策略
- **階段訓練**: 先在簡化環境訓練，再逐步增加複雜度
- **多品種訓練**: 使用多個加密貨幣品種提升泛化能力
- **長期訓練**: 建議訓練2000+迭代以獲得穩定策略

### 2. 超參數調優
- **學習率**: 3e-4是良好的起點，可根據收斂情況調整
- **Clip範圍**: 0.2是PPO的經典設置
- **GAE λ**: 0.95平衡偏差和方差

### 3. 風險控制
- **最大回撤限制**: 在環境中設置最大回撤約束
- **倉位管理**: 限制單次最大倉位大小
- **止損機制**: 確保止損參數在合理範圍

## 🎯 性能優化

### 1. 計算優化
- 使用GPU加速訓練
- 批量處理技術指標計算
- 特徵緩存機制

### 2. 記憶體優化
- 梯度累積減少記憶體使用
- 動態序列長度處理
- 模型檢查點管理

### 3. 訓練穩定性
- 梯度裁剪防止梯度爆炸
- 學習率調度
- 正則化技術

## 📝 TODO

- [ ] 添加多品種聯合訓練
- [ ] 實現市場狀態檢測模組
- [ ] 集成更多技術指標
- [ ] 添加回測評估系統
- [ ] 實現模型解釋性分析
- [ ] 添加實時交易介面

## 🤝 貢獻

歡迎提交Issues和Pull Requests來改進這個項目！

## 📄 License

MIT License 