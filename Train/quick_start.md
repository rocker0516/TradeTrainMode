# SAC + LSTM 交易模型 - 快速入门指南

## 🚀 5分钟快速开始

### 步骤 1: 安装依赖

```bash
pip install -r requirements.txt
```

### 步骤 2: 快速测试（50回合）

```bash
python Train/train.py --mode quick_test
```

这将使用默认配置训练 50 个回合，用于验证代码是否正常工作。

### 步骤 3: 完整训练（1000回合）

```bash
python Train/train.py
```

训练完成后，模型将保存在 `./models/` 目录。

### 步骤 4: 评估模型

```bash
python Train/example_usage.py --model ./models/best_model.pth --data ./Data/BTCUSDT_futures_volume_5years_5min.csv --mode evaluate
```

## 📚 常见用法

### 使用不同的数据集

```bash
python Train/train.py --data ./Data/ETHUSDT_futures_volume_5years_5min.csv
```

### 指定训练回合数

```bash
python Train/train.py --episodes 2000
```

### 使用 CPU 训练（如果没有 GPU）

```bash
python Train/train.py --device cpu
```

### 继续训练已有模型

```bash
python Train/train.py --load_model ./models/checkpoint_ep100.pth --episodes 500
```

## 🎨 自定义配置

### 1. 创建配置文件

```python
from Train.config import Config

config = Config()

# 修改模型参数
config.model.lstm_hidden_dim = 256
config.model.lr = 1e-4

# 修改训练参数
config.training.total_episodes = 2000
config.training.batch_size = 512

# 保存配置
config.save('my_config.json')
```

### 2. 使用自定义配置训练

```bash
python Train/train.py --config my_config.json
```

## 📊 监控训练过程

训练期间，程序会输出：

- **每回合**: 回合奖励、回合长度
- **每10回合**: 平均奖励、平均长度、总步数
- **评估结果**: 平均奖励、标准差、最高/最低奖励

所有指标保存在 `./logs/` 目录。

## 🎯 输出文件

训练完成后，会生成以下文件：

```
models/
├── best_model.pth          # 最佳模型（评估奖励最高）
├── final_model.pth         # 最终模型
├── checkpoint_ep50.pth     # 定期检查点
├── checkpoint_ep100.pth
└── config.json             # 训练配置

logs/
└── (训练日志)
```

## 💡 使用训练好的模型

### 评估模型性能

```bash
python Train/example_usage.py \
  --model ./models/best_model.pth \
  --data ./Data/BTCUSDT_futures_volume_5years_5min.csv \
  --mode evaluate \
  --episodes 20
```

### 交互式演示

```bash
python Train/example_usage.py \
  --model ./models/best_model.pth \
  --data ./Data/BTCUSDT_futures_volume_5years_5min.csv \
  --mode interactive
```

## 🔧 性能调优

### 如果训练太慢

1. 减小批次大小: `--config` 文件中设置 `batch_size: 128`
2. 减小 LSTM 维度: 设置 `lstm_hidden_dim: 64`
3. 减小缓冲区: 设置 `buffer_size: 50000`

### 如果模型效果不好

1. 增加训练回合: `--episodes 5000`
2. 增加网络容量: `lstm_hidden_dim: 256, hidden_dim: 512`
3. 调整学习率: `lr: 1e-4` 或 `lr: 5e-4`
4. 增加预热步数: `warmup_steps: 5000`

### 如果训练不稳定

1. 降低学习率: `lr: 1e-4`
2. 增加软更新系数: `tau: 0.01`
3. 调整熵系数: `alpha: 0.1`

## 🐛 常见问题

### Q: CUDA out of memory

**A**: 减小 `batch_size` 或使用 CPU (`--device cpu`)

### Q: 奖励一直是负数

**A**: 这是正常的初期行为。继续训练，通常在 100-200 回合后会改善。

### Q: 训练中断怎么办

**A**: 按 Ctrl+C 会自动保存 `interrupted_model.pth`，可以用 `--load_model` 继续训练。

### Q: 如何更换其他交易对数据

**A**: 使用 `--data` 参数指定其他 CSV 文件，确保数据格式与现有数据一致。

## 🎓 进阶使用

### 添加自定义模型

查看 `Train/models/base_model.py` 了解如何创建新的模型架构。

### 修改奖励函数

修改 `Env/trading_env.py` 中的 `step()` 方法。

### 调整环境参数

在配置文件中修改 `environment` 部分。

## 📞 需要帮助？

- 查看详细文档: `Train/README.md`
- 查看代码示例: `Train/example_usage.py`
- 查看配置选项: `Train/config.py`

---

**祝训练愉快！🎉**

