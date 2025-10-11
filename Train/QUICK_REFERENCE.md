# SAC + LSTM 训练系统 - 快速参考

## 🚀 一分钟快速开始

```bash
# 1. 快速测试（50回合）
python Train/train.py --mode quick_test

# 2. 完整训练（1000回合）+ 自动测试
python Train/train.py --train_sdate 2020-01-01 --train_edate 2024-12-31 --test_sdate 2025-01-01 --test_edate 2025-10-10

# 3. 仅测试已训练模型
python Train/train.py --mode test_only --load_model ./models/best_model.pth --test_sdate 2025-01-01 --test_edate 2025-10-10
```

## 📁 文件结构

| 路径 | 说明 |
|------|------|
| `Train/train.py` | 主训练脚本 |
| `Train/config.py` | 配置管理 |
| `Train/example_usage.py` | 使用示例 |
| `Train/models/` | 模型架构 |
| `Train/trainers/` | 训练器 |
| `Train/utils/` | 工具函数 |

## ⚙️ 常用命令

### 训练相关
```bash
# 使用不同数据
python Train/train.py --data ./Data/ETHUSDT_futures_volume_5years_5min.csv

# 指定回合数
python Train/train.py --episodes 2000

# 使用 CPU
python Train/train.py --device cpu

# 加载模型继续训练
python Train/train.py --load_model ./models/checkpoint_ep100.pth --episodes 500

# 使用配置文件
python Train/train.py --config my_config.json
```

### 数据分割和测试（新功能）
```bash
# 按日期分割训练/测试集
python Train/train.py \
  --train_sdate 2020-01-01 --train_edate 2024-12-31 \
  --test_sdate 2025-01-01 --test_edate 2025-10-10 \
  --test_episodes 20

# 仅测试模式（不训练）
python Train/train.py --mode test_only \
  --load_model ./models/best_model.pth \
  --test_sdate 2025-01-01 --test_edate 2025-10-10

# 训练但不测试
python Train/train.py --no_test --episodes 1000

# 跳过训练，仅测试
python Train/train.py --no_train \
  --load_model ./models/best_model.pth \
  --test_sdate 2025-01-01 --test_edate 2025-10-10
```

## 🎛️ 主要超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lstm_hidden_dim` | 128 | LSTM 隐藏层维度 |
| `lstm_layers` | 2 | LSTM 层数 |
| `lr` | 3e-4 | 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `batch_size` | 256 | 批次大小 |
| `buffer_size` | 100000 | 经验回放大小 |
| `warmup_steps` | 1000 | 预热步数 |

## 📊 输出文件

| 文件 | 说明 |
|------|------|
| `models/best_model.pth` | 评估最佳模型 |
| `models/final_model.pth` | 最终模型 |
| `models/config.json` | 训练配置 |
| `models/checkpoint_ep*.pth` | 定期检查点 |

## 🔍 调试技巧

```bash
# 1. 运行系统测试
python test_train_setup.py

# 2. 快速验证（仅2回合）
# 修改 get_quick_test_config() 中的 total_episodes

# 3. 查看模型架构
python -c "from Train.models import SAC_LSTM_Model; import torch; model = SAC_LSTM_Model((15,50), 3, 'cpu'); print(model.encoder)"
```

## 📚 文档导航

- 🏃 快速入门 → `Train/quick_start.md`
- 📖 详细文档 → `Train/README.md`
- 🧪 测试指南 → `Train/TEST_GUIDE.md` ⭐ 新增
- 🏗️ 架构说明 → `TRAIN_ARCHITECTURE.md`
- 📋 完整指南 → `TRAINING_SYSTEM_SETUP.md`

## 💡 常见问题

**Q: 如何查看训练进度？**
A: 训练会实时打印回合奖励、长度等指标

**Q: 如何修改超参数？**
A: 修改 `Train/config.py` 或创建配置文件

**Q: 模型保存在哪里？**
A: `./models/` 目录（可在配置中修改）

**Q: 如何使用不同算法？**
A: 继承 `BaseRLModel` 创建新模型类

## 🎯 推荐工作流

1. **快速测试**: `--mode quick_test` 验证系统
2. **调整配置**: 根据需求修改超参数
3. **小规模训练**: 训练 100-200 回合观察
4. **完整训练**: 训练 1000+ 回合
5. **评估优化**: 使用 `example_usage.py` 评估
6. **迭代改进**: 根据结果调整配置

---

**需要更多帮助？** 查看详细文档或运行 `python test_train_setup.py` 验证系统。

