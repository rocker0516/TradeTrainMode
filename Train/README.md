# SAC + LSTM 交易模型训练

本模块实现了一个基于 **SAC (Soft Actor-Critic)** 算法和 **LSTM** 网络的加密货币交易模型训练系统。

## 📁 项目结构

```
Train/
├── models/              # 模型架构
│   ├── base_model.py    # 抽象基类
│   └── sac_lstm_model.py # SAC + LSTM 实现
├── trainers/            # 训练器
│   ├── base_trainer.py  # 抽象训练器
│   └── sac_trainer.py   # SAC 训练器
├── utils/               # 工具函数
│   ├── replay_buffer.py # 经验回放缓冲区
│   └── logger.py        # 日志记录器
├── config.py            # 配置管理
├── train.py             # 主训练脚本
└── README.md            # 本文件
```

## 🎯 特性

### 模块化设计
- **抽象基类**: 使用 ABC 定义接口，遵循 OCP 和 DIP 原则
- **易于扩展**: 可轻松添加新的模型架构（如 PPO、TD3 等）
- **配置分离**: 所有超参数集中管理，支持配置文件

### SAC + LSTM 模型
- **LSTM 编码器**: 处理时序市场数据
- **双 Q 网络**: 减少 Q 值过估计
- **自动熵调整**: 自适应调整探索-利用平衡
- **软更新**: 稳定训练过程

### 高效训练
- **经验回放**: 提高样本利用率
- **批量更新**: 支持 GPU 加速
- **实时监控**: 训练过程可视化
- **自动保存**: 定期保存最佳模型

## 🚀 快速开始

### 1. 基本训练

```bash
# 使用默认配置训练
python Train/train.py

# 指定数据文件
python Train/train.py --data ./Data/BTCUSDT_futures_volume_5years_5min.csv

# 快速测试模式（50 回合）
python Train/train.py --mode quick_test
```

### 2. 使用配置文件

```bash
# 保存默认配置
python -c "from Train.config import get_default_config; get_default_config().save('my_config.json')"

# 编辑配置文件后训练
python Train/train.py --config my_config.json

# 覆盖部分配置
python Train/train.py --config my_config.json --episodes 2000 --device cuda
```

### 3. 继续训练

```bash
# 加载预训练模型继续训练
python Train/train.py --load_model ./models/best_model.pth --episodes 1000
```

## ⚙️ 配置说明

### 模型配置 (ModelConfig)

```python
lstm_hidden_dim: int = 128      # LSTM 隐藏层维度
lstm_layers: int = 2            # LSTM 层数
hidden_dim: int = 256           # 全连接层维度
lr: float = 3e-4                # 学习率
gamma: float = 0.99             # 折扣因子
tau: float = 0.005              # 软更新系数
alpha: float = 0.2              # 熵系数
auto_alpha: bool = True         # 自动调整熵系数
```

### 训练配置 (TrainingConfig)

```python
total_episodes: int = 1000      # 总训练回合数
eval_interval: int = 10         # 评估间隔
buffer_size: int = 100000       # 经验回放缓冲区大小
batch_size: int = 256           # 批次大小
warmup_steps: int = 1000        # 预热步数（随机探索）
update_interval: int = 1        # 更新间隔
save_dir: str = './models'      # 模型保存目录
log_dir: str = './logs'         # 日志保存目录
device: str = 'cuda'            # 训练设备
```

### 环境配置 (EnvironmentConfig)

```python
data_path: str                  # 数据文件路径
initial_balance: float = 10000  # 初始资金
transaction_fee: float = 0.001  # 交易手续费
window_size: int = 288          # 时间窗口大小
leverage: float = 10            # 杠杆倍数
```

## 📊 训练监控

训练过程中会自动记录以下指标：

- **回合奖励**: 每个回合的总奖励
- **回合长度**: 每个回合的步数
- **Critic 损失**: Q 网络的训练损失
- **Actor 损失**: 策略网络的训练损失
- **Alpha 值**: 熵正则化系数

这些指标会保存在 `logs/` 目录下，并可生成可视化图表。

## 🔧 添加新模型

如果想使用其他算法（如 PPO、TD3），只需：

1. **继承 BaseRLModel**:

```python
from Train.models.base_model import BaseRLModel

class MyNewModel(BaseRLModel):
    def select_action(self, state, evaluate=False):
        # 实现动作选择逻辑
        pass
    
    def update(self, batch):
        # 实现模型更新逻辑
        pass
    
    def save(self, filepath):
        # 实现模型保存逻辑
        pass
    
    def load(self, filepath):
        # 实现模型加载逻辑
        pass
    
    def reset_hidden_state(self):
        # 实现隐藏状态重置逻辑
        pass
```

2. **创建对应的训练器**（可选，或使用通用训练器）

3. **在 train.py 中导入并使用**

## 📈 性能优化建议

1. **GPU 加速**: 确保 CUDA 可用，使用 `--device cuda`
2. **批次大小**: 根据 GPU 内存调整 `batch_size`
3. **经验回放**: 增大 `buffer_size` 可提高样本多样性
4. **学习率**: 如果训练不稳定，可降低 `lr`
5. **网络容量**: 复杂市场可增大 `lstm_hidden_dim` 和 `hidden_dim`

## 🐛 调试建议

1. **快速测试**: 使用 `--mode quick_test` 验证代码
2. **检查数据**: 确保数据包含所需列（open, high, low, close, volume 等）
3. **监控指标**: 观察 Critic/Actor 损失是否正常收敛
4. **调整预热**: 如果初期崩溃，增加 `warmup_steps`

## 📝 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--config` | 配置文件路径 | `--config config.json` |
| `--data` | 数据文件路径 | `--data ./Data/ETHUSDT.csv` |
| `--mode` | 训练模式 | `--mode quick_test` |
| `--load_model` | 加载预训练模型 | `--load_model ./models/best.pth` |
| `--episodes` | 训练回合数 | `--episodes 2000` |
| `--device` | 训练设备 | `--device cuda` |

## 📚 相关文档

- [SAC 论文](https://arxiv.org/abs/1801.01290)
- [LSTM 介绍](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Gymnasium 文档](https://gymnasium.farama.org/)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License

