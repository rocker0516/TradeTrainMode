# SAC + LSTM 训练系统架构文档

## 📐 总体架构

本项目采用**模块化设计**，严格遵循 **SOLID 原则**，实现了训练与模型的完全分离。

```
┌─────────────────────────────────────────────────────────────┐
│                        主程序 (train.py)                      │
│                     配置管理 (config.py)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │           训练器 (Trainer)               │
        │   - SACTrainer (当前实现)                │
        │   - PPOTrainer (未来扩展)                │
        │   - TD3Trainer (未来扩展)                │
        └─────────────────────────────────────────┘
                   │                    │
         ┌─────────┴─────────┐         │
         ▼                   ▼         ▼
    ┌─────────┐        ┌─────────┐  ┌─────────┐
    │  模型    │        │  环境    │  │  工具    │
    │ Models  │        │   Env   │  │  Utils  │
    └─────────┘        └─────────┘  └─────────┘
         │                   │            │
         │                   │            │
    SAC+LSTM          TradingEnv    ReplayBuffer
    PPO+LSTM          Executor      Logger
    TD3+LSTM          (future)      (future)
```

## 🏗️ 模块详解

### 1. 模型层 (Train/models/)

**职责**: 定义和实现强化学习算法

```
models/
├── base_model.py          # 抽象基类 (ABC)
└── sac_lstm_model.py      # SAC + LSTM 实现
```

#### 设计原则

- **开放-封闭原则 (OCP)**: 通过继承 `BaseRLModel` 扩展新模型，无需修改现有代码
- **依赖反转原则 (DIP)**: 训练器依赖抽象基类，不依赖具体实现
- **里氏替换原则 (LSP)**: 任何 `BaseRLModel` 的子类都可以替换使用

#### BaseRLModel 接口

```python
class BaseRLModel(ABC):
    @abstractmethod
    def select_action(self, state, evaluate=False) -> np.ndarray:
        """选择动作"""
        
    @abstractmethod
    def update(self, batch) -> dict:
        """更新模型"""
        
    @abstractmethod
    def save(self, filepath) -> None:
        """保存模型"""
        
    @abstractmethod
    def load(self, filepath) -> None:
        """加载模型"""
        
    @abstractmethod
    def reset_hidden_state(self) -> None:
        """重置隐藏状态"""
```

#### SAC_LSTM_Model 架构

```
输入观察 (n_features, window_size)
         │
         ▼
    ┌─────────────┐
    │ LSTM Encoder │  ← 提取时序特征
    └─────────────┘
         │
         ├────────────────┬────────────────┐
         ▼                ▼                ▼
    ┌────────┐      ┌─────────┐     ┌─────────┐
    │  Actor │      │ Critic1 │     │ Critic2 │
    └────────┘      └─────────┘     └─────────┘
         │                │                │
         ▼                ▼                ▼
      动作分布          Q值1             Q值2
```

**关键组件**:

1. **LSTMEncoder**: 双层 LSTM 编码时序特征
2. **Actor**: 策略网络，输出动作的高斯分布参数
3. **Critic**: 双 Q 网络，减少过估计偏差
4. **自动熵调整**: 动态平衡探索与利用

### 2. 训练器层 (Train/trainers/)

**职责**: 管理训练循环、经验收集、模型更新

```
trainers/
├── base_trainer.py        # 抽象训练器
└── sac_trainer.py         # SAC 训练器
```

#### 设计原则

- **单一职责原则 (SRP)**: 训练器只负责训练流程，不涉及模型细节
- **依赖注入**: 通过构造函数注入环境、模型、配置

#### 训练流程

```
开始训练
    │
    ▼
┌─────────────────────┐
│  初始化环境和模型    │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐     ┌──────────────┐
│  执行一个回合        │────▶│ 收集经验      │
│  (Episode Loop)     │     │ (Experience) │
└─────────────────────┘     └──────────────┘
    │                              │
    ▼                              ▼
┌─────────────────────┐     ┌──────────────┐
│  经验回放采样        │◀────│ 存入缓冲区    │
└─────────────────────┘     └──────────────┘
    │
    ▼
┌─────────────────────┐
│  更新模型参数        │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  定期评估和保存      │
└─────────────────────┘
    │
    ▼
  继续下一回合
```

### 3. 工具层 (Train/utils/)

**职责**: 提供训练相关的辅助功能

```
utils/
├── replay_buffer.py       # 经验回放缓冲区
└── logger.py              # 日志记录器
```

#### ReplayBuffer

- **循环缓冲区**: 固定大小，自动覆盖旧数据
- **预分配内存**: 避免频繁内存分配
- **高效采样**: O(1) 随机访问

#### TrainingLogger

- 记录训练指标
- 生成可视化图表
- 保存 JSON 格式日志

### 4. 配置层 (Train/config.py)

**职责**: 集中管理所有超参数

```python
@dataclass
class Config:
    model: ModelConfig           # 模型参数
    training: TrainingConfig     # 训练参数
    environment: EnvironmentConfig  # 环境参数
```

#### 配置分离的优势

1. **易于实验**: 快速切换不同配置
2. **版本控制**: 配置文件可追踪
3. **可复现**: 完整记录实验设置

### 5. 环境层 (Env/)

**职责**: 提供强化学习交互环境

```
Env/
├── trading_env.py         # 交易环境
└── trade_executor.py      # 交易执行器
```

## 🔄 数据流

### 训练阶段

```
市场数据 (OHLCV)
    ↓
TradingEnvironment
    ↓ (observation)
LSTM Encoder
    ↓ (encoded_state)
Actor Network
    ↓ (action)
TradeExecutor
    ↓ (reward)
ReplayBuffer
    ↓ (batch)
Model.update()
    ↓ (gradients)
反向传播更新参数
```

### 推理阶段

```
市场数据
    ↓
TradingEnvironment.reset()
    ↓ (observation)
Model.select_action(evaluate=True)
    ↓ (action)
TradingEnvironment.step(action)
    ↓ (next_observation, reward)
循环直到回合结束
```

## 🎯 扩展指南

### 添加新模型（例如 PPO）

1. **创建模型类**:

```python
# Train/models/ppo_lstm_model.py
from .base_model import BaseRLModel

class PPO_LSTM_Model(BaseRLModel):
    def select_action(self, state, evaluate=False):
        # PPO 动作选择逻辑
        pass
    
    def update(self, batch):
        # PPO 更新逻辑
        pass
    # ... 实现其他抽象方法
```

2. **创建训练器**（可选）:

```python
# Train/trainers/ppo_trainer.py
from .base_trainer import BaseTrainer

class PPOTrainer(BaseTrainer):
    def train(self, total_episodes, eval_interval):
        # PPO 特定的训练逻辑
        pass
```

3. **在主程序中使用**:

```python
# 修改 Train/train.py
from Train.models import PPO_LSTM_Model
from Train.trainers import PPOTrainer

model = PPO_LSTM_Model(...)
trainer = PPOTrainer(env, model, config)
```

### 添加新的训练特性

例如，添加优先经验回放 (PER):

1. **扩展 ReplayBuffer**:

```python
# Train/utils/prioritized_replay_buffer.py
from .replay_buffer import ReplayBuffer

class PrioritizedReplayBuffer(ReplayBuffer):
    def add(self, state, action, reward, next_state, done, priority):
        # 带优先级的添加逻辑
        pass
    
    def sample(self, batch_size):
        # 基于优先级的采样
        pass
```

2. **在训练器中使用**:

```python
# 修改 SACTrainer
self.replay_buffer = PrioritizedReplayBuffer(...)
```

## 📊 性能考虑

### GPU 利用率优化

1. **批量处理**: 使用较大的 `batch_size` (256-512)
2. **并行环境**: 可扩展为多环境并行收集经验
3. **混合精度**: 可添加 AMP 支持

### 内存优化

1. **经验回放**: 使用 `buffer_size` 控制内存占用
2. **LSTM 状态**: 训练时不保留跨样本的隐藏状态
3. **梯度累积**: 模拟更大批次

### 训练稳定性

1. **软更新**: 使用小的 `tau` (0.005)
2. **梯度裁剪**: 防止梯度爆炸
3. **预热阶段**: 随机探索填充缓冲区

## 🧪 测试策略

### 单元测试

```python
# 测试模型
test_model.py
- 测试动作选择
- 测试模型更新
- 测试保存/加载

# 测试环境
test_environment.py
- 测试状态空间
- 测试动作空间
- 测试奖励计算
```

### 集成测试

```python
test_train_setup.py
- 测试完整训练流程
- 测试配置系统
- 测试所有组件交互
```

### 性能测试

```python
benchmark_training.py
- 测量训练速度
- GPU/CPU 使用率
- 内存占用
```

## 📝 代码规范

### 命名约定

- **类**: `PascalCase` (例如 `SACTrainer`)
- **函数/变量**: `snake_case` (例如 `select_action`)
- **常量**: `UPPER_CASE` (例如 `MAX_STEPS`)
- **私有方法**: `_leading_underscore` (例如 `_soft_update`)

### 类型注解

所有 public 方法必须有类型注解:

```python
def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
    ...
```

### 文档字符串

所有 public 类和方法必须有 docstring:

```python
def update(self, batch: Tuple[Any, ...]) -> dict:
    """
    使用批次数据更新模型
    
    Args:
        batch: 从经验回放缓冲区采样的批次数据
        
    Returns:
        包含训练指标的字典
    """
```

## 🔐 最佳实践

1. **配置管理**: 使用配置文件，不要硬编码参数
2. **随机种子**: 设置固定种子以确保可复现
3. **定期保存**: 训练过程中定期保存检查点
4. **监控指标**: 实时跟踪损失和奖励变化
5. **版本控制**: 记录模型版本和对应的配置

## 📚 参考文献

- [SAC: Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
- [LSTM: Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Gymnasium: A Standard API for Reinforcement Learning](https://gymnasium.farama.org/)

---

**更新日期**: 2025-10-11  
**版本**: 1.0.0

