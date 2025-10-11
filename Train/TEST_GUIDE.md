# 模型测试功能使用指南

## 🎯 功能概述

训练脚本现在支持灵活的数据范围设定和模型测试功能：

1. **按日期分割训练/测试集**
2. **训练后自动测试**
3. **仅测试模式**（不训练）
4. **详细的测试报告**

---

## 📊 数据分割功能

### 按日期范围分割

使用 `timestamp` 列（毫秒时间戳）自动分割数据：

```bash
python Train/train.py \
  --train_sdate 2020-01-01 \
  --train_edate 2024-12-31 \
  --test_sdate 2025-01-01 \
  --test_edate 2025-10-10
```

### 默认行为

如果数据没有 `timestamp` 列，将按 80/20 比例自动分割：
- 前 80% 用于训练
- 后 20% 用于测试

---

## 🚀 使用场景

### 场景 1: 完整训练 + 测试（默认）

训练完成后自动在测试集上评估：

```bash
python Train/train.py \
  --data ./Data/BTCUSDT_futures_volume_5years_5min.csv \
  --train_sdate 2020-01-01 \
  --train_edate 2024-12-31 \
  --test_sdate 2025-01-01 \
  --test_edate 2025-10-10 \
  --episodes 1000 \
  --test_episodes 10
```

**说明**:
- 使用 2020-2024 年数据训练
- 使用 2025 年数据测试
- 训练 1000 回合
- 测试 10 回合

---

### 场景 2: 快速测试（验证系统）

```bash
python Train/train.py \
  --mode quick_test \
  --train_sdate 2024-01-01 \
  --train_edate 2024-06-30 \
  --test_sdate 2024-07-01 \
  --test_edate 2024-12-31
```

**说明**:
- 使用 quick_test 配置（50 回合）
- 快速验证系统是否正常工作

---

### 场景 3: 仅测试模式（不训练）

已有训练好的模型，仅在新数据上测试：

```bash
python Train/train.py \
  --mode test_only \
  --load_model ./models/best_model.pth \
  --test_sdate 2025-01-01 \
  --test_edate 2025-10-10 \
  --test_episodes 20
```

或使用 `--no_train` 标志：

```bash
python Train/train.py \
  --no_train \
  --load_model ./models/best_model.pth \
  --test_sdate 2025-01-01 \
  --test_edate 2025-10-10
```

**说明**:
- 跳过训练阶段
- 仅在指定日期范围的数据上测试
- 必须提供 `--load_model` 参数

---

### 场景 4: 训练但不测试

只想训练模型，不进行测试：

```bash
python Train/train.py \
  --no_test \
  --train_sdate 2020-01-01 \
  --train_edate 2024-12-31 \
  --episodes 1000
```

---

### 场景 5: 继续训练 + 测试

加载已有模型继续训练，训练后测试：

```bash
python Train/train.py \
  --load_model ./models/checkpoint_ep500.pth \
  --train_sdate 2020-01-01 \
  --train_edate 2024-12-31 \
  --test_sdate 2025-01-01 \
  --test_edate 2025-10-10 \
  --episodes 500
```

---

## 📋 命令行参数说明

### 数据和日期参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | BTCUSDT 数据 | 数据文件路径 |
| `--train_sdate` | 2020-01-01 | 训练数据开始日期 |
| `--train_edate` | 2024-12-31 | 训练数据结束日期 |
| `--test_sdate` | 2025-01-01 | 测试数据开始日期 |
| `--test_edate` | 2025-10-10 | 测试数据结束日期 |

### 运行模式参数

| 参数 | 说明 |
|------|------|
| `--mode default` | 完整训练（1000 回合） |
| `--mode quick_test` | 快速测试（50 回合） |
| `--mode test_only` | 仅测试模式 |
| `--no_train` | 跳过训练（等同 test_only） |
| `--no_test` | 训练后不测试 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--episodes` | 依配置 | 训练回合数 |
| `--test_episodes` | 5 | 测试回合数 |
| `--load_model` | None | 加载预训练模型 |
| `--device` | cuda | 训练设备 |
| `--config` | None | 配置文件路径 |

---

## 📊 测试结果

### 输出内容

测试完成后会显示：

```
============================================================
测试结果摘要
============================================================
测试回合数: 10

平均奖励: 15.23 ± 3.45
最高奖励: 21.56
最低奖励: 8.92

平均收益率: 12.34% ± 5.67%
最高收益率: 23.45%
最低收益率: 3.21%

平均最终权益: $11234.56
平均回合长度: 1234.5
胜率: 80.0%
============================================================
```

### 保存文件

测试结果保存在 `models/test_results.json`，包含：

```json
{
  "num_episodes": 10,
  "mean_reward": 15.23,
  "std_reward": 3.45,
  "max_reward": 21.56,
  "min_reward": 8.92,
  "mean_return_pct": 12.34,
  "std_return_pct": 5.67,
  "max_return_pct": 23.45,
  "min_return_pct": 3.21,
  "mean_final_equity": 11234.56,
  "mean_length": 1234.5,
  "win_rate": 80.0,
  "all_rewards": [...],
  "all_returns": [...],
  "all_final_equities": [...]
}
```

训练配置中也会追加测试结果：

```json
{
  "model": {...},
  "training": {...},
  "environment": {...},
  "test_results": {...},
  "test_date_range": {
    "start": "2025-01-01",
    "end": "2025-10-10"
  }
}
```

---

## 💡 最佳实践

### 1. 时间序列分割

**推荐做法**：按时间顺序分割
```bash
# 训练集: 2020-2024（历史数据）
# 测试集: 2025（未来数据）
--train_sdate 2020-01-01 --train_edate 2024-12-31
--test_sdate 2025-01-01 --test_edate 2025-10-10
```

**避免**：随机分割或测试集时间早于训练集
```bash
# ❌ 错误：测试集时间早于训练集
--train_sdate 2024-01-01 --train_edate 2025-12-31
--test_sdate 2020-01-01 --test_edate 2023-12-31
```

### 2. 数据量建议

- **训练集**: 至少需要 `window_size` 行（默认 288 行）
- **测试集**: 至少需要 `window_size` 行
- **推荐**: 训练集越大越好，测试集至少几千行

### 3. 测试回合数

- **快速验证**: 5-10 回合
- **完整评估**: 20-50 回合
- **统计显著性**: 100+ 回合

### 4. 典型工作流

```bash
# 步骤 1: 快速验证系统
python Train/train.py --mode quick_test \
  --train_sdate 2024-01-01 --train_edate 2024-06-30 \
  --test_sdate 2024-07-01 --test_edate 2024-12-31

# 步骤 2: 完整训练
python Train/train.py \
  --train_sdate 2020-01-01 --train_edate 2024-12-31 \
  --test_sdate 2025-01-01 --test_edate 2025-10-10 \
  --episodes 1000 --test_episodes 20

# 步骤 3: 在其他时间段测试
python Train/train.py --mode test_only \
  --load_model ./models/best_model.pth \
  --test_sdate 2023-01-01 --test_edate 2023-12-31 \
  --test_episodes 30
```

---

## 🔍 故障排除

### 问题 1: 数据不足

```
ValueError: 训练数据不足！需要至少 288 行，实际 100 行
```

**解决方法**:
- 扩大日期范围
- 检查数据文件是否完整
- 确认日期范围内有数据

### 问题 2: 没有 timestamp 列

```
警告: 数据中没有 timestamp 列，将按 80/20 比例分割
```

**说明**: 这不是错误，系统会自动按比例分割

**如果需要按日期分割**: 在数据文件中添加 `timestamp` 列（毫秒时间戳）

### 问题 3: 测试模式缺少模型

```
ValueError: 测试模式需要指定 --load_model 参数加载模型
```

**解决方法**:
```bash
python Train/train.py --mode test_only \
  --load_model ./models/best_model.pth  # 必须指定
```

---

## 📈 评估指标说明

| 指标 | 说明 | 理想值 |
|------|------|--------|
| **平均奖励** | 所有测试回合的平均累积奖励 | 越高越好 |
| **平均收益率** | 最终权益相对初始资金的平均收益 | 正值，越高越好 |
| **胜率** | 收益率为正的回合占比 | > 50% |
| **标准差** | 结果的波动性 | 较小表示稳定 |
| **最高/最低** | 最好和最坏的情况 | 了解风险范围 |

---

## 🎯 示例：完整的训练和测试流程

```bash
# 1. 使用 2020-2024 年数据训练，2025 年数据测试
python Train/train.py \
  --data ./Data/BTCUSDT_futures_volume_5years_5min.csv \
  --train_sdate 2020-01-01 \
  --train_edate 2024-12-31 \
  --test_sdate 2025-01-01 \
  --test_edate 2025-10-10 \
  --episodes 1000 \
  --test_episodes 20 \
  --device cuda

# 输出文件:
# - models/best_model.pth (最佳模型)
# - models/final_model.pth (最终模型)
# - models/test_results.json (测试结果)
# - models/config.json (配置+测试结果)
```

---

## 📞 需要帮助？

- 查看主文档: `Train/README.md`
- 快速参考: `Train/QUICK_REFERENCE.md`
- 架构说明: `TRAIN_ARCHITECTURE.md`

---

**祝测试顺利！** 📊✅

