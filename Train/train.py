"""
主训练脚本

使用 SAC + LSTM 训练交易模型。
支持模块化配置，便于更换模型和调整参数。
"""

import os
import sys
import argparse
import pandas as pd
import json
from datetime import datetime
from typing import Tuple, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Env.trading_env import TradingEnvironment
from Train.models import SAC_LSTM_Model
from Train.trainers import SACTrainer
from Train.config import Config, get_default_config, get_quick_test_config


def load_data(data_path: str, parse_dates: bool = True) -> pd.DataFrame:
    """
    加载交易数据
    
    Args:
        data_path: 数据文件路径
        parse_dates: 是否解析日期列
        
    Returns:
        DataFrame
    """
    print(f"加载数据: {data_path}")
    
    # 尝试加载数据
    df = pd.read_csv(data_path)
    
    # 如果存在 timestamp 列，转换为 datetime（兼容 秒 / 毫秒 / 已格式化字符串）
    if parse_dates and 'timestamp' in df.columns:
        ts = df['timestamp']
        try:
            if pd.api.types.is_numeric_dtype(ts):
                # 数值时间戳：根据数量级判断单位（>1e11 视为毫秒，否则视为秒）
                max_val = float(pd.to_numeric(ts, errors='coerce').max())
                unit = 'ms' if max_val > 1e11 else 's'
                df['timestamp'] = pd.to_datetime(ts, unit=unit, utc=False)
            else:
                # 字符串时间戳：直接解析常见格式
                parsed = pd.to_datetime(ts, errors='coerce', utc=False)
                # 如果解析失败过多，尝试当作数字再按单位推断
                if parsed.isna().mean() > 0.5:
                    as_num = pd.to_numeric(ts, errors='coerce')
                    max_val = float(as_num.max()) if not pd.isna(as_num.max()) else 0.0
                    unit = 'ms' if max_val > 1e11 else 's'
                    parsed = pd.to_datetime(as_num, unit=unit, errors='coerce', utc=False)
                if parsed.isna().any():
                    bad_idx = int(parsed[parsed.isna()].index[0])
                    bad_val = ts.iloc[bad_idx]
                    raise ValueError(f"无法解析 timestamp，示例问题值: index={bad_idx}, value={bad_val}")
                df['timestamp'] = parsed
        except Exception as e:
            raise ValueError(f"解析 timestamp 失败: {e}")
        print(f"数据时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
    
    print(f"数据形状: {df.shape}")
    print(f"数据列: {list(df.columns)}")
    
    return df


def split_data_by_date(
    df: pd.DataFrame,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按日期范围分割训练集和测试集
    
    Args:
        df: 完整数据
        train_start: 训练数据开始日期 (YYYY-MM-DD)
        train_end: 训练数据结束日期 (YYYY-MM-DD)
        test_start: 测试数据开始日期 (YYYY-MM-DD)
        test_end: 测试数据结束日期 (YYYY-MM-DD)
        
    Returns:
        (训练数据, 测试数据)
    """
    # 如果没有 timestamp 列，按索引比例分割
    if 'timestamp' not in df.columns:
        print("警告: 数据中没有 timestamp 列，将按 80/20 比例分割")
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"\n数据分割（按索引）:")
        print(f"  训练集: {len(train_df)} 行")
        print(f"  测试集: {len(test_df)} 行")
        
        return train_df, test_df
    
    # 确保 timestamp 是 datetime 类型（与 load_data 的兼容策略一致）
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        ts = df['timestamp']
        if pd.api.types.is_numeric_dtype(ts):
            max_val = float(pd.to_numeric(ts, errors='coerce').max())
            unit = 'ms' if max_val > 1e11 else 's'
            df['timestamp'] = pd.to_datetime(ts, unit=unit, errors='coerce', utc=False)
        else:
            parsed = pd.to_datetime(ts, errors='coerce', utc=False)
            if parsed.isna().mean() > 0.5:
                as_num = pd.to_numeric(ts, errors='coerce')
                max_val = float(as_num.max()) if not pd.isna(as_num.max()) else 0.0
                unit = 'ms' if max_val > 1e11 else 's'
                parsed = pd.to_datetime(as_num, unit=unit, errors='coerce', utc=False)
            df['timestamp'] = parsed
    
    # 分割训练集
    train_df = df.copy()
    if train_start:
        train_start_dt = pd.to_datetime(train_start)
        train_df = train_df[train_df['timestamp'] >= train_start_dt]
    if train_end:
        train_end_dt = pd.to_datetime(train_end)
        train_df = train_df[train_df['timestamp'] <= train_end_dt]
    
    # 分割测试集
    test_df = df.copy()
    if test_start:
        test_start_dt = pd.to_datetime(test_start)
        test_df = test_df[test_df['timestamp'] >= test_start_dt]
    if test_end:
        test_end_dt = pd.to_datetime(test_end)
        test_df = test_df[test_df['timestamp'] <= test_end_dt]
    
    # 删除 timestamp 列（环境不需要）
    if 'timestamp' in train_df.columns:
        train_df = train_df.drop(columns=['timestamp'])
    if 'timestamp' in test_df.columns:
        test_df = test_df.drop(columns=['timestamp'])
    
    print(f"\n数据分割（按日期）:")
    print(f"  训练集: {len(train_df)} 行 ({train_start} 至 {train_end})")
    print(f"  测试集: {len(test_df)} 行 ({test_start} 至 {test_end})")
    
    return train_df, test_df


def create_environment(df: pd.DataFrame, config: Config) -> TradingEnvironment:
    """
    创建交易环境
    
    Args:
        df: 交易数据
        config: 配置对象
        
    Returns:
        交易环境
    """
    env_config = config.environment
    
    env = TradingEnvironment(
        df=df,
        initial_balance=env_config.initial_balance,
        transaction_fee=env_config.transaction_fee,
        window_size=env_config.window_size,
        leverage=env_config.leverage,
        min_balance=env_config.min_balance,
        min_trade_qty=env_config.min_trade_qty,
        max_stop_loss_percent=env_config.max_stop_loss_percent,
        max_take_profit_percent=env_config.max_take_profit_percent,
    )
    
    print(f"\n环境信息:")
    print(f"  观察空间: {env.observation_space.shape}")
    print(f"  动作空间: {env.action_space.shape}")
    print(f"  初始资金: {env_config.initial_balance}")
    print(f"  杠杆倍数: {env_config.leverage}")
    
    return env


def create_model(env: TradingEnvironment, config: Config) -> SAC_LSTM_Model:
    """
    创建 SAC + LSTM 模型
    
    Args:
        env: 交易环境
        config: 配置对象
        
    Returns:
        SAC + LSTM 模型
    """
    model_config = config.model
    
    model = SAC_LSTM_Model(
        observation_shape=env.observation_space.shape,
        action_dim=env.action_space.shape[0],
        device=config.training.device,
        lstm_hidden_dim=model_config.lstm_hidden_dim,
        lstm_layers=model_config.lstm_layers,
        hidden_dim=model_config.hidden_dim,
        lr=model_config.lr,
        gamma=model_config.gamma,
        tau=model_config.tau,
        alpha=model_config.alpha,
        auto_alpha=model_config.auto_alpha,
    )
    
    print(f"\n模型信息:")
    print(f"  LSTM 隐藏维度: {model_config.lstm_hidden_dim}")
    print(f"  LSTM 层数: {model_config.lstm_layers}")
    print(f"  全连接隐藏维度: {model_config.hidden_dim}")
    print(f"  学习率: {model_config.lr}")
    print(f"  折扣因子: {model_config.gamma}")
    print(f"  设备: {model.device}")
    
    return model


def create_trainer(env: TradingEnvironment, model: SAC_LSTM_Model, config: Config) -> SACTrainer:
    """
    创建训练器
    
    Args:
        env: 交易环境
        model: SAC 模型
        config: 配置对象
        
    Returns:
        SAC 训练器
    """
    trainer_config = config.training.to_dict()
    
    trainer = SACTrainer(
        env=env,
        model=model,
        config=trainer_config,
    )
    
    print(f"\n训练器信息:")
    print(f"  经验回放缓冲区: {trainer_config['buffer_size']}")
    print(f"  批次大小: {trainer_config['batch_size']}")
    print(f"  预热步数: {trainer_config['warmup_steps']}")
    
    return trainer


def test_model(
    model: SAC_LSTM_Model,
    test_env: TradingEnvironment,
    num_episodes: int = 5,
    save_dir: str = './models'
) -> dict:
    """
    测试模型性能
    
    Args:
        model: 训练好的模型
        test_env: 测试环境
        num_episodes: 测试回合数
        save_dir: 结果保存目录
        
    Returns:
        测试结果字典
    """
    print("\n" + "=" * 60)
    print("开始测试模型")
    print("=" * 60)
    
    test_rewards = []
    test_lengths = []
    test_returns = []
    final_equities = []
    
    for episode in range(1, num_episodes + 1):
        state, _ = test_env.reset()
        model.reset_hidden_state()
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        print(f"\n测试回合 {episode}/{num_episodes}...", end=" ")
        
        while not done:
            # 使用评估模式（不添加探索噪声）
            action = model.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # 计算最终权益和收益率
        current_price = float(test_env.df.iloc[test_env.current_step - 1]['close'])
        final_equity = test_env.executor.equity(current_price)
        return_pct = (final_equity / test_env.initial_balance - 1) * 100
        
        test_rewards.append(episode_reward)
        test_lengths.append(episode_length)
        test_returns.append(return_pct)
        final_equities.append(final_equity)
        
        print(f"奖励: {episode_reward:.2f}, 收益率: {return_pct:.2f}%, 最终权益: ${final_equity:.2f}")
    
    # 计算统计结果
    import numpy as np
    results = {
        'num_episodes': num_episodes,
        'mean_reward': float(np.mean(test_rewards)),
        'std_reward': float(np.std(test_rewards)),
        'max_reward': float(np.max(test_rewards)),
        'min_reward': float(np.min(test_rewards)),
        'mean_return_pct': float(np.mean(test_returns)),
        'std_return_pct': float(np.std(test_returns)),
        'max_return_pct': float(np.max(test_returns)),
        'min_return_pct': float(np.min(test_returns)),
        'mean_final_equity': float(np.mean(final_equities)),
        'mean_length': float(np.mean(test_lengths)),
        'win_rate': float(sum(1 for r in test_returns if r > 0) / len(test_returns) * 100),
        'all_rewards': test_rewards,
        'all_returns': test_returns,
        'all_final_equities': final_equities,
    }
    
    # 打印测试摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    print(f"测试回合数: {results['num_episodes']}")
    print(f"\n平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"最高奖励: {results['max_reward']:.2f}")
    print(f"最低奖励: {results['min_reward']:.2f}")
    print(f"\n平均收益率: {results['mean_return_pct']:.2f}% ± {results['std_return_pct']:.2f}%")
    print(f"最高收益率: {results['max_return_pct']:.2f}%")
    print(f"最低收益率: {results['min_return_pct']:.2f}%")
    print(f"\n平均最终权益: ${results['mean_final_equity']:.2f}")
    print(f"平均回合长度: {results['mean_length']:.1f}")
    print(f"胜率: {results['win_rate']:.1f}%")
    print("=" * 60)
    
    # 保存测试结果
    results_file = os.path.join(save_dir, 'test_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n测试结果已保存至: {results_file}")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练 SAC + LSTM 交易模型')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--data', type=str, default="./Data/BTCUSDT_futures_volume_5years_5min.csv", help='数据文件路径')
    parser.add_argument('--train_sdate', type=str, default='2020-01-01', help='训练数据开始日期 (YYYY-MM-DD)')
    parser.add_argument('--train_edate', type=str, default='2024-12-31', help='训练数据结束日期 (YYYY-MM-DD)')
    parser.add_argument('--test_sdate', type=str, default='2025-01-01', help='测试数据开始日期 (YYYY-MM-DD)')
    parser.add_argument('--test_edate', type=str, default='2025-10-10', help='测试数据结束日期 (YYYY-MM-DD)')
    parser.add_argument('--mode', type=str, default='default', choices=['default', 'quick_test', 'test_only'], 
                        help='运行模式 (default: 完整训练, quick_test: 快速测试, test_only: 仅测试)')
    parser.add_argument('--load_model', type=str, default=None, help='加载预训练模型路径')
    parser.add_argument('--episodes', type=int, default=None, help='训练回合数（覆盖配置）')
    parser.add_argument('--test_episodes', type=int, default=5, help='测试回合数')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
                        help='训练设备（覆盖配置）')
    parser.add_argument('--no_train', action='store_true', help='跳过训练，仅测试模型')
    parser.add_argument('--no_test', action='store_true', help='训练后不进行测试')
    
    args = parser.parse_args()
    
    # 判断运行模式
    test_only = args.mode == 'test_only' or args.no_train
    skip_test = args.no_test and not test_only
    
    # 加载配置
    if args.config:
        config = Config.load(args.config)
    elif args.mode == 'quick_test':
        config = get_quick_test_config()
        print("使用快速测试配置")
    else:
        config = get_default_config()
        print("使用默认配置")
    
    # 覆盖配置
    if args.data:
        config.environment.data_path = args.data
    if args.episodes:
        config.training.total_episodes = args.episodes
    if args.device:
        config.training.device = args.device
    
    # 创建保存目录
    os.makedirs(config.training.save_dir, exist_ok=True)
    os.makedirs(config.training.log_dir, exist_ok=True)
    
    # 保存配置
    config.save(os.path.join(config.training.save_dir, 'config.json'))
    
    # 加载完整数据并分割训练/测试集
    print("\n" + "=" * 60)
    print("数据加载与分割")
    print("=" * 60)
    
    full_df = load_data(config.environment.data_path, parse_dates=True)
    
    train_df, test_df = split_data_by_date(
        full_df,
        train_start=args.train_sdate,
        train_end=args.train_edate,
        test_start=args.test_sdate,
        test_end=args.test_edate
    )
    
    # 验证数据
    if len(train_df) < config.environment.window_size:
        raise ValueError(f"训练数据不足！需要至少 {config.environment.window_size} 行，实际 {len(train_df)} 行")
    if len(test_df) < config.environment.window_size:
        raise ValueError(f"测试数据不足！需要至少 {config.environment.window_size} 行，实际 {len(test_df)} 行")
    
    # 创建训练环境和模型
    if not test_only:
        print("\n" + "=" * 60)
        print("创建训练环境")
        print("=" * 60)
        train_env = create_environment(train_df, config)
        model = create_model(train_env, config)
        
        # 加载预训练模型（继续训练）
        if args.load_model:
            print(f"\n加载预训练模型: {args.load_model}")
            model.load(args.load_model)
        
        # 创建训练器
        trainer = create_trainer(train_env, model, config)
        
        # 开始训练
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        try:
            trainer.train(
                total_episodes=config.training.total_episodes,
                eval_interval=config.training.eval_interval,
            )
        except KeyboardInterrupt:
            print("\n\n训练被用户中断")
            print("保存当前模型...")
            trainer.save_checkpoint(f"{config.training.save_dir}/interrupted_model.pth")
        
        print("\n训练完成！")
        print(f"模型保存在: {config.training.save_dir}")
        print(f"日志保存在: {config.training.log_dir}")
    else:
        # 仅测试模式：加载已训练模型
        print("\n" + "=" * 60)
        print("仅测试模式：加载已训练模型")
        print("=" * 60)
        
        if not args.load_model:
            raise ValueError("测试模式需要指定 --load_model 参数加载模型")
        
        # 创建一个临时环境用于获取观察和动作空间
        temp_env = create_environment(test_df, config)
        model = create_model(temp_env, config)
        
        print(f"\n加载模型: {args.load_model}")
        model.load(args.load_model)
    
    # 测试模型
    if not skip_test:
        print("\n" + "=" * 60)
        print("创建测试环境")
        print("=" * 60)
        test_env = create_environment(test_df, config)
        
        test_results = test_model(
            model=model,
            test_env=test_env,
            num_episodes=args.test_episodes,
            save_dir=config.training.save_dir
        )
        
        # 如果是训练后测试，追加结果到训练配置
        if not test_only:
            config_path = os.path.join(config.training.save_dir, 'config.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
            saved_config['test_results'] = test_results
            saved_config['test_date_range'] = {
                'start': args.test_sdate,
                'end': args.test_edate
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(saved_config, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("任务完成！")
    print("=" * 60)
    if not test_only:
        print(f"模型保存在: {config.training.save_dir}")
    if not skip_test:
        print(f"测试结果保存在: {os.path.join(config.training.save_dir, 'test_results.json')}")


if __name__ == '__main__':
    main()

