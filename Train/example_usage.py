"""
使用示例

演示如何使用训练好的模型进行交易。
"""

import os
import sys
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Env.trading_env import TradingEnvironment
from Train.models import SAC_LSTM_Model
from Train.config import Config


def load_trained_model(model_path: str, env: TradingEnvironment, device: str = 'cuda') -> SAC_LSTM_Model:
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        env: 交易环境
        device: 设备
        
    Returns:
        加载的模型
    """
    model = SAC_LSTM_Model(
        observation_shape=env.observation_space.shape,
        action_dim=env.action_space.shape[0],
        device=device,
    )
    model.load(model_path)
    return model


def run_episode(env: TradingEnvironment, model: SAC_LSTM_Model, evaluate: bool = True) -> dict:
    """
    运行一个完整回合
    
    Args:
        env: 交易环境
        model: 训练好的模型
        evaluate: 是否为评估模式（不添加探索噪声）
        
    Returns:
        回合统计信息
    """
    state, _ = env.reset()
    model.reset_hidden_state()
    
    episode_reward = 0.0
    episode_length = 0
    done = False
    
    actions_history = []
    rewards_history = []
    positions_history = []
    equity_history = []
    
    while not done:
        # 选择动作
        action = model.select_action(state, evaluate=evaluate)
        
        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 记录信息
        actions_history.append(action)
        rewards_history.append(reward)
        positions_history.append(env.executor.position.size)
        equity_history.append(env.executor.equity(float(env.df.iloc[env.current_step - 1]['close'])))
        
        state = next_state
        episode_reward += reward
        episode_length += 1
    
    return {
        'total_reward': episode_reward,
        'episode_length': episode_length,
        'final_equity': equity_history[-1] if equity_history else env.initial_balance,
        'return_pct': (equity_history[-1] / env.initial_balance - 1) * 100 if equity_history else 0,
        'actions': np.array(actions_history),
        'rewards': np.array(rewards_history),
        'positions': np.array(positions_history),
        'equity': np.array(equity_history),
    }


def evaluate_model(
    model_path: str,
    data_path: str,
    num_episodes: int = 10,
    initial_balance: float = 10000.0,
    device: str = 'cuda'
) -> None:
    """
    评估训练好的模型
    
    Args:
        model_path: 模型文件路径
        data_path: 数据文件路径
        num_episodes: 评估回合数
        initial_balance: 初始资金
        device: 设备
    """
    print("=" * 60)
    print("模型评估")
    print("=" * 60)
    
    # 加载数据
    print(f"\n加载数据: {data_path}")
    df = pd.read_csv(data_path)
    print(f"数据形状: {df.shape}")
    
    # 创建环境
    env = TradingEnvironment(
        df=df,
        initial_balance=initial_balance,
        transaction_fee=0.001,
        window_size=24 * 60 // 5,
        leverage=10.0,
    )
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    model = load_trained_model(model_path, env, device)
    
    # 运行多个回合
    print(f"\n运行 {num_episodes} 个回合进行评估...\n")
    
    all_results = []
    for i in range(num_episodes):
        print(f"回合 {i + 1}/{num_episodes}...", end=" ")
        result = run_episode(env, model, evaluate=True)
        all_results.append(result)
        print(f"奖励: {result['total_reward']:.2f}, 收益率: {result['return_pct']:.2f}%, "
              f"最终权益: ${result['final_equity']:.2f}")
    
    # 统计结果
    rewards = [r['total_reward'] for r in all_results]
    returns = [r['return_pct'] for r in all_results]
    final_equities = [r['final_equity'] for r in all_results]
    
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)
    print(f"平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"最高奖励: {np.max(rewards):.2f}")
    print(f"最低奖励: {np.min(rewards):.2f}")
    print(f"\n平均收益率: {np.mean(returns):.2f}% ± {np.std(returns):.2f}%")
    print(f"最高收益率: {np.max(returns):.2f}%")
    print(f"最低收益率: {np.min(returns):.2f}%")
    print(f"\n平均最终权益: ${np.mean(final_equities):.2f}")
    print(f"胜率: {sum(1 for r in returns if r > 0) / len(returns) * 100:.1f}%")
    print("=" * 60)


def interactive_trading(
    model_path: str,
    data_path: str,
    initial_balance: float = 10000.0,
    device: str = 'cuda'
) -> None:
    """
    交互式交易演示
    
    Args:
        model_path: 模型文件路径
        data_path: 数据文件路径
        initial_balance: 初始资金
        device: 设备
    """
    print("=" * 60)
    print("交互式交易演示")
    print("=" * 60)
    
    # 加载数据和模型
    df = pd.read_csv(data_path)
    env = TradingEnvironment(
        df=df,
        initial_balance=initial_balance,
        transaction_fee=0.001,
        window_size=24 * 60 // 5,
        leverage=10.0,
    )
    model = load_trained_model(model_path, env, device)
    
    # 重置环境
    state, _ = env.reset()
    model.reset_hidden_state()
    
    print("\n开始交易...")
    print("按 Enter 继续下一步，输入 'q' 退出\n")
    
    step = 0
    done = False
    
    while not done:
        # 获取当前价格
        current_price = float(env.df.iloc[env.current_step - 1]['close'])
        
        # 选择动作
        action = model.select_action(state, evaluate=True)
        
        # 显示信息
        print(f"\n步骤 {step + 1}")
        print(f"  当前价格: ${current_price:.2f}")
        print(f"  当前权益: ${env.executor.equity(current_price):.2f}")
        print(f"  当前持仓: {env.executor.position.size:.4f}")
        print(f"  模型动作: 仓位={action[0]:.2f}, 止盈={action[1]*100:.1f}%, 止损={action[2]*100:.1f}%")
        
        # 等待用户输入
        user_input = input("\n[Enter] 继续 / [q] 退出: ").strip().lower()
        if user_input == 'q':
            print("交易演示结束")
            break
        
        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        print(f"  步骤奖励: {reward:.4f}")
        
        state = next_state
        step += 1
    
    # 显示最终结果
    final_price = float(env.df.iloc[env.current_step - 1]['close'])
    final_equity = env.executor.equity(final_price)
    print(f"\n" + "=" * 60)
    print(f"交易结束")
    print(f"  初始资金: ${initial_balance:.2f}")
    print(f"  最终权益: ${final_equity:.2f}")
    print(f"  收益率: {(final_equity / initial_balance - 1) * 100:.2f}%")
    print(f"  总步数: {step}")
    print("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='使用训练好的模型进行交易')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--data', type=str, required=True, help='数据文件路径')
    parser.add_argument('--mode', type=str, default='evaluate', 
                        choices=['evaluate', 'interactive'],
                        help='运行模式')
    parser.add_argument('--episodes', type=int, default=10, help='评估回合数')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--balance', type=float, default=10000.0, help='初始资金')
    
    args = parser.parse_args()
    
    if args.mode == 'evaluate':
        evaluate_model(
            model_path=args.model,
            data_path=args.data,
            num_episodes=args.episodes,
            initial_balance=args.balance,
            device=args.device,
        )
    elif args.mode == 'interactive':
        interactive_trading(
            model_path=args.model,
            data_path=args.data,
            initial_balance=args.balance,
            device=args.device,
        )


if __name__ == '__main__':
    # 示例用法
    # python Train/example_usage.py --model ./models/best_model.pth --data ./Data/BTCUSDT_futures_volume_5years_5min.csv --mode evaluate
    # python Train/example_usage.py --model ./models/best_model.pth --data ./Data/BTCUSDT_futures_volume_5years_5min.csv --mode interactive
    main()

