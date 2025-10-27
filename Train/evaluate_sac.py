"""
SAC 模型評估腳本

評估訓練好的 SAC 模型性能。
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加父目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from Env.trading_env import TradingEnvironment
from Env.reward import RewardCalculator


def create_environment(
    data_path: str,
    initial_balance: float = 10000.0,
    leverage: float = 10.0,
    transaction_fee: float = 0.001,
    window_size: int = 288,
    reward_mode: str = 'delta_equity',
    margin_mode: str = 'isolated'
) -> TradingEnvironment:
    """創建交易環境"""
    df = pd.read_csv(data_path)
    reward_calculator = RewardCalculator(mode=reward_mode, scale=1.0)
    
    env = TradingEnvironment(
        df=df,
        initial_balance=initial_balance,
        transaction_fee=transaction_fee,
        window_size=window_size,
        leverage=leverage,
        min_balance=0.0,
        min_trade_qty=0.001,
        margin_mode=margin_mode,
        reward_calculator=reward_calculator
    )
    
    return env


def evaluate_model(
    model_path: str,
    data_path: str,
    num_episodes: int = 10,
    initial_balance: float = 10000.0,
    leverage: float = 10.0,
    window_size: int = 288,
    verbose: bool = True
) -> None:
    """
    評估模型性能
    
    Args:
        model_path: 模型路徑
        data_path: 數據路徑
        num_episodes: 評估回合數
        initial_balance: 初始資金
        leverage: 槓桿倍數
        window_size: 觀察窗口大小
        verbose: 是否輸出詳細信息
    """
    # 加載模型
    print(f"加載模型: {model_path}")
    model = SAC.load(model_path)
    print("模型加載成功\n")
    
    # 創建環境
    env = create_environment(
        data_path=data_path,
        initial_balance=initial_balance,
        leverage=leverage,
        window_size=window_size
    )
    
    # 運行多個回合
    all_rewards = []
    all_balances = []
    all_profits = []
    all_profit_rates = []
    
    print(f"{'='*60}")
    print(f"開始評估 - 共 {num_episodes} 回合")
    print(f"{'='*60}\n")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # 使用模型預測動作
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            # 詳細輸出
            if verbose and steps % 100 == 0:
                current_price = float(env.df.iloc[env.current_step - 1]['close'])
                print(
                    f"  Episode {episode+1} | Step {steps:4d} | "
                    f"Action: {action[0]:6.3f} | "
                    f"Position: {env.executor.position.size:7.4f} | "
                    f"Price: {current_price:10.2f} | "
                    f"Balance: {env.total_value:10.2f} | "
                    f"Reward: {reward:8.4f}"
                )
        
        # 記錄結果
        final_balance = float(env.total_value)
        profit = final_balance - initial_balance
        profit_rate = (profit / initial_balance) * 100
        
        all_rewards.append(episode_reward)
        all_balances.append(final_balance)
        all_profits.append(profit)
        all_profit_rates.append(profit_rate)
        
        print(
            f"\nEpisode {episode+1}/{num_episodes} 完成:\n"
            f"  總獎勵: {episode_reward:.2f}\n"
            f"  步數: {steps}\n"
            f"  最終資金: {final_balance:.2f}\n"
            f"  盈虧: {profit:.2f} ({profit_rate:+.2f}%)\n"
        )
    
    # 總結統計
    print(f"{'='*60}")
    print("評估總結:")
    print(f"{'='*60}")
    print(f"  平均獎勵: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"  平均資金: {np.mean(all_balances):.2f} ± {np.std(all_balances):.2f}")
    print(f"  平均盈虧: {np.mean(all_profits):.2f} ± {np.std(all_profits):.2f}")
    print(f"  平均盈虧率: {np.mean(all_profit_rates):+.2f}% ± {np.std(all_profit_rates):.2f}%")
    print(f"  勝率: {sum(1 for p in all_profits if p > 0) / len(all_profits) * 100:.1f}%")
    print(f"  最佳盈虧: {max(all_profits):.2f} ({max(all_profit_rates):+.2f}%)")
    print(f"  最差盈虧: {min(all_profits):.2f} ({min(all_profit_rates):+.2f}%)")
    print(f"{'='*60}")


def main() -> None:
    """主函數"""
    parser = argparse.ArgumentParser(description='SAC 模型評估')
    
    parser.add_argument('--model', type=str, required=True,
                       help='模型路徑（.zip 文件）')
    parser.add_argument('--data', type=str, 
                       default='./Data/BTCUSDT_futures_volume_5years_5min.csv',
                       help='數據路徑')
    parser.add_argument('--episodes', type=int, default=10,
                       help='評估回合數')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                       help='初始資金')
    parser.add_argument('--leverage', type=float, default=10.0,
                       help='槓桿倍數')
    parser.add_argument('--window_size', type=int, default=288,
                       help='觀察窗口大小')
    parser.add_argument('--quiet', action='store_true',
                       help='安靜模式（不輸出詳細信息）')
    
    args = parser.parse_args()
    
    # 檢查模型文件
    if not Path(args.model).exists():
        print(f"錯誤: 模型文件不存在: {args.model}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("SAC 模型評估")
    print(f"{'='*60}\n")
    
    try:
        evaluate_model(
            model_path=args.model,
            data_path=args.data,
            num_episodes=args.episodes,
            initial_balance=args.initial_balance,
            leverage=args.leverage,
            window_size=args.window_size,
            verbose=not args.quiet
        )
        
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

