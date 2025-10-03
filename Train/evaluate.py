"""
模型評估腳本 - 簡潔版

用於測試訓練好的 SAC 模型，提供簡潔的命令行接口。

使用範例：
    # 基本測試
    python Train/evaluate.py --model models/sac_btc.zip --episodes 10
    
    # 完整測試
    python Train/evaluate.py \
        --model models/sac_btc.zip \
        --csv Data/BTCUSDT_futures_volume_5years_5min.csv \
        --start_date 20240101 \
        --end_date 20240331 \
        --episodes 50 \
        --output results/eval_report
"""

import os
import sys
import pathlib
import argparse
from typing import Dict, List
from datetime import datetime

import numpy as np
import pandas as pd

# 添加項目根目錄到 sys.path
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from stable_baselines3 import SAC
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from Env.trading_env import TradingEnvironment
from Train.train import ActionTransformWrapper, read_csv_date_range
from Train.visualization import EpisodeStatsPlotter


class EvaluationLogger:
    """評估日誌記錄器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.episodes_data = []
        os.makedirs(log_dir, exist_ok=True)
        
    def log_episode(self, episode_num: int, episode_data: Dict) -> None:
        """記錄單個回合數據"""
        episode_data['episode'] = episode_num
        self.episodes_data.append(episode_data)
        
    def save_results(self) -> str:
        """保存評估結果到 CSV"""
        if not self.episodes_data:
            return None
        
        df = pd.DataFrame(self.episodes_data)
        csv_path = os.path.join(self.log_dir, 'evaluation_results.csv')
        df.to_csv(csv_path, index=False)
        return csv_path
    
    def print_summary(self) -> None:
        """打印評估摘要"""
        if not self.episodes_data:
            print("無評估數據")
            return
        
        df = pd.DataFrame(self.episodes_data)
        
        print(f"\n{'='*70}")
        print(f"評估摘要")
        print(f"{'='*70}")
        print(f"總回合數：{len(df)}")
        print(f"平均收益率：{df['return_rate'].mean()*100:.2f}%")
        print(f"勝率：{(df['return_rate'] > 0).sum() / len(df) * 100:.2f}%")
        print(f"平均最終資產：${df['final_balance'].mean():.2f}")
        print(f"平均最大回撤：{df['max_drawdown'].mean()*100:.2f}%")
        print(f"平均交易次數：{df['trade_count'].mean():.1f}")
        print(f"平均回合步數：{df['steps'].mean():.1f}")
        print(f"{'='*70}\n")
        
        # 詳細統計
        print(f"收益率分佈：")
        print(f"  最小值：{df['return_rate'].min()*100:.2f}%")
        print(f"  25分位：{df['return_rate'].quantile(0.25)*100:.2f}%")
        print(f"  中位數：{df['return_rate'].median()*100:.2f}%")
        print(f"  75分位：{df['return_rate'].quantile(0.75)*100:.2f}%")
        print(f"  最大值：{df['return_rate'].max()*100:.2f}%")


def create_eval_env(df: pd.DataFrame, 
                    initial_balance: float = 10000.0,
                    transaction_fee: float = 0.001,
                    window_size: int = 288,
                    leverage: int = 5,
                    min_trade_amount: float = 10.0,
                    episode_steps: int = 0,
                    position_scale: float = 0.5) -> gym.Env:
    """
    創建評估環境
    
    Args:
        df: 交易數據
        initial_balance: 初始資金
        transaction_fee: 手續費率
        window_size: 觀察窗口大小
        leverage: 槓桿倍數
        min_trade_amount: 最小交易金額
        episode_steps: 每回合最大步數（0表示無限制）
        position_scale: 倉位比例上限
    
    Returns:
        配置好的評估環境
    """
    env = TradingEnvironment(
        df=df,
        initial_balance=initial_balance,
        transaction_fee=transaction_fee,
        window_size=window_size,
        leverage=leverage,
        min_trade_amount=min_trade_amount
    )
    
    # 時間限制（如果設置）
    if episode_steps > 0:
        env = TimeLimit(env, max_episode_steps=episode_steps)
    
    # 動作映射
    env = ActionTransformWrapper(env, position_scale=position_scale)
    
    return env


def evaluate_model(model_path: str,
                   df: pd.DataFrame,
                   num_episodes: int = 10,
                   env_config: Dict = None,
                   logger: EvaluationLogger = None,
                   deterministic: bool = True,
                   render: bool = False) -> List[Dict]:
    """
    評估訓練好的模型
    
    Args:
        model_path: 模型路徑
        df: 測試數據
        num_episodes: 測試回合數
        env_config: 環境配置
        logger: 日誌記錄器
        deterministic: 是否使用確定性策略
        render: 是否渲染（暫不支持）
    
    Returns:
        評估結果列表
    """
    # 載入模型
    print(f"載入模型：{model_path}")
    model = SAC.load(model_path)
    
    # 創建環境
    env_config = env_config or {}
    env = create_eval_env(df, **env_config)
    
    # 評估
    print(f"\n開始評估：{num_episodes} 回合...")
    results = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        trade_count = 0
        last_position = 0
        max_balance = env.unwrapped.initial_balance
        min_balance = env.unwrapped.initial_balance
        
        while not done and not truncated:
            # 模型預測
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # 執行動作
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            # 統計交易次數
            current_position = env.unwrapped.btc_held
            if current_position != last_position:
                trade_count += 1
            last_position = current_position
            
            # 追蹤資產變化
            current_balance = env.unwrapped.total_value
            max_balance = max(max_balance, current_balance)
            min_balance = min(min_balance, current_balance)
        
        # 回合結束統計
        final_balance = env.unwrapped.total_value
        initial_balance = env.unwrapped.initial_balance
        return_rate = (final_balance - initial_balance) / initial_balance
        max_drawdown = (max_balance - min_balance) / max_balance if max_balance > 0 else 0
        
        episode_data = {
            'final_balance': final_balance,
            'return_rate': return_rate,
            'episode_reward': episode_reward,
            'steps': steps,
            'trade_count': trade_count,
            'max_drawdown': max_drawdown,
            'done_reason': 'completed' if not done else 'terminated'
        }
        
        results.append(episode_data)
        
        # 記錄日誌
        if logger:
            logger.log_episode(episode, episode_data)
        
        # 打印進度
        print(f"回合 {episode+1}/{num_episodes}: "
              f"收益率={return_rate*100:.2f}%, "
              f"最終資產=${final_balance:.2f}, "
              f"交易次數={trade_count}")
    
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description='評估訓練好的 SAC 模型')
    
    # 必需參數
    parser.add_argument('--model', type=str, required=True, help='模型路徑（.zip 文件）')
    
    # 測試數據參數
    parser.add_argument('--csv', type=str, 
                       default=os.path.join('Data', 'BTCUSDT_futures_volume_5years_5min.csv'),
                       help='測試數據 CSV 路徑')
    parser.add_argument('--start_date', type=int, required=True, help='測試起始日期（YYYYMMDD）')
    parser.add_argument('--end_date', type=int, required=True, help='測試結束日期（YYYYMMDD）')
    
    # 評估參數
    parser.add_argument('--episodes', type=int, default=10, help='測試回合數（默認10）')
    parser.add_argument('--episode_steps', type=int, default=0, help='每回合最大步數（0=無限制）')
    parser.add_argument('--deterministic', action='store_true', help='使用確定性策略（推薦）')
    
    # 環境參數（應與訓練時保持一致）
    parser.add_argument('--window_size', type=int, default=288, help='觀察窗口大小')
    parser.add_argument('--initial_balance', type=float, default=10000.0, help='初始資金')
    parser.add_argument('--leverage', type=int, default=5, help='槓桿倍數')
    parser.add_argument('--position_scale', type=float, default=0.5, help='倉位比例上限')
    parser.add_argument('--transaction_fee', type=float, default=0.001, help='手續費率')
    parser.add_argument('--min_trade_amount', type=float, default=10.0, help='最小交易金額')
    
    # 輸出參數
    parser.add_argument('--output', type=str, default='evaluation_results', help='輸出目錄')
    
    args = parser.parse_args()
    
    # 驗證模型文件
    if not os.path.exists(args.model):
        print(f"錯誤：模型文件不存在 - {args.model}")
        sys.exit(1)
    
    # 載入測試數據
    print(f"載入測試數據：{args.csv}")
    print(f"時間範圍：{args.start_date} - {args.end_date}")
    
    try:
        df = read_csv_date_range(args.csv, args.start_date, args.end_date)
        print(f"數據載入成功：{len(df)} 筆資料")
    except Exception as e:
        print(f"錯誤：無法載入數據 - {e}")
        sys.exit(1)
    
    # 配置環境參數
    env_config = {
        'initial_balance': args.initial_balance,
        'transaction_fee': args.transaction_fee,
        'window_size': args.window_size,
        'leverage': args.leverage,
        'min_trade_amount': args.min_trade_amount,
        'episode_steps': args.episode_steps,
        'position_scale': args.position_scale
    }
    
    # 創建日誌記錄器
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, timestamp)
    logger = EvaluationLogger(output_dir)
    
    # 打印評估配置
    print(f"\n{'='*70}")
    print(f"評估配置")
    print(f"{'='*70}")
    print(f"模型：{args.model}")
    print(f"測試回合數：{args.episodes}")
    print(f"每回合步數：{args.episode_steps if args.episode_steps > 0 else '無限制'}")
    print(f"策略模式：{'確定性' if args.deterministic else '隨機'}")
    print(f"初始資金：${args.initial_balance}")
    print(f"槓桿倍數：{args.leverage}x")
    print(f"倉位上限：{args.position_scale*100}%")
    print(f"輸出目錄：{output_dir}")
    print(f"{'='*70}\n")
    
    # 執行評估
    try:
        results = evaluate_model(
            model_path=args.model,
            df=df,
            num_episodes=args.episodes,
            env_config=env_config,
            logger=logger,
            deterministic=args.deterministic
        )
        
        # 保存結果
        csv_path = logger.save_results()
        print(f"\n結果已保存：{csv_path}")
        
        # 打印摘要
        logger.print_summary()
        
        # 生成評估報告圖表（如果有足夠數據）
        if len(results) >= 5:
            print("生成評估報告圖表...")
            try:
                # 創建臨時統計文件供繪圖使用
                df_results = pd.DataFrame(results)
                df_results['env_id'] = 0
                df_results['episode'] = range(len(results))
                df_results['end_total_value'] = df_results['final_balance']
                df_results['global_episode'] = range(1, len(results) + 1)
                
                # 保存為標準格式
                stats_dir = os.path.join(output_dir, 'episode_stats')
                os.makedirs(stats_dir, exist_ok=True)
                stats_file = os.path.join(stats_dir, 'env_0.csv')
                df_results.to_csv(stats_file, index=False)
                
                # 繪製圖表
                plotter = EpisodeStatsPlotter()
                figure_path = os.path.join(output_dir, 'evaluation_stats.png')
                plotter.plot(stats_dir, figure_path)
                print(f"圖表已保存：{figure_path}")
                
            except Exception as e:
                print(f"警告：圖表生成失敗 - {e}")
        
        print(f"\n✅ 評估完成！")
        
    except Exception as e:
        print(f"\n❌ 評估失敗：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

