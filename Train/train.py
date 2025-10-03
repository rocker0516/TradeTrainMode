import os
import math
from sqlite3 import Date
import time
import pathlib
import warnings
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, CallbackList

import sys
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))
from Env.trading_env import TradingEnvironment
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from Train.visualization import plot_training_curve, plot_step_reward_curve, plot_episode_stats

# 讀取 CSV 文件並返回 DataFrame
def read_csv_date_range(csv_path: str, start_date: Date, end_date: Date) -> pd.DataFrame:
    '''
    讀取 CSV 文件並返回 DataFrame
    
    Args:
        csv_path (str): CSV 文件路徑
        start_date (Date): 起始日期
        end_date (Date): 結束日期
    '''
    df = pd.read_csv(csv_path)
    if 'timestamp' not in df.columns:
        raise ValueError("CSV 必須包含 'timestamp' 欄位")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').set_index('timestamp')

    # TradingEnvironment 需要的所有欄位
    required = ['open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume', 
                'volume_ratio', 'long_short_ratio', 'trades', 'quote_volume']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV 缺少必要欄位: {c}")
    df = df[required].astype(float)

    if len(df) == 0:
        raise ValueError('CSV 無資料')

    df_slice = df.loc[df.index >= start_date and df.index <= end_date].copy()
    if len(df_slice) < 1000:
        warnings.warn('近三個月資料樣本過少，訓練效果可能受限')
    return df_slice

# 將 DataFrame 切分為多個 shard
def split_dataframe_into_shards(df: pd.DataFrame, num_shards: int, min_len: int) -> List[pd.DataFrame]:
    '''
    將 DataFrame 切分為多個 shard
    
    Args:
        df (pd.DataFrame): 要切分的 DataFrame
        num_shards (int): 切分的 shard 數量
        min_len (int): 每個 shard 的最小長度
    '''
    n = len(df)
    if n < num_shards * min_len:
        # 若資料不足以平均切分，改為滑動切片
        step = max(min_len, n // num_shards)
        shards = []
        for i in range(num_shards):
            start = max(0, n - (i + 1) * step)
            end = n - i * step
            shard = df.iloc[start:end]
            shards.append(shard)
        shards = list(reversed(shards))
        return shards

    shard_size = n // num_shards
    shards = []
    for i in range(num_shards):
        start = i * shard_size
        end = (i + 1) * shard_size if i < num_shards - 1 else n
        shard = df.iloc[start:end]
        shards.append(shard)
    return shards


class RewardLogWrapper(gym.Wrapper):
    """逐步記錄 reward（以及部分帳戶狀態）到 CSV 的包裝器。"""
    def __init__(self, env: gym.Env, log_file: str):
        super().__init__(env)
        self.log_file = log_file
        self._step_counter = 0
        # 準備資料夾與檔頭
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write('step,reward,total_value,balance,btc_held,current_step\n')

    def reset(self, **kwargs):
        self._step_counter = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._step_counter += 1
        try:
            # 從最底層環境讀取狀態（unwrapped 跳過所有 wrapper）
            base_env = getattr(self.env, 'unwrapped', self.env)
            total_value = getattr(base_env, 'total_value', np.nan)
            balance = getattr(base_env, 'balance', np.nan)
            btc_held = getattr(base_env, 'btc_held', np.nan)
            current_step = getattr(base_env, 'current_step', np.nan)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"{self._step_counter},{float(reward)},{total_value},{balance},{btc_held},{current_step}\n")
        except Exception:
            pass
        return obs, reward, done, truncated, info


class ActionTransformWrapper(gym.Wrapper):
    """將 SAC 動作映射到環境需求，並限制倉位比例。

    - a0 ∈ [-1,1] → 倉位比例 [-position_scale, position_scale]
    - a1 ∈ [-1,1] → 止盈 [0, 50]
    - a2 ∈ [-1,1] → 止損 [0, 20]
    """
    def __init__(self, env: gym.Env, position_scale: float = 0.5):
        super().__init__(env)
        self.position_scale = float(max(0.0, min(position_scale, 1.0)))

    def step(self, action):
        if isinstance(action, np.ndarray) and action.shape[-1] == 3:
            a0 = float(action[0])
            a1 = float(action[1])
            a2 = float(action[2])
            # 倉位比例縮放
            a0_scaled = float(np.clip(a0, -1.0, 1.0)) * self.position_scale
            # 止盈/止損映射
            a1_mapped = (float(np.clip(a1, -1.0, 1.0)) + 1.0) * 0.5 * 50.0
            a2_mapped = (float(np.clip(a2, -1.0, 1.0)) + 1.0) * 0.5 * 20.0
            mapped = np.array([a0_scaled, a1_mapped, a2_mapped], dtype=np.float32)
        else:
            mapped = action
        return self.env.step(mapped)


class EpisodeStatsWrapper(gym.Wrapper):
    """收集每回合統計：期末資金、期間最小資金、最大回撤、回合總 reward、收益率、交易次數。"""
    def __init__(self, env: gym.Env, log_file: str, env_id: int, aggressive_threshold: float = 0.4):
        super().__init__(env)
        self.log_file = log_file
        self.env_id = env_id
        self.aggressive_threshold = aggressive_threshold
        self.episode_idx = 0
        self._episode_reward = 0.0
        self._min_total_value = None
        self._peak_total_value = None
        self._max_drawdown = 0.0
        self._aggressive_count = 0
        self._trade_count = 0
        self._steps = 0
        self._initial_balance = None
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write('env_id,episode,steps,end_total_value,min_total_value,max_drawdown,episode_reward,return_rate,trade_count,aggressive_step_ratio,done_reason\n')

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 初始化統計
        # 從最底層環境讀取資產
        base_env = getattr(self.env, 'unwrapped', self.env)
        current_tv = getattr(base_env, 'total_value', np.nan)
        self._initial_balance = getattr(base_env, 'initial_balance', 10000)
        self._episode_reward = 0.0
        self._min_total_value = current_tv
        self._peak_total_value = current_tv
        self._max_drawdown = 0.0
        self._steps = 0
        self._aggressive_count = 0
        self._trade_count = 0
        self._last_position = 0
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._steps += 1
        self._episode_reward += float(reward)
        # 計算激進步比例（基於實際送進環境的 a0）
        try:
            a0_actual = float(action[0])
            if abs(a0_actual) > self.aggressive_threshold:
                self._aggressive_count += 1
        except Exception:
            pass
        
        # 計算交易次數（倉位變化即為交易）
        base_env = getattr(self.env, 'unwrapped', self.env)
        current_position = getattr(base_env, 'btc_held', 0)
        if hasattr(self, '_last_position') and current_position != self._last_position:
            self._trade_count += 1
        self._last_position = current_position
        
        current_tv = getattr(base_env, 'total_value', np.nan)
        if self._min_total_value is None or (not np.isnan(current_tv) and current_tv < self._min_total_value):
            self._min_total_value = current_tv
        if self._peak_total_value is None or (not np.isnan(current_tv) and current_tv > self._peak_total_value):
            self._peak_total_value = current_tv
        # 計算即時回撤
        if self._peak_total_value and self._peak_total_value > 0 and not np.isnan(current_tv):
            dd = (self._peak_total_value - current_tv) / self._peak_total_value
            if dd > self._max_drawdown:
                self._max_drawdown = dd

        if done or truncated:
            base_env = getattr(self.env, 'unwrapped', self.env)
            end_tv = getattr(base_env, 'total_value', np.nan)
            # 判斷結束原因
            done_reason = None
            if truncated:
                done_reason = 'time_limit'
            else:
                # 優先讀取環境標記
                done_reason = getattr(base_env, 'last_done_reason', None)
            aggressive_ratio = (self._aggressive_count / self._steps) if self._steps > 0 else 0.0
            # 計算收益率
            return_rate = (end_tv - self._initial_balance) / self._initial_balance if self._initial_balance > 0 else 0.0
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{self.env_id},{self.episode_idx},{self._steps},{end_tv},{self._min_total_value},{self._max_drawdown},{self._episode_reward},{return_rate},{self._trade_count},{aggressive_ratio},{done_reason}\n")
            except Exception:
                pass
            self.episode_idx += 1

        return obs, reward, done, truncated, info


def make_env_fn(df_slice: pd.DataFrame, window_size: int, initial_balance: float,
                transaction_fee: float, leverage: int, min_trade_amount: float, seed: int,
                log_dir: str, env_index: int, episode_steps: int = 0, position_scale: float = 0.5):
    def _thunk():
        env = TradingEnvironment(
            df=df_slice,
            initial_balance=initial_balance,
            transaction_fee=transaction_fee,
            window_size=window_size,
            leverage=leverage,
            min_trade_amount=min_trade_amount
        )
        # 每回合步數上限（TimeLimit）
        if episode_steps and episode_steps > 0:
            env = TimeLimit(env, max_episode_steps=episode_steps)
        # 動作映射與限制（放最外層，使後續 wrapper 看到轉換後的動作）
        env = ActionTransformWrapper(env, position_scale=position_scale)
        # 每回合統計
        ep_log_dir = os.path.join(log_dir, 'episode_stats')
        ep_log_file = os.path.join(ep_log_dir, f'env_{env_index}.csv')
        env = EpisodeStatsWrapper(env, log_file=ep_log_file, env_id=env_index, aggressive_threshold=0.4)
        # 逐步 reward 記錄
        step_log_dir = os.path.join(log_dir, 'step_rewards')
        step_log_file = os.path.join(step_log_dir, f'env_{env_index}.csv')
        env = RewardLogWrapper(env, log_file=step_log_file)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _thunk


def determine_device() -> str:
    """確定使用 GPU 或 CPU 進行訓練"""
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def main():
    parser = argparse.ArgumentParser()
    # 指定包含歷史交易數據的 CSV 文件路徑的參數。
    parser.add_argument('--csv', type=str, default=os.path.join('Data', 'BTCUSDT_futures_volume_5years_5min.csv'))
    
    # 指定從 CSV 文件中使用的數據起訖日期的參數。
    parser.add_argument('--start_date', type=int, required=True, help='起始日期')
    parser.add_argument('--end_date', type=int, required=True, help='結束日期')
    
    # 指定用於並行訓練的向量化環境數量的參數。
    parser.add_argument('--vec_envs', type=int, default=16, help='並行環境數量')
    
    # 指定訓練的回合數的參數（每個環境跑的回合數）【必需】
    parser.add_argument('--episodes', type=int, required=True, help='每個環境訓練的回合數（必需參數）')
    
    # 指定每回合的最大步數的參數【必需】
    parser.add_argument('--episode_steps', type=int, required=True, help='每回合最大步數（必需參數）')
    
    # 指定窗口大小的參數，決定每次觀察使用的數據點數量。
    parser.add_argument('--window_size', type=int, default=288)
    
    # 指定交易環境的初始資金的參數。
    parser.add_argument('--initial_balance', type=float, default=10000.0)
    
    # 指定每次交易應用的交易手續費率的參數。
    parser.add_argument('--transaction_fee', type=float, default=0.001)
    
    # 指定交易中使用的槓桿倍數的參數。
    parser.add_argument('--leverage', type=int, default=5)
    
    # 指定交易環境中允許的最小交易金額的參數。
    parser.add_argument('--min_trade_amount', type=float, default=10.0)
    
    # 指定保存訓練日誌和結果的目錄的參數。
    parser.add_argument('--logdir', type=str, default=os.path.join('training_results', 'sac_btc_3m'))
    
    # 指定動作倉位比例縮放的參數（a0 將被限制在 [-position_scale, position_scale]）
    parser.add_argument('--position_scale', type=float, default=0.5, help='倉位比例上限 (0~1)')
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    df = read_csv_date_range(args.csv, args.start_date, args.end_date)

    # 建立 16 個 shard，盡量平均切分 3 個月資料
    min_len = args.window_size + 500  # 保證每個 shard 至少有可交易步數

    shards = split_dataframe_into_shards(df, args.vec_envs, min_len)
    if len(shards) < args.vec_envs:
        # 若仍不足，重複使用最後一個 shard
        while len(shards) < args.vec_envs:
            shards.append(shards[-1])

    # 建立向量化環境
    env_fns = []
    for i in range(args.vec_envs):
        seed = 1000 + i
        env_fns.append(make_env_fn(shards[i], args.window_size, args.initial_balance,
                                   args.transaction_fee, args.leverage, args.min_trade_amount, seed,
                                   log_dir=args.logdir, env_index=i, episode_steps=args.episode_steps,
                                   position_scale=args.position_scale))

    # Windows 上 SubprocVecEnv 需要 if __name__ == '__main__' 保護，這裡已符合
    vec_env = SubprocVecEnv(env_fns) if args.vec_envs > 1 else DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, filename=os.path.join(args.logdir, 'monitor'))

    device = determine_device()#確定使用 GPU 或 CPU 進行訓練
    print(f"Using device: {device}")

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256])
    )#網絡架構

    model = SAC(
        policy='MlpPolicy',#使用 MLP 網絡作為策略網絡
        env=vec_env,#向量化環境
        verbose=1,#打印日志
        tensorboard_log=args.logdir,
        learning_rate=3e-4,#學習率
        buffer_size=1_000_000,#經驗回放緩衝區大小
        batch_size=256,#每次訓練的批次大小
        tau=0.005,#軟更新係數
        gamma=0.99,#折扣因子
        train_freq=(1, 'step'),#每步更新一次參數
        gradient_steps=1,#每次訓練的梯度步數
        target_entropy='auto',#自動調整目標熵
        use_sde=True,#使用狀態依賴探索
        device=device,
        policy_kwargs=policy_kwargs,
    )#創建 SAC 模型

    # 計算訓練總步數：vec_envs × episodes × episode_steps
    total_episodes = args.vec_envs * args.episodes
    total_timesteps = total_episodes * args.episode_steps
    
    # 訓練參數摘要
    print(f"\n{'='*60}")
    print(f"訓練配置摘要")
    print(f"{'='*60}")
    print(f"並行環境數：{args.vec_envs}")
    print(f"每環境回合數：{args.episodes}")
    print(f"每回合步數：{args.episode_steps}")
    print(f"總回合數：{total_episodes} (= {args.vec_envs} × {args.episodes})")
    print(f"總訓練步數：{total_timesteps:,} (= {args.vec_envs} × {args.episodes} × {args.episode_steps})")
    print(f"{'='*60}\n")
    
    # 設置停止條件：以回合數為準
    callbacks = [StopTrainingOnMaxEpisodes(max_episodes=total_episodes, verbose=1)]
    callback = callbacks[0]
    
    # 訓練（留 20% 緩衝以防提前結束）
    start = time.time()
    model.learn(total_timesteps=int(total_timesteps * 1.2), progress_bar=True, callback=callback)
    elapsed = time.time() - start
    print(f"Training finished in {elapsed/60:.2f} min")

    # 保存模型
    pathlib.Path(args.logdir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(args.logdir, 'sac_btc_3m_model.zip')
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # 繪製訓練曲線
    curve_path = os.path.join(args.logdir, 'training_curve.png')
    plot_training_curve(args.logdir, curve_path)
    print(f"Training curve saved to {curve_path}")

    # 繪製逐步 reward 曲線（平滑）
    step_curve_path = os.path.join(args.logdir, 'step_reward_curve.png')
    plot_step_reward_curve(os.path.join(args.logdir, 'step_rewards'), step_curve_path)
    print(f"Step reward curve saved to {step_curve_path}")

    # 繪製每回合統計圖表並匯出彙總 CSV
    ep_stats_dir = os.path.join(args.logdir, 'episode_stats')
    ep_summary_csv = os.path.join(args.logdir, 'episode_stats_summary.csv')
    ep_figure_path = os.path.join(args.logdir, 'episode_stats.png')
    plot_episode_stats(ep_stats_dir, ep_figure_path, combined_csv_out=ep_summary_csv)
    print(f"Episode stats saved to {ep_figure_path} and {ep_summary_csv}")


if __name__ == '__main__':
    main()


