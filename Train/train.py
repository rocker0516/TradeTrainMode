import os
import math
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
from trading_env import TradingEnvironment
import gymnasium as gym
from gymnasium.wrappers import TimeLimit


def read_csv_last_months(csv_path: str, months: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'timestamp' not in df.columns:
        raise ValueError("CSV 必須包含 'timestamp' 欄位")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').set_index('timestamp')

    required = ['open', 'high', 'low', 'close', 'volume']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV 缺少必要欄位: {c}")
    df = df[required].astype(float)

    if len(df) == 0:
        raise ValueError('CSV 無資料')

    end_time = df.index.max()
    start_time = end_time - pd.DateOffset(months=months)
    df_slice = df.loc[df.index >= start_time].copy()
    if len(df_slice) < 1000:
        warnings.warn('近三個月資料樣本過少，訓練效果可能受限')
    return df_slice


def split_dataframe_into_shards(df: pd.DataFrame, num_shards: int, min_len: int) -> List[pd.DataFrame]:
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
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"{self._step_counter},{float(reward)},{getattr(self.env, 'total_value', np.nan)},{getattr(self.env, 'balance', np.nan)},{getattr(self.env, 'btc_held', np.nan)},{getattr(self.env, 'current_step', np.nan)}\n")
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
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def plot_training_curve(monitor_logs_dir: str, output_path: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import glob

    # 兼容不同命名格式：monitor.csv、monitor_*.csv、*.monitor.csv
    csv_files = []
    csv_files += glob.glob(os.path.join(monitor_logs_dir, 'monitor.csv'))
    csv_files += glob.glob(os.path.join(monitor_logs_dir, 'monitor_*.csv'))
    csv_files += glob.glob(os.path.join(monitor_logs_dir, '*.monitor.csv'))
    if len(csv_files) == 0:
        # SubprocVecEnv + Monitor 會將單環境日誌命名為 monitor.csv 於每個子資料夾
        csv_files += glob.glob(os.path.join(monitor_logs_dir, '*', 'monitor.csv'))
        csv_files += glob.glob(os.path.join(monitor_logs_dir, '*', 'monitor_*.csv'))
        csv_files += glob.glob(os.path.join(monitor_logs_dir, '*', '*.monitor.csv'))

    episode_points: List[Tuple[float, float]] = []  # (timestep, episode_reward)
    for f in csv_files:
        try:
            # 略過前兩行 header (Monitor 格式)
            df = pd.read_csv(f, skiprows=2)
            if {'l', 'r', 't'}.issubset(df.columns):
                # l: episode length, r: reward sum, t: time
                df['timestep'] = df['l'].cumsum()
                for _, row in df.iterrows():
                    episode_points.append((row['timestep'], row['r']))
        except Exception:
            continue

    if len(episode_points) == 0:
        return

    episode_points.sort(key=lambda x: x[0])
    timesteps = np.array([p[0] for p in episode_points])
    rewards = np.array([p[1] for p in episode_points])

    # 平滑處理
    window = max(1, len(rewards) // 20)
    if window > 1:
        kernel = np.ones(window) / window
        rewards_smooth = np.convolve(rewards, kernel, mode='same')
    else:
        rewards_smooth = rewards

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, rewards, alpha=0.3, label='Episode reward')
    plt.plot(timesteps, rewards_smooth, color='C1', label='Smoothed')
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title('SAC Training Curve (Episode Returns)')
    plt.legend()
    plt.tight_layout()
    pathlib.Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_step_reward_curve(step_logs_dir: str, output_path: str, smooth_window: int = 500) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import glob

    csv_files = glob.glob(os.path.join(step_logs_dir, 'env_*.csv'))
    if len(csv_files) == 0:
        return

    def smooth(y: np.ndarray, w: int) -> np.ndarray:
        if len(y) < 3 or w <= 1:
            return y
        w = min(w, max(1, len(y)//10))
        kernel = np.ones(w, dtype=float) / w
        return np.convolve(y, kernel, mode='same')

    plt.figure(figsize=(10, 5))
    total_series = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if 'step' in df.columns and 'reward' in df.columns:
                y = df['reward'].values.astype(float)
                y_s = smooth(y, smooth_window)
                plt.plot(df['step'].values, y_s, alpha=0.35)
                total_series.append(y)
        except Exception:
            continue

    if len(total_series) > 0:
        # 繪製平均曲線
        max_len = max(len(s) for s in total_series)
        padded = []
        for s in total_series:
            if len(s) < max_len:
                pad = np.full(max_len - len(s), np.nan)
                s = np.concatenate([s, pad])
            padded.append(s)
        arr = np.vstack(padded)
        mean_series = np.nanmean(arr, axis=0)
        mean_series = smooth(mean_series, smooth_window)
        plt.plot(np.arange(1, len(mean_series)+1), mean_series, color='C1', linewidth=2.0, label='Mean (smoothed)')

    plt.title('Per-Step Reward (smoothed)')
    plt.xlabel('Env Step (per-env)')
    plt.ylabel('Reward')
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_episode_stats(ep_logs_dir: str, output_path: str, combined_csv_out: str | None = None) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import glob

    csv_files = glob.glob(os.path.join(ep_logs_dir, 'env_*.csv'))
    if len(csv_files) == 0:
        return

    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception:
            continue
    if len(frames) == 0:
        return

    df = pd.concat(frames, ignore_index=True)
    # 統一排序與全域回合編號
    df = df.sort_values(['env_id', 'episode']).reset_index(drop=True)
    df['global_episode'] = np.arange(1, len(df) + 1)

    # 可選：匯出彙總 CSV
    if combined_csv_out is not None:
        try:
            df.to_csv(combined_csv_out, index=False)
        except Exception:
            pass

    # 計算勝率與交易統計（加入數值保護）
    df['return_rate'] = df['return_rate'].clip(-10, 10)  # 限制收益率在合理範圍 (-1000% ~ +1000%)
    win_rate = (df['return_rate'] > 0).sum() / len(df) if len(df) > 0 else 0
    avg_return = df['return_rate'].mean() if 'return_rate' in df.columns else 0
    avg_trades = df['trade_count'].mean() if 'trade_count' in df.columns else 0
    
    # 作圖：2x3 子圖
    fig = plt.figure(figsize=(15, 10))
    
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(df['global_episode'], df['return_rate'] * 100, marker='o', linewidth=1, markersize=3, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax1.set_title(f'Return Rate per Episode (Win Rate: {win_rate*100:.1f}%)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return Rate (%)')
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(df['global_episode'], df['end_total_value'], marker='o', linewidth=1, markersize=3, alpha=0.6, color='C1')
    ax2.axhline(y=df['end_total_value'].iloc[0] if len(df) > 0 else 10000, color='r', linestyle='--', alpha=0.3, label='Initial')
    ax2.set_title('End Total Value per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('End Total Value')
    ax2.legend()

    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(df['global_episode'], df['max_drawdown'], marker='o', linewidth=1, markersize=3, alpha=0.6, color='C2')
    ax3.set_title('Max Drawdown per Episode')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Max Drawdown')

    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(df['global_episode'], df['trade_count'], marker='o', linewidth=1, markersize=3, alpha=0.6, color='C3')
    ax4.set_title(f'Trade Count per Episode (Avg: {avg_trades:.1f})')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Trade Count')
    
    ax5 = plt.subplot(2, 3, 5)
    # 繪製累積收益率（加入溢出保護）
    returns_clamped = df['return_rate'].clip(-0.5, 2.0)  # 單回合收益率限制在 -50% ~ +200%
    cumulative_return = (returns_clamped + 1).cumprod() - 1
    cumulative_return = cumulative_return.clip(-1, 100)  # 累積收益限制在 -100% ~ +10000%
    ax5.plot(df['global_episode'], cumulative_return * 100, linewidth=2, color='C4')
    ax5.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax5.set_title('Cumulative Return Rate (capped)')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Cumulative Return (%)')
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(df['global_episode'], df['episode_reward'], marker='o', linewidth=1, markersize=3, alpha=0.6, color='C5')
    ax6.set_title('Episode Total Reward (for reference)')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Total Reward')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    # 輸出統計摘要
    print(f"\n=== 訓練統計摘要 ===")
    print(f"總回合數: {len(df)}")
    print(f"勝率: {win_rate*100:.2f}%")
    print(f"平均收益率: {avg_return*100:.2f}%")
    print(f"平均交易次數/回合: {avg_trades:.1f}")
    final_cum_return = cumulative_return.iloc[-1]*100 if len(cumulative_return) > 0 else 0
    print(f"最終累積收益: {final_cum_return:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    # 指定包含歷史交易數據的 CSV 文件路徑的參數。
    parser.add_argument('--csv', type=str, default=os.path.join('Data', 'BTCUSDT_futures_volume_5years_5min.csv'))
    
    # 指定從 CSV 文件中使用的數據月份數的參數。
    parser.add_argument('--months', type=int, default=3)
    
    # 指定用於並行訓練的向量化環境數量的參數。
    parser.add_argument('--vec_envs', type=int, default=16)
    
    # 指定訓練的總步數的參數。
    parser.add_argument('--total_timesteps', type=int, default=100000)  # 總訓練步數
    
    # 指定訓練的回合數的參數（每個環境平均跑的回合數）。若 >0，總回合數 = episodes × vec_envs
    parser.add_argument('--episodes', type=int, default=0, help='>0 時以回合數為停止條件（每個環境平均跑此數量）')
    
    # 指定每回合的最大步數的參數。用於設置每回合的時間限制。
    parser.add_argument('--episode_steps', type=int, default=0, help='每回合最大步數（TimeLimit）')
    
    # 指定窗口大小的參數，決定每次觀察使用的數據點數量。
    parser.add_argument('--window_size', type=int, default=120)
    
    # 指定交易環境的初始資金的參數。
    parser.add_argument('--initial_balance', type=float, default=100.0)
    
    # 指定每次交易應用的交易手續費率的參數。
    parser.add_argument('--transaction_fee', type=float, default=0.001)
    
    # 指定交易中使用的槓桿倍數的參數。
    parser.add_argument('--leverage', type=int, default=10)
    
    # 指定交易環境中允許的最小交易金額的參數。
    parser.add_argument('--min_trade_amount', type=float, default=10.0)
    
    # 指定保存訓練日誌和結果的目錄的參數。
    parser.add_argument('--logdir', type=str, default=os.path.join('training_results', 'sac_btc_3m'))
    
    # 指定動作倉位比例縮放的參數（a0 將被限制在 [-position_scale, position_scale]）
    parser.add_argument('--position_scale', type=float, default=0.5, help='倉位比例上限 (0~1)')
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    df = read_csv_last_months(args.csv, args.months)

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

    device = determine_device()
    print(f"Using device: {device}")

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256])
    )

    model = SAC(
        policy='MlpPolicy',
        env=vec_env,
        verbose=1,
        tensorboard_log=args.logdir,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, 'step'),
        gradient_steps=1,
        target_entropy='auto',
        use_sde=True,
        device=device,
        policy_kwargs=policy_kwargs,
    )

    # 停止條件：以回合數優先
    callbacks = []
    total_episodes_target = None
    if args.episodes and args.episodes > 0:
        # 總回合數 = 每環境回合數 × 環境數
        total_episodes_target = args.episodes * args.vec_envs
        callbacks.append(StopTrainingOnMaxEpisodes(max_episodes=total_episodes_target, verbose=1))
        print(f"訓練目標：{args.episodes} 回合/環境 × {args.vec_envs} 環境 = {total_episodes_target} 總回合")
    callback = CallbackList(callbacks) if len(callbacks) > 1 else (callbacks[0] if callbacks else None)

    # 估算 timesteps 上限（實際由 callback 截止）
    if total_episodes_target and args.episode_steps > 0:
        estimated_steps = total_episodes_target * args.episode_steps * 2  # 2 倍保險
    else:
        estimated_steps = args.total_timesteps

    start = time.time()
    model.learn(total_timesteps=estimated_steps, progress_bar=True, callback=callback)
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


