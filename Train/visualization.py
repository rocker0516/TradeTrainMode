"""
訓練結果可視化模組

此模組提供訓練過程中的各種圖表繪製功能：
1. 訓練曲線（Episode Reward）
2. 逐步獎勵曲線（Step Reward）
3. 回合統計圖表（收益率、回撤、交易次數等）

符合 Python OOP + SOLID 原則：
- 單一職責原則：專注於可視化功能
- 開放封閉原則：易於擴展新的圖表類型
"""

import os
import pathlib
import glob
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端
import matplotlib.pyplot as plt


class TrainingVisualizer:
    """訓練結果可視化器基類"""
    
    def __init__(self, dpi: int = 150):
        """
        初始化可視化器
        
        Args:
            dpi: 圖片解析度，默認 150
        """
        self.dpi = dpi
    
    def _ensure_output_dir(self, output_path: str) -> None:
        """確保輸出目錄存在"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)


class TrainingCurvePlotter(TrainingVisualizer):
    """訓練曲線繪製器 - 繪製 Episode Reward 趨勢"""
    
    def plot(self, monitor_logs_dir: str, output_path: str) -> None:
        """
        繪製訓練曲線（Episode Returns）
        
        Args:
            monitor_logs_dir: Monitor 日誌目錄
            output_path: 輸出圖片路徑
        """
        csv_files = self._find_monitor_csv_files(monitor_logs_dir)
        episode_points = self._extract_episode_points(csv_files)
        
        if len(episode_points) == 0:
            print(f"警告：未找到任何訓練數據於 {monitor_logs_dir}")
            return
        
        self._plot_curve(episode_points, output_path)
    
    def _find_monitor_csv_files(self, monitor_logs_dir: str) -> List[str]:
        """查找所有 Monitor CSV 文件"""
        csv_files = []
        # 兼容不同命名格式：monitor.csv、monitor_*.csv、*.monitor.csv
        csv_files += glob.glob(os.path.join(monitor_logs_dir, 'monitor.csv'))
        csv_files += glob.glob(os.path.join(monitor_logs_dir, 'monitor_*.csv'))
        csv_files += glob.glob(os.path.join(monitor_logs_dir, '*.monitor.csv'))
        
        if len(csv_files) == 0:
            # SubprocVecEnv + Monitor 會將單環境日誌命名為 monitor.csv 於每個子資料夾
            csv_files += glob.glob(os.path.join(monitor_logs_dir, '*', 'monitor.csv'))
            csv_files += glob.glob(os.path.join(monitor_logs_dir, '*', 'monitor_*.csv'))
            csv_files += glob.glob(os.path.join(monitor_logs_dir, '*', '*.monitor.csv'))
        
        return csv_files
    
    def _extract_episode_points(self, csv_files: List[str]) -> List[Tuple[float, float]]:
        """從 CSV 文件提取 (timestep, episode_reward) 數據點"""
        episode_points: List[Tuple[float, float]] = []
        
        for f in csv_files:
            try:
                # 略過前兩行 header (Monitor 格式)
                df = pd.read_csv(f, skiprows=2)
                if {'l', 'r', 't'}.issubset(df.columns):
                    # l: episode length, r: reward sum, t: time
                    df['timestep'] = df['l'].cumsum()
                    for _, row in df.iterrows():
                        episode_points.append((row['timestep'], row['r']))
            except Exception as e:
                print(f"警告：讀取 {f} 失敗: {e}")
                continue
        
        return episode_points
    
    def _plot_curve(self, episode_points: List[Tuple[float, float]], output_path: str) -> None:
        """繪製訓練曲線"""
        episode_points.sort(key=lambda x: x[0])
        timesteps = np.array([p[0] for p in episode_points])
        rewards = np.array([p[1] for p in episode_points])
        
        # 平滑處理
        rewards_smooth = self._smooth_data(rewards, window_ratio=20)
        
        # 繪圖
        plt.figure(figsize=(10, 5))
        plt.plot(timesteps, rewards, alpha=0.3, label='Episode reward')
        plt.plot(timesteps, rewards_smooth, color='C1', label='Smoothed')
        plt.xlabel('Timesteps')
        plt.ylabel('Episode Reward')
        plt.title('SAC Training Curve (Episode Returns)')
        plt.legend()
        plt.tight_layout()
        
        self._ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
    
    def _smooth_data(self, data: np.ndarray, window_ratio: int = 20) -> np.ndarray:
        """
        數據平滑處理
        
        Args:
            data: 原始數據
            window_ratio: 窗口大小比例（數據長度 / window_ratio）
        
        Returns:
            平滑後的數據
        """
        window = max(1, len(data) // window_ratio)
        if window > 1:
            kernel = np.ones(window) / window
            return np.convolve(data, kernel, mode='same')
        return data


class StepRewardPlotter(TrainingVisualizer):
    """逐步獎勵繪製器 - 繪製每步的 Reward"""
    
    def plot(self, step_logs_dir: str, output_path: str, smooth_window: int = 500) -> None:
        """
        繪製逐步獎勵曲線
        
        Args:
            step_logs_dir: Step 日誌目錄
            output_path: 輸出圖片路徑
            smooth_window: 平滑窗口大小
        """
        csv_files = glob.glob(os.path.join(step_logs_dir, 'env_*.csv'))
        if len(csv_files) == 0:
            print(f"警告：未找到任何 step reward 數據於 {step_logs_dir}")
            return
        
        total_series = self._load_step_reward_series(csv_files, smooth_window)
        
        if len(total_series) == 0:
            print(f"警告：無法載入任何有效的 step reward 數據")
            return
        
        self._plot_step_rewards(total_series, output_path, smooth_window)
    
    def _load_step_reward_series(self, csv_files: List[str], smooth_window: int) -> List[np.ndarray]:
        """載入所有環境的 step reward 序列"""
        total_series = []
        
        plt.figure(figsize=(10, 5))
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                if 'step' in df.columns and 'reward' in df.columns:
                    y = df['reward'].values.astype(float)
                    y_smoothed = self._smooth(y, smooth_window)
                    plt.plot(df['step'].values, y_smoothed, alpha=0.35)
                    total_series.append(y)
            except Exception as e:
                print(f"警告：讀取 {f} 失敗: {e}")
                continue
        
        return total_series
    
    def _plot_step_rewards(self, total_series: List[np.ndarray], output_path: str, smooth_window: int) -> None:
        """繪製 step reward 曲線及平均線"""
        # 計算平均曲線
        max_len = max(len(s) for s in total_series)
        padded = []
        for s in total_series:
            if len(s) < max_len:
                pad = np.full(max_len - len(s), np.nan)
                s = np.concatenate([s, pad])
            padded.append(s)
        
        arr = np.vstack(padded)
        mean_series = np.nanmean(arr, axis=0)
        mean_series = self._smooth(mean_series, smooth_window)
        plt.plot(np.arange(1, len(mean_series)+1), mean_series, 
                color='C1', linewidth=2.0, label='Mean (smoothed)')
        
        plt.title('Per-Step Reward (smoothed)')
        plt.xlabel('Env Step (per-env)')
        plt.ylabel('Reward')
        plt.legend()
        plt.tight_layout()
        
        self._ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
    
    def _smooth(self, y: np.ndarray, w: int) -> np.ndarray:
        """
        平滑處理
        
        Args:
            y: 原始數據
            w: 窗口大小
        
        Returns:
            平滑後的數據
        """
        if len(y) < 3 or w <= 1:
            return y
        w = min(w, max(1, len(y)//10))
        kernel = np.ones(w, dtype=float) / w
        return np.convolve(y, kernel, mode='same')


class EpisodeStatsPlotter(TrainingVisualizer):
    """回合統計繪製器 - 繪製收益率、回撤、交易次數等統計圖表"""
    
    def plot(self, ep_logs_dir: str, output_path: str, combined_csv_out: Optional[str] = None) -> None:
        """
        繪製回合統計圖表（2x3 子圖）
        
        Args:
            ep_logs_dir: Episode 統計日誌目錄
            output_path: 輸出圖片路徑
            combined_csv_out: 可選的彙總 CSV 輸出路徑
        """
        df = self._load_episode_stats(ep_logs_dir)
        
        if df is None or len(df) == 0:
            print(f"警告：未找到任何 episode 統計數據於 {ep_logs_dir}")
            return
        
        # 可選：匯出彙總 CSV
        if combined_csv_out is not None:
            self._export_combined_csv(df, combined_csv_out)
        
        # 計算統計指標
        stats = self._calculate_statistics(df)
        
        # 繪製 2x3 子圖
        self._plot_all_stats(df, stats, output_path)
        
        # 輸出統計摘要
        self._print_summary(df, stats)
    
    def _load_episode_stats(self, ep_logs_dir: str) -> Optional[pd.DataFrame]:
        """載入並合併所有環境的 episode 統計"""
        csv_files = glob.glob(os.path.join(ep_logs_dir, 'env_*.csv'))
        if len(csv_files) == 0:
            return None
        
        frames = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                frames.append(df)
            except Exception as e:
                print(f"警告：讀取 {f} 失敗: {e}")
                continue
        
        if len(frames) == 0:
            return None
        
        df = pd.concat(frames, ignore_index=True)
        # 統一排序與全域回合編號
        df = df.sort_values(['env_id', 'episode']).reset_index(drop=True)
        df['global_episode'] = np.arange(1, len(df) + 1)
        
        return df
    
    def _export_combined_csv(self, df: pd.DataFrame, output_path: str) -> None:
        """匯出彙總 CSV"""
        try:
            self._ensure_output_dir(output_path)
            df.to_csv(output_path, index=False)
        except Exception as e:
            print(f"警告：匯出 CSV 失敗: {e}")
    
    def _calculate_statistics(self, df: pd.DataFrame) -> dict:
        """計算統計指標"""
        # 數值保護
        df['return_rate'] = df['return_rate'].clip(-10, 10)
        
        win_rate = (df['return_rate'] > 0).sum() / len(df) if len(df) > 0 else 0
        avg_return = df['return_rate'].mean() if 'return_rate' in df.columns else 0
        avg_trades = df['trade_count'].mean() if 'trade_count' in df.columns else 0
        
        # 計算累積收益率（加入溢出保護）
        returns_clamped = df['return_rate'].clip(-0.5, 2.0)
        cumulative_return = (returns_clamped + 1).cumprod() - 1
        cumulative_return = cumulative_return.clip(-1, 100)
        
        return {
            'win_rate': win_rate,
            'avg_return': avg_return,
            'avg_trades': avg_trades,
            'cumulative_return': cumulative_return
        }
    
    def _plot_all_stats(self, df: pd.DataFrame, stats: dict, output_path: str) -> None:
        """繪製所有統計子圖"""
        fig = plt.figure(figsize=(15, 10))
        
        # 子圖 1: 收益率趨勢
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(df['global_episode'], df['return_rate'] * 100, 
                marker='o', linewidth=1, markersize=3, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax1.set_title(f"Return Rate per Episode (Win Rate: {stats['win_rate']*100:.1f}%)")
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Return Rate (%)')
        
        # 子圖 2: 期末資金
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(df['global_episode'], df['end_total_value'], 
                marker='o', linewidth=1, markersize=3, alpha=0.6, color='C1')
        initial_value = df['end_total_value'].iloc[0] if len(df) > 0 else 10000
        ax2.axhline(y=initial_value, color='r', linestyle='--', alpha=0.3, label='Initial')
        ax2.set_title('End Total Value per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('End Total Value')
        ax2.legend()
        
        # 子圖 3: 最大回撤
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(df['global_episode'], df['max_drawdown'], 
                marker='o', linewidth=1, markersize=3, alpha=0.6, color='C2')
        ax3.set_title('Max Drawdown per Episode')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Max Drawdown')
        
        # 子圖 4: 交易次數
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(df['global_episode'], df['trade_count'], 
                marker='o', linewidth=1, markersize=3, alpha=0.6, color='C3')
        ax4.set_title(f"Trade Count per Episode (Avg: {stats['avg_trades']:.1f})")
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Trade Count')
        
        # 子圖 5: 累積收益率
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(df['global_episode'], stats['cumulative_return'] * 100, 
                linewidth=2, color='C4')
        ax5.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax5.set_title('Cumulative Return Rate (capped)')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Cumulative Return (%)')
        
        # 子圖 6: 回合總獎勵
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(df['global_episode'], df['episode_reward'], 
                marker='o', linewidth=1, markersize=3, alpha=0.6, color='C5')
        ax6.set_title('Episode Total Reward (for reference)')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Total Reward')
        
        plt.tight_layout()
        self._ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
    
    def _print_summary(self, df: pd.DataFrame, stats: dict) -> None:
        """輸出統計摘要"""
        print(f"\n=== 訓練統計摘要 ===")
        print(f"總回合數: {len(df)}")
        print(f"勝率: {stats['win_rate']*100:.2f}%")
        print(f"平均收益率: {stats['avg_return']*100:.2f}%")
        print(f"平均交易次數/回合: {stats['avg_trades']:.1f}")
        final_cum_return = stats['cumulative_return'].iloc[-1]*100 if len(stats['cumulative_return']) > 0 else 0
        print(f"最終累積收益: {final_cum_return:.2f}%")


# 便利函數：向後兼容原有調用方式
def plot_training_curve(monitor_logs_dir: str, output_path: str) -> None:
    """
    繪製訓練曲線（便利函數）
    
    Args:
        monitor_logs_dir: Monitor 日誌目錄
        output_path: 輸出圖片路徑
    """
    plotter = TrainingCurvePlotter()
    plotter.plot(monitor_logs_dir, output_path)


def plot_step_reward_curve(step_logs_dir: str, output_path: str, smooth_window: int = 500) -> None:
    """
    繪製逐步獎勵曲線（便利函數）
    
    Args:
        step_logs_dir: Step 日誌目錄
        output_path: 輸出圖片路徑
        smooth_window: 平滑窗口大小
    """
    plotter = StepRewardPlotter()
    plotter.plot(step_logs_dir, output_path, smooth_window)


def plot_episode_stats(ep_logs_dir: str, output_path: str, combined_csv_out: Optional[str] = None) -> None:
    """
    繪製回合統計圖表（便利函數）
    
    Args:
        ep_logs_dir: Episode 統計日誌目錄
        output_path: 輸出圖片路徑
        combined_csv_out: 可選的彙總 CSV 輸出路徑
    """
    plotter = EpisodeStatsPlotter()
    plotter.plot(ep_logs_dir, output_path, combined_csv_out)

