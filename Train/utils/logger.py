"""
训练日志记录器

用于记录和可视化训练过程。
"""

import json
import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class TrainingLogger:
    """
    训练日志记录器
    
    记录训练指标并提供可视化功能。
    """
    
    def __init__(self, log_dir: str = './logs') -> None:
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志保存目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics: Dict[str, List[Any]] = {
            'episode': [],
            'reward': [],
            'length': [],
            'critic_loss': [],
            'actor_loss': [],
            'alpha': [],
        }
    
    def log_episode(self, episode: int, reward: float, length: int) -> None:
        """
        记录回合信息
        
        Args:
            episode: 回合数
            reward: 回合总奖励
            length: 回合长度
        """
        self.metrics['episode'].append(episode)
        self.metrics['reward'].append(reward)
        self.metrics['length'].append(length)
    
    def log_update(self, critic_loss: float, actor_loss: float, alpha: float) -> None:
        """
        记录更新信息
        
        Args:
            critic_loss: Critic 损失
            actor_loss: Actor 损失
            alpha: 熵系数
        """
        self.metrics['critic_loss'].append(critic_loss)
        self.metrics['actor_loss'].append(actor_loss)
        self.metrics['alpha'].append(alpha)
    
    def save_metrics(self, filename: str = 'metrics.json') -> None:
        """
        保存指标到 JSON 文件
        
        Args:
            filename: 文件名
        """
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"训练指标已保存至: {filepath}")
    
    def plot_training_curves(self, save_path: str = None) -> None:
        """
        绘制训练曲线
        
        Args:
            save_path: 保存路径（如果为 None，则显示图表）
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 奖励曲线
        if len(self.metrics['reward']) > 0:
            ax = axes[0, 0]
            episodes = self.metrics['episode']
            rewards = self.metrics['reward']
            ax.plot(episodes, rewards, alpha=0.6, label='Episode Reward')
            
            # 计算移动平均
            if len(rewards) > 10:
                window = min(50, len(rewards) // 10)
                moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
                ax.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Training Reward')
            ax.legend()
            ax.grid(True)
        
        # 回合长度
        if len(self.metrics['length']) > 0:
            ax = axes[0, 1]
            ax.plot(self.metrics['episode'], self.metrics['length'], alpha=0.6)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Length')
            ax.set_title('Episode Length')
            ax.grid(True)
        
        # Critic 损失
        if len(self.metrics['critic_loss']) > 0:
            ax = axes[1, 0]
            ax.plot(self.metrics['critic_loss'], alpha=0.6)
            ax.set_xlabel('Update Step')
            ax.set_ylabel('Loss')
            ax.set_title('Critic Loss')
            ax.grid(True)
        
        # Actor 损失和 Alpha
        if len(self.metrics['actor_loss']) > 0:
            ax = axes[1, 1]
            ax.plot(self.metrics['actor_loss'], alpha=0.6, label='Actor Loss')
            ax.set_xlabel('Update Step')
            ax.set_ylabel('Loss')
            ax.set_title('Actor Loss')
            
            # 在第二个 y 轴上绘制 alpha
            if len(self.metrics['alpha']) > 0:
                ax2 = ax.twinx()
                ax2.plot(self.metrics['alpha'], 'r-', alpha=0.6, label='Alpha')
                ax2.set_ylabel('Alpha', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
            
            ax.legend(loc='upper left')
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"训练曲线已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def print_summary(self) -> None:
        """打印训练摘要"""
        if len(self.metrics['reward']) == 0:
            print("暂无训练数据")
            return
        
        print("\n" + "=" * 60)
        print("训练摘要")
        print("=" * 60)
        print(f"总回合数: {len(self.metrics['reward'])}")
        print(f"平均奖励: {np.mean(self.metrics['reward']):.2f} ± {np.std(self.metrics['reward']):.2f}")
        print(f"最高奖励: {np.max(self.metrics['reward']):.2f}")
        print(f"最低奖励: {np.min(self.metrics['reward']):.2f}")
        print(f"平均回合长度: {np.mean(self.metrics['length']):.1f}")
        
        if len(self.metrics['reward']) >= 10:
            last_10_avg = np.mean(self.metrics['reward'][-10:])
            print(f"最近10回合平均奖励: {last_10_avg:.2f}")
        
        print("=" * 60 + "\n")

