"""
SAC 训练器实现

专门用于训练 SAC 类型的强化学习模型。
"""

from typing import Dict, Any, Optional
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import time

from .base_trainer import BaseTrainer
from ..models.base_model import BaseRLModel
from ..utils.replay_buffer import ReplayBuffer


class SACTrainer(BaseTrainer):
    """
    SAC 训练器
    
    负责管理训练循环、经验回放和模型更新。
    """
    
    def __init__(
        self,
        env: gym.Env,
        model: BaseRLModel,
        config: Dict[str, Any],
    ) -> None:
        """
        初始化 SAC 训练器
        
        Args:
            env: 训练环境
            model: SAC 模型
            config: 训练配置，包含以下键：
                - buffer_size: 经验回放缓冲区大小
                - batch_size: 批次大小
                - warmup_steps: 预热步数（随机探索）
                - update_interval: 更新间隔（每隔多少步更新一次）
                - save_dir: 模型保存目录
                - log_interval: 日志记录间隔
        """
        super().__init__(env, model, config)
        
        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer(
            buffer_size=config.get('buffer_size', 100000),
            observation_shape=env.observation_space.shape,
            action_dim=env.action_space.shape[0],
        )
        
        self.batch_size = config.get('batch_size', 256)
        self.warmup_steps = config.get('warmup_steps', 1000)
        self.update_interval = config.get('update_interval', 1)
        self.save_dir = config.get('save_dir', './models')
        self.log_interval = config.get('log_interval', 100)
        
        # 训练指标
        self.update_count = 0
        self.best_eval_reward = -np.inf
        
    def train(self, total_episodes: int, eval_interval: int = 10) -> None:
        """
        执行训练
        
        Args:
            total_episodes: 总训练回合数
            eval_interval: 评估间隔
        """
        print("=" * 50)
        print("开始训练 SAC + LSTM 模型")
        print(f"总回合数: {total_episodes}")
        print(f"评估间隔: {eval_interval}")
        print(f"缓冲区大小: {self.replay_buffer.buffer_size}")
        print(f"批次大小: {self.batch_size}")
        print(f"预热步数: {self.warmup_steps}")
        print("=" * 50)
        
        for episode in range(1, total_episodes + 1):
            episode_reward, episode_length = self._train_episode(episode)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # 记录训练信息
            if episode % self.log_interval == 0 or episode == 1:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                print(f"\n[Episode {episode}/{total_episodes}]")
                print(f"  本回合奖励: {episode_reward:.2f}")
                print(f"  本回合长度: {episode_length}")
                print(f"  近10回合平均奖励: {avg_reward:.2f}")
                print(f"  近10回合平均长度: {avg_length:.1f}")
                print(f"  总步数: {self.total_steps}")
                print(f"  缓冲区大小: {len(self.replay_buffer)}")
            
            # 定期评估
            if episode % eval_interval == 0:
                eval_metrics = self.evaluate(num_episodes=3)
                print(f"\n{'='*50}")
                print(f"[评估 @ Episode {episode}]")
                for key, value in eval_metrics.items():
                    print(f"  {key}: {value:.2f}")
                print(f"{'='*50}\n")
                
                # 保存最佳模型
                if eval_metrics['mean_reward'] > self.best_eval_reward:
                    self.best_eval_reward = eval_metrics['mean_reward']
                    self.save_checkpoint(
                        f"{self.save_dir}/best_model.pth",
                        metadata={'episode': episode, 'eval_reward': self.best_eval_reward}
                    )
                    print(f"  新的最佳模型已保存！奖励: {self.best_eval_reward:.2f}\n")
            
            # 定期保存检查点
            if episode % (eval_interval * 5) == 0:
                self.save_checkpoint(f"{self.save_dir}/checkpoint_ep{episode}.pth")
        
        # 训练结束，保存最终模型
        self.save_checkpoint(f"{self.save_dir}/final_model.pth")
        print("\n训练完成！")
    
    def _train_episode(self, episode: int) -> tuple:
        """
        执行单个训练回合
        
        Args:
            episode: 当前回合数
            
        Returns:
            (回合总奖励, 回合长度)
        """
        state, _ = self.env.reset()
        self.model.reset_hidden_state()
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        progress_bar = tqdm(desc=f"Episode {episode}", leave=False)
        
        while not done:
            # 选择动作
            if self.total_steps < self.warmup_steps:
                # 预热阶段：随机探索
                action = self.env.action_space.sample()
            else:
                # 使用策略选择动作
                action = self.model.select_action(state, evaluate=False)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # 存储经验
            self.replay_buffer.add(state, action, reward, next_state, float(done))
            
            # 更新模型
            if self.total_steps >= self.warmup_steps and self.total_steps % self.update_interval == 0:
                if len(self.replay_buffer) >= self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    metrics = self.model.update(batch)
                    self.update_count += 1
                    
                    # 记录更新指标
                    if self.update_count % 100 == 0:
                        progress_bar.set_postfix(metrics)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            progress_bar.update(1)
        
        progress_bar.close()
        
        return episode_reward, episode_length
    
    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            num_episodes: 评估回合数
            
        Returns:
            评估指标字典
        """
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            self.model.reset_hidden_state()
            
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                # 评估模式：不添加探索噪声
                action = self.model.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
                episode_length += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'mean_length': np.mean(eval_lengths),
        }

