import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import time
from typing import List, Dict, Tuple
import argparse

from configs.model_config import ModelConfig
from transformer_env_adapter_simple import TransformerTradingEnvironment
from utils.data_preprocessing import TradingDataPreprocessor

class PPOBuffer:
    """PPO經驗緩衝區"""
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        
        # 緩衝區
        self.states = torch.zeros(buffer_size, state_dim, device=device)
        self.actions = torch.zeros(buffer_size, action_dim, device=device)
        self.log_probs = torch.zeros(buffer_size, action_dim, device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.values = torch.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, device=device)
        self.advantages = torch.zeros(buffer_size, device=device)
        self.returns = torch.zeros(buffer_size, device=device)
        
        self.ptr = 0
        self.size = 0
    
    def store(self, state: torch.Tensor, action: torch.Tensor, log_prob: torch.Tensor, 
              reward: float, value: torch.Tensor, done: bool):
        """存儲經驗"""
        if self.ptr >= self.buffer_size:
            # 緩衝區滿了，重置指針
            self.ptr = 0
            
        # 確保狀態維度正確
        if len(state.shape) == 3:
            state = state.squeeze(1)  # [1, embed_dim]
        if len(state.shape) == 2:
            state = state.squeeze(0)  # [embed_dim]
            
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = torch.tensor(reward, dtype=torch.float32).sum() if hasattr(reward, "__len__") else torch.tensor(reward, dtype=torch.float32)
        self.values[self.ptr] = value.squeeze()
        self.dones[self.ptr] = done
        
        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_gae(self, next_value: torch.Tensor, gamma: float, gae_lambda: float):
        """計算廣義優勢估計(GAE)"""
        if self.size == 0:
            return
            
        # 添加下一個狀態的價值
        values = torch.cat([self.values[:self.size], next_value.flatten()])
        
        # 計算時間差分誤差
        deltas = self.rewards[:self.size] + gamma * values[1:] * (1 - self.dones[:self.size]) - values[:-1]
        
        # 計算GAE
        advantages = torch.zeros_like(deltas)
        advantage = 0
        
        for t in reversed(range(self.size)):
            advantage = deltas[t] + gamma * gae_lambda * advantage * (1 - self.dones[t])
            advantages[t] = advantage
        
        self.advantages[:self.size] = advantages
        self.returns[:self.size] = advantages + self.values[:self.size]
    
    def get_batch(self, batch_size: int):
        """獲取訓練批次"""
        if self.size == 0:
            return None
            
        # 隨機採樣
        indices = torch.randperm(self.size, device=self.device)[:batch_size]
        
        return {
            'states': self.states[indices].detach(),
            'actions': self.actions[indices].detach(),
            'log_probs': self.log_probs[indices].detach(),
            'advantages': self.advantages[indices].detach(),
            'returns': self.returns[indices].detach(),
            'values': self.values[indices].detach()
        }
    
    def clear(self):
        """清空緩衝區"""
        self.ptr = 0
        self.size = 0

class PPOTrainer:
    """PPO訓練器"""
    
    def __init__(self, env: TransformerTradingEnvironment, config: ModelConfig):
        self.env = env
        self.config = config
        self.device = config.DEVICE
        
        # 訓練配置
        self.learning_rate = config.TRAINING_CONFIG['learning_rate']
        self.gamma = config.TRAINING_CONFIG['gamma']
        self.gae_lambda = config.TRAINING_CONFIG['gae_lambda']
        self.clip_epsilon = config.TRAINING_CONFIG['clip_epsilon']
        self.entropy_coef = config.TRAINING_CONFIG['entropy_coef']
        self.value_loss_coef = config.TRAINING_CONFIG['value_loss_coef']
        self.max_grad_norm = config.TRAINING_CONFIG['max_grad_norm']
        self.batch_size = config.TRAINING_CONFIG['batch_size']
        self.epochs_per_update = config.TRAINING_CONFIG['epochs_per_update']
        self.buffer_size = config.TRAINING_CONFIG['buffer_size']
        
        # 優化器
        self.optimizer = optim.Adam(
            env.get_model_parameters(), 
            lr=self.learning_rate
        )
        
        # 學習率調度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=1000, 
            gamma=0.95
        )
        
        # 經驗緩衝區
        self.buffer = PPOBuffer(
            self.buffer_size, 
            config.TRANSFORMER_CONFIG['embed_dim'],
            config.ACTOR_CRITIC_CONFIG['action_dim'],
            self.device
        )
        
        # 訓練統計
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        
    def collect_rollouts(self, n_steps: int) -> Dict:
        """收集經驗"""
        self.env.train_mode()
        
        total_reward = 0
        episode_length = 0
        episodes_completed = 0
        
        state, _ = self.env.reset()
        
        for step in range(n_steps):
            # 獲取動作
            action, log_prob, value = self.env.actor_critic(state)
            
            # 執行動作
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # 存儲經驗
            self.buffer.store(state, action, log_prob, reward, value, done)
            
            total_reward += reward
            episode_length += 1
            
            # 更新狀態
            state = next_state
            
            # 處理episode結束
            if done or truncated:
                self.episode_rewards.append(total_reward)
                self.episode_lengths.append(episode_length)
                episodes_completed += 1
                
                # 重置環境
                state, _ = self.env.reset()
                total_reward = 0
                episode_length = 0
        
        # 計算下一個狀態的價值
        with torch.no_grad():
            next_value = self.env.actor_critic.critic(state)
        
        # 計算GAE
        self.buffer.compute_gae(next_value, self.gamma, self.gae_lambda)
        
        return {
            'episodes_completed': episodes_completed,
            'avg_reward': np.mean(self.episode_rewards[-episodes_completed:]) if episodes_completed > 0 else 0,
            'avg_length': np.mean(self.episode_lengths[-episodes_completed:]) if episodes_completed > 0 else 0
        }
    
    def update_policy(self) -> Dict:
        """更新策略"""
        self.env.train_mode()
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for epoch in range(self.epochs_per_update):
            batch = self.buffer.get_batch(self.batch_size)
            if batch is None:
                continue
                
            # 重新評估動作
            log_probs, values, entropy = self.env.evaluate_state_action(
                batch['states'], batch['actions']
            )
            
            # 計算比率
            ratio = torch.exp(log_probs.sum(dim=-1) - batch["log_probs"].sum(dim=-1).detach())
            
            # 正規化優勢
            advantages = batch['advantages']
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 策略損失 (PPO Clip)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 價值損失
            value_loss = nn.MSELoss()(values.squeeze(), batch['returns'])
            
            # 熵損失（鼓勵探索）
            entropy_loss = -entropy.mean()
            
            # 總損失
            total_loss = (policy_loss + 
                         self.value_loss_coef * value_loss + 
                         self.entropy_coef * entropy_loss)
            
            # 反向傳播
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.env.get_model_parameters(), 
                self.max_grad_norm
            )
            
            self.optimizer.step()
            
            # 統計
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            n_updates += 1
        
        # 更新學習率
        self.scheduler.step()
        
        # 清空緩衝區
        self.buffer.clear()
        
        # 記錄統計
        if n_updates > 0:
            avg_policy_loss = total_policy_loss / n_updates
            avg_value_loss = total_value_loss / n_updates
            avg_entropy = total_entropy / n_updates
            
            self.policy_losses.append(avg_policy_loss)
            self.value_losses.append(avg_value_loss)
            self.entropies.append(avg_entropy)
            
            return {
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'entropy': avg_entropy,
                'learning_rate': self.scheduler.get_last_lr()[0]
            }
        
        return {}
    
    def train(self, n_iterations: int, rollout_steps: int, save_interval: int = 100):
        """訓練主循環"""
        print(f"開始訓練 - 總迭代數: {n_iterations}")
        print(f"模型參數數量: {self.env.get_parameter_count()}")
        
        start_time = time.time()
        best_reward = float('-inf')
        
        for iteration in range(n_iterations):
            # 收集經驗
            rollout_stats = self.collect_rollouts(rollout_steps)
            
            # 更新策略
            update_stats = self.update_policy()
            
            # 記錄統計
            if iteration % 10 == 0:
                elapsed_time = time.time() - start_time
                avg_reward = rollout_stats.get('avg_reward', 0)
                
                print(f"迭代 {iteration:4d} | "
                      f"獎勵: {avg_reward:8.2f} | "
                      f"Episode長度: {rollout_stats.get('avg_length', 0):6.1f} | "
                      f"策略損失: {update_stats.get('policy_loss', 0):8.4f} | "
                      f"價值損失: {update_stats.get('value_loss', 0):8.4f} | "
                      f"熵: {update_stats.get('entropy', 0):8.4f} | "
                      f"時間: {elapsed_time:6.1f}s")
                
                # 保存最佳模型
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.env.save_model('best_model.pth')
                    print(f"保存最佳模型，獎勵: {best_reward:.2f}")
            
            # 定期保存檢查點
            if iteration % save_interval == 0 and iteration > 0:
                self.env.save_model(f'checkpoint_{iteration}.pth')
                self.save_training_stats(f'training_stats_{iteration}.npz')
        
        print(f"訓練完成！最佳獎勵: {best_reward:.2f}")
        print(f"總時間: {time.time() - start_time:.1f}秒")
    
    def save_training_stats(self, path: str):
        """保存訓練統計"""
        np.savez(path,
                episode_rewards=np.array(self.episode_rewards),
                episode_lengths=np.array(self.episode_lengths),
                policy_losses=np.array(self.policy_losses),
                value_losses=np.array(self.value_losses),
                entropies=np.array(self.entropies))
    
    def plot_training_progress(self, save_path: str = 'training_progress.png'):
        """繪製訓練進度"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 獎勵曲線
        if len(self.episode_rewards) > 0:
            axes[0, 0].plot(self.episode_rewards)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
        
        # Episode長度
        if len(self.episode_lengths) > 0:
            axes[0, 1].plot(self.episode_lengths)
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Length')
            axes[0, 1].grid(True)
        
        # 策略損失
        if len(self.policy_losses) > 0:
            axes[1, 0].plot(self.policy_losses)
            axes[1, 0].set_title('Policy Loss')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        
        # 價值損失
        if len(self.value_losses) > 0:
            axes[1, 1].plot(self.value_losses)
            axes[1, 1].set_title('Value Loss')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='訓練Transformer交易模型')
    parser.add_argument('--data_path', type=str, default='Data/BTCUSDT_futures_volume_5years_5min.csv',
                       help='數據文件路徑')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='訓練迭代數')
    parser.add_argument('--rollout_steps', type=int, default=2048,
                       help='每次rollout的步數')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='保存檢查點的間隔')
    
    args = parser.parse_args()
    
    # 加載配置
    config = ModelConfig()
    
    # 加載數據
    print("加載數據...")
    df = pd.read_csv(args.data_path)
    print(f"數據量: {len(df)} 行")
    
    # 創建環境
    print("創建環境...")
    env = TransformerTradingEnvironment(df, config)
    
    # 創建訓練器
    print("創建訓練器...")
    trainer = PPOTrainer(env, config)
    
    # 開始訓練
    trainer.train(
        n_iterations=args.iterations,
        rollout_steps=args.rollout_steps,
        save_interval=args.save_interval
    )
    
    # 繪製訓練進度
    trainer.plot_training_progress()
    
    # 保存最終統計
    trainer.save_training_stats('final_training_stats.npz')

if __name__ == "__main__":
    main() 