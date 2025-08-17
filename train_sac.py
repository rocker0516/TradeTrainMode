import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import pandas as pd
import gymnasium as gym
from trading_env import TradingEnvironment
import random
from collections import deque
import os
import matplotlib.pyplot as plt

# 檢查是否有可用的 CUDA 設備
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 重播緩衝區 ---
class ReplayBuffer:
    """一個固定大小的緩衝區，用於儲存經驗元組。"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """儲存一個經驗元組。"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """從緩衝區中隨機取樣一個批次的經驗。"""
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward, dtype=np.float32), np.array(next_state), np.array(done, dtype=np.bool_)

    def __len__(self):
        """返回緩衝區的當前大小。"""
        return len(self.buffer)

# --- 神經網路定義 ---
class Critic(nn.Module):
    """Critic (Q-Network) 模型，使用 Twin-Q 架構以提高穩定性。"""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        num_features, window_size = state_dim

        # Q1 網路
        self.conv1_1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1_1 = nn.Linear(64 * window_size + action_dim, 256)
        self.fc2_1 = nn.Linear(256, 128)
        self.fc3_1 = nn.Linear(128, 1)

        # Q2 網路
        self.conv1_2 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1_2 = nn.Linear(64 * window_size + action_dim, 256)
        self.fc2_2 = nn.Linear(256, 128)
        self.fc3_2 = nn.Linear(128, 1)

    def forward(self, state, action):
        # Q1 前向傳播
        x1 = F.relu(self.conv1_1(state))
        x1 = F.relu(self.conv2_1(x1))
        x1 = x1.view(x1.size(0), -1)  # 展平
        q1 = F.relu(self.fc1_1(torch.cat([x1, action], 1)))
        q1 = F.relu(self.fc2_1(q1))
        q1 = self.fc3_1(q1)

        # Q2 前向傳播
        x2 = F.relu(self.conv1_2(state))
        x2 = F.relu(self.conv2_2(x2))
        x2 = x2.view(x2.size(0), -1)  # 展平
        q2 = F.relu(self.fc1_2(torch.cat([x2, action], 1)))
        q2 = F.relu(self.fc2_2(q2))
        q2 = self.fc3_2(q2)
        
        return q1, q2

class Actor(nn.Module):
    """Actor (Policy-Network) 模型，輸出一個高斯分佈的參數。"""
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        num_features, window_size = state_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * window_size, 256)
        self.fc2 = nn.Linear(256, 128)
        
        self.mean_layer = nn.Linear(128, action_dim)
        self.log_std_layer = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

    def get_action(self, state, deterministic=False):
        """根據狀態獲取動作和其對數機率。"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        
        if deterministic:
            z = mean
        else:
            z = normal.rsample()  # 使用重參數化技巧
        
        action = torch.tanh(z)  # 使用 Tanh 將動作壓縮到 [-1, 1]
        
        # 計算對數機率，考慮到 Tanh 變換
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob

# --- SAC 智能體 ---
class SACAgent:
    def __init__(self, state_dim, action_dim, action_space, gamma=0.99, tau=0.005, lr=3e-4):
        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space

        # 網路
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 優化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # 自動熵調整
        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(DEVICE)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = self.log_alpha.exp().item()

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        action_tanh, _ = self.actor.get_action(state, deterministic)
        action_tanh = action_tanh.detach().cpu().numpy()[0]
        
        # 將 Tanh 輸出的 [-1, 1] 範圍的動作重新縮放到環境的實際動作空間
        low = self.action_space.low
        high = self.action_space.high
        scaled_action = low + (0.5 * (action_tanh + 1.0) * (high - low))
        return scaled_action

    def update_parameters(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state_batch = torch.FloatTensor(state).to(DEVICE)
        next_state_batch = torch.FloatTensor(next_state).to(DEVICE)
        action_batch = torch.FloatTensor(action).to(DEVICE)
        reward_batch = torch.FloatTensor(reward).unsqueeze(1).to(DEVICE)
        done_batch = torch.FloatTensor(done.astype(np.float32)).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            next_action_tanh, next_log_prob = self.actor.get_action(next_state_batch)
            q1_next_target, q2_next_target = self.critic_target(next_state_batch, next_action_tanh)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_prob
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * min_q_next_target

        # Critic 更新
        q1, q2 = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Actor 更新
        pi_tanh, log_pi = self.actor.get_action(state_batch)
        q1_pi, q2_pi = self.critic(state_batch, pi_tanh)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Alpha (溫度) 更新
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # 目標網路軟更新
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.actor.state_dict(), filename + "_actor.pth")

def plot_training_results(rewards, equities, long_trades, short_trades, save_path="plots/training_results.png"):
    """繪製訓練結果並儲存圖表。"""
    print("繪製訓練圖表中...")
    
    # 確保儲存目錄存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    episodes = range(1, len(rewards) + 1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('SAC Training Performance', fontsize=16)
    
    # 1. 回合獎勵
    ax1.plot(episodes, rewards, label='Episode Reward', color='dodgerblue')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards Over Time')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 2. 最終資產 (損益)
    initial_balance = equities[0] if len(equities) > 0 else 10000
    final_pnl = [e - initial_balance for e in equities]
    ax2.plot(episodes, final_pnl, label='Final PnL', color='seagreen')
    ax2.axhline(y=0, color='r', linestyle='--', label='Breakeven')
    ax2.set_ylabel('Final Profit/Loss ($)')
    ax2.set_title('Final Profit/Loss Over Time')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # 3. 交易次數
    ax3.plot(episodes, long_trades, label='Long Trades', color='orange')
    ax3.plot(episodes, short_trades, label='Short Trades', color='purple')
    ax3.set_ylabel('Count')
    ax3.set_xlabel('Episode')
    ax3.set_title('Trade Counts Per Episode')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path)
    plt.close()
    print(f"圖表已儲存至 {save_path}")

# --- 主訓練函數 ---
def main():
    # 超參數
    DATA_PATH = './Data/BTCUSDT_futures_volume_5years_5min.csv'
    MODELS_DIR = './models'
    EPISODES = 1000
    BATCH_SIZE = 256
    BUFFER_SIZE = int(1e6)
    START_STEPS = 1000 # 在開始訓練前，使用隨機動作填充緩衝區的步數
    
    # 建立模型儲存目錄
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # 設定
    print(f"從 {DATA_PATH} 載入數據...")
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.dropna()
    df = df.astype(np.float32)

    print("初始化環境...")
    env = TradingEnvironment(df=df)
    
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    agent = SACAgent(state_dim, action_dim, env.action_space)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    total_steps = 0
    rewards_history = []
    equity_history = []
    longs_history = []
    shorts_history = []
    
    print("開始訓練...")
    for i_episode in range(1, EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        info = {}

        while not done:
            if total_steps < START_STEPS:
                action = env.action_space.sample()  # 隨機動作
            else:
                action = agent.select_action(state)

            if len(replay_buffer) > BATCH_SIZE:
                agent.update_parameters(replay_buffer, BATCH_SIZE)
            
            next_state, reward, done, _, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            total_steps += 1
            episode_steps += 1
        
        # 收集每回合的數據
        rewards_history.append(episode_reward)
        equity_history.append(env.total_value)
        longs_history.append(info.get('long_trades', 0))
        shorts_history.append(info.get('short_trades', 0))

        print(f"回合 {i_episode}, 總步數: {total_steps}, 回合獎勵: {episode_reward:.4f}, 最終資產: {env.total_value:.2f}")

        if i_episode % 50 == 0:
            model_path = os.path.join(MODELS_DIR, f"sac_model_episode_{i_episode}")
            agent.save(model_path)
            print(f"--- 模型已儲存至 {model_path} ---")
            
    # 訓練結束後繪製圖表
    plot_training_results(rewards_history, equity_history, longs_history, shorts_history)
    env.close()

if __name__ == '__main__':
    main() 