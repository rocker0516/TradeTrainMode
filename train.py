import torch
import numpy as np
from ppo_sac_hybrid import PPOSACHybrid
from replay_buffer import ReplayBuffer
from trading_env import TradingEnvironment
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def train(
    env,
    agent,
    replay_buffer,
    n_episodes=1000,
    batch_size=256,
    update_interval=10,
    eval_interval=10,
    save_interval=100,
    max_steps=1000,
    warmup_steps=1000
):
    # 訓練記錄
    episode_rewards = []
    eval_rewards = []
    best_eval_reward = float('-inf')
    
    # 預熱階段
    print("Warming up replay buffer...")
    state, _ = env.reset()
    for _ in tqdm(range(warmup_steps)):
        action = env.action_space.sample()  # 隨機動作
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()[0]
    
    print("Starting training...")
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < max_steps:
            # 確保狀態是有效的
            if np.isnan(state).any():
                print(f"Warning: NaN values in state at episode {episode}")
                state = np.nan_to_num(state, nan=0.0)
            
            # 選擇動作
            action = agent.select_action(state)
            
            # 執行動作
            next_state, reward, done, _, info = env.step(action)
            
            # 確保下一個狀態是有效的
            if np.isnan(next_state).any():
                print(f"Warning: NaN values in next_state at episode {episode}" )
                next_state = np.nan_to_num(next_state, nan=0.0)
            
            # 存儲經驗
            replay_buffer.push(state, action, reward, next_state, done)
            
            # 更新狀態和獎勵
            state = next_state
            episode_reward += reward
            step += 1
            
            # 定期更新網絡
            if len(replay_buffer) >= batch_size and len(replay_buffer) % update_interval == 0:
                batch = replay_buffer.sample(batch_size)
                loss = agent.update(batch)
                if episode % 10 == 0:
                    print(f"Episode {episode}, Loss: {loss:.4f}")
        
        # 記錄獎勵
        episode_rewards.append(episode_reward)
        
        # 評估
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate(env, agent, n_episodes=5)
            eval_rewards.append(eval_reward)
            print(f"Episode {episode + 1}, Eval Reward: {eval_reward:.2f}")
            
            # 保存最佳模型
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                save_model(agent, f"best_model.pth")
        
        # 定期保存模型
        if (episode + 1) % save_interval == 0:
            save_model(agent, f"model_episode_{episode + 1}.pth")
        
        # 打印進度
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}")
    
    # 繪製訓練曲線
    plot_training_curves(episode_rewards, eval_rewards)
    
    return episode_rewards, eval_rewards

def evaluate(env, agent, n_episodes=5):
    total_reward = 0
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 確保狀態是有效的
            if np.isnan(state).any():
                print("Warning: NaN values in state during evaluation")
                state = np.nan_to_num(state, nan=0.0)
            
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, _, _ = env.step(action)
            
            # 確保下一個狀態是有效的
            if np.isnan(next_state).any():
                print("Warning: NaN values in next_state during evaluation")
                next_state = np.nan_to_num(next_state, nan=0.0)
            
            state = next_state
            episode_reward += reward
        
        total_reward += episode_reward
    
    return total_reward / n_episodes

def save_model(agent, filename):
    torch.save({
        'feature_extractor': agent.feature_extractor.state_dict(),
        'ppo_actor': agent.ppo_actor.state_dict(),
        'ppo_critic': agent.ppo_critic.state_dict(),
        'sac_actor': agent.sac_actor.state_dict(),
        'sac_critic1': agent.sac_critic1.state_dict(),
        'sac_critic2': agent.sac_critic2.state_dict(),
        'target_critic1': agent.target_critic1.state_dict(),
        'target_critic2': agent.target_critic2.state_dict()
    }, filename)

def plot_training_curves(episode_rewards, eval_rewards):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(eval_rewards)
    plt.title('Evaluation Rewards')
    plt.xlabel('Evaluation')
    plt.ylabel('Reward')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == "__main__":
    # 加載數據
    # 這裡需要根據您的數據格式進行調整
    df = pd.read_csv('Data/BTCUSDT_futures_volume_5years_5min.csv')
    
    # 創建環境
    env = TradingEnvironment(df)
    
    # 獲取狀態和動作空間的維度
    state_dim = env.observation_space.shape[0]  # 特徵數量
    action_dim = env.action_space.shape[0]      # 動作維度
    
    # 創建智能體
    agent = PPOSACHybrid(state_dim, action_dim)
    
    # 創建經驗回放緩衝區
    replay_buffer = ReplayBuffer()
    
    # 開始訓練
    episode_rewards, eval_rewards = train(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        n_episodes=1000,
        batch_size=256,
        update_interval=10,
        eval_interval=10,
        save_interval=100,
        max_steps=1000,
        warmup_steps=1000
    ) 