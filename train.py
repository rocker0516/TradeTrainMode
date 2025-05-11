import pandas as pd
import numpy as np
from trading_env import TradingEnvironment
from dqn_model import DQNAgent
import matplotlib.pyplot as plt
import torch

def load_data(file_path):
    df = pd.read_csv(file_path)
    # 確保數據按時間排序
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return df

def train_agent(env, agent, episodes, batch_size=32):
    scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        scores.append(total_reward)
        
        if episode % 10 == 0:
            print(f"Episode: {episode}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            agent.update_target_model()
    
    return scores

def plot_results(scores):
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('training_results.png')
    plt.close()

def main():
    # 加載數據
    df = load_data('Data/BTCUSDT_futures_volume_5years_5min.csv')
    
    # 創建環境
    env = TradingEnvironment(df)
    
    # 創建智能體
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # 訓練智能體
    episodes = 100
    scores = train_agent(env, agent, episodes)
    
    # 繪製結果
    plot_results(scores)
    
    # 保存模型
    torch.save(agent.model.state_dict(), 'trading_model.pth')

if __name__ == "__main__":
    main() 