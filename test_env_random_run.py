import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import glob
import os
from Env.trading_env import TradingEnvironment



def run_random_test(num_episodes=300, max_steps=3000):
    """
    執行隨機動作測試
    Args:
        num_episodes: 總回合數
        max_steps: 每回合最大步數
    """
    # 初始化環境
    # window_size 設為 288 (一天)
    env = TradingEnvironment(
        env_id=0,
        window_size=288,
        min_episode_steps=max_steps, 
        max_step_pos_change_pct=0.1
    )
    
    print(f"\nStarting Random Action Test: {num_episodes} episodes, {max_steps} steps/ep")
    start_time = time.time()
    
    total_rewards = []
    
    # 使用 tqdm 顯示進度
    pbar = tqdm(range(num_episodes), desc="Running Episodes")
    
    for ep in pbar:
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0
        
        while not (done or truncated) and steps < max_steps:
            # 隨機動作 (-1.0 ~ 1.0)
            action = env.action_space.sample()
            
            obs, reward, done, truncated, info = env.step(action)
            
            ep_reward += reward
            steps += 1
            
            # 簡單檢查 Observation 形狀 (第一步檢查即可)
            if steps == 1 and ep == 0:
                print("\n[Check] Observation Shapes:")
                for k, v in obs.items():
                    print(f"  {k}: {v.shape}")
        
        total_rewards.append(ep_reward)
        pbar.set_postfix({'last_reward': f"{ep_reward:.2f}", 'avg_reward': f"{np.mean(total_rewards):.2f}"})
        
    elapsed = time.time() - start_time
    print(f"\nTest Completed in {elapsed:.2f}s")
    if total_rewards:
        print(f"Average Reward: {np.mean(total_rewards):.4f} ± {np.std(total_rewards):.4f}")
    
    env.close()

if __name__ == "__main__":
    run_random_test()
