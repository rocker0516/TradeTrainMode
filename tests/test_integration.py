
print("Start of script")
import sys
import os
print("Imports start")
import numpy as np
import pandas as pd
import logging

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

print("Importing project modules...")
from Env.trading_env import TradingEnvironment
from Train.cost import CombinedCostCalculator
from Train.config import Config
print("Imports done")

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger("IntegrationTest")

def generate_synthetic_data(length=500):
    """Generate synthetic OHLCV data"""
    # Simple sine wave price + trend
    t = np.linspace(0, 4*np.pi, length)
    price = 10000 + 500 * np.sin(t) + 5 * np.arange(length)
    
    # Add some noise
    noise = np.random.normal(0, 10, length)
    close = price + noise
    
    # Construct OHLC
    high = close + np.random.uniform(5, 20, length)
    low = close - np.random.uniform(5, 20, length)
    open_p = np.roll(close, 1)
    open_p[0] = close[0]
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=length, freq='5min'),
        'open': open_p,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.uniform(100, 1000, length),
        'buy_volume': np.random.uniform(50, 500, length),
        'sell_volume': np.random.uniform(50, 500, length),
        'volume_ratio': np.random.uniform(0.4, 0.6, length),
        'long_short_ratio': np.random.uniform(0.8, 1.2, length),
        'trades': np.random.randint(10, 100, length),
        'quote_volume': np.random.uniform(1000000, 10000000, length)
    })
    df.set_index('timestamp', inplace=True)
    return df

def run_test():
    print("=== Starting Integration Test: Reward + 3 Cost Lines ===")
    
    # 1. Setup Environment with Synthetic Data
    df = generate_synthetic_data(length=1000)
    
    # Ensure Config uses these params for visibility
    Config.REWARD_TURNOVER_PENALTY = 1.0 # Explicitly set as requested previously
    
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        transaction_fee=0.0004, # 0.04%
        window_size=20, # Short window for test
        leverage=10,
        min_episode_steps=50,
        turnover_penalty=Config.REWARD_TURNOVER_PENALTY,
        dd_penalty_coef=Config.REWARD_DD_PENALTY
    )
    
    cost_calculator = CombinedCostCalculator(num_envs=1)
    cost_calculator.reset([0], [10000])
    
    obs, _ = env.reset(seed=42)
    
    # ==========================================
    # Case 1: Normal Trading (Conservative)
    # ==========================================
    print("\n[Case 1: Normal/Safe Trading] (50 Steps)")
    print("Strategy: Hold small long position, minimal changes.")
    
    normal_logs = []
    
    # Step 1: Open small Long (0.3)
    action = np.array([0.3], dtype=np.float32)
    
    for i in range(50):
        # Slight random noise to action to simulate active agent but stable
        # action = 0.3 +/- 0.01
        current_action = np.array([0.3 + np.random.uniform(-0.01, 0.01)], dtype=np.float32)
        
        next_obs, reward, done, truncated, info = env.step(current_action)
        costs = cost_calculator.calculate_costs([info])[0] # [C1, C2, C3]
        
        log_entry = {
            'step': i,
            'price': info.get('equity', 0), # Log equity roughly
            'action': current_action[0],
            'pos_change': info.get('position_change_norm', 0),
            'reward': reward,
            'cost_margin': costs[0],
            'cost_dd': costs[1],
            'cost_fee': costs[2],
            'equity': info.get('equity'),
            'fee_ratio': info.get('step_fee_ratio', 0)
        }
        normal_logs.append(log_entry)
        if done:
            break
            
    # Analysis Case 1
    avg_reward_norm = np.mean([x['reward'] for x in normal_logs])
    avg_cost_fee_norm = np.mean([x['cost_fee'] for x in normal_logs])
    print(f"Avg Reward: {avg_reward_norm:.6f}")
    print(f"Avg Fee Cost: {avg_cost_fee_norm:.6f}")
    print(f"Avg Margin Cost: {np.mean([x['cost_margin'] for x in normal_logs]):.6f}")
    print(f"Final Equity: {normal_logs[-1]['equity']:.2f}")

    # ==========================================
    # Case 2: Abnormal Trading (High Risk/Churn)
    # ==========================================
    print("\n[Case 2: Abnormal/High Frequency Trading] (50 Steps)")
    print("Strategy: Flip position -1.0 to 1.0 every step (Max Fees, Turnover).")
    
    # Reset Env for clean comparison
    env.reset(seed=42)
    cost_calculator.reset([0], [10000])
    
    abnormal_logs = []
    
    for i in range(50):
        # Alternate full Long / full Short
        target = 1.0 if i % 2 == 0 else -1.0
        action = np.array([target], dtype=np.float32)
        
        next_obs, reward, done, truncated, info = env.step(action)
        costs = cost_calculator.calculate_costs([info])[0]
        
        log_entry = {
            'step': i,
            'action': action[0],
            'reward': reward,
            'cost_margin': costs[0],
            'cost_dd': costs[1],
            'cost_fee': costs[2],
            'equity': info.get('equity'),
            'pos_change_norm': info.get('position_change_norm', 0)
        }
        abnormal_logs.append(log_entry)
        if done: 
            print("Episode ended early (likely liquidation/bankruptcy)")
            break

    # Analysis Case 2
    avg_reward_abn = np.mean([x['reward'] for x in abnormal_logs])
    avg_cost_fee_abn = np.mean([x['cost_fee'] for x in abnormal_logs])
    
    print(f"Avg Reward: {avg_reward_abn:.6f}")
    print(f"Avg Fee Cost: {avg_cost_fee_abn:.6f}")
    print(f"Avg Margin Cost: {np.mean([x['cost_margin'] for x in abnormal_logs]):.6f}")
    print(f"Final Equity: {abnormal_logs[-1]['equity']:.2f}")
    
    # ==========================================
    # Comparison Table
    # ==========================================
    print("\n" + "="*80)
    print(f"{'Metric':<20} | {'Normal (Conservative)':<25} | {'Abnormal (Churning)':<25}")
    print("-" * 80)
    print(f"{'Avg Reward':<20} | {avg_reward_norm:<25.6f} | {avg_reward_abn:<25.6f}")
    print(f"{'Avg Fee Cost (C3)':<20} | {avg_cost_fee_norm:<25.6f} | {avg_cost_fee_abn:<25.6f}")
    print(f"{'Avg Margin Cost(C1)':<20} | {np.mean([x['cost_margin'] for x in normal_logs]):<25.6f} | {np.mean([x['cost_margin'] for x in abnormal_logs]):<25.6f}")
    print(f"{'Final Equity':<20} | {normal_logs[-1]['equity']:<25.2f} | {abnormal_logs[-1]['equity']:<25.2f}")
    print("="*80)
    print("\nInterpretation:")
    print("1. Reward: Normal should be > Abnormal (Abnormal punished by turnover penalty & fees in equity).")
    print("2. Fee Cost: Abnormal should be significantly higher than Normal.")
    print("3. Net Equity: Abnormal should deplete rapidly due to fees.")

if __name__ == "__main__":
    print("Main called")
    run_test()
