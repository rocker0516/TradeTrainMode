
import sys
print("Start of detailed integration test")
import os
import numpy as np
import pandas as pd
import logging

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from Env.trading_env import TradingEnvironment
    from Train.cost import CombinedCostCalculator
    from Train.config import Config
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

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
    print("=== Starting Detailed Integration Test (20 Steps) ===")
    print(f"Config Check: MIN_POSITION_CHANGE={Config.MIN_POSITION_CHANGE}")
    
    # 1. Setup Environment with Synthetic Data
    df = generate_synthetic_data(length=1000)
    
    Config.MIN_POSITION_CHANGE = 0.01 # 1%
    Config.MIN_EPISODE_STEPS = 50
    Config.WINDOW_SIZE = 20
    
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        transaction_fee=0.0004, # 0.04%
        window_size=20,
        leverage=10,
        min_episode_steps=50,
        min_position_change=Config.MIN_POSITION_CHANGE
    )
    
    cost_calculator = CombinedCostCalculator(num_envs=1)
    cost_calculator.reset([0], [10000])
    
    obs, _ = env.reset(seed=42)
    
    # ==========================================
    # Case 1: Normal Trading (Small / Micro moves)
    # Goal: Verify Deadband (Min Position Change)
    # ==========================================
    print("\n[Case 1: Normal/Micro Trading] (20 Steps)")
    print("Strategy: 0.5 -> 0.505 (Change 0.5% < 1%) -> Should be ignored")
    
    logs_c1 = []
    
    # Start with a position
    # Step 0: Open Long 0.5
    env.step(np.array([0.5], dtype=np.float32))
    
    for i in range(20):
        # Try to move from 0.5 to 0.505 (0.5% change)
        target = 0.5 + 0.005 * (1 if i % 2 == 0 else -1) 
        action = np.array([target], dtype=np.float32)
        
        prev_pos_size = env.executor.position.size
        next_obs, reward, done, truncated, info = env.step(action)
        curr_pos_size = env.executor.position.size
        
        pos_changed = abs(curr_pos_size - prev_pos_size) > 1e-8
        
        costs = cost_calculator.calculate_costs([info])[0]
        
        log_entry = {
            'step': i,
            'action': action[0],
            'real_pos': curr_pos_size,
            'changed': pos_changed,
            'reward': reward,
            'cost_turnover': costs[0],
            'equity': info.get('equity'),
            'fee_ratio': info.get('step_fee_ratio', 0)
        }
        logs_c1.append(log_entry)
        
    # Verify Case 1
    hold_count = sum(1 for x in logs_c1 if not x['changed'])
    avg_reward_c1 = np.mean([x['reward'] for x in logs_c1])
    print(f"Steps Ignored (Hold): {hold_count}/20")
    print(f"Avg Reward: {avg_reward_c1:.6f}")
    if hold_count >= 19: # Allow 1 step for initial adjust or similar
        print("SUCCESS: Micro-moves ignored (Deadband works).")
    else:
        print(f"WARNING: Some moves executed. Check min_position_change logic.")

    # ==========================================
    # Case 2: Abnormal Trading (Churning)
    # Goal: Verify Turnover Cost
    # ==========================================
    print("\n[Case 2: Abnormal Trading] (20 Steps)")
    print("Strategy: Flip -1.0 to 1.0 every step (Max Fee, Turnover)")
    
    env.reset(seed=42)
    logs_c2 = []
    
    for i in range(20):
        target = 1.0 if i % 2 == 0 else -1.0
        action = np.array([target], dtype=np.float32)
        
        next_obs, reward, done, truncated, info = env.step(action)
        costs = cost_calculator.calculate_costs([info])[0]
        
        log_entry = {
            'step': i,
            'reward': reward,
            'cost_turnover': costs[0],
            'equity': info.get('equity'),
            'step_fee': info.get('step_fee_ratio', 0) * 10000 # approx amount
        }
        logs_c2.append(log_entry)
        if done: break
        
    # Analysis Case 2
    avg_reward_c2 = np.mean([x['reward'] for x in logs_c2])
    avg_turnover_cost_c2 = np.mean([x['cost_turnover'] for x in logs_c2])
    total_equity_loss = 10000 - logs_c2[-1]['equity']
    
    print(f"Avg Reward: {avg_reward_c2:.6f}")
    print(f"Avg Turnover Cost (C1): {avg_turnover_cost_c2:.6f}")
    print(f"Total Equity Loss: {total_equity_loss:.2f}")
    
    print("\n=== Summary Comparison ===")
    print(f"{'Case':<15} | {'Avg Reward':<15} | {'Turnover Cost':<15} | {'Action'}")
    print("-" * 60)
    print(f"{'Micro (Hold)':<15} | {avg_reward_c1:<15.6f} | {0.0:<15.6f} | Ignored")
    print(f"{'Churning':<15} | {avg_reward_c2:<15.6f} | {avg_turnover_cost_c2:<15.6f} | Executed")
    
    # Validation Logic
    if avg_reward_c1 > avg_reward_c2:
        print("\nVALIDATION PASSED: Holding/Micro-moves yield better reward than churning.")
    else:
        print("\nVALIDATION FAILED: Churning reward is higher (check penalties).")
        
    if avg_turnover_cost_c2 > 0:
        print("VALIDATION PASSED: Turnover Cost correctly tracked for churning.")
    else:
        print("VALIDATION FAILED: Turnover Cost is 0 for churning.")

if __name__ == "__main__":
    print("Main called")
    run_test()
