import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import CallbackList
from Train.lagrangian import SharedLagrangianController, LagrangianRewardWrapper, LagrangianCallback

class MockEnv(gym.Env):
    """模擬會發送 Cost 的環境"""
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.steps = 0
        
    def reset(self, seed=None, options=None):
        self.steps = 0
        return np.zeros(10, dtype=np.float32), {}
        
    def step(self, action):
        self.steps += 1
        # 模擬：每步都有 Cost，且最後一步回傳 episode metrics
        reward = 0.01  # Log Return
        terminated = self.steps >= 10
        truncated = False
        
        info = {
            "cost": 0.5, # High cost
            "cost_breakdown": {"death_cost": 0.0, "dense_buffer_cost": 0.5},
            "profit": 100.0,
        }
        
        if terminated:
            # VecMonitor 行為模擬
            info["episode"] = {"r": 0.1, "l": 10, "t": 1.0}
            info["final_balance"] = 10100.0
            # TradingEnv 行為模擬
            info["episode_metrics"] = {
                "return_orig": 0.1,
                "return_orig_scaled": 1.0,
                "return_total": 0.5, # 1.0 - (0.1 * 0.5 * 10)
                "return_cost": 5.0,
                "cost_breakdown": {"dense_buffer_cost": 5.0},
                "cost_penalty_total": 0.5,
                "cost_penalty_breakdown": {"dense_buffer_cost": 0.5},
            }
            
        return np.zeros(10, dtype=np.float32), reward, terminated, truncated, info

def test_callback_dump_stats():
    """測試 Callback 是否能正確執行 _dump_stats 而不崩潰"""
    # 1. Setup
    controller = SharedLagrangianController(cost_limit=0.1, lambda_init=0.1)
    
    # 建立 Callback，設定 log_freq=1 讓它馬上印
    callback = LagrangianCallback(
        controller=controller, 
        update_freq=10, 
        log_freq=1, 
        reward_scale=10.0,
        verbose=1
    )
    
    # 建立 Dummy Model 讓 callback 讀取 (mock replay buffer)
    class DummyModel:
        def __init__(self):
            self.replay_buffer = None # Skip Q-value stats
            self.logger = None
            
    # Mock Logger
    from stable_baselines3.common.logger import configure
    callback.init_callback(DummyModel())
    callback.logger = configure(None, ["stdout"])

    # 2. Run simulation
    # 模擬 VecEnv 的 locals
    infos = []
    
    # Run 10 steps to finish one episode
    print("\n--- Simulating Episode ---")
    for _ in range(10):
        # 模擬 Step 結束
        if _ == 9: # Last step
             infos = [{
                 "cost": 0.5,
                 "episode": {"r": 0.1, "l": 10},
                 "episode_metrics": {
                     "return_orig": 0.1, 
                     "return_orig_scaled": 1.0,
                     "return_total": 0.5,
                     "return_cost": 5.0,
                     "cost_breakdown": {"dense_buffer_cost": 5.0},
                     "cost_penalty_total": 0.5,
                     "cost_penalty_breakdown": {"dense_buffer_cost": 0.5}
                 },
                 "profit": 100.0,
                 "final_balance": 10100.0
             }]
        else:
             infos = [{"cost": 0.5}]
             
        callback.update_locals({"infos": infos})
        callback.on_step()
        
    print("\n--- Test Finished Successfully ---")

if __name__ == "__main__":
    test_callback_dump_stats()
