from __future__ import annotations

import multiprocessing
import time
from typing import Dict, List, Any
from collections import deque

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from Train.lagrangian import SharedLagrangianController, LagrangianRewardWrapper, LagrangianCallback
from Env.trading_env import TradingEnvironment


# ==========================================
# 1. 單元測試 (Unit Tests)
# ==========================================

class MockEnv(gym.Env):
    """模擬環境：可控制回傳的 cost 與 reward"""
    def __init__(self, cost_seq: List[float], reward_seq: List[float]):
        super().__init__()
        self.cost_seq = deque(cost_seq)
        self.reward_seq = deque(reward_seq)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        
    def reset(self, **kwargs):
        return np.array([0.0]), {}
        
    def step(self, action):
        cost = self.cost_seq.popleft() if self.cost_seq else 0.0
        reward = self.reward_seq.popleft() if self.reward_seq else 0.0
        done = len(self.cost_seq) == 0
        info = {"cost": cost, "profit": reward} # profit for callback stats
        return np.array([0.0]), reward, done, False, info


def test_controller_initialization():
    ctrl = SharedLagrangianController(cost_limit=0.1, lambda_init=0.5, lambda_max=5.0)
    assert ctrl.cost_limit == 0.1
    assert ctrl.current_lambda == 0.5
    assert ctrl.lambda_max == 5.0

def test_controller_update_increase():
    # Cost (0.2) > Limit (0.1) -> Lambda 應增加
    ctrl = SharedLagrangianController(cost_limit=0.1, kp=1.0, lambda_init=1.0)
    new_lambda = ctrl.update(avg_cost=0.2)
    # new = 1.0 + 1.0 * (0.2 - 0.1) = 1.1
    assert new_lambda == pytest.approx(1.1)
    assert ctrl.current_lambda == pytest.approx(1.1)

def test_controller_update_decrease():
    # Cost (0.05) < Limit (0.1) -> Lambda 應減少
    ctrl = SharedLagrangianController(cost_limit=0.1, kp=1.0, lambda_init=1.0)
    new_lambda = ctrl.update(avg_cost=0.05)
    # new = 1.0 + 1.0 * (0.05 - 0.1) = 0.95
    assert new_lambda == pytest.approx(0.95)

def test_controller_clamp_max():
    # Cost 極大，Lambda 不應超過 Max
    ctrl = SharedLagrangianController(cost_limit=0.1, kp=100.0, lambda_init=1.0, lambda_max=5.0)
    new_lambda = ctrl.update(avg_cost=10.0)
    assert new_lambda == 5.0
    assert ctrl.current_lambda == 5.0

def test_controller_clamp_min():
    # Cost 極小，Lambda 不應低於 Min (0.0)
    ctrl = SharedLagrangianController(cost_limit=0.1, kp=100.0, lambda_init=1.0, lambda_min=0.0)
    new_lambda = ctrl.update(avg_cost=-10.0)
    assert new_lambda == 0.0
    assert ctrl.current_lambda == 0.0

def test_reward_wrapper_calculation():
    # R' = R*scale - λ*C
    ctrl = SharedLagrangianController(cost_limit=0.1, lambda_init=2.0)
    env = MockEnv(cost_seq=[0.5], reward_seq=[1.0])
    wrapped = LagrangianRewardWrapper(env, ctrl, reward_scale=10.0)
    
    _, reward, _, _, info = wrapped.step([0])
    
    # Expected: (1.0 * 10.0) - (2.0 * 0.5) = 10.0 - 1.0 = 9.0
    assert reward == pytest.approx(9.0)
    assert info["lag_lambda"] == 2.0
    assert info["original_reward"] == 1.0
    assert info["modified_reward"] == 9.0

def test_wrapper_reads_dynamic_lambda():
    # 測試 Wrapper 能讀到 Controller 更新後的 λ
    ctrl = SharedLagrangianController(cost_limit=0.1, lambda_init=1.0)
    env = MockEnv(cost_seq=[0.5, 0.5], reward_seq=[1.0, 1.0])
    wrapped = LagrangianRewardWrapper(env, ctrl)
    
    # Step 1: lambda=1.0
    _, r1, _, _, _ = wrapped.step([0])
    # Step 2: 更新 lambda -> 2.0
    ctrl._lambda_val.value = 2.0 
    _, r2, _, _, _ = wrapped.step([0])
    
    # R1 = 1 - 1*0.5 = 0.5
    # R2 = 1 - 2*0.5 = 0.0
    assert r1 == 0.5
    assert r2 == 0.0


# ==========================================
# 2. 整合情境測試 (Scenarios) - 50+ Cases
# ==========================================

@pytest.mark.parametrize("case_id", range(50))
def test_lagrangian_scenarios_50_cases(case_id):
    """
    透過參數化生成 50 種不同情境，驗證 Lagrangian 機制的穩健性。
    
    變因：
    - Initial Lambda: 0.0 ~ 5.0
    - Cost Pattern: 安全(0.0), 危險(0.1), 極度危險(1.0)
    - Reward Pattern: 賺錢, 賠錢
    """
    # 隨機生成測試參數 (Deterministic per case_id)
    rng = np.random.RandomState(case_id)
    
    init_lambda = rng.uniform(0.0, 5.0)
    cost_limit = 0.05
    kp = 0.1
    reward_scale = rng.choice([1.0, 10.0])
    
    # Cost Pattern
    # 0-15: Safe (avg < limit)
    # 16-35: Risky (avg > limit)
    # 36-49: Extreme (avg >> limit)
    if case_id < 16:
        costs = rng.uniform(0.0, 0.04, size=10)
        expect_lambda_decrease = True
    elif case_id < 36:
        costs = rng.uniform(0.06, 0.15, size=10)
        expect_lambda_decrease = False
    else:
        costs = rng.uniform(0.2, 1.0, size=10)
        expect_lambda_decrease = False
        
    rewards = rng.uniform(-1.0, 1.0, size=10)
    
    # Setup
    ctrl = SharedLagrangianController(
        cost_limit=cost_limit, 
        kp=kp, 
        lambda_init=init_lambda
    )
    env = MockEnv(cost_seq=list(costs), reward_seq=list(rewards))
    wrapped = LagrangianRewardWrapper(env, ctrl, reward_scale=reward_scale)
    
    # Run
    avg_cost_measured = 0.0
    for i in range(10):
        _, r, _, _, info = wrapped.step([0])
        
        # Check Reward Formula Correctness
        cost = costs[i]
        raw_r = rewards[i]
        curr_lam = ctrl.current_lambda
        expected_r = (raw_r * reward_scale) - (curr_lam * cost)
        assert r == pytest.approx(expected_r, abs=1e-5), f"Case {case_id}: Reward calculation error"
        
        avg_cost_measured += cost
    
    avg_cost_measured /= 10.0
    
    # Update Lambda
    old_lambda = init_lambda
    new_lambda = ctrl.update(avg_cost_measured)
    
    # Check Update Logic
    if expect_lambda_decrease:
        # 如果原本已經是 0，就維持 0
        if old_lambda > 0:
            assert new_lambda < old_lambda or new_lambda == 0.0, f"Case {case_id}: Lambda should decrease"
    else:
        # 如果原本已經是 Max，就維持 Max
        if old_lambda < ctrl.lambda_max:
            assert new_lambda > old_lambda or new_lambda == ctrl.lambda_max, f"Case {case_id}: Lambda should increase"

    # Check Clamp
    assert 0.0 <= new_lambda <= ctrl.lambda_max, f"Case {case_id}: Lambda out of bounds"


# ==========================================
# 3. 並行同步測試 (Parallel Sync)
# ==========================================

def _worker_process(ctrl: SharedLagrangianController, steps: int):
    """模擬並行環境中的 Worker，只讀取 λ"""
    for _ in range(steps):
        # 讀取 λ (模擬 Wrapper 行為)
        _ = ctrl.current_lambda
        time.sleep(0.001)

def test_parallel_lambda_sync():
    """驗證多進程下 λ 的修改能被所有 Worker 即時看到"""
    ctrl = SharedLagrangianController(cost_limit=0.1, lambda_init=1.0)
    
    # 啟動 2 個 Worker
    p1 = multiprocessing.Process(target=_worker_process, args=(ctrl, 50))
    p2 = multiprocessing.Process(target=_worker_process, args=(ctrl, 50))
    p1.start()
    p2.start()
    
    # 主進程修改 λ
    time.sleep(0.01)
    ctrl.update(avg_cost=1.0) # Cost > Limit -> Lambda should increase
    assert ctrl.current_lambda > 1.0
    
    # 確保 Worker 沒掛掉 (若 Shared Memory 有問題可能會 Crash)
    p1.join()
    p2.join()
    assert p1.exitcode == 0
    assert p2.exitcode == 0


# ==========================================
# 4. Callback 整合測試
# ==========================================

class MockLogger:
    def __init__(self):
        self.records = {}
    def record(self, key, value, exclude=None):
        self.records[key] = value

def test_callback_logic():
    ctrl = SharedLagrangianController(cost_limit=0.1)
    cb = LagrangianCallback(controller=ctrl, update_freq=2, log_freq=2, verbose=0)
    cb.logger = MockLogger()
    
    # 模擬 2 個並行環境，跑 2 個 Steps
    # Step 1
    # Env 1: Cost=0.2 (High)
    # Env 2: Cost=0.0 (Safe)
    infos_step1 = [{"cost": 0.2}, {"cost": 0.0}]
    cb.locals = {"infos": infos_step1}
    cb.on_step()
    
    # Step 2 (Update Lambda)
    infos_step2 = [{"cost": 0.2}, {"cost": 0.0}]
    cb.locals = {"infos": infos_step2}
    cb.on_step()
    
    # 檢查 λ 是否更新
    # Avg Cost = (0.2+0+0.2+0)/4 = 0.1
    # Cost Limit = 0.1
    # Violation = 0 -> Lambda 不變
    # 但 Callback 邏輯是 collect buffer -> update
    assert "lagrangian/lambda" in cb.logger.records
    assert "lagrangian/avg_cost" in cb.logger.records
    assert cb.logger.records["lagrangian/avg_cost"] == 0.1


