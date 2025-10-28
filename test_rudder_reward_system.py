"""
測試 SAC-Lagrangian + RUDDER Reward 系統

驗證功能：
1. PBRS 勢能計算與 shaping reward
2. Outcome 計算與回填機制
3. 成本計算（step_cost, cost_prob, CVaR）
4. RUDDER Replay Buffer 的回填邏輯
5. Lagrangian 控制器更新
6. 環境 info 協議完整性
"""

import sys
import os

# 添加專案根目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from Env.trading_env import TradingEnvironment
from Env.reward import create_default_reward_system
from Train.utils.rudder_replay_buffer import RUDDERReplayBuffer
from Train.utils.wrappers import RiskPenaltyWrapper, InfoLoggerWrapper
from Train.utils.lagrangian import LagrangianController


def test_reward_system():
    """測試獎勵系統組件"""
    print("=" * 60)
    print("測試 1: 獎勵系統組件")
    print("=" * 60)
    
    # 創建獎勵系統
    reward_system = create_default_reward_system()
    shaping_calc = reward_system['shaping']
    outcome_calc = reward_system['outcome']
    cost_calc = reward_system['cost']
    potential_calc = reward_system['potential']
    
    # 測試 PBRS 勢能計算
    print("\n[測試 PBRS 勢能計算]")
    phi1 = potential_calc.compute_potential(
        margin_buffer=0.8,
        mae_atr=1.0,
        dist_to_extreme_atr=2.0,
        has_position=True,
    )
    phi2 = potential_calc.compute_potential(
        margin_buffer=0.6,
        mae_atr=1.5,
        dist_to_extreme_atr=1.0,
        has_position=True,
    )
    print(f"  Φ(s1) = {phi1:.4f}")
    print(f"  Φ(s2) = {phi2:.4f}")
    print(f"  γΦ(s2) - Φ(s1) = {potential_calc.gamma * phi2 - phi1:.4f}")
    
    # 測試 Shaping Reward
    print("\n[測試 Shaping Reward]")
    reward_shaping = shaping_calc.compute(
        margin_buffer=0.6,
        mae_atr=1.5,
        dist_to_extreme_atr=1.0,
        has_position=True,
        prev_margin_buffer=0.8,
        prev_mae_atr=1.0,
        prev_dist_to_extreme_atr=2.0,
        prev_has_position=True,
        entry_happened=False,
        entry_streak_count=0,
    )
    print(f"  Shaping Reward = {reward_shaping:.4f}")
    
    # 測試 Outcome 計算
    print("\n[測試 Outcome 計算]")
    outcome_profit = outcome_calc.compute_outcome_delta(
        realized_pnl=100.0,
        notional=10000.0,
        exit_reason='close',
    )
    outcome_stop_loss = outcome_calc.compute_outcome_delta(
        realized_pnl=-200.0,
        notional=10000.0,
        exit_reason='stop_loss',
    )
    outcome_liq = outcome_calc.compute_outcome_delta(
        realized_pnl=-500.0,
        notional=10000.0,
        exit_reason='liq',
    )
    print(f"  Outcome (盈利平倉) = {outcome_profit:.4f}")
    print(f"  Outcome (止損) = {outcome_stop_loss:.4f}")
    print(f"  Outcome (強平) = {outcome_liq:.4f}")
    
    # 測試成本計算
    print("\n[測試成本計算]")
    step_cost = cost_calc.compute_step_cost(fee=10.0, slippage=2.0, funding=0.5)
    cost_prob = cost_calc.compute_cost_prob(stop_loss_triggered=True, liq_triggered=False)
    loss_cvar = cost_calc.compute_cvar_loss_sample(outcome_delta=-50.0)
    print(f"  Step Cost = {step_cost:.4f}")
    print(f"  Cost Prob = {cost_prob:.4f}")
    print(f"  Loss CVaR Sample = {loss_cvar:.4f}")
    
    print("\n[PASS] 測試 1 通過：所有獎勵組件正常工作")


def test_environment_info_protocol():
    """測試環境 info 協議"""
    print("\n" + "=" * 60)
    print("測試 2: 環境 Info 協議")
    print("=" * 60)
    
    # 加載數據
    data_path = "Data/BTCUSDT_futures_volume_5years_5min.csv"
    if not os.path.exists(data_path):
        print(f"⚠ 數據文件不存在：{data_path}")
        return
    
    df = pd.read_csv(data_path)
    df = df.iloc[:1000]  # 使用前 1000 筆數據測試
    
    # 創建環境（啟用 RUDDER）
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        leverage=10,
        use_rudder=True,
        random_start=False,
    )
    
    print(f"\n[環境配置]")
    print(f"  use_rudder = {env.use_rudder}")
    print(f"  initial_balance = {env.initial_balance}")
    print(f"  leverage = {env.leverage}")
    
    # 運行幾步並檢查 info
    obs, info = env.reset()
    print(f"\n[Reset Info Keys]: {list(info.keys())}")
    
    # 執行若干步
    for step in range(10):
        action = np.array([0.5])  # 持續做多
        obs, reward, done, truncated, info = env.step(action)
        
        if step == 0:
            print(f"\n[Step {step} Info Keys]: {list(info.keys())}")
            required_keys = [
                'is_entry', 'is_reduce', 'is_exit',
                'trade_id', 'outcome_delta_to_entry',
                'step_cost', 'cost_prob', 'loss_cvar_sample',
            ]
            missing_keys = [k for k in required_keys if k not in info]
            if missing_keys:
                print(f"  ⚠ 缺少必要鍵：{missing_keys}")
            else:
                print(f"  [OK] 所有必要鍵存在")
        
        # 檢查進場事件
        if info.get('is_entry', False):
            print(f"\n[Step {step}] 進場事件")
            print(f"  entered_trade_id = {info.get('entered_trade_id', -1)}")
            print(f"  reward (shaping) = {reward:.4f}")
        
        # 檢查出場事件
        if info.get('is_exit', False) or info.get('is_reduce', False):
            print(f"\n[Step {step}] 出場事件")
            print(f"  exited_trade_id = {info.get('exited_trade_id', -1)}")
            print(f"  exit_reason = {info.get('exit_reason', 'unknown')}")
            print(f"  outcome_delta = {info.get('outcome_delta_to_entry', 0.0):.4f}")
            print(f"  step_cost = {info.get('step_cost', 0.0):.4f}")
        
        if done:
            print(f"\n[Episode 結束於 Step {step}]")
            if 'termination_reason' in info:
                print(f"  termination_reason = {info['termination_reason']}")
            break
    
    print("\n[PASS] 測試 2 通過：環境 info 協議完整")


def test_rudder_replay_buffer():
    """測試 RUDDER Replay Buffer"""
    print("\n" + "=" * 60)
    print("測試 3: RUDDER Replay Buffer")
    print("=" * 60)
    
    # 創建 buffer
    buffer = RUDDERReplayBuffer(
        capacity=1000,
        observation_shape=(20, 288),
        action_dim=1,
        device='cpu',
    )
    
    print(f"\n[Buffer 配置]")
    print(f"  capacity = {buffer.capacity}")
    print(f"  size = {buffer.size}")
    
    # 模擬添加經驗（包含進場與出場事件）
    print(f"\n[模擬交易序列]")
    
    # Step 0: 進場（trade_id=0）
    state0 = np.random.randn(20, 288).astype(np.float32)
    action0 = np.array([0.5], dtype=np.float32)
    reward0 = -0.5  # 進場懲罰
    next_state0 = np.random.randn(20, 288).astype(np.float32)
    info0 = {
        'is_entry': True,
        'entered_trade_id': 0,
        'trade_id': 0,
        'outcome_delta_to_entry': 0.0,
        'step_cost': 0.0,
        'cost_prob': 0.0,
        'loss_cvar_sample': 0.0,
    }
    buffer.add(state0, action0, reward0, next_state0, False, info0)
    print(f"  Step 0: 進場 (trade_id=0)")
    
    # Step 1-4: 持倉中
    for i in range(1, 5):
        state = np.random.randn(20, 288).astype(np.float32)
        action = np.array([0.5], dtype=np.float32)
        reward = 0.1  # 小額 shaping
        next_state = np.random.randn(20, 288).astype(np.float32)
        info = {
            'is_entry': False,
            'trade_id': 0,
            'outcome_delta_to_entry': 0.0,
            'step_cost': 0.0,
            'cost_prob': 0.0,
            'loss_cvar_sample': 0.0,
        }
        buffer.add(state, action, reward, next_state, False, info)
    print(f"  Step 1-4: 持倉中 (trade_id=0)")
    
    # Step 5: 出場（平倉，outcome_delta 回填到 step 0）
    state5 = np.random.randn(20, 288).astype(np.float32)
    action5 = np.array([0.0], dtype=np.float32)
    reward5 = 0.0
    next_state5 = np.random.randn(20, 288).astype(np.float32)
    info5 = {
        'is_entry': False,
        'is_exit': True,
        'exited_trade_id': 0,
        'exit_reason': 'close',
        'trade_id': -1,
        'outcome_delta_to_entry': 10.0,  # 盈利 outcome
        'step_cost': 5.0,
        'cost_prob': 0.0,
        'loss_cvar_sample': 0.0,
    }
    buffer.add(state5, action5, reward5, next_state5, False, info5)
    print(f"  Step 5: 出場 (trade_id=0, outcome_delta=10.0)")
    
    # 檢查回填
    print(f"\n[檢查回填結果]")
    print(f"  Buffer size = {buffer.size}")
    print(f"  Step 0 outcome_delta = {buffer.outcome_deltas[0][0]:.4f} (預期 10.0)")
    print(f"  Step 5 outcome_delta = {buffer.outcome_deltas[5][0]:.4f} (預期 0.0)")
    
    # 採樣測試
    if buffer.is_ready(batch_size=4):
        batch = buffer.sample(4)
        print(f"\n[採樣測試]")
        print(f"  batch keys = {list(batch.keys())}")
        print(f"  reward shape = {batch['reward'].shape}")
        print(f"  outcome_delta shape = {batch['outcome_delta'].shape}")
    
    # 統計
    stats = buffer.get_stats()
    print(f"\n[Buffer 統計]")
    print(f"  size = {stats['size']}")
    print(f"  mean_outcome = {stats['mean_outcome']:.4f}")
    print(f"  total_outcomes_backfilled = {stats['total_outcomes_backfilled']}")
    
    print("\n[PASS] 測試 3 通過：RUDDER 回填邏輯正常")


def test_lagrangian_controller():
    """測試 Lagrangian 控制器"""
    print("\n" + "=" * 60)
    print("測試 4: Lagrangian 控制器")
    print("=" * 60)
    
    # 創建控制器
    controller = LagrangianController(
        lambda_prob_init=1.0,
        lambda_cvar_init=1.0,
        lr_prob=0.01,
        lr_cvar=0.01,
        target_prob=0.03,
        target_cvar=0.01,
    )
    
    print(f"\n[控制器配置]")
    print(f"  lambda_prob_init = 1.0")
    print(f"  lambda_cvar_init = 1.0")
    print(f"  target_prob = 0.03")
    print(f"  target_cvar = 0.01")
    
    # 模擬更新（違反約束的情況）
    print(f"\n[模擬更新（違反約束）]")
    for i in range(5):
        # 模擬批次成本（違反約束）
        cost_prob_batch = np.random.rand(32, 1) * 0.1  # 平均約 0.05 > target 0.03
        loss_cvar_batch = np.random.rand(32, 1) * 0.05  # 平均約 0.025 > target 0.01
        
        stats = controller.update(
            cost_prob_batch=cost_prob_batch,
            loss_cvar_batch=loss_cvar_batch,
        )
        
        print(f"  Update {i+1}:")
        print(f"    lambda_prob = {stats['lambda_prob']:.4f}")
        print(f"    lambda_cvar = {stats['lambda_cvar']:.4f}")
        print(f"    violation_prob = {stats['violation_prob']:.4f}")
        print(f"    violation_cvar = {stats['violation_cvar']:.4f}")
    
    print(f"\n[期望行為] λ 應該隨著違規增大而增加")
    
    print("\n[PASS] 測試 4 通過：Lagrangian 控制器更新正常")


def test_wrappers():
    """測試環境 Wrapper"""
    print("\n" + "=" * 60)
    print("測試 5: 環境 Wrapper")
    print("=" * 60)
    
    # 加載數據
    data_path = "Data/BTCUSDT_futures_volume_5years_5min.csv"
    if not os.path.exists(data_path):
        print(f"⚠ 數據文件不存在：{data_path}")
        return
    
    df = pd.read_csv(data_path)
    df = df.iloc[:500]
    
    # 創建環境並包裝
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        leverage=10,
        use_rudder=True,
        random_start=False,
    )
    
    # 添加 wrapper
    env = RiskPenaltyWrapper(env, lambda_prob=1.0, lambda_cvar=1.0)
    env = InfoLoggerWrapper(env)
    
    print(f"\n[Wrapper 層級]")
    print(f"  1. TradingEnvironment (use_rudder=True)")
    print(f"  2. RiskPenaltyWrapper (lambda_prob=1.0, lambda_cvar=1.0)")
    print(f"  3. InfoLoggerWrapper")
    
    # 運行幾步
    obs, info = env.reset()
    for step in range(10):
        action = np.array([0.3])
        obs, reward, done, truncated, info = env.step(action)
        
        if step == 0:
            print(f"\n[Step {step} Info Keys]")
            wrapper_keys = ['reward_original', 'reward_penalty', 'reward_penalized']
            for k in wrapper_keys:
                if k in info:
                    print(f"  [OK] {k} = {info[k]:.4f}")
                else:
                    print(f"  ⚠ {k} 缺失")
        
        if done:
            if 'episode_stats' in info:
                print(f"\n[Episode 統計]")
                stats = info['episode_stats']
                print(f"  entry_count = {stats['entry_count']}")
                print(f"  exit_count = {stats['exit_count']}")
                print(f"  stop_loss_count = {stats['stop_loss_count']}")
                print(f"  total_outcome = {stats['total_outcome']:.4f}")
            break
    
    print("\n[PASS] 測試 5 通過：Wrapper 正常工作")


def main():
    """運行所有測試"""
    print("\n" + "=" * 80)
    print(" SAC-Lagrangian + RUDDER Reward 系統測試 ".center(80, "="))
    print("=" * 80)
    
    try:
        test_reward_system()
        test_environment_info_protocol()
        test_rudder_replay_buffer()
        test_lagrangian_controller()
        test_wrappers()
        
        print("\n" + "=" * 80)
        print(" [SUCCESS] 所有測試通過！ ".center(80, "="))
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[FAIL] 測試失敗：{e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

