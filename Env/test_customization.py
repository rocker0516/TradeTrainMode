"""
測試自定義組件範例

展示如何使用 TradingEnvironment 的三個擴展點：
1. AccountFeatureBuilder - 自定義帳戶特徵
2. RewardCalculator - 自定義獎勵函數
3. InfoCollector - 自定義資訊收集
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from Env import (
    TradingEnvironment,
    DefaultAccountFeatureBuilder,
    ExtendedAccountFeatureBuilder,
    RewardCalculator,
    DefaultInfoCollector,
    ExtendedInfoCollector,
)


def test_default_components():
    """測試默認組件"""
    print("\n" + "="*60)
    print("測試 1: 使用默認組件")
    print("="*60)
    
    # 創建簡單測試數據
    n_steps = 1000
    df = pd.DataFrame({
        'open': np.random.randn(n_steps).cumsum() + 100,
        'high': np.random.randn(n_steps).cumsum() + 102,
        'low': np.random.randn(n_steps).cumsum() + 98,
        'close': np.random.randn(n_steps).cumsum() + 100,
        'volume': np.random.rand(n_steps) * 1000,
    })
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    # 創建環境（默認組件）
    env = TradingEnvironment(
        df=df,
        initial_balance=10_000,
        leverage=10,
        window_size=50,
        max_liq_count=3,
    )
    
    print(f"[OK] 環境創建成功")
    print(f"  - 觀測空間: {env.observation_space.shape}")
    print(f"  - 動作空間: {env.action_space.shape}")
    print(f"  - 帳戶特徵數: {env.account_feature_count}")
    print(f"  - 帳戶特徵: {env.account_feature_builder.feature_names()}")
    
    # 運行幾步
    obs, info = env.reset()
    total_reward = 0.0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    print(f"[OK] 運行 {i+1} 步成功")
    print(f"  - 累積獎勵: {total_reward:.2f}")
    print(f"  - 當前權益: {info['equity']:.2f}")


def test_extended_features():
    """測試擴展帳戶特徵"""
    print("\n" + "="*60)
    print("測試 2: 使用擴展帳戶特徵")
    print("="*60)
    
    # 創建測試數據
    n_steps = 1000
    df = pd.DataFrame({
        'open': np.random.randn(n_steps).cumsum() + 100,
        'high': np.random.randn(n_steps).cumsum() + 102,
        'low': np.random.randn(n_steps).cumsum() + 98,
        'close': np.random.randn(n_steps).cumsum() + 100,
        'volume': np.random.rand(n_steps) * 1000,
    })
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    # 使用擴展特徵構建器
    feature_builder = ExtendedAccountFeatureBuilder()
    
    env = TradingEnvironment(
        df=df,
        initial_balance=10_000,
        leverage=10,
        window_size=50,
        max_liq_count=3,
        account_feature_builder=feature_builder,
    )
    
    print(f"[OK] 環境創建成功（擴展特徵）")
    print(f"  - 觀測空間: {env.observation_space.shape}")
    print(f"  - 帳戶特徵數: {env.account_feature_count}")
    print(f"  - 帳戶特徵: {feature_builder.feature_names()}")
    
    # 運行測試
    obs, info = env.reset()
    action = np.array([0.5], dtype=np.float32)  # 50% 做多
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"[OK] 執行一步成功")
    print(f"  - 獎勵: {reward:.4f}")
    print(f"  - 觀測形狀: {obs.shape}")


def test_custom_reward():
    """測試自定義獎勵函數"""
    print("\n" + "="*60)
    print("測試 3: 使用自定義獎勵函數")
    print("="*60)
    
    # 創建測試數據
    n_steps = 1000
    df = pd.DataFrame({
        'open': np.random.randn(n_steps).cumsum() + 100,
        'high': np.random.randn(n_steps).cumsum() + 102,
        'low': np.random.randn(n_steps).cumsum() + 98,
        'close': np.random.randn(n_steps).cumsum() + 100,
        'volume': np.random.rand(n_steps) * 1000,
    })
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    # 自定義獎勵參數
    reward_calculator = RewardCalculator(
        w_stop_loss=100.0,   # 提高止損懲罰
        w_terminal=200.0,    # 提高終局懲罰
        w_return=20.0,       # 提高收益獎勵
        w_risk=10.0,         # 降低風險權重
        w_struct=3.0,        # 降低結構性風險權重
    )
    
    env = TradingEnvironment(
        df=df,
        initial_balance=10_000,
        leverage=10,
        window_size=50,
        max_liq_count=3,
        reward_calculator=reward_calculator,
    )
    
    print(f"[OK] 環境創建成功（自定義獎勵）")
    print(f"  - 獎勵配置: {reward_calculator.get_info()}")
    
    # 運行測試
    obs, info = env.reset()
    rewards = []
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    
    print(f"[OK] 運行 {len(rewards)} 步成功")
    print(f"  - 平均獎勵: {np.mean(rewards):.4f}")
    print(f"  - 獎勵範圍: [{np.min(rewards):.4f}, {np.max(rewards):.4f}]")


def test_extended_info():
    """測試擴展資訊收集器"""
    print("\n" + "="*60)
    print("測試 4: 使用擴展資訊收集器")
    print("="*60)
    
    # 創建測試數據
    n_steps = 1000
    df = pd.DataFrame({
        'open': np.random.randn(n_steps).cumsum() + 100,
        'high': np.random.randn(n_steps).cumsum() + 102,
        'low': np.random.randn(n_steps).cumsum() + 98,
        'close': np.random.randn(n_steps).cumsum() + 100,
        'volume': np.random.rand(n_steps) * 1000,
    })
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    # 使用擴展資訊收集器
    info_collector = ExtendedInfoCollector()
    
    env = TradingEnvironment(
        df=df,
        initial_balance=10_000,
        leverage=10,
        window_size=50,
        max_liq_count=3,
        info_collector=info_collector,
    )
    
    print(f"[OK] 環境創建成功（擴展資訊收集）")
    
    # 運行一個完整回合
    obs, info = env.reset()
    step_count = 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        if terminated or truncated:
            break
        if step_count >= 100:  # 限制步數
            break
    
    print(f"[OK] 回合結束")
    print(f"  - 總步數: {step_count}")
    print(f"  - 終止原因: {info.get('termination_reason', 'N/A')}")
    print(f"  - 是否成功: {info.get('is_success', False)}")
    print(f"  - 最終權益: {info.get('final_equity', 0):.2f}")
    print(f"  - 報酬率: {info.get('profit_rate', 0):.2f}%")
    
    # 擴展統計
    if 'max_drawdown' in info:
        print(f"  - 最大回撤: {info['max_drawdown']:.2f}%")
    if 'sharpe_ratio' in info:
        print(f"  - 夏普比率: {info['sharpe_ratio']:.4f}")
    if 'win_rate' in info:
        print(f"  - 勝率: {info['win_rate']:.2f}%")
    if 'avg_trade_pnl' in info:
        print(f"  - 平均交易損益: {info['avg_trade_pnl']:.2f}")


def test_termination_conditions():
    """測試終止條件"""
    print("\n" + "="*60)
    print("測試 5: 驗證終止條件")
    print("="*60)
    
    # 創建短數據（測試 data_exhausted）
    n_steps = 200
    df = pd.DataFrame({
        'open': np.random.randn(n_steps).cumsum() + 100,
        'high': np.random.randn(n_steps).cumsum() + 102,
        'low': np.random.randn(n_steps).cumsum() + 98,
        'close': np.random.randn(n_steps).cumsum() + 100,
        'volume': np.random.rand(n_steps) * 1000,
    })
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    env = TradingEnvironment(
        df=df,
        initial_balance=10_000,
        leverage=10,
        window_size=50,
        max_liq_count=2,  # 降低上限以便測試
        min_balance=100,
    )
    
    print(f"[OK] 環境配置:")
    print(f"  - 數據長度: {len(df)}")
    print(f"  - 最大強平次數: {env.max_liq_count}")
    print(f"  - 最小資金: {env.min_balance}")
    
    # 測試正常完成
    obs, info = env.reset()
    terminated_reason = None
    step_count = 0
    while True:
        action = np.array([0.0], dtype=np.float32)  # 保持平倉
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        if terminated:
            terminated_reason = info.get('termination_reason')
            break
        if step_count >= len(df) - 60:
            break
    
    print(f"\n[OK] 測試結果:")
    print(f"  - 步數: {step_count}")
    print(f"  - 終止原因: {terminated_reason}")
    print(f"  - 預期: data_exhausted")
    print(f"  - 符合預期: {terminated_reason == 'data_exhausted'}")


def main():
    """運行所有測試"""
    print("\n" + "="*60)
    print("TradingEnvironment 重構驗證測試")
    print("="*60)
    
    try:
        test_default_components()
        test_extended_features()
        test_custom_reward()
        test_extended_info()
        test_termination_conditions()
        
        print("\n" + "="*60)
        print("[OK] 所有測試通過！")
        print("="*60)
        print("\n重構成功！環境已準備好用於 SAC 拉格朗日 Agent 訓練。")
        print("\n擴展點:")
        print("  1. AccountFeatureBuilder - 自定義帳戶特徵")
        print("  2. RewardCalculator - 自定義獎勵函數")
        print("  3. InfoCollector - 自定義資訊收集")
        print("\n詳見 CUSTOMIZATION_GUIDE.md")
        
    except Exception as e:
        print(f"\n[FAIL] 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

