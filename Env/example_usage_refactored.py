"""
重構後環境使用範例

展示如何使用新版 TradingEnvironment，包含：
1. 基礎使用（預設配置）
2. 自定義帳戶特徵
3. 自定義獎勵函數
4. 自定義資訊收集器
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

from trading_env import TradingEnvironment
from account_features import (
    AccountFeatureBuilder,
    DefaultAccountFeatureBuilder,
    ExtendedAccountFeatureBuilder
)
from info_collector import (
    InfoCollector,
    DefaultInfoCollector,
    ExtendedInfoCollector
)
from reward import RewardCalculator


# ============================================================
# 範例 1: 基礎使用（預設配置）
# ============================================================
def example_basic_usage():
    """基礎使用範例"""
    print("=" * 60)
    print("範例 1: 基礎使用（預設配置）")
    print("=" * 60)
    
    # 載入數據
    df = pd.read_csv('../Data/BTCUSDT_futures_volume_5years_5min.csv')
    
    # 創建環境（使用預設配置）
    env = TradingEnvironment(
        df=df,
        initial_balance=10_000,
        transaction_fee=0.001,
        window_size=288,  # 24小時
        leverage=10,
        max_liq_count=3,  # 最多強平 3 次
        random_start=True
    )
    
    # 執行一個回合
    obs, info = env.reset()
    print(f"觀測形狀: {obs.shape}")
    print(f"動作空間: {env.action_space}")
    
    done = False
    total_reward = 0
    step_count = 0
    
    while not done and step_count < 100:  # 限制步數以快速展示
        action = env.action_space.sample()  # 隨機動作
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
    
    print(f"\n回合結束:")
    print(f"  步數: {step_count}")
    print(f"  總獎勵: {total_reward:.4f}")
    if done:
        print(f"  終止原因: {info.get('termination_reason', 'unknown')}")
        print(f"  最終權益: {info.get('final_balance', 0):.2f}")
        print(f"  收益率: {info.get('profit_rate', 0):.2f}%")
    print()


# ============================================================
# 範例 2: 使用擴展帳戶特徵
# ============================================================
def example_extended_features():
    """使用擴展帳戶特徵範例"""
    print("=" * 60)
    print("範例 2: 使用擴展帳戶特徵（含槓桿比率、可用保證金）")
    print("=" * 60)
    
    df = pd.read_csv('../Data/BTCUSDT_futures_volume_5years_5min.csv')
    
    # 使用擴展特徵構建器
    feature_builder = ExtendedAccountFeatureBuilder(max_leverage=20.0)
    
    env = TradingEnvironment(
        df=df,
        initial_balance=10_000,
        window_size=288,
        leverage=10,
        account_feature_builder=feature_builder,  # 注入自定義特徵構建器
        max_liq_count=5
    )
    
    print(f"帳戶特徵名稱: {feature_builder.feature_names()}")
    print(f"帳戶特徵數量: {feature_builder.feature_count()}")
    
    obs, info = env.reset()
    print(f"觀測形狀: {obs.shape}")
    print(f"  數據特徵: {env.data_feature_count}")
    print(f"  帳戶特徵: {env.account_feature_count}")
    print()


# ============================================================
# 範例 3: 自定義獎勵函數
# ============================================================
class SimpleRewardCalculator(RewardCalculator):
    """
    簡化獎勵計算器範例
    
    只關注權益變化，不考慮其他因素。
    """
    
    def compute(self, **kwargs) -> float:
        """
        計算簡化獎勵：只看權益變化率
        
        Args:
            **kwargs: 包含 last_equity, new_equity 等
            
        Returns:
            float: 獎勵值
        """
        last_equity = kwargs.get('last_equity', 0.0)
        new_equity = kwargs.get('new_equity', 0.0)
        done = kwargs.get('done', False)
        termination_reason = kwargs.get('termination_reason', '')
        
        # 基礎獎勵：權益變化率
        if last_equity > 0:
            equity_change_rate = (new_equity - last_equity) / last_equity
            reward = equity_change_rate * 100  # 放大 100 倍
        else:
            reward = 0.0
        
        # 終局獎勵
        if done:
            if termination_reason == 'data_exhausted':
                reward += 10.0  # 成功存活獎勵
            elif termination_reason in ['liq_limit', 'balance_insufficient']:
                reward -= 20.0  # 失敗懲罰
        
        return float(reward)


def example_custom_reward():
    """自定義獎勵函數範例"""
    print("=" * 60)
    print("範例 3: 自定義獎勵函數（簡化版）")
    print("=" * 60)
    
    df = pd.read_csv('../Data/BTCUSDT_futures_volume_5years_5min.csv')
    
    # 使用自定義獎勵計算器
    reward_calculator = SimpleRewardCalculator()
    
    env = TradingEnvironment(
        df=df,
        initial_balance=10_000,
        window_size=288,
        reward_calculator=reward_calculator,  # 注入自定義獎勵計算器
        max_liq_count=3
    )
    
    obs, info = env.reset()
    
    # 執行幾步觀察獎勵
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"步數 {i+1}: 獎勵={reward:.4f}")
        if done:
            break
    print()


# ============================================================
# 範例 4: 使用擴展資訊收集器
# ============================================================
def example_extended_info():
    """使用擴展資訊收集器範例"""
    print("=" * 60)
    print("範例 4: 使用擴展資訊收集器（含最大回撤、夏普比率）")
    print("=" * 60)
    
    df = pd.read_csv('../Data/BTCUSDT_futures_volume_5years_5min.csv')
    
    # 使用擴展資訊收集器
    info_collector = ExtendedInfoCollector()
    
    env = TradingEnvironment(
        df=df,
        initial_balance=10_000,
        window_size=288,
        info_collector=info_collector,  # 注入自定義資訊收集器
        max_liq_count=3
    )
    
    obs, info = env.reset()
    
    done = False
    step_count = 0
    
    while not done and step_count < 200:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1
        
        # 打印單步資訊（含當前回撤）
        if 'current_drawdown' in info:
            print(f"步數 {step_count}: 當前回撤={info['current_drawdown']:.4f}")
    
    # 打印回合結束資訊（含最大回撤、夏普比率）
    if done:
        print(f"\n回合結束:")
        print(f"  終止原因: {info.get('termination_reason', 'unknown')}")
        print(f"  最終權益: {info.get('final_balance', 0):.2f}")
        print(f"  收益率: {info.get('profit_rate', 0):.2f}%")
        print(f"  最大回撤: {info.get('max_drawdown', 0):.4f}")
        print(f"  夏普比率: {info.get('sharpe_ratio', 0):.4f}")
    print()


# ============================================================
# 範例 5: SAC 訓練設定範例（偽代碼）
# ============================================================
def example_sac_training_setup():
    """
    SAC 訓練設定範例（偽代碼）
    
    展示如何配置環境以供 SAC 拉格朗日 agent 訓練。
    """
    print("=" * 60)
    print("範例 5: SAC 訓練設定範例（偽代碼）")
    print("=" * 60)
    
    df = pd.read_csv('../Data/BTCUSDT_futures_volume_5years_5min.csv')
    
    # 訓練環境配置
    env = TradingEnvironment(
        df=df,
        initial_balance=10_000,
        transaction_fee=0.001,
        window_size=288,
        leverage=10,
        max_liq_count=3,  # 最多強平 3 次後失敗
        random_start=True,  # 訓練時使用隨機起點
        account_feature_builder=ExtendedAccountFeatureBuilder(max_leverage=20.0),
        info_collector=ExtendedInfoCollector()
    )
    
    print("環境配置完成，適用於 SAC 訓練:")
    print(f"  觀測空間: {env.observation_space.shape}")
    print(f"  動作空間: {env.action_space}")
    print(f"  成功條件: 資料耗盡")
    print(f"  失敗條件: 強平次數 >= {env.max_liq_count} 或 資金耗盡")
    print(f"  隨機起點: {env.random_start}")
    
    # 偽代碼：SAC 訓練迴圈
    print("\n# SAC 訓練迴圈（偽代碼）:")
    print("# from stable_baselines3 import SAC")
    print("# model = SAC('MlpPolicy', env, verbose=1)")
    print("# model.learn(total_timesteps=1_000_000)")
    print("# model.save('sac_trading_agent')")
    print()


# ============================================================
# 主函數
# ============================================================
if __name__ == '__main__':
    # 執行所有範例
    try:
        example_basic_usage()
    except Exception as e:
        print(f"範例 1 錯誤: {e}\n")
    
    try:
        example_extended_features()
    except Exception as e:
        print(f"範例 2 錯誤: {e}\n")
    
    try:
        example_custom_reward()
    except Exception as e:
        print(f"範例 3 錯誤: {e}\n")
    
    try:
        example_extended_info()
    except Exception as e:
        print(f"範例 4 錯誤: {e}\n")
    
    try:
        example_sac_training_setup()
    except Exception as e:
        print(f"範例 5 錯誤: {e}\n")
    
    print("=" * 60)
    print("所有範例執行完成")
    print("=" * 60)

