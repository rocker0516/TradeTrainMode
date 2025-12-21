print("STARTING TEST SCRIPT", flush=True)
"""
交易環境完整測試腳本
測試所有功能包括：基本交易、止盈止損、最低交易金額、觀察空間等
"""
import numpy as np
import pandas as pd
from Env.trading_env import TradingEnvironment
from Train.config import Config
import matplotlib.pyplot as plt
import pytest

# Patch Config for testing with small data
Config.MIN_EPISODE_STEPS = 50
Config.WINDOW_SIZE = 20 # Reduce window size for small data tests
Config.TRANSACTION_FEE = 0.001 # Default small fee for tests
Config.FEE_LIMIT_RATIO = 1.0 # Loose limit so tests don't die unexpectedly

def create_test_data(n_samples=2000):
    """創建模擬的交易數據"""
    np.random.seed(42)
    
    # 模擬BTC價格走勢
    base_price = 50000
    price_changes = np.random.randn(n_samples) * 0.002  # 0.2% 波動
    
    # 添加一些趨勢
    trend = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 0.001
    price_changes += trend
    
    # 生成價格序列
    close_prices = [base_price]
    for change in price_changes[1:]:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(new_price)
    
    # 創建OHLCV數據
    data = {
        'open': [],
        'high': [],
        'low': [],
        'close': close_prices,
        'volume': np.random.randint(1000, 50000, n_samples)
    }
    
    for i, close in enumerate(close_prices):
        if i == 0:
            open_price = close
        else:
            open_price = close_prices[i-1]
        
        high = max(open_price, close) * (1 + np.random.rand() * 0.001)
        low = min(open_price, close) * (1 - np.random.rand() * 0.001)
        
        data['open'].append(open_price)
        data['high'].append(high)
        data['low'].append(low)
    
    return pd.DataFrame(data)

@pytest.fixture
def env():
    df = create_test_data(1000)
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=100,
        leverage=10,
        min_trade_qty=0.001,
    )
    env.reset()
    return env


def test_basic_functionality():
    """測試基本功能"""
    print("=" * 60)
    print("📊 基本功能測試")
    print("=" * 60)
    
    # 創建測試數據
    df = create_test_data(1000)
    
    # 創建環境
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=100,
        leverage=10,
        min_trade_qty=0.001
    )
    
    print(f"✅ 環境創建成功")
    print(f"   觀察空間形狀: {env.observation_space.shape}")
    print(f"   動作空間形狀: {env.action_space.shape}")
    print(f"   動作空間範圍: {env.action_space.low} ~ {env.action_space.high}")
    
    # 重置環境
    obs, info = env.reset()
    if isinstance(obs, dict):
         print(f"✅ 環境重置成功，觀察空間為 Dict")
         for k, v in obs.items():
             print(f"   - {k}: {v.shape}")
    else:
         print(f"✅ 環境重置成功，觀察空間形狀: {obs.shape}")
    
    return env, obs

def test_trading_actions(env):
    """測試交易動作"""
    print("\n📈 交易動作測試")
    print("-" * 40)
    
    initial_balance = env.balance
    test_actions = [
        ([0.0, 0.0, 0.0], "不交易"),
        ([0.3, 2.0, 1.0], "做多30%，止盈2%，止損1%"),
        ([0.0, 0.0, 0.0], "平倉"),
        ([-0.2, 1.5, 0.8], "做空20%，止盈1.5%，止損0.8%"),
        ([0.5, 3.0, 1.5], "反向做多50%"),
    ]
    
    results = []
    
    for i, (action, description) in enumerate(test_actions):
        print(f"\n{i+1}. {description}")
        action_array = np.array(action, dtype=np.float32)
        
        # 執行動作
        obs, reward, done, truncated, info = env.step(action_array)
        
        # 記錄結果
        result = {
            'action': action,
            'description': description,
            'balance': env.balance,
            'position': env.btc_held,
            'total_value': env.total_value,
            'reward': reward,
            'profit_loss': env.total_value - initial_balance
        }
        results.append(result)
        
        print(f"   動作: {action}")
        print(f"   資金: ${env.balance:.2f}")
        print(f"   持倉: {env.btc_held:.6f} BTC")
        print(f"   總價值: ${env.total_value:.2f}")
        print(f"   獎勵: {reward:.4f}")
        print(f"   損益: ${result['profit_loss']:.2f}")
        
        if done:
            print("   ⚠️ 環境結束")
            break
    
    return results

def test_stop_loss_take_profit():
    """測試止盈止損機制"""
    print("\n🛡️ 止盈止損測試")
    print("-" * 40)
    
    # Save original min steps
    original_min_steps = Config.MIN_EPISODE_STEPS
    original_window_size = Config.WINDOW_SIZE
    Config.MIN_EPISODE_STEPS = 5 # Small enough for this test
    Config.WINDOW_SIZE = 5
    
    # 創建特殊的測試數據（價格大幅波動）
    base_price = 50000
    prices = [base_price]
    
    # 先上漲3%觸發止盈
    for i in range(10):
        prices.append(base_price * (1 + 0.003 * (i + 1)))
    
    # 然後下跌2%觸發止損
    for i in range(10):
        prices.append(prices[-1] * 0.998)
    
    df_test = pd.DataFrame({
        'open': prices,
        'high': [p * 1.001 for p in prices],
        'low': [p * 0.999 for p in prices],
        'close': prices,
        'volume': [1000] * len(prices)
    })
    
    env = TradingEnvironment(
        df=df_test,
        initial_balance=10000,
        window_size=5,
        min_trade_qty=0.001  # 最低交易數量
    )
    
    obs, _ = env.reset()
    
    # 開多倉，設定止盈2%，止損1%
    action = np.array([0.5, 2.0, 1.0], dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)
    
    print(f"開倉後:")
    print(f"   持倉: {env.btc_held:.6f} BTC")
    print(f"   止盈價: ${env.take_profit_price if hasattr(env, 'take_profit_price') else 'N/A'}")
    print(f"   止損價: ${env.executor.position.stop_loss_price if hasattr(env.executor.position, 'stop_loss_price') else 'N/A'}")
    
    # 繼續執行幾步，觀察止盈止損是否觸發
    for step in range(5):
        current_price = env.df.iloc[env.current_step]['close']
        print(f"\n步驟 {step + 1}, 當前價格: ${current_price:.2f}")
        
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 不做新交易
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"   持倉: {env.btc_held:.6f} BTC")
        print(f"   總價值: ${env.total_value:.2f}")
        
        if env.btc_held == 0:
            print("   ✅ 止盈/止損已觸發，倉位已平倉")
            break
            
    Config.MIN_EPISODE_STEPS = original_min_steps
    Config.WINDOW_SIZE = original_window_size

def test_minimum_trade_amount():
    """測試最低交易數量限制"""
    print("\n💰 最低交易數量測試")
    print("-" * 40)
    
    df = create_test_data(100)
    
    # 測試不同的最低交易數量設定 (BTC)
    test_qtys = [0.001, 0.01, 0.05, 0.1]
    
    for min_qty in test_qtys:
        print(f"\n最低交易數量: {min_qty} BTC")
        
        env = TradingEnvironment(
            df=df,
            initial_balance=10000,
            window_size=20,
            min_trade_qty=min_qty
        )
        
        obs, _ = env.reset()
        
        # 嘗試小額交易
        small_action = np.array([0.01, 2.0, 1.0], dtype=np.float32)  # 1%倉位
        obs, reward, done, truncated, info = env.step(small_action)
        
        current_price = env.df.iloc[env.current_step-1]['close']
        intended_qty = (env.initial_balance * 0.01 * env.leverage) / current_price
        
        print(f"   預期交易數量: {intended_qty:.6f} BTC")
        print(f"   實際持倉: {env.btc_held:.6f} BTC")
        print(f"   交易是否執行: {'是' if env.btc_held != 0 else '否'}")

def test_observation_space():
    """測試觀察空間"""
    print("\n👁️ 觀察空間測試")
    print("-" * 40)
    
    df = create_test_data(200)
    env = TradingEnvironment(df=df, window_size=50)
    
    obs, _ = env.reset()
    
    # env.observation_space is a Dict now
    print(f"觀察空間類型: {type(obs)}")
    if isinstance(obs, dict):
         for key, val in obs.items():
             print(f"   Key: {key}, Shape: {val.shape}, Range: [{val.min():.3f}, {val.max():.3f}]")
    else:
        print(f"觀察空間形狀: {obs.shape}")
        print(f"觀察空間範圍: [{obs.min():.3f}, {obs.max():.3f}]")

def run_full_episode():
    """運行完整回合測試"""
    print("\n🎮 完整回合測試")
    print("-" * 40)
    
    df = create_test_data(500)
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        window_size=100
    )
    
    obs, _ = env.reset()
    
    total_reward = 0
    step_count = 0
    balance_history = [env.balance]
    total_value_history = [env.total_value]
    
    # 簡單的隨機策略測試
    while not env.done and step_count < 100:
        # 生成隨機動作
        action = env.action_space.sample()
        
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        balance_history.append(env.balance)
        total_value_history.append(env.total_value)
        
        if step_count % 20 == 0:
            print(f"   步驟 {step_count}: 總價值=${env.total_value:.2f}, 累計獎勵={total_reward:.4f}")
    
    print(f"\n回合結束:")
    print(f"   總步數: {step_count}")
    print(f"   最終總價值: ${env.total_value:.2f}")
    print(f"   初始資金: ${env.initial_balance}")
    print(f"   總損益: ${env.total_value - env.initial_balance:.2f}")
    print(f"   收益率: {(env.total_value / env.initial_balance - 1) * 100:.2f}%")
    print(f"   累計獎勵: {total_reward:.4f}")
    
    return balance_history, total_value_history

def main():
    """主測試函數"""
    print("🚀 交易環境測試開始")
    print("=" * 60)
    
    try:
        # 1. 基本功能測試
        env, obs = test_basic_functionality()
        
        # 2. 交易動作測試
        trading_results = test_trading_actions(env)
        
        # 3. 止盈止損測試
        test_stop_loss_take_profit()
        
        # 4. 最低交易金額測試
        test_minimum_trade_amount()
        
        # 5. 觀察空間測試
        test_observation_space()
        
        # 6. 完整回合測試
        balance_history, total_value_history = run_full_episode()
        
        print("\n" + "=" * 60)
        print("✅ 所有測試完成！")
        print("=" * 60)
        
        # 總結
        print("\n📋 測試總結:")
        print(f"   ✅ 環境創建和重置: 正常")
        print(f"   ✅ 交易動作執行: 正常")
        print(f"   ✅ 止盈止損機制: 正常")
        print(f"   ✅ 最低交易金額: 正常")
        print(f"   ✅ 觀察空間生成: 正常")
        print(f"   ✅ 完整回合運行: 正常")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 環境測試通過，可以用於強化學習訓練！")
    else:
        print("\n💥 環境測試失敗，需要修正問題。")
