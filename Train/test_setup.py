"""
System setup test script

Test if all components are working correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test if all imports work"""
    print("Testing imports...")
    
    try:
        import stable_baselines3
        print(f"  - stable-baselines3: {stable_baselines3.__version__}")
    except ImportError as e:
        print(f"  - ERROR: stable-baselines3 not installed: {e}")
        return False
    
    try:
        import torch
        print(f"  - torch: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"  - ERROR: torch not installed: {e}")
        return False
    
    try:
        import pandas
        print(f"  - pandas: {pandas.__version__}")
    except ImportError as e:
        print(f"  - ERROR: pandas not installed: {e}")
        return False
    
    try:
        import numpy
        print(f"  - numpy: {numpy.__version__}")
    except ImportError as e:
        print(f"  - ERROR: numpy not installed: {e}")
        return False
    
    try:
        import gymnasium
        print(f"  - gymnasium: {gymnasium.__version__}")
    except ImportError as e:
        print(f"  - ERROR: gymnasium not installed: {e}")
        return False
    
    print("All imports successful!\n")
    return True


def test_environment():
    """Test if trading environment can be created"""
    print("Testing trading environment...")
    
    try:
        from Env.trading_env import TradingEnvironment
        from Env.reward import RewardCalculator
        import pandas as pd
        import numpy as np
        
        # Create test data
        test_data = pd.DataFrame({
            'open': np.random.randn(1000) * 100 + 50000,
            'high': np.random.randn(1000) * 100 + 50100,
            'low': np.random.randn(1000) * 100 + 49900,
            'close': np.random.randn(1000) * 100 + 50000,
            'volume': np.random.randn(1000) * 1000 + 10000,
            'buy_volume': np.random.randn(1000) * 500 + 5000,
            'sell_volume': np.random.randn(1000) * 500 + 5000,
            'volume_ratio': np.random.randn(1000) * 0.1 + 0.5,
            'long_short_ratio': np.random.randn(1000) * 0.1 + 0.5,
            'trades': np.random.randint(100, 1000, 1000),
            'quote_volume': np.random.randn(1000) * 100000 + 500000,
        })
        
        reward_calculator = RewardCalculator(mode='delta_equity', scale=1.0)
        
        env = TradingEnvironment(
            df=test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=50,
            leverage=10,
            reward_calculator=reward_calculator
        )
        
        # Test environment
        state, _ = env.reset()
        print(f"  - Observation shape: {state.shape}")
        
        action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)
        print(f"  - Step executed successfully")
        print(f"  - Reward: {reward}")
        
        print("Trading environment test passed!\n")
        return True
        
    except Exception as e:
        print(f"  - ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sac():
    """Test if SAC can be created"""
    print("Testing SAC model...")
    
    try:
        from stable_baselines3 import SAC
        from Env.trading_env import TradingEnvironment
        from Env.reward import RewardCalculator
        import pandas as pd
        import numpy as np
        
        # Create test environment
        test_data = pd.DataFrame({
            'open': np.random.randn(1000) * 100 + 50000,
            'high': np.random.randn(1000) * 100 + 50100,
            'low': np.random.randn(1000) * 100 + 49900,
            'close': np.random.randn(1000) * 100 + 50000,
            'volume': np.random.randn(1000) * 1000 + 10000,
            'buy_volume': np.random.randn(1000) * 500 + 5000,
            'sell_volume': np.random.randn(1000) * 500 + 5000,
            'volume_ratio': np.random.randn(1000) * 0.1 + 0.5,
            'long_short_ratio': np.random.randn(1000) * 0.1 + 0.5,
            'trades': np.random.randint(100, 1000, 1000),
            'quote_volume': np.random.randn(1000) * 100000 + 500000,
        })
        
        reward_calculator = RewardCalculator(mode='delta_equity', scale=1.0)
        env = TradingEnvironment(
            df=test_data,
            initial_balance=10000,
            window_size=50,
            leverage=10,
            reward_calculator=reward_calculator
        )
        
        # Create SAC model
        model = SAC('MlpPolicy', env, verbose=0)
        print(f"  - SAC model created successfully")
        
        # Test prediction
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        print(f"  - Action prediction: {action}")
        
        print("SAC model test passed!\n")
        return True
        
    except Exception as e:
        print(f"  - ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("="*60)
    print("SAC Trading System Setup Test")
    print("="*60)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Trading Environment", test_environment),
        ("SAC Model", test_sac),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"FATAL ERROR in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("="*60)
    print("Test Summary:")
    print("="*60)
    
    all_passed = True
    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "[OK]" if result else "[FAIL]"
        print(f"  {symbol} {name}: {status}")
        if not result:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\nAll tests passed! System is ready to use.")
        print("\nQuick start:")
        print("  python Train/train_sac.py --mode quick_test")
        return 0
    else:
        print("\nSome tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

