#!/usr/bin/env python3
"""
Multi-Scale Transformer + Actor-Critic 模型測試腳本
用於驗證整個系統是否能正常運行
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from configs.model_config import ModelConfig
from transformer_env_adapter_simple import TransformerTradingEnvironment

def test_model_components():
    """測試模型各個組件"""
    print("🔧 測試模型組件...")
    
    # 創建配置
    config = ModelConfig()
    
    # 創建虛擬數據
    print("📊 創建測試數據...")
    n_samples = 1000
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='5min'),
        'open': np.random.randn(n_samples).cumsum() + 50000,
        'high': np.random.randn(n_samples).cumsum() + 50100,
        'low': np.random.randn(n_samples).cumsum() + 49900,
        'close': np.random.randn(n_samples).cumsum() + 50000,
        'volume': np.random.exponential(1000, n_samples),
        'buy_volume': np.random.exponential(500, n_samples),
        'sell_volume': np.random.exponential(500, n_samples),
        'volume_ratio': np.random.normal(1.0, 0.2, n_samples),
        'long_short_ratio': np.random.normal(1.0, 0.3, n_samples),
        'trades': np.random.poisson(100, n_samples),
        'quote_volume': np.random.exponential(50000000, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # 修正數據邏輯
    df['high'] = np.maximum(df[['open', 'close']].max(axis=1), df['high'])
    df['low'] = np.minimum(df[['open', 'close']].min(axis=1), df['low'])
    df['buy_volume'] = np.minimum(df['buy_volume'], df['volume'])
    df['sell_volume'] = df['volume'] - df['buy_volume']
    
    print(f"   ✓ 數據形狀: {df.shape}")
    print(f"   ✓ 時間範圍: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    
    # 測試環境創建
    print("🏗️ 測試環境創建...")
    try:
        env = TransformerTradingEnvironment(df, config)
        print(f"   ✓ 環境創建成功")
        print(f"   ✓ 觀察空間: {env.observation_space}")
        print(f"   ✓ 動作空間: {env.action_space}")
    except Exception as e:
        print(f"   ❌ 環境創建失敗: {e}")
        return False
    
    # 測試模型參數
    print("📊 測試模型參數...")
    param_count = env.get_parameter_count()
    print(f"   ✓ 特徵提取器參數: {param_count['feature_extractor']:,}")
    print(f"   ✓ Transformer參數: {param_count['transformer']:,}")
    print(f"   ✓ Actor-Critic參數: {param_count['actor_critic']:,}")
    print(f"   ✓ 總參數量: {param_count['total']:,}")
    
    return env

def test_environment_interaction(env):
    """測試環境交互"""
    print("🎮 測試環境交互...")
    
    try:
        # 重置環境
        state, info = env.reset()
        print(f"   ✓ 環境重置成功，狀態形狀: {state.shape}")
        
        # 測試動作採樣
        action = env.get_action(state, deterministic=False)
        print(f"   ✓ 動作採樣成功: {action}")
        print(f"     - 交易方向: {action[0]:.3f}")
        print(f"     - 止盈比例: {action[1]:.3f}")
        print(f"     - 止損比例: {action[2]:.3f}")
        
        # 測試環境步進
        next_state, reward, done, truncated, info = env.step(action)
        print(f"   ✓ 環境步進成功")
        print(f"     - 獎勵: {reward:.6f}")
        print(f"     - 完成: {done}")
        print(f"     - 截斷: {truncated}")
        print(f"     - 下一狀態形狀: {next_state.shape}")
        
        # 測試多步交互
        print("🔄 測試多步交互...")
        total_reward = 0
        for step in range(10):
            action = env.get_action(next_state, deterministic=False)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                next_state, info = env.reset()
                print(f"   ✓ Episode結束，重置環境")
        
        print(f"   ✓ 10步累計獎勵: {total_reward:.6f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 環境交互失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_forward_pass(env):
    """測試模型前向傳播"""
    print("⚡ 測試模型前向傳播...")
    
    try:
        # 設為評估模式
        env.eval_mode()
        
        # 創建批量狀態
        batch_size = 4
        state_dim = env.config.TRANSFORMER_CONFIG['embed_dim']
        batch_states = torch.randn(batch_size, 1, state_dim).to(env.device)
        
        # 測試Actor-Critic前向傳播
        with torch.no_grad():
            actions, log_probs, values = env.actor_critic(batch_states)
            
        print(f"   ✓ 批量前向傳播成功")
        print(f"     - 批量大小: {batch_size}")
        print(f"     - 動作形狀: {actions.shape}")
        print(f"     - 對數概率形狀: {log_probs.shape}")
        print(f"     - 價值形狀: {values.shape}")
        
        # 測試評估
        batch_actions = torch.randn(batch_size, 3).to(env.device)
        batch_actions[:, 0] = torch.tanh(batch_actions[:, 0])  # 方向 [-1, 1]
        batch_actions[:, 1] = torch.sigmoid(batch_actions[:, 1]) * 9.8 + 0.2  # 止盈 [0.2, 10]
        batch_actions[:, 2] = torch.sigmoid(batch_actions[:, 2]) * 0.2 + 0.1  # 止損 [0.1, 0.3]
        
        log_probs_eval, values_eval, entropy = env.evaluate_state_action(
            batch_states.squeeze(1), batch_actions
        )
        
        print(f"   ✓ 批量評估成功")
        print(f"     - 評估對數概率形狀: {log_probs_eval.shape}")
        print(f"     - 評估價值形狀: {values_eval.shape}")
        print(f"     - 熵形狀: {entropy.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 模型前向傳播失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_save_load(env):
    """測試模型保存和加載"""
    print("💾 測試模型保存和加載...")
    
    try:
        # 保存模型
        save_path = 'test_model.pth'
        env.save_model(save_path)
        print(f"   ✓ 模型保存成功: {save_path}")
        
        # 獲取原始輸出
        state, _ = env.reset()
        original_action = env.get_action(state, deterministic=True)
        
        # 加載模型
        env.load_model(save_path)
        print(f"   ✓ 模型加載成功")
        
        # 比較輸出
        loaded_action = env.get_action(state, deterministic=True)
        
        action_diff = np.abs(original_action - loaded_action).max()
        print(f"   ✓ 動作差異: {action_diff:.8f}")
        
        if action_diff < 1e-6:
            print(f"   ✓ 保存/加載一致性驗證通過")
        else:
            print(f"   ⚠️ 保存/加載存在差異")
        
        # 清理測試文件
        os.remove(save_path)
        print(f"   ✓ 清理測試文件")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 模型保存/加載失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主測試函數"""
    print("🚀 Multi-Scale Transformer + Actor-Critic 模型測試")
    print("=" * 60)
    
    # 設備信息
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用設備: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # 測試流程
    tests = [
        ("模型組件", test_model_components),
        ("環境交互", lambda: test_environment_interaction(env)),
        ("模型前向傳播", lambda: test_model_forward_pass(env)),
        ("模型保存/加載", lambda: test_model_save_load(env)),
    ]
    
    results = []
    env = None
    
    for test_name, test_func in tests:
        print(f"🧪 執行測試: {test_name}")
        print("-" * 40)
        
        if test_name == "模型組件":
            env = test_func()
            success = env is not None
        else:
            success = test_func()
        
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name} 測試通過")
        else:
            print(f"❌ {test_name} 測試失敗")
        print()
    
    # 測試總結
    print("📋 測試總結")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"{test_name:20s} {status}")
    
    print("-" * 40)
    print(f"總計: {passed}/{total} 測試通過")
    
    if passed == total:
        print("🎉 所有測試通過！系統可以正常運行。")
        print("\n🚀 你現在可以開始訓練:")
        print("   cd transformer_model")
        print("   python train_transformer.py --data_path ../Data/BTCUSDT_futures_volume_5years_5min.csv")
    else:
        print("⚠️ 部分測試失敗，請檢查錯誤信息。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 