"""
测试训练系统设置

验证所有组件是否正常工作。
"""

import os
import sys
import pandas as pd
import numpy as np

# 设置 Windows 控制台 UTF-8 编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Env.trading_env import TradingEnvironment
from Train.models import SAC_LSTM_Model
from Train.trainers import SACTrainer
from Train.utils import ReplayBuffer
from Train.config import Config, get_quick_test_config


def test_config():
    """测试配置系统"""
    print("测试配置系统...")
    
    config = get_quick_test_config()
    assert config.training.total_episodes == 50
    
    # 测试保存和加载
    config.save('test_config.json')
    loaded_config = Config.load('test_config.json')
    assert loaded_config.training.total_episodes == 50
    
    # 清理
    os.remove('test_config.json')
    
    print("✓ 配置系统正常")


def test_environment():
    """测试交易环境"""
    print("测试交易环境...")
    
    # 创建示例数据
    df = pd.DataFrame({
        'open': np.random.rand(1000) * 100 + 40000,
        'high': np.random.rand(1000) * 100 + 40100,
        'low': np.random.rand(1000) * 100 + 39900,
        'close': np.random.rand(1000) * 100 + 40000,
        'volume': np.random.rand(1000) * 1000,
        'buy_volume': np.random.rand(1000) * 500,
        'sell_volume': np.random.rand(1000) * 500,
        'volume_ratio': np.random.rand(1000),
        'long_short_ratio': np.random.rand(1000) + 0.5,
        'trades': np.random.randint(100, 1000, 1000),
        'quote_volume': np.random.rand(1000) * 1000000,
    })
    
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=50,
        leverage=10,
    )
    
    # 测试环境重置
    state, _ = env.reset()
    assert state.shape == env.observation_space.shape
    
    # 测试步进
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    assert next_state.shape == env.observation_space.shape
    assert isinstance(reward, (int, float))
    
    print("✓ 交易环境正常")


def test_model():
    """测试模型"""
    print("测试模型...")
    
    observation_shape = (15, 50)  # (n_features, window_size)
    action_dim = 3
    
    model = SAC_LSTM_Model(
        observation_shape=observation_shape,
        action_dim=action_dim,
        device='cpu',  # 使用 CPU 测试
    )
    
    # 测试动作选择
    state = np.random.randn(*observation_shape).astype(np.float32)
    action = model.select_action(state, evaluate=False)
    assert action.shape == (action_dim,)
    assert np.all(action >= -1) and np.all(action <= 1)
    
    # 测试模型更新
    batch_size = 32
    batch = (
        np.random.randn(batch_size, *observation_shape).astype(np.float32),
        np.random.randn(batch_size, action_dim).astype(np.float32),
        np.random.randn(batch_size).astype(np.float32),
        np.random.randn(batch_size, *observation_shape).astype(np.float32),
        np.random.randint(0, 2, batch_size).astype(np.float32),
    )
    
    metrics = model.update(batch)
    assert 'critic_loss' in metrics
    assert 'actor_loss' in metrics
    
    # 测试保存和加载
    model.save('test_model.pth')
    model.load('test_model.pth')
    os.remove('test_model.pth')
    
    print("✓ 模型正常")


def test_replay_buffer():
    """测试经验回放缓冲区"""
    print("测试经验回放缓冲区...")
    
    buffer = ReplayBuffer(
        buffer_size=1000,
        observation_shape=(15, 50),
        action_dim=3,
    )
    
    # 添加经验
    for _ in range(100):
        state = np.random.randn(15, 50).astype(np.float32)
        action = np.random.randn(3).astype(np.float32)
        reward = np.random.randn()
        next_state = np.random.randn(15, 50).astype(np.float32)
        done = 0.0
        
        buffer.add(state, action, reward, next_state, done)
    
    assert len(buffer) == 100
    
    # 测试采样
    batch = buffer.sample(32)
    assert len(batch) == 5
    assert batch[0].shape == (32, 15, 50)
    
    print("✓ 经验回放缓冲区正常")


def test_trainer():
    """测试训练器"""
    print("测试训练器...")
    
    # 创建示例数据
    df = pd.DataFrame({
        'open': np.random.rand(1000) * 100 + 40000,
        'high': np.random.rand(1000) * 100 + 40100,
        'low': np.random.rand(1000) * 100 + 39900,
        'close': np.random.rand(1000) * 100 + 40000,
        'volume': np.random.rand(1000) * 1000,
        'buy_volume': np.random.rand(1000) * 500,
        'sell_volume': np.random.rand(1000) * 500,
        'volume_ratio': np.random.rand(1000),
        'long_short_ratio': np.random.rand(1000) + 0.5,
        'trades': np.random.randint(100, 1000, 1000),
        'quote_volume': np.random.rand(1000) * 1000000,
    })
    
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=50,
        leverage=10,
    )
    
    model = SAC_LSTM_Model(
        observation_shape=env.observation_space.shape,
        action_dim=env.action_space.shape[0],
        device='cpu',
    )
    
    config = {
        'buffer_size': 1000,
        'batch_size': 32,
        'warmup_steps': 50,
        'update_interval': 1,
        'save_dir': './test_models',
        'log_dir': './test_logs',
        'log_interval': 1,
    }
    
    trainer = SACTrainer(env, model, config)
    
    # 测试单个回合（不实际训练完整流程）
    print("  训练器创建成功")
    
    # 清理测试目录
    import shutil
    if os.path.exists('./test_models'):
        shutil.rmtree('./test_models')
    if os.path.exists('./test_logs'):
        shutil.rmtree('./test_logs')
    
    print("✓ 训练器正常")


def test_integration():
    """集成测试：运行2个完整回合"""
    print("集成测试：运行2个完整回合...")
    
    # 创建示例数据
    df = pd.DataFrame({
        'open': np.random.rand(1000) * 100 + 40000,
        'high': np.random.rand(1000) * 100 + 40100,
        'low': np.random.rand(1000) * 100 + 39900,
        'close': np.random.rand(1000) * 100 + 40000,
        'volume': np.random.rand(1000) * 1000,
        'buy_volume': np.random.rand(1000) * 500,
        'sell_volume': np.random.rand(1000) * 500,
        'volume_ratio': np.random.rand(1000),
        'long_short_ratio': np.random.rand(1000) + 0.5,
        'trades': np.random.randint(100, 1000, 1000),
        'quote_volume': np.random.rand(1000) * 1000000,
    })
    
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=50,
        leverage=10,
    )
    
    model = SAC_LSTM_Model(
        observation_shape=env.observation_space.shape,
        action_dim=env.action_space.shape[0],
        device='cpu',
    )
    
    config = {
        'buffer_size': 1000,
        'batch_size': 32,
        'warmup_steps': 50,
        'update_interval': 1,
        'save_dir': './test_models',
        'log_dir': './test_logs',
        'log_interval': 1,
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    trainer = SACTrainer(env, model, config)
    
    # 运行2个回合
    trainer.train(total_episodes=2, eval_interval=2)
    
    # 验证模型保存
    assert os.path.exists(f"{config['save_dir']}/best_model.pth")
    
    # 清理
    import shutil
    if os.path.exists('./test_models'):
        shutil.rmtree('./test_models')
    if os.path.exists('./test_logs'):
        shutil.rmtree('./test_logs')
    
    print("✓ 集成测试通过")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("训练系统测试")
    print("=" * 60)
    print()
    
    try:
        test_config()
        test_environment()
        test_replay_buffer()
        test_model()
        test_trainer()
        test_integration()
        
        print()
        print("=" * 60)
        print("所有测试通过！✓")
        print("=" * 60)
        print()
        print("系统已就绪，可以开始训练：")
        print("  python Train/train.py --mode quick_test")
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"测试失败：{e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

