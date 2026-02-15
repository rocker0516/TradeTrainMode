"""
测试双CNN分离架构的实现

验证：
1. 分离特征提取（5m 和 1d）
2. 双CNN输出维度
3. cross-attention 机制
"""

import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
import pytest
from Env.feature_transformer import FeatureTransformer
from Env.Components.market_data import MarketData
from Train.sb3_cnn_policy import (
    Target5mCNN,
    Others5mCNN,
    Target1dCNN,
    Others1dCNN,
    CrossTimeframeAttention,
    DualCnnFeatureExtractor,
)
import gymnasium as gym
from gymnasium import spaces


def create_mock_5m_data(n: int = 1000) -> pd.DataFrame:
    """创建模拟 5m 数据"""
    dates = pd.date_range(start='2020-01-01', periods=n, freq='5min')
    df = pd.DataFrame(index=dates)
    df['timestamp'] = dates
    
    # BTCUSDT 数据
    np.random.seed(42)
    df['BTCUSDT_close'] = 50000 + np.cumsum(np.random.randn(n) * 100)
    df['BTCUSDT_open'] = df['BTCUSDT_close'].shift(1).fillna(df['BTCUSDT_close'])
    df['BTCUSDT_high'] = df['BTCUSDT_close'] * (1 + np.abs(np.random.randn(n) * 0.01))
    df['BTCUSDT_low'] = df['BTCUSDT_close'] * (1 - np.abs(np.random.randn(n) * 0.01))
    df['BTCUSDT_volume'] = np.random.rand(n) * 1000
    df['BTCUSDT_quote_volume'] = df['BTCUSDT_volume'] * df['BTCUSDT_close']
    df['BTCUSDT_trades'] = np.random.randint(100, 1000, n)
    df['BTCUSDT_buy_volume'] = df['BTCUSDT_volume'] * 0.5
    df['BTCUSDT_sell_volume'] = df['BTCUSDT_volume'] * 0.5
    
    # ETHUSDT 数据（用于 others）
    df['ETHUSDT_close'] = 3000 + np.cumsum(np.random.randn(n) * 10)
    df['ETHUSDT_open'] = df['ETHUSDT_close'].shift(1).fillna(df['ETHUSDT_close'])
    df['ETHUSDT_high'] = df['ETHUSDT_close'] * (1 + np.abs(np.random.randn(n) * 0.01))
    df['ETHUSDT_low'] = df['ETHUSDT_close'] * (1 - np.abs(np.random.randn(n) * 0.01))
    df['ETHUSDT_volume'] = np.random.rand(n) * 500
    df['ETHUSDT_quote_volume'] = df['ETHUSDT_volume'] * df['ETHUSDT_close']
    df['ETHUSDT_trades'] = np.random.randint(50, 500, n)
    df['ETHUSDT_buy_volume'] = df['ETHUSDT_volume'] * 0.5
    df['ETHUSDT_sell_volume'] = df['ETHUSDT_volume'] * 0.5
    
    return df.fillna(0.0)


def create_mock_1d_data(n: int = 100) -> pd.DataFrame:
    """创建模拟 1d 数据"""
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    df = pd.DataFrame(index=dates)
    df['timestamp'] = dates
    
    # BTCUSDT 1d 数据
    np.random.seed(42)
    df['BTCUSDT_close'] = 50000 + np.cumsum(np.random.randn(n) * 1000)
    df['BTCUSDT_high'] = df['BTCUSDT_close'] * (1 + np.abs(np.random.randn(n) * 0.02))
    df['BTCUSDT_low'] = df['BTCUSDT_close'] * (1 - np.abs(np.random.randn(n) * 0.02))
    df['BTCUSDT_open_interest_close'] = np.random.rand(n) * 1e9
    df['BTCUSDT_funding_rate_close'] = np.random.randn(n) * 0.0001
    df['BTCUSDT_global_long_short_account_ratio_global_account_long_short_ratio'] = 0.5 + np.random.randn(n) * 0.1
    df['BTCUSDT_top_long_short_position_ratio_top_position_long_short_ratio'] = 0.5 + np.random.randn(n) * 0.1
    df['BTCUSDT_liquidation_long_liquidation_usd'] = np.random.rand(n) * 1e6
    df['BTCUSDT_liquidation_short_liquidation_usd'] = np.random.rand(n) * 1e6
    df['BTCUSDT_ask_bids_bids_usd'] = np.random.rand(n) * 1e7
    df['BTCUSDT_ask_bids_asks_usd'] = np.random.rand(n) * 1e7
    
    # ETHUSDT 1d 数据
    df['ETHUSDT_close'] = 3000 + np.cumsum(np.random.randn(n) * 50)
    df['ETHUSDT_high'] = df['ETHUSDT_close'] * (1 + np.abs(np.random.randn(n) * 0.02))
    df['ETHUSDT_low'] = df['ETHUSDT_close'] * (1 - np.abs(np.random.randn(n) * 0.02))
    df['ETHUSDT_open_interest_close'] = np.random.rand(n) * 5e8
    df['ETHUSDT_funding_rate_close'] = np.random.randn(n) * 0.0001
    df['ETHUSDT_global_long_short_account_ratio_global_account_long_short_ratio'] = 0.5 + np.random.randn(n) * 0.1
    df['ETHUSDT_ask_bids_bids_usd'] = np.random.rand(n) * 5e6
    df['ETHUSDT_ask_bids_asks_usd'] = np.random.rand(n) * 5e6
    
    # Macro 指标
    df['fear_greed_fear_greed_index'] = 50 + np.random.randn(n) * 20
    df['altcoin_season_altcoin_index'] = 50 + np.random.randn(n) * 20
    df['bitcoin_macro_oscillator_bmo_value'] = np.random.randn(n) * 10
    df['bitcoin_sth_sopr_lth_sopr'] = 1.0 + np.random.randn(n) * 0.1
    
    return df.fillna(0.0)


def test_5m_features_split():
    """测试 5m 特征分离"""
    transformer = FeatureTransformer()
    df_5m = create_mock_5m_data(1000)
    
    atr_ratio_arr = np.random.rand(1000) * 0.01
    rv_ratio_arr = np.random.rand(1000) * 0.5
    
    target_features, others_features, target_cols, others_cols = transformer.build_5m_features_split(
        df_5m,
        target_symbol='BTCUSDT',
        atr_ratio_arr=atr_ratio_arr,
        rv_ratio_arr=rv_ratio_arr,
        feature_symbols=['ETHUSDT'],
    )
    
    # 验证维度
    assert target_features.shape[0] == 1000
    assert target_features.shape[1] == len(transformer.OPTIMIZED_TARGET_5M_COLS)
    assert others_features.shape[0] == 1000
    assert others_features.shape[1] == len(transformer.OTHERS_5M_COLS_PER_SYMBOL)
    assert len(target_cols) == len(transformer.OPTIMIZED_TARGET_5M_COLS)
    assert len(others_cols) == len(transformer.OTHERS_5M_COLS_PER_SYMBOL)
    
    # 验证没有 NaN/Inf
    assert not np.isnan(target_features).any()
    assert not np.isinf(target_features).any()
    assert not np.isnan(others_features).any()
    assert not np.isinf(others_features).any()


def test_1d_features_split():
    """测试 1d 特征分离"""
    transformer = FeatureTransformer()
    df_1d = create_mock_1d_data(100)
    df_5m = create_mock_5m_data(288 * 100)  # 确保有足够的数据用于 288 窗口
    
    target_features, others_features, target_cols, others_cols = transformer.build_1d_features_split(
        df_1d,
        df_5m,
        target_symbol='BTCUSDT',
        feature_symbols=['ETHUSDT'],
    )
    
    # 验证维度
    assert target_features.shape[0] == 100
    assert target_features.shape[1] == len(transformer.TARGET_1D_COLS)
    assert others_features.shape[0] == 100
    # others: 1 symbol * 6 + 4 macro = 10
    expected_others_dim = len(transformer.OTHERS_1D_COLS_PER_SYMBOL) + len(transformer.PRICE_SEQ_1D_MACRO_COLS)
    assert others_features.shape[1] == expected_others_dim
    assert len(target_cols) == len(transformer.TARGET_1D_COLS)
    
    # 验证没有 NaN/Inf
    assert not np.isnan(target_features).any()
    assert not np.isinf(target_features).any()
    assert not np.isnan(others_features).any()
    assert not np.isinf(others_features).any()


def test_cnn_output_dims():
    """测试 CNN 输出维度"""
    batch_size = 4
    window_5m = 432
    window_1d = 30
    
    # Target 5m CNN
    cnn_5m_target = Target5mCNN(in_channels=30, emb_dim=128)
    x_5m_target = torch.randn(batch_size, window_5m, 30)
    out_5m_target = cnn_5m_target(x_5m_target)
    assert out_5m_target.shape == (batch_size, 128)
    
    # Others 5m CNN
    cnn_5m_others = Others5mCNN(in_channels=10, emb_dim=64)
    x_5m_others = torch.randn(batch_size, window_5m, 10)
    out_5m_others = cnn_5m_others(x_5m_others)
    assert out_5m_others.shape == (batch_size, 64)
    
    # Target 1d CNN
    cnn_1d_target = Target1dCNN(in_channels=18, emb_dim=64)
    x_1d_target = torch.randn(batch_size, window_1d, 18)
    out_1d_target = cnn_1d_target(x_1d_target)
    assert out_1d_target.shape == (batch_size, 64)
    
    # Others 1d CNN
    cnn_1d_others = Others1dCNN(in_channels=10, emb_dim=32)
    x_1d_others = torch.randn(batch_size, window_1d, 10)
    out_1d_others = cnn_1d_others(x_1d_others)
    assert out_1d_others.shape == (batch_size, 32)


def test_cross_attention():
    """测试 cross-attention 机制"""
    batch_size = 4
    emb_5m = 192
    emb_1d = 96
    
    cross_attn = CrossTimeframeAttention(emb_5m=emb_5m, emb_1d=emb_1d, hidden_dim=128)
    
    emb_5m_tensor = torch.randn(batch_size, emb_5m)
    emb_1d_tensor = torch.randn(batch_size, emb_1d)
    
    out = cross_attn(emb_5m_tensor, emb_1d_tensor)
    
    # 输出维度应该与 emb_5m 相同（残差连接）
    assert out.shape == (batch_size, emb_5m)


def test_dual_cnn_feature_extractor():
    """测试完整的 DualCnnFeatureExtractor"""
    # 创建 observation space
    observation_space = gym.spaces.Dict({
        "price_seq_target": spaces.Box(low=-np.inf, high=np.inf, shape=(432, 30), dtype=np.float32),
        "price_seq_others": spaces.Box(low=-np.inf, high=np.inf, shape=(432, 10), dtype=np.float32),
        "price_seq_1d_target": spaces.Box(low=-np.inf, high=np.inf, shape=(30, 18), dtype=np.float32),
        "price_seq_1d_others": spaces.Box(low=-np.inf, high=np.inf, shape=(30, 10), dtype=np.float32),
        "account_state": spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32),
    })
    
    extractor = DualCnnFeatureExtractor(
        observation_space,
        emb_5m_target=128,
        emb_5m_others=64,
        emb_1d_target=64,
        emb_1d_others=32,
        emb_vec=128,
        out_dim=256,
        use_cross_attention=True,
    )
    
    # 创建模拟观察
    batch_size = 4
    observations = {
        "price_seq_target": torch.randn(batch_size, 432, 30),
        "price_seq_others": torch.randn(batch_size, 432, 10),
        "price_seq_1d_target": torch.randn(batch_size, 30, 18),
        "price_seq_1d_others": torch.randn(batch_size, 30, 10),
        "account_state": torch.randn(batch_size, 23),
    }
    
    # 前向传播
    output = extractor(observations)
    
    # 验证输出维度
    assert output.shape == (batch_size, 256)


if __name__ == "__main__":
    test_5m_features_split()
    print("✓ test_5m_features_split passed")
    
    test_1d_features_split()
    print("✓ test_1d_features_split passed")
    
    test_cnn_output_dims()
    print("✓ test_cnn_output_dims passed")
    
    test_cross_attention()
    print("✓ test_cross_attention passed")
    
    test_dual_cnn_feature_extractor()
    print("✓ test_dual_cnn_feature_extractor passed")
    
    print("\n所有测试通过！")

