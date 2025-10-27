"""
SAC + LSTM 策略網絡

整合 LSTM 層到 SAC 的 Actor-Critic 網絡中，以處理序列資訊。
適用於部分可觀測馬可夫決策過程（POMDP）的交易環境。
"""
from __future__ import annotations
import torch as th
import torch.nn as nn
from typing import Dict, List, Tuple, Type, Optional, Any
from gymnasium import spaces
import numpy as np

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import SACPolicy


class LSTMExtractor(BaseFeaturesExtractor):
    """
    LSTM 特徵提取器
    
    將 (n_features, window_size) 的觀測轉換為 LSTM 隱藏狀態。
    
    Args:
        observation_space: 觀測空間 (n_features, window_size)
        features_dim: 輸出特徵維度（LSTM hidden size）
        lstm_layers: LSTM 層數
        lstm_hidden_size: LSTM 隱藏層大小
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        lstm_layers: int = 2,
        lstm_hidden_size: int = 128,
    ):
        super().__init__(observation_space, features_dim)
        
        # 觀測形狀: (n_features, window_size)
        self.n_features = observation_space.shape[0]
        self.window_size = observation_space.shape[1]
        
        # LSTM 參數
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        
        # 降維層：先將 n_features 壓縮（減少 LSTM 輸入維度）
        self.input_projection = nn.Linear(self.n_features, min(self.n_features, 16))
        
        # LSTM 層（輸入: (batch, seq_len, input_size)）
        self.lstm = nn.LSTM(
            input_size=min(self.n_features, 16),
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        
        # 投影層：LSTM 輸出 → features_dim
        self.projection = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        前向傳播
        
        Args:
            observations: (batch, n_features, window_size)
            
        Returns:
            features: (batch, features_dim)
        """
        # 轉置為 LSTM 輸入格式: (batch, seq_len, input_size)
        # (batch, n_features, window_size) -> (batch, window_size, n_features)
        lstm_input = observations.transpose(1, 2)  # (batch, window_size, n_features)
        
        # 降維投影（逐時間步）
        # (batch, window_size, n_features) -> (batch, window_size, compressed_dim)
        compressed = self.input_projection(lstm_input)
        
        # LSTM 前向（訓練時不使用記憶狀態，每個 batch 獨立）
        # lstm_out: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(compressed)
        
        # 取最後時間步的輸出
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # 投影到特徵維度
        features = self.projection(last_output)  # (batch, features_dim)
        
        return features


class RecurrentSACPolicy(SACPolicy):
    """
    SAC + LSTM 策略
    
    使用 LSTM 特徵提取器替代標準 MLP extractor。
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        activation_fn: Type[nn.Module] = nn.ReLU,
        **kwargs,
    ):
        # 設定 LSTM 參數
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        
        # net_arch 預設
        if net_arch is None:
            net_arch = [256, 256]
        
        # 設定自訂 features_extractor_class
        kwargs['features_extractor_class'] = LSTMExtractor
        kwargs['features_extractor_kwargs'] = {
            'features_dim': 128,
            'lstm_layers': lstm_layers,
            'lstm_hidden_size': lstm_hidden_size,
        }
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            **kwargs,
        )
