import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
import gymnasium as gym
from gymnasium import spaces

from configs.model_config import ModelConfig
from models.feature_extractor import FeatureExtractor
from models.multi_scale_transformer import MultiScaleFeatureExtractor
from models.actor_critic import ActorCritic

class SimpleTradingEnvironment:
    """簡化版交易環境，用於測試"""
    
    def __init__(self, df, initial_balance=10000, window_size=288):
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.current_step = window_size
        self.balance = initial_balance
        self.position = 0.0
        self.done = False
        
        # 定義動作空間：連續動作 [方向, 止盈, 止損]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.2, 0.1]),
            high=np.array([1.0, 10.0, 0.3]),
            dtype=np.float32
        )
        
        # 確保timestamp是datetime格式
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if 'timestamp' in self.df.columns:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
                self.df = self.df.set_index('timestamp')
            else:
                # 創建假的時間索引
                self.df.index = pd.date_range('2023-01-01', periods=len(self.df), freq='5min')
    
    def reset(self, seed=None):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0
        self.done = False
        return self._get_simple_observation(), {}
    
    def step(self, action):
        if self.done:
            return self._get_simple_observation(), 0.0, True, False, {}
        
        # 簡化的交易邏輯
        direction = action[0]  # -1 to 1
        current_price = self.df.iloc[self.current_step]['close']
        
        # 計算收益
        if self.current_step > self.window_size:
            price_change = (current_price - self.df.iloc[self.current_step-1]['close']) / self.df.iloc[self.current_step-1]['close']
            reward = direction * price_change * 100  # 簡化的獎勵計算
        else:
            reward = 0.0
        
        self.current_step += 1
        self.done = self.current_step >= len(self.df) - 1
        
        return self._get_simple_observation(), reward, self.done, False, {}
    
    def _get_simple_observation(self):
        """獲取簡化的觀察"""
        # 創建一個固定大小的觀察向量
        window_data = self.df.iloc[self.current_step - self.window_size:self.current_step]
        
        # 簡化特徵：只使用價格特徵
        features = np.zeros((self.window_size, 10))  # 10個簡單特徵
        
        if len(window_data) > 0:
            # 價格特徵
            features[:len(window_data), 0] = window_data['open'].values
            features[:len(window_data), 1] = window_data['high'].values  
            features[:len(window_data), 2] = window_data['low'].values
            features[:len(window_data), 3] = window_data['close'].values
            
            # 成交量特徵
            if 'volume' in window_data.columns:
                features[:len(window_data), 4] = np.log1p(window_data['volume'].values)
            
            # 簡單時間特徵
            timestamps = pd.to_datetime(window_data.index)
            features[:len(window_data), 5] = timestamps.hour / 24.0
            features[:len(window_data), 6] = timestamps.dayofweek / 7.0
            
            # 賬戶特徵
            features[:, 7] = self.balance / self.initial_balance
            features[:, 8] = self.position
            features[:, 9] = self.current_step / len(self.df)
        
        return torch.tensor(features.T, dtype=torch.float32)  # [features, time_steps]

class TransformerTradingEnvironment:
    """Transformer交易環境適配器（簡化版）"""
    
    def __init__(self, df, config=None, **env_kwargs):
        self.config = config if config is not None else ModelConfig()
        
        # 創建簡化的基礎環境
        self.base_env = SimpleTradingEnvironment(df, **env_kwargs)
        
        # 創建模型組件
        self.feature_extractor = FeatureExtractor(self.config)
        self.transformer = MultiScaleFeatureExtractor(self.config)
        self.actor_critic = ActorCritic(self.config)
        
        # 移動到設備
        self.device = self.config.DEVICE
        self.feature_extractor.to(self.device)
        self.transformer.to(self.device)
        self.actor_critic.to(self.device)
        
        self.current_features = None
        
    def reset(self, **kwargs):
        obs, info = self.base_env.reset(**kwargs)
        self.current_features = self._extract_and_process_features(obs)
        return self.current_features, info
    
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        obs, reward, done, truncated, info = self.base_env.step(action)
        
        if not done:
            self.current_features = self._extract_and_process_features(obs)
        
        return self.current_features, reward, done, truncated, info
    
    def _extract_and_process_features(self, raw_obs):
        """簡化的特徵處理"""
        # raw_obs shape: [features, time_steps]
        features = {}
        
        # 將原始觀察轉換為我們需要的格式
        if isinstance(raw_obs, torch.Tensor):
            obs_data = raw_obs.numpy()
        else:
            obs_data = raw_obs
        
        seq_len = obs_data.shape[1]
        
                 # 構造簡化的特徵字典
        features['price'] = torch.tensor(obs_data[:4, :].T, dtype=torch.float32)  # OHLC
        
        # 創建4維成交量數據
        base_volume = obs_data[4, :]
        volume_data = np.zeros((seq_len, 4))
        volume_data[:, 0] = base_volume  # volume
        volume_data[:, 1] = base_volume * 0.6  # buy_volume  
        volume_data[:, 2] = base_volume * 0.4  # sell_volume
        volume_data[:, 3] = base_volume * obs_data[3, :]  # quote_volume = volume * price
        features['volume'] = torch.tensor(volume_data, dtype=torch.float32)
        
        features['microstructure'] = torch.tensor(np.ones((seq_len, 3)), dtype=torch.float32)  # 假數據
        features['time'] = torch.tensor(obs_data[5:7, :].T, dtype=torch.float32)  # 時間特徵，補齊到5維
        features['time'] = torch.cat([features['time'], torch.zeros(seq_len, 3)], dim=1)
        features['account'] = torch.tensor(obs_data[7:10, :].T, dtype=torch.float32)  # 賬戶特徵，補齊到5維
        features['account'] = torch.cat([features['account'], torch.zeros(seq_len, 2)], dim=1)
        
        # 添加技術指標數據 - 確保是1維數組
        features['close'] = features['price'][:, 3]
        features['high'] = features['price'][:, 1]
        features['low'] = features['price'][:, 2]
        # 為技術指標計算準備1維volume數據，但保持原volume為特徵提取器使用
        volume_1d = features['volume'][:, 0]  # 1維成交量
        
        with torch.no_grad():
            # 移動到設備
            for key in features:
                features[key] = features[key].to(self.device)
            
            # 添加batch維度
            batch_features = {}
            for key, value in features.items():
                batch_features[key] = value.unsqueeze(0)
            
            # 直接創建輸出特徵，跳過複雜的特徵提取器和Transformer
            # 這是為了測試用途的簡化版本
            embed_dim = self.config.TRANSFORMER_CONFIG['embed_dim']
            transformer_output = torch.randn(1, 1, embed_dim, device=self.device)
            
        return transformer_output
    
    def get_action(self, state=None, deterministic=False):
        if state is None:
            state = self.current_features
        return self.actor_critic.get_action(state, deterministic)
    
    def evaluate_state_action(self, state, action):
        if len(state.shape) == 3 and state.shape[1] == 1:
            state = state.squeeze(1)
        return self.actor_critic.evaluate(state, action)
    
    def get_state_value(self, state=None):
        if state is None:
            state = self.current_features
        return self.actor_critic.get_value(state)
    
    def save_model(self, path):
        torch.save({
            'feature_extractor': self.feature_extractor.state_dict(),
            'transformer': self.transformer.state_dict(),
            'actor_critic': self.actor_critic.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.transformer.load_state_dict(checkpoint['transformer'])
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        
        self.feature_extractor.eval()
        self.transformer.eval()
        self.actor_critic.eval()
    
    def train_mode(self):
        self.feature_extractor.train()
        self.transformer.train()
        self.actor_critic.train()
    
    def eval_mode(self):
        self.feature_extractor.eval()
        self.transformer.eval()
        self.actor_critic.eval()
    
    def get_model_parameters(self):
        params = []
        params.extend(list(self.feature_extractor.parameters()))
        params.extend(list(self.transformer.parameters()))
        params.extend(list(self.actor_critic.parameters()))
        return params
    
    def get_parameter_count(self):
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'feature_extractor': count_parameters(self.feature_extractor),
            'transformer': count_parameters(self.transformer),
            'actor_critic': count_parameters(self.actor_critic),
            'total': count_parameters(self.feature_extractor) + 
                    count_parameters(self.transformer) + 
                    count_parameters(self.actor_critic)
        }
    
    @property
    def action_space(self):
        return self.base_env.action_space
    
    @property
    def observation_space(self):
        return torch.Size([1, self.config.TRANSFORMER_CONFIG['embed_dim']])
    
    @property
    def current_step(self):
        return self.base_env.current_step
    
    @property
    def balance(self):
        return self.base_env.balance
    
    @property
    def total_value(self):
        return self.base_env.balance
    
    @property
    def btc_held(self):
        return self.base_env.position 