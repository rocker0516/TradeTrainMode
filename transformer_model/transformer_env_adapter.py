import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from typing import Dict, Tuple, Any
from trading_env import TradingEnvironment
from configs.model_config import ModelConfig
from models.feature_extractor import FeatureExtractor
from models.multi_scale_transformer import MultiScaleFeatureExtractor
from models.actor_critic import ActorCritic
from utils.data_preprocessing import TradingDataPreprocessor

class TransformerTradingEnvironment:
    """
    Transformer交易環境適配器
    將原有的TradingEnvironment與新的Transformer模型架構集成
    """
    
    def __init__(self, df, config=None, **env_kwargs):
        # 使用配置
        self.config = config if config is not None else ModelConfig()
        
        # 創建原始交易環境
        self.base_env = TradingEnvironment(df, **env_kwargs)
        
        # 數據預處理器
        self.preprocessor = TradingDataPreprocessor(self.config)
        
        # 預處理數據
        self.processed_df = self.preprocessor.load_and_preprocess_data(None)  # 已經有DataFrame
        self.processed_df = df  # 使用原始數據，因為已經在TradingEnvironment中處理
        
        # 特徵提取器
        self.feature_extractor = FeatureExtractor(self.config)
        
        # Multi-Scale Transformer
        self.transformer = MultiScaleFeatureExtractor(self.config)
        
        # Actor-Critic網絡
        self.actor_critic = ActorCritic(self.config)
        
        # 移動到設備
        self.device = self.config.DEVICE
        self.feature_extractor.to(self.device)
        self.transformer.to(self.device)
        self.actor_critic.to(self.device)
        
        # 狀態緩存
        self.current_features = None
        
    def reset(self, **kwargs):
        """重置環境"""
        # 重置基礎環境
        obs, info = self.base_env.reset(**kwargs)
        
        # 提取並處理特徵
        self.current_features = self._extract_and_process_features()
        
        return self.current_features, info
    
    def step(self, action):
        """執行一步"""
        # 確保動作格式正確
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        # 執行動作
        obs, reward, done, truncated, info = self.base_env.step(action)
        
        # 更新特徵
        if not done:
            self.current_features = self._extract_and_process_features()
        
        return self.current_features, reward, done, truncated, info
    
    def _extract_and_process_features(self) -> torch.Tensor:
        """提取並處理特徵"""
        # 從基礎環境獲取原始觀察
        raw_obs = self.base_env._get_observation()
        
        # 構造特徵字典
        features = self._construct_feature_dict(raw_obs)
        
        # 特徵提取
        with torch.no_grad():
            # 將特徵移動到設备
            for key in features:
                if isinstance(features[key], torch.Tensor):
                    features[key] = features[key].to(self.device)
            
            # 添加batch維度
            batch_features = {}
            for key, value in features.items():
                if isinstance(value, torch.Tensor):
                    if len(value.shape) == 2:  # [seq_len, feature_dim]
                        batch_features[key] = value.unsqueeze(0)  # [1, seq_len, feature_dim]
                    else:
                        batch_features[key] = value.unsqueeze(0)
                else:
                    batch_features[key] = value
            
            # 特徵提取和融合
            fused_features = self.feature_extractor(batch_features)
            
            # Transformer處理
            transformer_output = self.transformer(fused_features)
            
        return transformer_output  # [1, 1, embed_dim]
    
    def _construct_feature_dict_safe(self, raw_obs: np.ndarray) -> Dict[str, torch.Tensor]:
        """構造特徵字典"""
        # raw_obs shape: [n_features, window_size]
        n_features, window_size = raw_obs.shape
        
        # 轉置到 [window_size, n_features]
        obs_transposed = raw_obs.T
        
        # 根據原始環境的特徵順序解析
        features = {}
        
        # 原始數據特徵（來自DataFrame）
        df_features = len(self.base_env.df.columns)
        
        # 1. 價格特徵 (OHLC) - 假設是前4列
        if df_features >= 4:
            features['price'] = torch.tensor(obs_transposed[:, :4], dtype=torch.float32)
        else:
            features['price'] = torch.zeros(window_size, 4, dtype=torch.float32)
        
        # 2. 成交量特徵 - 下4列
        if df_features >= 8:
            features['volume'] = torch.tensor(obs_transposed[:, 4:8], dtype=torch.float32)
        else:
            features['volume'] = torch.zeros(window_size, 4, dtype=torch.float32)
        
        # 3. 市場微觀結構特徵 - 下3列
        if df_features >= 11:
            features['microstructure'] = torch.tensor(obs_transposed[:, 8:11], dtype=torch.float32)
        else:
            features['microstructure'] = torch.zeros(window_size, 3, dtype=torch.float32)
        
        # 4. 賬戶狀態特徵 (5個)
        account_start = df_features
        account_features = obs_transposed[:, account_start:account_start+5]
        features['account'] = torch.tensor(account_features, dtype=torch.float32)
        
        # 5. 時間特徵 (5個)
        time_start = account_start + 5
        time_features = obs_transposed[:, time_start:time_start+5]
        features['time'] = torch.tensor(time_features, dtype=torch.float32)
        
        # 6. 技術指標特徵 (8個)
        tech_start = time_start + 5
        tech_features = obs_transposed[:, tech_start:tech_start+8]
        features['technical'] = torch.tensor(tech_features, dtype=torch.float32)
        
        # 為技術指標計算添加額外的價格數據
        if df_features >= 4:
            features['close'] = features['price'][:, 3]  # close price
            features['high'] = features['price'][:, 1]   # high price
            features['low'] = features['price'][:, 2]    # low price
        else:
            features['close'] = torch.zeros(window_size, dtype=torch.float32)
            features['high'] = torch.zeros(window_size, dtype=torch.float32)
            features['low'] = torch.zeros(window_size, dtype=torch.float32)
        
        return features
    
    def get_action(self, state: torch.Tensor = None, deterministic: bool = False) -> np.ndarray:
        """獲取動作"""
        if state is None:
            state = self.current_features
            
        return self.actor_critic.get_action(state, deterministic)
    
    def evaluate_state_action(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """評估狀態-動作對"""
        # 確保狀態維度正確
        if len(state.shape) == 3 and state.shape[1] == 1:
            state = state.squeeze(1)  # [batch, embed_dim]
        
        return self.actor_critic.evaluate(state, action)
    
    def get_state_value(self, state: torch.Tensor = None) -> float:
        """獲取狀態價值"""
        if state is None:
            state = self.current_features
            
        return self.actor_critic.get_value(state)
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'feature_extractor': self.feature_extractor.state_dict(),
            'transformer': self.transformer.state_dict(),
            'actor_critic': self.actor_critic.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path: str):
        """加載模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.transformer.load_state_dict(checkpoint['transformer'])
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        
        # 設為評估模式
        self.feature_extractor.eval()
        self.transformer.eval()
        self.actor_critic.eval()
    
    def train_mode(self):
        """設為訓練模式"""
        self.feature_extractor.train()
        self.transformer.train()
        self.actor_critic.train()
    
    def eval_mode(self):
        """設為評估模式"""
        self.feature_extractor.eval()
        self.transformer.eval()
        self.actor_critic.eval()
    
    def get_model_parameters(self):
        """獲取所有模型參數"""
        params = []
        params.extend(list(self.feature_extractor.parameters()))
        params.extend(list(self.transformer.parameters()))
        params.extend(list(self.actor_critic.parameters()))
        return params
    
    def get_parameter_count(self) -> Dict[str, int]:
        """獲取參數數量統計"""
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
    
    # 代理基礎環境的屬性
    @property
    def action_space(self):
        return self.base_env.action_space
    
    @property
    def observation_space(self):
        # 返回Transformer輸出的觀察空間
        return torch.Size([1, self.config.TRANSFORMER_CONFIG['embed_dim']])
    
    @property
    def current_step(self):
        return self.base_env.current_step
    
    @property
    def balance(self):
        return self.base_env.balance
    
    @property
    def total_value(self):
        return self.base_env.total_value
    
    @property
    def btc_held(self):
        return self.base_env.btc_held 