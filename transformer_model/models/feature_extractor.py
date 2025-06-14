import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import talib
from typing import Dict, Tuple

class FeatureExtractor(nn.Module):
    """
    多維特徵提取器，處理：
    1. 價格特徵 (OHLC)
    2. 成交量特徵 (volume, buy_volume, sell_volume, quote_volume)
    3. 市場微觀結構特徵 (volume_ratio, long_short_ratio, trades)
    4. 技術指標特徵
    5. 時間特徵
    6. 賬戶狀態特徵
    """
    
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()
        self.config = config
        self.feature_dims = config.FEATURE_DIMS
        
        # 特徵歸一化層
        self.price_norm = nn.LayerNorm(self.feature_dims['price'])
        self.volume_norm = nn.LayerNorm(self.feature_dims['volume'])
        self.micro_norm = nn.LayerNorm(self.feature_dims['microstructure'])
        self.tech_norm = nn.LayerNorm(self.feature_dims['technical'])
        self.time_norm = nn.LayerNorm(self.feature_dims['time'])
        self.account_norm = nn.LayerNorm(self.feature_dims['account'])
        
        # 特徵投影層
        embed_dim = config.TRANSFORMER_CONFIG['embed_dim']
        self.price_proj = nn.Linear(self.feature_dims['price'], embed_dim // 6)
        self.volume_proj = nn.Linear(self.feature_dims['volume'], embed_dim // 6)
        self.micro_proj = nn.Linear(self.feature_dims['microstructure'], embed_dim // 6)
        self.tech_proj = nn.Linear(self.feature_dims['technical'], embed_dim // 6)
        self.time_proj = nn.Linear(self.feature_dims['time'], embed_dim // 6)
        self.account_proj = nn.Linear(self.feature_dims['account'], embed_dim // 6)
        
        # 最終融合層
        self.fusion = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(config.TRANSFORMER_CONFIG['dropout'])
        
    def extract_price_features(self, data: torch.Tensor) -> torch.Tensor:
        """提取價格特徵 (OHLC)"""
        # data shape: [batch, seq_len, 4] (OHLC)
        return data
    
    def extract_volume_features(self, data: torch.Tensor) -> torch.Tensor:
        """提取成交量特徵"""
        # data shape: [batch, seq_len, 4] (volume, buy_volume, sell_volume, quote_volume)
        volume_features = torch.zeros_like(data)
        
        # 標準化成交量
        volume_features[:, :, 0] = torch.log1p(data[:, :, 0])  # log(1+volume)
        volume_features[:, :, 1] = torch.log1p(data[:, :, 1])  # log(1+buy_volume)
        volume_features[:, :, 2] = torch.log1p(data[:, :, 2])  # log(1+sell_volume)
        volume_features[:, :, 3] = torch.log1p(data[:, :, 3])  # log(1+quote_volume)
        
        return volume_features
    
    def extract_microstructure_features(self, data: torch.Tensor) -> torch.Tensor:
        """提取市場微觀結構特徵"""
        # data shape: [batch, seq_len, 3] (volume_ratio, long_short_ratio, trades)
        micro_features = torch.zeros_like(data)
        
        # 買賣比例變換
        micro_features[:, :, 0] = torch.tanh(data[:, :, 0])  # volume_ratio
        micro_features[:, :, 1] = torch.tanh(data[:, :, 1])  # long_short_ratio
        micro_features[:, :, 2] = torch.log1p(data[:, :, 2])  # log(1+trades)
        
        return micro_features
    
    def extract_technical_features(self, window_data: Dict) -> torch.Tensor:
        """提取技術指標特徵"""
        batch_size, seq_len = window_data['close'].shape
        device = window_data['close'].device
        
        tech_features = torch.zeros(batch_size, seq_len, self.feature_dims['technical'], device=device)
        
        for i in range(batch_size):
            close = window_data["close"][i].cpu().numpy().astype(np.float64)
            high = window_data["high"][i].cpu().numpy().astype(np.float64)
            low = window_data["low"][i].cpu().numpy().astype(np.float64)
            volume = window_data["volume"][i].cpu().numpy().astype(np.float64)
            
            # RSI
            if len(close) >= 14:
                rsi = talib.RSI(close, timeperiod=14)
                tech_features[i, :, 0] = torch.tensor(np.nan_to_num(rsi, nan=50.0) / 100.0)
            
            # MACD
            if len(close) >= 26:
                _, _, macd_hist = talib.MACD(close)
                tech_features[i, :, 1] = torch.tensor(np.nan_to_num(macd_hist, nan=0.0))
            
            # 布林帶位置
            if len(close) >= 20:
                bb_upper, _, bb_lower = talib.BBANDS(close, timeperiod=20)
                bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
                tech_features[i, :, 2] = torch.tensor(np.nan_to_num(bb_position, nan=0.5))
            
            # ROC
            if len(close) >= 10:
                roc = talib.ROC(close, timeperiod=10)
                tech_features[i, :, 3] = torch.tensor(np.nan_to_num(roc, nan=0.0) / 100.0)
            
            # Volume ROC
            if len(volume) >= 10:
                vol_roc = talib.ROC(volume, timeperiod=10)
                tech_features[i, :, 4] = torch.tensor(np.nan_to_num(vol_roc, nan=0.0) / 100.0)
            
            # ATR
            if len(close) >= 14:
                atr = talib.ATR(high, low, close, timeperiod=14)
                atr_pct = atr / (close + 1e-8)
                tech_features[i, :, 5] = torch.tensor(np.nan_to_num(atr_pct, nan=0.01))
            
            # Williams %R
            if len(close) >= 14:
                williams_r = talib.WILLR(high, low, close, timeperiod=14)
                tech_features[i, :, 6] = torch.tensor(np.nan_to_num(williams_r, nan=-50.0) / -100.0)
            
            # Stochastic %K
            if len(close) >= 14:
                slowk, _ = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
                tech_features[i, :, 7] = torch.tensor(np.nan_to_num(slowk, nan=50.0) / 100.0)
        
        return tech_features
    
    def extract_time_features(self, timestamps: torch.Tensor) -> torch.Tensor:
        """提取時間特徵"""
        # timestamps: [batch, seq_len]
        batch_size, seq_len = timestamps.shape
        device = timestamps.device
        
        time_features = torch.zeros(batch_size, seq_len, self.feature_dims['time'], device=device)
        
        # 假設 timestamps 是 datetime 對象的數值表示
        # 這裡簡化處理，實際使用時需要根據具體的時間戳格式調整
        time_features[:, :, 0] = torch.sin(timestamps * 2 * np.pi / 24)  # 小時週期
        time_features[:, :, 1] = torch.cos(timestamps * 2 * np.pi / 24)
        time_features[:, :, 2] = torch.sin(timestamps * 2 * np.pi / 7)   # 週期週期
        time_features[:, :, 3] = torch.cos(timestamps * 2 * np.pi / 7)
        time_features[:, :, 4] = torch.sin(timestamps * 2 * np.pi / 365) # 年週期
        
        return time_features
    
    def extract_account_features(self, account_data: torch.Tensor) -> torch.Tensor:
        """提取賬戶狀態特徵"""
        # account_data: [batch, seq_len, 5] (step_progress, btc_held, position_value, total_value, balance)
        return account_data
    
    def forward(self, raw_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向傳播
        Args:
            raw_features: 包含各種原始特徵的字典
        Returns:
            融合後的特徵張量 [batch, seq_len, embed_dim]
        """
        # 提取各類特徵
        price_feat = self.extract_price_features(raw_features['price'])
        volume_feat = self.extract_volume_features(raw_features['volume'])
        micro_feat = self.extract_microstructure_features(raw_features['microstructure'])
        tech_feat = self.extract_technical_features(raw_features)
        time_feat = self.extract_time_features(raw_features['time'])
        account_feat = self.extract_account_features(raw_features['account'])
        
        # 歸一化
        price_feat = self.price_norm(price_feat)
        volume_feat = self.volume_norm(volume_feat)
        micro_feat = self.micro_norm(micro_feat)
        tech_feat = self.tech_norm(tech_feat)
        time_feat = self.time_norm(time_feat)
        account_feat = self.account_norm(account_feat)
        
        # 投影到嵌入空間
        price_emb = self.price_proj(price_feat)
        volume_emb = self.volume_proj(volume_feat)
        micro_emb = self.micro_proj(micro_feat)
        tech_emb = self.tech_proj(tech_feat)
        time_emb = self.time_proj(time_feat)
        account_emb = self.account_proj(account_feat)
        
        # 特徵融合
        combined_features = torch.cat([
            price_emb, volume_emb, micro_emb, 
            tech_emb, time_emb, account_emb
        ], dim=-1)
        
        # 最終融合層
        fused_features = self.fusion(combined_features)
        fused_features = self.dropout(fused_features)
        
        return fused_features 