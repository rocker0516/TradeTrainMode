import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """位置編碼"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """多頭注意力機制"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        return self.w_o(attention_output)
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                   V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V)

class TransformerBlock(nn.Module):
    """Transformer塊"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class MultiScaleTransformer(nn.Module):
    """
    多尺度Transformer模型
    處理不同時間窗口的特徵：短期(32步)、中期(96步)、長期(288步)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 配置參數
        self.embed_dim = config.TRANSFORMER_CONFIG['embed_dim']
        self.num_heads = config.TRANSFORMER_CONFIG['num_heads']
        self.num_layers = config.TRANSFORMER_CONFIG['num_layers']
        self.dropout = config.TRANSFORMER_CONFIG['dropout']
        self.ff_dim = config.TRANSFORMER_CONFIG['ff_dim']
        self.max_seq_len = config.TRANSFORMER_CONFIG['max_seq_len']
        
        # 多尺度配置
        self.short_term = config.MULTI_SCALE_CONFIG['short_term']
        self.medium_term = config.MULTI_SCALE_CONFIG['medium_term']
        self.long_term = config.MULTI_SCALE_CONFIG['long_term']
        self.scale_weights = config.MULTI_SCALE_CONFIG['scale_weights']
        
        # 位置編碼
        self.pos_encoding = PositionalEncoding(self.embed_dim, self.max_seq_len)
        
        # 多尺度Transformer層
        self.short_term_layers = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.dropout)
            for _ in range(self.num_layers // 3)
        ])
        
        self.medium_term_layers = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.dropout)
            for _ in range(self.num_layers // 3)
        ])
        
        self.long_term_layers = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.dropout)
            for _ in range(self.num_layers // 3)
        ])
        
        # 尺度融合層
        self.scale_fusion = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        )
        
        # 全局注意力層
        self.global_attention = MultiHeadAttention(self.embed_dim, self.num_heads, self.dropout)
        self.global_norm = nn.LayerNorm(self.embed_dim)
        
        # 輸出投影層
        self.output_projection = nn.Linear(self.embed_dim, self.embed_dim)
        
    def create_padding_mask(self, seq_len: int, target_len: int) -> torch.Tensor:
        """創建填充遮罩"""
        if seq_len >= target_len:
            return torch.ones(1, 1, target_len, target_len)
        else:
            mask = torch.zeros(1, 1, target_len, target_len)
            mask[:, :, :seq_len, :seq_len] = 1
            return mask
    
    def extract_multi_scale_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """提取多尺度特徵"""
        batch_size, seq_len, embed_dim = x.shape
        
        # 短期特徵 (最近32步)
        if seq_len >= self.short_term:
            short_term_x = x[:, -self.short_term:, :]
        else:
            # 如果序列長度不足，進行填充
            short_term_x = F.pad(x, (0, 0, self.short_term - seq_len, 0), value=0)
        
        # 中期特徵 (最近96步)
        if seq_len >= self.medium_term:
            medium_term_x = x[:, -self.medium_term:, :]
        else:
            medium_term_x = F.pad(x, (0, 0, self.medium_term - seq_len, 0), value=0)
        
        # 長期特徵 (全部288步)
        if seq_len >= self.long_term:
            long_term_x = x[:, -self.long_term:, :]
        else:
            long_term_x = F.pad(x, (0, 0, self.long_term - seq_len, 0), value=0)
        
        return short_term_x, medium_term_x, long_term_x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        Args:
            x: 輸入特徵 [batch_size, seq_len, embed_dim]
        Returns:
            多尺度融合後的特徵 [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # 添加位置編碼
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # 提取多尺度特徵
        short_x, medium_x, long_x = self.extract_multi_scale_features(x)
        
        # 短期特徵處理
        for layer in self.short_term_layers:
            short_x = layer(short_x)
        short_features = short_x[:, -1:, :]  # 取最後一個時間步
        
        # 中期特徵處理
        for layer in self.medium_term_layers:
            medium_x = layer(medium_x)
        medium_features = medium_x[:, -1:, :]  # 取最後一個時間步
        
        # 長期特徵處理
        for layer in self.long_term_layers:
            long_x = layer(long_x)
        long_features = long_x[:, -1:, :]  # 取最後一個時間步
        
        # 多尺度特徵融合
        multi_scale_features = torch.cat([
            short_features, medium_features, long_features
        ], dim=-1)
        
        # 尺度融合
        fused_features = self.scale_fusion(multi_scale_features)
        
        # 全局注意力
        global_context = self.global_attention(fused_features, fused_features, fused_features)
        fused_features = self.global_norm(fused_features + global_context)
        
        # 輸出投影
        output = self.output_projection(fused_features)
        
        return output  # [batch_size, 1, embed_dim]
    
    def get_attention_weights(self, x: torch.Tensor) -> dict:
        """獲取注意力權重，用於可視化"""
        # 這個方法可以用於分析模型關注的重點
        # 實現細節可以根據需要添加
        return {}

class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特徵提取器，結合CNN和Transformer"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1D CNN for local pattern extraction
        self.local_cnn = nn.Sequential(
            nn.Conv1d(config.get_total_feature_dim(), 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, config.TRANSFORMER_CONFIG['embed_dim'], kernel_size=7, padding=3),
            nn.ReLU(),
        )
        
        # Multi-scale Transformer
        self.transformer = MultiScaleTransformer(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, feature_dim]
        Returns:
            [batch_size, 1, embed_dim]
        """
        # CNN for local pattern extraction
        x_cnn = x.transpose(1, 2)  # [batch, feature_dim, seq_len]
        x_cnn = self.local_cnn(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)  # [batch, seq_len, embed_dim]
        
        # Transformer for long-range dependencies
        output = self.transformer(x_cnn)
        
        return output 