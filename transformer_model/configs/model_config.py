import torch

class ModelConfig:
    # 數據相關配置
    WINDOW_SIZE = 288  # 24小時 * 60分鐘 / 5分鐘
    FEATURE_DIMS = {
        'price': 4,  # OHLC
        'volume': 4,  # volume, buy_volume, sell_volume, quote_volume
        'microstructure': 3,  # volume_ratio, long_short_ratio, trades
        'technical': 8,  # RSI, MACD, BB, ROC, ATR, Williams%R, Stoch等
        'time': 5,  # 時間特徵
        'account': 5,  # 賬戶狀態
    }
    
    # Transformer架構配置
    TRANSFORMER_CONFIG = {
        'embed_dim': 256,
        'num_heads': 8,
        'num_layers': 6,
        'dropout': 0.1,
        'ff_dim': 1024,
        'max_seq_len': WINDOW_SIZE,
    }
    
    # Multi-Scale配置
    MULTI_SCALE_CONFIG = {
        'short_term': 32,   # 最近32個時間步
        'medium_term': 96,  # 最近96個時間步  
        'long_term': 288,   # 全部288個時間步
        'scale_weights': [0.3, 0.3, 0.4],  # 不同尺度的權重
    }
    
    # Actor-Critic配置
    ACTOR_CRITIC_CONFIG = {
        'hidden_dim': 512,
        'num_layers': 3,
        'dropout': 0.2,
        'activation': 'relu',
        'action_dim': 3,  # [方向, 止盈, 止損]
        'action_bounds': [(-1.0, 1.0), (0.2, 10.0), (0.1, 0.3)],
    }
    
    # 訓練配置
    TRAINING_CONFIG = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        'max_grad_norm': 0.5,
        'batch_size': 64,
        'epochs_per_update': 10,
        'buffer_size': 10000,
    }
    
    # 設備配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def get_total_feature_dim(cls):
        return sum(cls.FEATURE_DIMS.values()) 