"""
训练配置

集中管理所有训练相关的超参数和配置。
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ModelConfig:
    """
    模型配置
    """
    # LSTM 参数
    lstm_hidden_dim: int = 128
    lstm_layers: int = 2
    
    # 网络参数
    hidden_dim: int = 256
    
    # SAC 超参数
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'lstm_hidden_dim': self.lstm_hidden_dim,
            'lstm_layers': self.lstm_layers,
            'hidden_dim': self.hidden_dim,
            'lr': self.lr,
            'gamma': self.gamma,
            'tau': self.tau,
            'alpha': self.alpha,
            'auto_alpha': self.auto_alpha,
        }


@dataclass
class TrainingConfig:
    """
    训练配置
    """
    # 训练参数
    total_episodes: int = 1000
    eval_interval: int = 10
    
    # 经验回放
    buffer_size: int = 100000
    batch_size: int = 256
    warmup_steps: int = 1000
    update_interval: int = 1
    
    # 保存和日志
    save_dir: str = './models'
    log_dir: str = './logs'
    log_interval: int = 1
    
    # 设备
    device: str = 'cuda'  # 'cuda' 或 'cpu'
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_episodes': self.total_episodes,
            'eval_interval': self.eval_interval,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'warmup_steps': self.warmup_steps,
            'update_interval': self.update_interval,
            'save_dir': self.save_dir,
            'log_dir': self.log_dir,
            'log_interval': self.log_interval,
            'device': self.device,
        }


@dataclass
class EnvironmentConfig:
    """
    环境配置
    """
    # 数据路径
    data_path: str = './Data/BTCUSDT_futures_volume_5years_5min.csv'
    
    # 环境参数
    initial_balance: float = 10000.0
    transaction_fee: float = 0.001
    window_size: int = 24 * 60 // 5  # 24小时，5分钟K线
    leverage: float = 10.0
    min_balance: float = 0.0
    min_trade_qty: float = 0.001
    max_stop_loss_percent: float = 1000.0
    max_take_profit_percent: float = 50.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data_path': self.data_path,
            'initial_balance': self.initial_balance,
            'transaction_fee': self.transaction_fee,
            'window_size': self.window_size,
            'leverage': self.leverage,
            'min_balance': self.min_balance,
            'min_trade_qty': self.min_trade_qty,
            'max_stop_loss_percent': self.max_stop_loss_percent,
            'max_take_profit_percent': self.max_take_profit_percent,
        }


@dataclass
class Config:
    """
    完整配置
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    
    def save(self, filepath: str) -> None:
        """
        保存配置到文件
        
        Args:
            filepath: 保存路径
        """
        import json
        
        config_dict = {
            'model': self.model.to_dict(),
            'training': self.training.to_dict(),
            'environment': self.environment.to_dict(),
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"配置已保存至: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """
        从文件加载配置
        
        Args:
            filepath: 配置文件路径
            
        Returns:
            配置对象
        """
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = cls()
        
        # 更新模型配置
        for key, value in config_dict.get('model', {}).items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)
        
        # 更新训练配置
        for key, value in config_dict.get('training', {}).items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)
        
        # 更新环境配置
        for key, value in config_dict.get('environment', {}).items():
            if hasattr(config.environment, key):
                setattr(config.environment, key, value)
        
        print(f"配置已加载自: {filepath}")
        return config


def get_default_config() -> Config:
    """
    获取默认配置
    
    Returns:
        默认配置对象
    """
    return Config()


def get_quick_test_config() -> Config:
    """
    获取快速测试配置（用于调试）
    
    Returns:
        快速测试配置对象
    """
    config = Config()
    config.training.total_episodes = 50
    config.training.eval_interval = 5
    config.training.buffer_size = 10000
    config.training.warmup_steps = 100
    return config

