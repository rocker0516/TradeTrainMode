"""
配置管理系統

提供集中式的配置管理，遵循單一職責原則 (SRP)。
所有訓練參數、模型參數、環境參數都在此定義。
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Literal
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """
    模型配置（純 SAC）
    
    Args:
        hidden_dim: 隱藏層維度
        num_layers: 網絡層數
        lr: 學習率
        alpha: 熵係數（探索程度）
        gamma: 折扣因子
        tau: 軟更新係數
        auto_entropy_tuning: 是否自動調整熵係數
    """
    hidden_dim: int = 256
    num_layers: int = 2
    lr: float = 3e-4
    alpha: float = 0.2
    gamma: float = 0.99
    tau: float = 0.005
    auto_entropy_tuning: bool = True


@dataclass
class TrainingConfig:
    """
    訓練配置
    
    Args:
        total_episodes: 總訓練回合數
        batch_size: 批次大小
        buffer_size: 經驗回放緩衝區大小
        warmup_steps: 預熱步數（隨機探索）
        update_interval: 更新間隔（每 N 步更新一次）
        eval_interval: 評估間隔（每 N 回合評估一次）
        eval_episodes: 評估回合數
        save_interval: 保存間隔（每 N 回合保存一次）
        device: 訓練設備 ('cuda' 或 'cpu')
    """
    total_episodes: int = 1000
    batch_size: int = 256
    buffer_size: int = 100000
    warmup_steps: int = 1000
    update_interval: int = 1
    eval_interval: int = 50
    eval_episodes: int = 5
    save_interval: int = 50
    device: Literal['cuda', 'cpu'] = 'cuda'


@dataclass
class EnvironmentConfig:
    """
    交易環境配置
    
    Args:
        initial_balance: 初始資金
        transaction_fee: 交易手續費率
        window_size: 觀察窗口大小（K線數量）
        leverage: 槓桿倍數
        min_balance: 最小資金（低於此值則結束）
        min_trade_qty: 最低交易數量
        margin_mode: 保證金模式 ('cross' 或 'isolated')
        reward_mode: 獎勵模式 ('delta_equity', 'pct', 'log')
        reward_scale: 獎勵縮放係數
    """
    initial_balance: float = 10000.0
    transaction_fee: float = 0.001
    window_size: int = 288  # 24小時 * 60分鐘 / 5分鐘
    leverage: float = 10.0
    min_balance: float = 0.0
    min_trade_qty: float = 0.001
    margin_mode: Literal['cross', 'isolated'] = 'isolated'
    reward_mode: Literal['delta_equity', 'pct', 'log'] = 'delta_equity'
    reward_scale: float = 1.0


@dataclass
class PathConfig:
    """
    路徑配置
    
    Args:
        data_path: 訓練數據路徑
        model_dir: 模型保存目錄
        log_dir: 日誌保存目錄
    """
    data_path: str = './Data/BTCUSDT_futures_volume_5years_5min.csv'
    model_dir: str = './models'
    log_dir: str = './logs'


@dataclass
class Config:
    """
    完整配置類別
    
    整合所有配置項目，提供保存和加載功能。
    遵循開放封閉原則 (OCP)，可通過繼承擴展。
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        將配置轉換為字典
        
        Returns:
            配置字典
        """
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'environment': asdict(self.environment),
            'paths': asdict(self.paths),
        }
    
    def save(self, path: str) -> None:
        """
        保存配置到 JSON 檔案
        
        Args:
            path: 保存路徑
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> Config:
        """
        從 JSON 檔案加載配置
        
        Args:
            path: 配置檔案路徑
            
        Returns:
            Config 實例
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            model=ModelConfig(**data.get('model', {})),
            training=TrainingConfig(**data.get('training', {})),
            environment=EnvironmentConfig(**data.get('environment', {})),
            paths=PathConfig(**data.get('paths', {})),
        )
    
    def update_from_args(self, **kwargs: Any) -> None:
        """
        從命令行參數更新配置
        
        Args:
            **kwargs: 參數字典
        """
        # 更新訓練配置
        if 'episodes' in kwargs and kwargs['episodes'] is not None:
            self.training.total_episodes = int(kwargs['episodes'])
        if 'device' in kwargs and kwargs['device'] is not None:
            self.training.device = kwargs['device']
        if 'batch_size' in kwargs and kwargs['batch_size'] is not None:
            self.training.batch_size = int(kwargs['batch_size'])
        
        # 更新路徑配置
        if 'data' in kwargs and kwargs['data'] is not None:
            self.paths.data_path = kwargs['data']
        if 'model_dir' in kwargs and kwargs['model_dir'] is not None:
            self.paths.model_dir = kwargs['model_dir']
        
        # 更新環境配置
        if 'leverage' in kwargs and kwargs['leverage'] is not None:
            self.environment.leverage = float(kwargs['leverage'])
        if 'initial_balance' in kwargs and kwargs['initial_balance'] is not None:
            self.environment.initial_balance = float(kwargs['initial_balance'])

