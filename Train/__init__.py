"""
SAC (Soft Actor-Critic) 訓練系統

此模組提供完整的強化學習訓練系統，專為交易環境設計。
遵循 SOLID 原則，提供模組化、可擴展的架構。

主要組件:
    - config: 配置管理系統
    - models: 強化學習模型 (純 SAC)
    - trainers: 訓練器實作
    - utils: 工具函數 (經驗回放、日誌記錄)
"""

from .config import Config

__all__ = ['Config']
__version__ = '1.0.0'

