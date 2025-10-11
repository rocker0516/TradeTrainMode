"""
训练器模块

包含不同的训练器实现。
"""

from .base_trainer import BaseTrainer
from .sac_trainer import SACTrainer

__all__ = ['BaseTrainer', 'SACTrainer']

