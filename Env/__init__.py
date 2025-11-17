"""
Env 套件初始化

導出核心類別與工廠函數，便於外部使用。
"""

# 核心環境
from .trading_env import TradingEnvironment

# 交易執行器
from .trade_executor import TradeExecutor

# 獎勵計算器
from .reward import (
    RewardCalculator,
    create_default_calculator,
)

# 帳戶特徵構建器
from .account_features import (
    AccountFeatureBuilder,
    DefaultAccountFeatureBuilder,
    ExtendedAccountFeatureBuilder,
)

# 資訊收集器
from .info_collector import (
    InfoCollector,
    DefaultInfoCollector,
    ExtendedInfoCollector,
)

# 特徵工具（保留舊版相容）
try:
    from .features import build_all_features
except ImportError:
    pass

__all__ = [
    # 核心環境
    'TradingEnvironment',
    'TradeExecutor',
    # 獎勵系統
    'RewardCalculator',
    'create_default_calculator',
    # 帳戶特徵
    'AccountFeatureBuilder',
    'DefaultAccountFeatureBuilder',
    'ExtendedAccountFeatureBuilder',
    # 資訊收集
    'InfoCollector',
    'DefaultInfoCollector',
    'ExtendedInfoCollector',
]
