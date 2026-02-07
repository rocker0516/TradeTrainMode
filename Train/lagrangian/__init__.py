"""
Lagrangian 相關模組。

提供：
- 控制器：SharedLagrangianController, MultiSharedLagrangianController
- Wrapper：LagrangianRewardWrapper
- Callback：LagrangianCallback
- 統計函數：compute_trade_stats, compute_end_result_stats
"""

from Train.lagrangian.controllers import (
    SharedLagrangianController,
    MultiSharedLagrangianController,
    LagrangianChannelConfig,
)
from Train.lagrangian.wrapper import LagrangianRewardWrapper
from Train.lagrangian.callback import LagrangianCallback
from Train.lagrangian.stats import (
    compute_trade_stats,
    compute_end_result_stats,
)

__all__ = [
    "SharedLagrangianController",
    "MultiSharedLagrangianController",
    "LagrangianChannelConfig",
    "LagrangianRewardWrapper",
    "LagrangianCallback",
    "compute_trade_stats",
    "compute_end_result_stats",
]

