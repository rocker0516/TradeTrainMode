"""
評估相關模組。

提供：
- 配置：EvalConfig, EvalConstraints, TrainEvalStartGateConfig
- 數據模型：EpisodeEval, EvalResults
- Callback：ConstraintEvalCallback
- Gate：TrainEpisodeWindowGate
- 工具函數：格式化、轉換等
"""

from Train.eval.callback import ConstraintEvalCallback
from Train.eval.config import (
    EvalConfig,
    EvalConstraints,
    TrainEvalStartGateConfig,
)
from Train.eval.gate import TrainEpisodeWindowGate
from Train.eval.models import EpisodeEval, EvalResults
from Train.eval.utils import (
    fmt_float,
    fmt_int,
    fmt_pct,
    format_episode_line,
    mean_or_none,
    to_float_or_none,
    to_int_or_none,
)

__all__ = [
    "ConstraintEvalCallback",
    "EvalConfig",
    "EvalConstraints",
    "TrainEvalStartGateConfig",
    "TrainEpisodeWindowGate",
    "EpisodeEval",
    "EvalResults",
    "fmt_float",
    "fmt_int",
    "fmt_pct",
    "format_episode_line",
    "mean_or_none",
    "to_float_or_none",
    "to_int_or_none",
]

