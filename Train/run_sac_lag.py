from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
from typing import Any, Dict

# ---------------------------
# 路徑修正（Windows 直跑腳本常見問題）
# ---------------------------
# 你使用：
#   & .conda/python.exe Train/run_sac_lag.py
# 時，Python 的 sys.path 不一定包含「專案根目錄」，導致 `import Env` 失敗。
# 這裡強制把專案根（Train/ 的上一層）加入 sys.path，確保可直接執行腳本。
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Train.train_config import TrainConfig
from Train.config_builder import TrainingConfigBuilder
from Train.env_builder import EnvironmentBuilder
from Train.model_builder import ModelBuilder
from Train.callback_builder import CallbackBuilder
from Train.trainer import SACLagrangianTrainer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# 注意：EnvFactory 已移至 EnvironmentBuilder，这里保留空注释以保持兼容性


def parse_args() -> argparse.Namespace:
    """解析命令行参数。
    
    Returns:
        解析后的命令行参数
    """
    parser = argparse.ArgumentParser(description="SAC Lagrangian Training")
    # 預設值全部由 TrainConfig 提供；CLI 參數只做覆寫
    parser.add_argument("--symbol", type=str, default=TrainConfig.SYMBOL, help="交易標的符號")
    parser.add_argument("--total_timesteps", type=int, default=TrainConfig.TOTAL_TIMESTEPS, help="總訓練步數")
    parser.add_argument("--n_envs", type=int, default=TrainConfig.N_ENVS, help="並行環境數量")
    parser.add_argument("--cost_limit", type=float, default=TrainConfig.COST_LIMIT, help="平均成本限制（單一 Lambda 模式，已棄用）")
    parser.add_argument("--risk_cost_limit", type=float, default=TrainConfig.RISK_COST_LIMIT, help="風險成本限制（死亡風險通道）")
    parser.add_argument("--fric_cost_limit", type=float, default=TrainConfig.FRIC_COST_LIMIT, help="摩擦成本限制（手續費通道）")
    parser.add_argument("--device", type=str, default=TrainConfig.DEVICE, help="計算設備：'cuda' / 'cpu' / 'auto'")
    parser.add_argument("--log_every_episodes", type=int, default=TrainConfig.LOG_EVERY_EPISODES, help="每 N 回合輸出交易統計")
    parser.add_argument("--update_lambda_every_steps", type=int, default=TrainConfig.UPDATE_LAMBDA_EVERY_STEPS, help="每 N steps 更新一次 lambda")
    # 進度條：預設開啟（避免你忘記加參數而覺得「沒有進度」）
    parser.add_argument("--no_progress_bar", action="store_true", help="關閉 SB3 進度條（預設會顯示）")
    parser.add_argument("--verbose", type=int, default=TrainConfig.SB3_VERBOSE_DEFAULT, help="SB3 verbose 等級（預設：開進度條時=0，否則=1）")
    return parser.parse_args()


def main() -> None:
    """
    SAC Lagrangian 训练主入口。
    
    功能：
    1. 解析命令行参数
    2. 构建训练配置
    3. 创建构建器（环境、模型、回调）
    4. 创建训练器
    5. 执行训练
    
    Raises:
        ValueError: 配置参数无效
        RuntimeError: 环境或模型创建失败
        KeyboardInterrupt: 用户中断训练
    """
    try:
        # 1. 解析命令行参数
        args = parse_args()
        
        # 2. 构建训练配置
        config = TrainingConfigBuilder.from_cli_args(args)
        
        # 3. 创建构建器
        env_builder = EnvironmentBuilder(
            config=config.env_config,
            max_position_pct=config.max_position_pct,
            action_repeat=config.action_repeat,
            reward_scale=config.reward_scale,
        )
        model_builder = ModelBuilder(config.model_config)
        callback_builder = CallbackBuilder(config)
        
        # 4. 创建训练器
        trainer = SACLagrangianTrainer(
            config=config,
            env_builder=env_builder,
            model_builder=model_builder,
            callback_builder=callback_builder,
        )
        
        # 5. 执行训练
        trainer.train()
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        raise
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Training failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise RuntimeError(f"Training failed: {e}") from e


if __name__ == "__main__":
    # Windows 下 multiprocessing 必須在 if __name__ == "__main__": 下執行
    multiprocessing.freeze_support()
    main()
