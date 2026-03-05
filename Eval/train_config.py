from __future__ import annotations

"""
訓練端設定檔（TrainConfig）

說明：
- 原本的 `Train/config.py` 是「相容性 shim」：只做 `from Env.config import Config`，不應塞訓練參數。
- 因此訓練腳本 `Train/run_sac_lag.py` 的參數預設值，集中放在此檔，避免 Env/Train 設定混雜。

使用方式：
    from Eval.train_config import TrainConfig
    default_n_envs = TrainConfig.N_ENVS
"""


class TrainConfig:
    """
    SAC-Lagrangian 訓練參數預設值（可由 CLI 覆寫）。
    
    所有參數都可在 `Train/run_sac_lag.py` 中透過命令行參數覆寫。
    """

    # ==================== 基本訓練參數 ====================
    SYMBOL: str = "BTCUSDT"
    """交易標的符號"""
    
    FEATURE_SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT")
    """
    特徵 symbols（固定順序、固定維度）
    - 用於 5m 跨市場摘要 +（後續可擴充）多幣 1d regime
    - 注意：Gym observation_space 必須固定 shape，因此這裡用「固定清單」，
      而不是隨 Data 目錄動態增減。
    """
    

