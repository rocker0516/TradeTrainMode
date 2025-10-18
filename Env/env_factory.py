"""
可序列化的環境工廠（適用 Windows 多進程）

將環境創建邏輯封裝為頂層可調用對象，
避免本地閉包函數在 Windows 下無法被 pickle 的問題。
"""

from typing import Any, Dict
import pandas as pd

from .trading_env import TradingEnvironment


class EnvFactory:
    """頂層可調用工廠，為每個進程構建一個新的環境實例。"""

    def __init__(self, df: pd.DataFrame, env_kwargs: Dict[str, Any]) -> None:
        self.df = df
        self.env_kwargs = env_kwargs

    def __call__(self) -> TradingEnvironment:
        # 為每個進程提供獨立的數據副本，避免共享狀態
        return TradingEnvironment(df=self.df.copy(), **self.env_kwargs)


