from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseEpisodeRenderer(ABC):
    """Episode renderer 抽象介面（遵循 SRP/ISP）。

    設計原則：
    - Env 不做繪圖細節，只負責提供資料與呼叫 renderer。
    - Renderer 可獨立擴充（例如 ANSI / rgb_array / mplfinance）。
    """

    @abstractmethod
    def render_episode(self, *, env: Any, info: Optional[dict[str, Any]] = None) -> Optional[str]:
        """Render 當前 episode（通常在 episode 結束時呼叫）。

        Args:
            env: TradingEnvironment（以 Any 取代 import，避免循環依賴）
            info: step() 最後一次回傳的 info（可用於 title/終止原因/統計）

        Returns:
            - 若有存檔：回傳輸出路徑字串
            - 否則回傳 None
        """
        raise NotImplementedError

