"""TradeTrainMode DataUpdaterService package.

此 package 提供：
- 增量更新 Data/ CSV 的 updater（5m / 1d）
- Windows Service（pywin32）常駐排程更新

設計原則：SRP + DIP，Updater 依賴抽象 client（Protocol），避免高層模組直接耦合第三方 SDK。
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"


