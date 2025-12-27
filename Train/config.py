"""
Train-side Config compatibility shim.

目的：
- 保留舊 import 路徑：`from Train.config import Config`
- 統一設定來源：實際定義在 `Env.config.Config`

注意：
- 請勿在此檔新增/覆寫參數，避免 Env/Train 設定分岔。
"""

from __future__ import annotations

from Env.config import Config


