"""Account series updating utilities."""

from __future__ import annotations

from typing import Any, Dict


class AccountSeriesUpdater:
    """Maintains滾動帳戶狀態時間序列。"""

    def update(
        self,
        *,
        account_series: Dict[str, Any],
        current_step: int,
        initial_balance: float,
        current_price: float,
        executor: Any,
    ) -> None:
        """Write最新帳戶狀態至序列。"""

        if not (0 <= current_step < len(account_series['position'])):
            return

        if initial_balance <= 0 or current_price <= 0:
            pos_norm = 0.0
            pos_value_norm = 0.0
            equity_norm = 0.0
            wallet_norm = 0.0
        else:
            pos_norm = float(executor.position.size / (initial_balance / current_price))
            pos_value_norm = float((executor.position.size * current_price) / initial_balance)
            equity_norm = float(executor.equity(current_price) / initial_balance)
            wallet_norm = float(executor.wallet_balance / initial_balance)

        account_series['position'][current_step] = pos_norm
        account_series['position_value'][current_step] = pos_value_norm
        account_series['equity'][current_step] = equity_norm
        account_series['wallet'][current_step] = wallet_norm

