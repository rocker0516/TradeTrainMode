from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class PositionState:
    size: float = 0.0  # 正數為多單，負數為空單（合約大小，資產單位）
    entry_price: float = 0.0
    take_profit_price: float = 0.0
    stop_loss_price: float = 0.0


class TradeExecutor:
    """執行槓桿合約交易的執行器，負責保證金與手續費計算。

    權益（Equity）= 錢包餘額（wallet_balance）+ 未實現損益（unrealized_pnl）
    手續費：開倉/平倉依名目金額（notional）計收
    初始保證金：abs(持倉數量) * 進場價 / 槓桿
    """

    def __init__(
        self,
        *,
        initial_balance: float,
        fee_rate: float,
        leverage: float,
        min_trade_qty: float,
        max_stop_loss_percent: float,
        max_take_profit_percent: float,
        maintenance_margin_rate: float = 0.005,
    ) -> None:
        self.fee_rate = float(fee_rate)
        self.leverage = float(leverage)
        self.min_trade_qty = float(min_trade_qty)
        self.max_stop_loss_percent = float(max_stop_loss_percent)
        self.max_take_profit_percent = float(max_take_profit_percent)
        self.maintenance_margin_rate = float(maintenance_margin_rate)

        self.wallet_balance: float = float(initial_balance)
        self.position = PositionState()
        self.used_margin: float = 0.0
        self.closed_trades: List[Dict[str, float]] = []

    def reset(self, initial_balance: float) -> None:
        self.wallet_balance = float(initial_balance)
        self.position = PositionState()
        self.used_margin = 0.0
        self.closed_trades = []

    # ---------- 查詢輔助方法 ----------
    def unrealized_pnl(self, current_price: float) -> float:
        if self.position.size == 0.0 or self.position.entry_price == 0.0:
            return 0.0
        return (current_price - self.position.entry_price) * self.position.size

    def equity(self, current_price: float) -> float:
        return self.wallet_balance + self.unrealized_pnl(current_price)

    def available_balance(self) -> float:
        return self.wallet_balance - self.used_margin

    # ---------- 核心執行邏輯 ----------
    def execute(
        self,
        *,
        position_percent: float,
        take_profit_percent: float,
        stop_loss_percent: float,
        current_price: float,
        high: float,
        low: float,
        equity: float,
    ) -> None:
        # 先檢查上一根K線設定的止盈/止損是否被觸發
        stop_triggered = False
        if self.position.size != 0.0:
            if self.position.size > 0:
                if low <= self.position.stop_loss_price:
                    self._close_position(self.position.stop_loss_price)
                    stop_triggered = True
                elif high >= self.position.take_profit_price:
                    self._close_position(self.position.take_profit_price)
                    stop_triggered = True
            else:
                if high >= self.position.stop_loss_price:
                    self._close_position(self.position.stop_loss_price)
                    stop_triggered = True
                elif low <= self.position.take_profit_price:
                    self._close_position(self.position.take_profit_price)
                    stop_triggered = True

        # 若本步因止盈/止損而平倉，當前K線不再重新開倉
        if stop_triggered:
            return

        # 止盈/止損未觸發時，再檢查是否觸發強平（以當根K線極值模擬盤中觸發）
        if self.position.size != 0.0:
            liq_price = self._calc_liquidation_price()
            if liq_price is not None and liq_price > 0.0:
                if (self.position.size > 0 and low <= liq_price) or (self.position.size < 0 and high >= liq_price):
                    self._close_position(liq_price)
                    return

        # 依最新權益與槓桿計算目標倉位數量
        current_equity = self.equity(current_price)
        target_size = (current_equity * position_percent * self.leverage) / current_price if current_price > 0 else 0.0

        if self.position.size == 0.0:
            if abs(target_size) >= self.min_trade_qty:
                self._increase_position(delta_size=target_size, price=current_price)
        else:
            # 若方向反轉，先平舊倉再依新方向開倉
            if self.position.size * target_size < 0:
                self._close_position(price=current_price)
                # 平倉後重算權益與目標數量（已實現手續費/損益）
                current_equity = self.equity(current_price)
                target_size = (current_equity * position_percent * self.leverage) / current_price if current_price > 0 else 0.0
                # 新方向倉位若達最小交易量，才開倉
                if abs(target_size) >= self.min_trade_qty:
                    self._increase_position(delta_size=target_size, price=current_price)
            else:
                delta = target_size - self.position.size
                if abs(delta) >= self.min_trade_qty:
                    if (self.position.size > 0 and delta < 0) or (self.position.size < 0 and delta > 0):
                        # 減倉
                        self._reduce_position(delta_size=delta, price=current_price)
                    else:
                        # 加倉
                        self._increase_position(delta_size=delta, price=current_price)

        # 根據最終倉位方向，從當前價格更新止盈/止損價格
        if self.position.size > 0:
            self.position.take_profit_price = current_price * (1 + max(0.0, min(take_profit_percent, self.max_take_profit_percent)) * 0.01)
            self.position.stop_loss_price = current_price * (1 - max(0.0, min(stop_loss_percent, self.max_stop_loss_percent)) * 0.01)
        elif self.position.size < 0:
            self.position.take_profit_price = current_price * (1 - max(0.0, min(take_profit_percent, self.max_take_profit_percent)) * 0.01)
            self.position.stop_loss_price = current_price * (1 + max(0.0, min(stop_loss_percent, self.max_stop_loss_percent)) * 0.01)
        else:
            self.position.take_profit_price = 0.0
            self.position.stop_loss_price = 0.0

    # ---------- 內部操作 ----------
    def _fee(self, notional: float) -> float:
        return abs(notional) * self.fee_rate

    def _required_margin(self, size: float, price: float) -> float:
        return abs(size) * price / self.leverage

    # 加倉
    def _increase_position(self, *, delta_size: float, price: float) -> None:
        # 確保可用資金足以覆蓋新增保證金與手續費
        desired_size = self.position.size + delta_size
        additional_margin = max(0.0, self._required_margin(desired_size, price) - self.used_margin)
        trade_notional = abs(delta_size) * price
        fee = self._fee(trade_notional)

        if self.available_balance() < additional_margin + fee:
            # 依可用資金上限縮小加倉數量
            cap = self.available_balance() - fee
            if cap <= 0:
                return
            max_notional_add = cap * self.leverage
            max_size_add = max_notional_add / price if price > 0 else 0.0
            delta_size = (max_size_add if delta_size > 0 else -max_size_add)
            trade_notional = abs(delta_size) * price
            additional_margin = max(0.0, self._required_margin(self.position.size + delta_size, price) - self.used_margin)
            fee = self._fee(trade_notional)
            if abs(delta_size) < self.min_trade_qty:
                return

        # 執行加倉
        if self.position.size == 0.0:
            self.position.entry_price = price
        else:
            # 同向加倉：以加權平均更新進場價格
            old_notional = abs(self.position.size) * self.position.entry_price
            new_notional = abs(delta_size) * price
            total_size = abs(self.position.size + delta_size)
            if total_size > 0:
                self.position.entry_price = (old_notional + new_notional) / total_size

        self.position.size += delta_size
        self.used_margin += additional_margin
        self.wallet_balance -= fee

    # 減倉
    def _reduce_position(self, *, delta_size: float, price: float) -> None:
        # delta_size 與當前持倉方向相反；以下計算實際平倉數量
        close_size = -delta_size  # 平倉數量（與現有持倉同號）
        if self.position.size > 0:
            close_size = min(abs(close_size), abs(self.position.size))
            realized_pnl = (price - self.position.entry_price) * close_size
            new_size = self.position.size - close_size
        else:
            close_size = min(abs(close_size), abs(self.position.size))
            realized_pnl = (self.position.entry_price - price) * close_size
            new_size = - (abs(self.position.size) - close_size)

        fee = self._fee(close_size * price)
        margin_release = self._required_margin(close_size, self.position.entry_price)

        self.wallet_balance += realized_pnl - fee
        self.used_margin = max(0.0, self.used_margin - margin_release)
        self.position.size = new_size
        if self.position.size == 0.0:
            self.position.entry_price = 0.0

    # 平倉
    def _close_position(self, price: float) -> None:
        if self.position.size == 0.0:
            return
        size_to_close = abs(self.position.size)
        if self.position.size > 0:
            realized_pnl = (price - self.position.entry_price) * size_to_close
        else:
            realized_pnl = (self.position.entry_price - price) * size_to_close

        fee = self._fee(size_to_close * price)
        self.wallet_balance += realized_pnl - fee
        self.used_margin = 0.0
        self.position = PositionState()

    # ---------- 風險控制與強平 ----------
    def _calc_liquidation_price(self) -> float | None:
        # 根據：equity(p) = wallet + (p - entry) * size
        # 維持保證金：|size| * p * mmr
        # 觸發強平條件：equity(p) <= maintenance_margin(p)
        if self.position.size == 0.0 or self.position.entry_price == 0.0:
            return None
        m = self.maintenance_margin_rate
        s = self.position.size
        e = self.position.entry_price
        w = self.wallet_balance
        if s > 0:
            denom = s * (1.0 - m)
            if denom <= 0:
                return None
            price = (s * e - w) / denom
            return max(0.0, price)
        else:
            u = abs(s)
            denom = u * (1.0 + m)
            if denom <= 0:
                return None
            price = (w + u * e) / denom
            return max(0.0, price)

