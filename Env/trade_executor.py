from __future__ import annotations

from dataclasses import dataclass
from typing import List, TypedDict, Literal, Optional


@dataclass
class PositionState:
    size: float = 0.0  # 正數為多單，負數為空單（合約大小，資產單位）
    entry_price: float = 0.0
    take_profit_price: float = 0.0
    stop_loss_price: float = 0.0
    opened_step: int = -1  # 開倉步數（持倉管理用）


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
        rebalance_qty_multiplier: float = 3.0,
    ) -> None:
        self.fee_rate = float(fee_rate)
        self.leverage = float(leverage)
        self.min_trade_qty = float(min_trade_qty)
        self.max_stop_loss_percent = float(max_stop_loss_percent)
        self.max_take_profit_percent = float(max_take_profit_percent)
        self.maintenance_margin_rate = float(maintenance_margin_rate)
        self.rebalance_qty_multiplier = float(rebalance_qty_multiplier)
        self.min_notional_usd: float = 0.0
        self.rebalance_cooldown_steps: int = 0
        self._last_rebalance_step: int = -10**9

        self.wallet_balance: float = float(initial_balance)
        self.position = PositionState()
        self.used_margin: float = 0.0

        # 平倉交易紀錄（每次完全平倉時寫入一筆）
        class ClosedTrade(TypedDict):
            side: Literal['long', 'short']#方向
            entry_price: float#進場價格
            exit_price: float#平倉價格
            size: float#持倉數量
            entry_step: int#進場步數
            exit_step: int#平倉步數
            holding_steps: int#持有步數
            pnl_after_fees: float#平倉後損益
            total_fee: float#總手續費
            notional_open: float#進場名目
            notional_close: float#平倉名目
            reason: Literal['take_profit', 'stop_loss', 'reverse', 'rebalance', 'liquidation', 'manual', 'end_of_data']#平倉原因

        self.closed_trades: List[ClosedTrade] = []  # 平倉交易

        # 進場即建立的持倉紀錄（持倉期間持續更新，平倉時計算並搬至 closed_trades）
        class OpenTrade(TypedDict):
            side: Literal['long', 'short']#方向
            entry_price: float#進場價格
            size: float#持倉數量（當前聚合倉位大小）
            entry_step: int#進場步數
            notional_open: float#進場名目
            fee_open: float#進場手續費（可累加）
            last_update_step: int#最後更新步數

        self.open_trade: Optional[OpenTrade] = None
        # 強平狀態標記（發生強平即視為爆倉）
        self.was_liquidated: bool = False
        self.last_liquidation_price: float | None = None
        # 交易次數統計（加倉、減倉、平倉視為一筆操作）
        self.trade_count: int = 0
        # 手續費與名目、成交率統計
        self.total_fee_spent: float = 0.0#總手續費
        self.total_notional_traded: float = 0.0#總名目
        self.orders_attempted: int = 0#嘗試下單次數
        self.orders_executed: int = 0#成功下單次數


    def reset(self, initial_balance: float) -> None:
        self.wallet_balance = float(initial_balance)
        self.position = PositionState()
        self.used_margin = 0.0
        self.closed_trades = []
        self.open_trade = None
        self.was_liquidated = False
        self.last_liquidation_price = None
        self.trade_count = 0
        self.total_fee_spent = 0.0
        self.total_notional_traded = 0.0
        self.orders_attempted = 0
        self.orders_executed = 0
        self._last_rebalance_step = -10**9#最後重平衡步數

    # ---------- 查詢輔助方法 ----------
    '''
    未實現損益 = (當前價格 - 進場價格) * 持倉數量
    '''
    def unrealized_pnl(self, current_price: float) -> float:
        if self.position.size == 0.0 or self.position.entry_price == 0.0:
            return 0.0
        return (current_price - self.position.entry_price) * self.position.size

    '''
    總資產 = 資金 + 未實現損益
    '''
    def equity(self, current_price: float) -> float:
        return self.wallet_balance + self.unrealized_pnl(current_price)

    '''
    可用資金 = 資金 - 已使用保證金
    '''
    def available_balance(self) -> float:
        return self.wallet_balance - self.used_margin

    # ---------- 核心執行邏輯 ----------
    '''
    執行交易
    1. 先檢查上一根K線設定的止盈/止損是否被觸發
    2. 若本步因止盈/止損而平倉，當前K線不再重新開倉
    3. 止盈/止損未觸發時，再檢查是否觸發強平（以當根K線極值模擬盤中觸發）
    4. 依最新權益與槓桿計算目標倉位數量
    5. 冷卻控制（使用外部傳入的 step_index）
    6. 名目檢查
    7. 若持倉數量為0，則檢查是否符合加倉條件
    8. 若持倉數量不為0，則檢查是否符合減倉條件
    9. 根據最終倉位方向，從當前價格更新止盈/止損價格
    '''
    def execute(
        self,
        *,  # 位置百分比（已歸一化相對 initial_balance）
        position_percent: float,  # 位置百分比
        take_profit_percent: float,  # 止盈百分比
        stop_loss_percent: float,  # 止損百分比
        current_price: float,  # 當前價格
        high: float,  # 當前高價
        low: float,  # 當前低價
        equity: float,  # 總資產
        step_index: int = 0,  # 步驟索引
    ) -> None:
        # 先檢查上一根K線設定的止盈/止損是否被觸發
        stop_triggered = False
        if self.position.size != 0.0:
            # 若持倉數量大於0，則檢查是否符合止損條件
            if self.position.size > 0:
                if low <= self.position.stop_loss_price:
                    self.orders_attempted += 1
                    self._close_position(self.position.stop_loss_price, step_index)
                    stop_triggered = True
                elif high >= self.position.take_profit_price:
                    self.orders_attempted += 1
                    self._close_position(self.position.take_profit_price, step_index)
                    stop_triggered = True
            else:#若持倉數量小於0，則檢查是否符合止損條件
                if high >= self.position.stop_loss_price:
                    self.orders_attempted += 1
                    self._close_position(self.position.stop_loss_price, step_index)
                    stop_triggered = True
                elif low <= self.position.take_profit_price:
                    self.orders_attempted += 1
                    self._close_position(self.position.take_profit_price, step_index)
                    stop_triggered = True

        # 若本步因止盈/止損而平倉，當前K線不再重新開倉
        if stop_triggered:
            return

        # 止盈/止損未觸發時，再檢查是否觸發強平（以當根K線極值模擬盤中觸發）
        if self.position.size != 0.0:
            liq_price = self._calc_liquidation_price()
            if liq_price is not None and liq_price > 0.0:
                if (self.position.size > 0 and low <= liq_price) or (self.position.size < 0 and high >= liq_price):
                    self.orders_attempted += 1
                    # 觸發強平：先以強平價平倉，再將錢包視為清零（全倉爆倉語義）
                    self._close_position(liq_price, step_index)
                    self.was_liquidated = True
                    self.last_liquidation_price = float(liq_price)
                    # 全倉爆倉：清零錢包與保證金
                    self.wallet_balance = 0.0
                    self.used_margin = 0.0
                    return

        # 依最新權益與槓桿計算目標倉位數量
        current_equity = self.equity(current_price)
        target_size = (current_equity * position_percent * self.leverage) / current_price if current_price > 0 else 0.0

        # 名目檢查 <- 確保每次交易至少有最小名目
        def notional_ok(delta_size: float) -> bool:
            return (abs(delta_size) * current_price) >= self.min_notional_usd

        # 若持倉數量為0，則檢查是否符合加倉條件
        if self.position.size == 0.0:
            if abs(target_size) >= self.min_trade_qty and notional_ok(target_size):
                self.orders_attempted += 1
                self._increase_position(delta_size=target_size, price=current_price, step_index=step_index)
                self._last_rebalance_step = step_index
        else:#若持倉數量不為0，則檢查是否符合減倉條件
            # 若方向反轉：當步僅平倉，不立即開新方向（避免單步雙交易與抖動）
            if self.position.size * target_size < 0:
                self.orders_attempted += 1
                self._close_position(price=current_price, step_index=step_index)
                return
            else:
                delta = target_size - self.position.size
                threshold = self.min_trade_qty * self.rebalance_qty_multiplier
                if abs(delta) >= threshold and notional_ok(delta):
                    self.orders_attempted += 1
                    if (self.position.size > 0 and delta < 0) or (self.position.size < 0 and delta > 0):
                        # 減倉
                        self._reduce_position(delta_size=delta, price=current_price, step_index=step_index)
                    else:
                        # 加倉
                        self._increase_position(delta_size=delta, price=current_price, step_index=step_index)
                    self._last_rebalance_step = step_index

        # 根據最終倉位方向，從當前價格更新止盈/止損價格
        if self.position.size > 0:
            self.position.take_profit_price = current_price * (1 + max(0.0, min(take_profit_percent, self.max_take_profit_percent)) * 0.01)
            self.position.stop_loss_price = current_price * (1 - max(0.0, min(stop_loss_percent, self.max_stop_loss_percent)) * 0.01)
            if self.position.entry_price == 0.0:
                self.position.opened_step = step_index
        elif self.position.size < 0:
            self.position.take_profit_price = current_price * (1 - max(0.0, min(take_profit_percent, self.max_take_profit_percent)) * 0.01)
            self.position.stop_loss_price = current_price * (1 + max(0.0, min(stop_loss_percent, self.max_stop_loss_percent)) * 0.01)
            if self.position.entry_price == 0.0:
                self.position.opened_step = step_index
        else:
            self.position.take_profit_price = 0.0
            self.position.stop_loss_price = 0.0

    # ---------- 內部操作 ----------
    def _fee(self, notional: float) -> float:
        return abs(notional) * self.fee_rate

    def _required_margin(self, size: float, price: float) -> float:
        return abs(size) * price / self.leverage

    # 加倉
    def _increase_position(self, *, delta_size: float, price: float, step_index: int) -> None:
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

        self.open_trade = {
            'side': 'long' if delta_size > 0 else 'short',
            'entry_price': price,
            'size': delta_size,
            'entry_step': step_index,
            'notional_open': trade_notional,
            'fee_open': fee,
            'last_update_step': step_index,
        }

        self.position.size += delta_size
        self.used_margin += additional_margin
        self.wallet_balance -= fee
        self.total_fee_spent += fee
        self.total_notional_traded += trade_notional
        self.trade_count += 1
        self.orders_executed += 1

    # 減倉
    def _reduce_position(self, *, delta_size: float, price: float, step_index: int) -> None:
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
        self.total_fee_spent += fee
        self.total_notional_traded += abs(close_size) * price
        self.trade_count += 1
        self.orders_executed += 1

        self.closed_trades.append({
            'side': 'long' if self.position.size > 0 else 'short',
            'entry_price': self.position.entry_price,
            'exit_price': price,
            'size': close_size,
            'entry_step': self.position.opened_step,
            'exit_step': step_index,
            'holding_steps': step_index - self.position.opened_step,
            'pnl_after_fees': realized_pnl - fee,
            'total_fee': fee,
            'notional_open': abs(close_size) * self.position.entry_price,
            'notional_close': abs(close_size) * price,
            'reason': 'take_profit' if self.position.size > 0 else 'stop_loss',
        })
        

    # 平倉
    def _close_position(self, price: float, step_index: int) -> None:
        if self.position.size == 0.0:
            return
        size_to_close = abs(self.position.size)#平倉數量
        if self.position.size > 0:
            realized_pnl = (price - self.position.entry_price) * size_to_close#實現損益
        else:
            realized_pnl = (self.position.entry_price - price) * size_to_close#實現損益

        fee = self._fee(size_to_close * price)#手續費
        self.wallet_balance += realized_pnl - fee
        self.used_margin = 0.0
        self.position = PositionState()#清空持倉
        self.total_fee_spent += fee
        self.total_notional_traded += size_to_close * price
        self.trade_count += 1
        self.orders_executed += 1

        self.closed_trades.append({
            'side': 'long' if self.position.size > 0 else 'short',
            'entry_price': self.position.entry_price,
            'exit_price': price,
            'size': size_to_close,
            'entry_step': self.position.opened_step,
            'exit_step': step_index,
            'holding_steps': step_index - self.position.opened_step,
            'pnl_after_fees': realized_pnl - fee,
            'total_fee': fee,
            'notional_open': size_to_close * self.position.entry_price,
            'notional_close': size_to_close * price,
            'reason': 'take_profit' if self.position.size > 0 else 'stop_loss',
        })

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

