from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class PositionState:
    '''
        持倉狀態
        
        Args:
            size: 持倉數量
            entry_price: 進場價格
    '''
    size: float = 0.0  # 正數為多單，負數為空單（合約大小，資產單位）
    entry_price: float = 0.0


class TradeExecutor:
    """執行槓桿合約交易的執行器，負責保證金與手續費計算。

    權益（Equity）= 錢包餘額（wallet_balance）+ 未實現損益（unrealized_pnl）
    手續費：開倉/平倉依名目金額（notional）計收
    初始保證金：abs(持倉數量) * 進場價 / 槓桿

    margin_mode:
        - 'cross' 全倉：以整體錢包餘額作為保護抵押，強平使用 wallet_balance
        - 'isolated' 逐倉：以倉位初始保證金作為抵押，強平使用 used_margin
    """

    def __init__(
        self,
        *,
        initial_balance: float, # 初始資金
        fee_rate: float, # 交易手續費
        leverage: float, # 槓桿倍數
        min_trade_qty: float, # 最低交易數量(BTC)
        maintenance_margin_rate: float = 0.005, # 維持保證金率
        margin_mode: str = 'cross', # 保證金模式：'cross' 或 'isolated'
    ) -> None:
        self.fee_rate = float(fee_rate)# 交易手續費
        self.leverage = float(leverage)# 槓桿倍數
        self.min_trade_qty = float(min_trade_qty)# 最低交易數量(BTC)
        self.maintenance_margin_rate = float(maintenance_margin_rate)# 維持保證金率
        self.initial_balance = float(initial_balance)# 初始資金
        mode = str(margin_mode).lower()
        if mode not in ("cross", "isolated"):
            raise ValueError("margin_mode must be 'cross' or 'isolated'")
        self.margin_mode = mode

        self.wallet_balance: float = float(initial_balance)# 錢包餘額
        self.position = PositionState()# 持倉狀態
        self.used_margin: float = 0.0# 已使用保證金
        self.closed_trades: List[Dict[str, float]] = []# 已平倉交易
        # 逐回合統計
        self.total_fees: float = 0.0
        self.long_close_count: int = 0
        self.short_close_count: int = 0
        self.long_entry_count: int = 0
        self.short_entry_count: int = 0

    def reset(self, initial_balance: float) -> None:
        self.wallet_balance = float(initial_balance)# 錢包餘額
        self.position = PositionState()# 持倉狀態
        self.used_margin = 0.0# 已使用保證金
        self.closed_trades = []# 已平倉交易
        self.total_fees = 0.0
        self.long_close_count = 0
        self.short_close_count = 0
        self.long_entry_count = 0
        self.short_entry_count = 0

    # ---------- 查詢輔助方法 ----------
    # 未實現損益
    def unrealized_pnl(self, current_price: float) -> float:
        if self.position.size == 0.0 or self.position.entry_price == 0.0:
            return 0.0
        return (current_price - self.position.entry_price) * self.position.size

    # 權益
    def equity(self, current_price: float) -> float:
        return self.wallet_balance + self.unrealized_pnl(current_price)

    # 可用資金
    def available_balance(self) -> float:
        return self.wallet_balance - self.used_margin

    # ---------- 核心執行邏輯 ----------
    def execute(
        self,
        *,
        position_percent: float, # 目標持倉比例 (-1.0 ~ 1.0)
        current_price: float, # 當前價格
        high: float, # 當前最高價
        low: float, # 當前最低價
        equity: float, # 當前權益
    ) -> None:
    #
     #   if self.position.size != 0.0:
     #       liq_price = self._calc_liquidation_price() # 強平價格
     #       if liq_price is not None and liq_price > 0.0:
     #           if (self.position.size > 0 and low <= liq_price) or (self.position.size < 0 and high >= liq_price):
     #               self._close_position(liq_price)
     #               return
    #

        # 檢查價格最低價是否觸發強平
        liq_price = self._calc_liquidation_price() # 強平價格
        if liq_price is not None and liq_price > 0.0:# 強平價格存在且大於0
            if (self.position.size > 0 and low <= liq_price) or (self.position.size < 0 and high >= liq_price):# 持倉方向與強平價格關係符合
                self._close_position(liq_price)
                self.done = True
                return

        # 以錢包餘額為風險基準計算目標倉位數量（避免未實現損益造成漂移）
        risk_base = self.wallet_balance
        target_size = (risk_base * position_percent * self.leverage) / current_price if current_price > 0 else 0.0

        # 若無持倉，則依目標倉位數量開倉
        if self.position.size == 0.0:
            if abs(target_size) >= self.min_trade_qty:
                self._increase_position(delta_size=target_size, price=current_price)
        else:
            # 若方向反轉，先平舊倉再依新方向開倉
            if self.position.size * target_size < 0:
                self._close_position(price=current_price)
                # 平倉後依錢包餘額重算目標數量（已實現手續費/損益已反映在餘額）
                risk_base = self.wallet_balance
                target_size = (risk_base * position_percent * self.leverage) / current_price if current_price > 0 else 0.0
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

       # # 根據最終倉位方向，從當前價格更新止盈/止損價格
        # if self.position.size > 0:
        #     self.position.take_profit_price = current_price * (1 + max(0.0, min(take_profit_percent, self.max_take_profit_percent)) * 0.01)
        #     self.position.stop_loss_price = current_price * (1 - max(0.0, min(stop_loss_percent, self.max_stop_loss_percent)) * 0.01)
        # elif self.position.size < 0:
        #     self.position.take_profit_price = current_price * (1 - max(0.0, min(take_profit_percent, self.max_take_profit_percent)) * 0.01)
        #     self.position.stop_loss_price = current_price * (1 + max(0.0, min(stop_loss_percent, self.max_stop_loss_percent)) * 0.01)
        # else:
        #     self.position.take_profit_price = 0.0
       #     self.position.stop_loss_price = 0.0

    # ---------- 內部操作 ----------
    def _fee(self, notional: float) -> float:
        return abs(notional) * self.fee_rate

    def _required_margin(self, size: float, price: float) -> float:
        return abs(size) * price / self.leverage

    # 加倉
    def _increase_position(self, *, delta_size: float, price: float) -> None:
        # 確保可用資金足以覆蓋新增保證金與手續費
        was_flat = (self.position.size == 0.0)
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
        self.total_fees += fee

        # 若原先為空倉，視為進場（記一次）
        if was_flat and abs(delta_size) > 0.0:
            if delta_size > 0:
                self.long_entry_count += 1
            else:
                self.short_entry_count += 1
        self.total_fees += fee

    # 減倉
    def _reduce_position(self, *, delta_size: float, price: float) -> None:
        # delta_size 與當前持倉方向相反；以下計算實際平倉數量
        close_size = -delta_size  # 平倉數量（與現有持倉同號）
        was_long = self.position.size > 0
        if was_long:
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
        # 統計
        self.total_fees += fee
        if close_size > 0:
            if was_long:
                self.long_close_count += 1
            else:
                self.short_close_count += 1
        # 記錄交易
        self.closed_trades.append({
            'side': 'long' if was_long else 'short',
            'size': float(close_size),
            'price': float(price),
            'fee': float(fee),
            'realized_pnl': float(realized_pnl),
            'type': 'reduce'
        })

    # 平倉
    def _close_position(self, price: float) -> None:
        if self.position.size == 0.0:
            return
        size_to_close = abs(self.position.size)
        was_long = self.position.size > 0
        if was_long:
            realized_pnl = (price - self.position.entry_price) * size_to_close
        else:
            realized_pnl = (self.position.entry_price - price) * size_to_close

        fee = self._fee(size_to_close * price)
        self.wallet_balance += realized_pnl - fee
        self.used_margin = 0.0
        self.position = PositionState()
        # 統計
        self.total_fees += fee
        if was_long:
            self.long_close_count += 1
        else:
            self.short_close_count += 1
        # 記錄交易
        self.closed_trades.append({
            'side': 'long' if was_long else 'short',
            'size': float(size_to_close),
            'price': float(price),
            'fee': float(fee),
            'realized_pnl': float(realized_pnl),
            'type': 'close'
        })

    # ---------- 風險控制與強平 ----------
    def _calc_liquidation_price(self) -> float | None:
        # 根據：equity(p) = wallet + (p - entry) * size
        # 維持保證金：|size| * p * mmr
        # 觸發強平條件：equity(p) <= maintenance_margin(p)
        if self.position.size == 0.0 or self.position.entry_price == 0.0:
            return None
        m = self.maintenance_margin_rate# 維持保證金率
        s = self.position.size# 持倉數量
        e = self.position.entry_price# 進場價格

        # 全倉使用錢包餘額作為抵押；逐倉使用倉位佔用保證金作為抵押
        collateral = self.wallet_balance if self.margin_mode == "cross" else self.used_margin
        if s > 0:
            denom = s * (1.0 - m)
            if denom <= 0:
                return None
            price = (s * e - collateral) / denom
            return max(0.0, price)
        else:
            u = abs(s)
            denom = u * (1.0 + m)
            if denom <= 0:
                return None
            price = (collateral + u * e) / denom
            return max(0.0, price)

