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
            stop_loss_price: 止損價格（固定規則計算）
            liq_price: 依維持保證金率推導的即時強平價
    '''
    size: float = 0.0  # 正數為多單，負數為空單（合約大小，資產單位）
    entry_price: float = 0.0
    stop_loss_price: float = 0.0  # 止損價格
    liq_price: float = 0.0  # 強平價快取，便於觀測/風險提示


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
        stop_loss_atr: float = 2.5, # 止損距離（ATR 倍數）
        min_position_change: float = 0.0, # 最小調倉幅度 (0.0 ~ 1.0)
    ) -> None:
        self.fee_rate = float(fee_rate)# 交易手續費
        self.leverage = float(leverage)# 槓桿倍數
        self.min_trade_qty = float(min_trade_qty)# 最低交易數量(BTC)
        self.min_position_change = float(min_position_change) # 最小調倉幅度
        self.liq_triggered = False# 強平觸發
        self.stop_loss_triggered = False# 止損觸發
        self.stop_loss_atr = float(stop_loss_atr)# 止損距離（ATR 倍數）
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
        self.max_trade_loss_pct: float = 0.0  # Max single trade loss percentage (ROI %)

    def reset(self, initial_balance: float) -> None:
        self.wallet_balance = float(initial_balance)# 錢包餘額
        self.position = PositionState()# 持倉狀態
        self.liq_triggered = False# 強平觸發
        self.stop_loss_triggered = False# 止損觸發
        self.used_margin = 0.0# 已使用保證金
        self.closed_trades = []# 已平倉交易
        self.total_fees = 0.0
        self.long_close_count = 0
        self.short_close_count = 0
        self.long_entry_count = 0
        self.short_entry_count = 0
        self.max_trade_loss_pct = 0.0

    def get_liquidation_price(self, current_price: float) -> float:
        """
        計算並快取當前持倉的強平價。
        
        Args:
            current_price: 最新標記價格
        
        Returns:
            最新強平價，若無倉位則回傳 0.0。
        """
        liq_price = self._calc_liquidation_price(current_price=current_price)
        self.position.liq_price = float(liq_price) if liq_price is not None else 0.0
        return self.position.liq_price

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
        atr: float = 0.0, # ATR（用於計算止損距離）
        risk_base: float = None # 用於計算倉位大小的基準金額 (預設為 None，若 None 則使用 wallet_balance)
    ) -> None:
        # 重置本步觸發標記
        self.stop_loss_triggered = False
        
        # 1. 優先檢查止損（先於清算，保護倉位）
        if self.position.size != 0.0 and self.position.stop_loss_price > 0.0:
            # 移動止損檢查 (Trailing Stop)
            # 邏輯：當價格朝有利方向移動，將止損價位移至 (CurrentPrice - ATR*Trail)
            # 但為了簡單起見，這裡只做 "損益兩平後保護" 或 "跟隨價格"
            # 這裡實作：標準移動止損 - 當價格創新高(多)/新低(空)，提升止損
            # 需注意：若太敏感會被洗掉。
            
            # 參數：Trailing 距離同 StopLoss 距離
            if atr > 0 and self.stop_loss_atr > 0:
                trail_dist = atr * self.stop_loss_atr
                if self.position.size > 0:
                    # 多單：若 (CurrentHigh - Trail) > CurrentSL，則提升 SL
                    # 使用 high 雖激進，但能鎖定利潤；保守可用 close
                    potential_new_sl = high - trail_dist
                    if potential_new_sl > self.position.stop_loss_price:
                         self.position.stop_loss_price = potential_new_sl
                else:
                    # 空單：若 (CurrentLow + Trail) < CurrentSL，則下移 SL
                    potential_new_sl = low + trail_dist
                    if potential_new_sl < self.position.stop_loss_price:
                         self.position.stop_loss_price = potential_new_sl

            # 執行止損檢查
            if (self.position.size > 0 and low <= self.position.stop_loss_price) or \
               (self.position.size < 0 and high >= self.position.stop_loss_price):
                # 觸發止損，強制平倉
                self._close_position(self.position.stop_loss_price)
                self.stop_loss_triggered = True
                return  # 止損後不再執行其他邏輯
        
        # 2. 檢查清算（極端情況兜底）
        liq_price = self._calc_liquidation_price(current_price=current_price)
        if liq_price is not None and liq_price > 0.0:
            if (self.position.size > 0 and low <= liq_price) or (self.position.size < 0 and high >= liq_price):
                self._close_position(liq_price)
                self.liq_triggered = True
                return

        # 以指定基準(risk_base)計算目標倉位數量
        # 若未指定，預設使用 wallet_balance (舊邏輯)
        # 但通常由外部 (Env) 傳入固定的基準金額 (例如每 288 步更新一次的餘額)
        if risk_base is not None:
            base_amount = risk_base
        else:
            # 使用可用資金與權益的保守基準，避免未實現虧損時放大倉位
            base_amount = min(self.wallet_balance, self.equity(current_price))
        
        # 避免 base_amount 小於等於 0
        base_amount = max(0.0, base_amount)
        
        target_size = (base_amount * position_percent * self.leverage) / current_price if current_price > 0 else 0.0

        # 檢查最小調倉幅度 (避免微小變動刷手續費)
        # 計算最大可持倉數量 (Max Capacity) based on base_amount
        max_capacity_size = (base_amount * self.leverage) / current_price if current_price > 0 else 1.0
        # 計算變動比例 (相对于总容量)
        change_ratio = abs(target_size - self.position.size) / max_capacity_size if max_capacity_size > 0 else 0.0
        
        if change_ratio < self.min_position_change:
            # 變動幅度太小，檢查是否為反向交易或平倉，如果是反向/平倉通常還是允許，除非非常小
            # 但如果只是微調 (例如 0.5 -> 0.51)，則忽略
            # 這裡簡單處理：只要變動小於閾值且不是為了觸發平倉(target=0)，就忽略
            if abs(target_size) > 1e-8: # 不是要全平
                 return

        # 3. 若無持倉，則依目標倉位數量開倉
        if self.position.size == 0.0:
            if abs(target_size) >= self.min_trade_qty:
                self._increase_position(delta_size=target_size, price=current_price, atr=atr)
        else:
            # 若方向反轉，先平舊倉再依新方向開倉
            if self.position.size * target_size < 0:
                self._close_position(price=current_price)
                # 平倉後依基準金額重算目標數量
                # 注意：這裡維持使用 base_amount，而非切換回 wallet_balance，保持基準一致性
                target_size = (base_amount * position_percent * self.leverage) / current_price if current_price > 0 else 0.0
                # 新方向倉位若達最小交易量，才開倉
                if abs(target_size) >= self.min_trade_qty:
                    self._increase_position(delta_size=target_size, price=current_price, atr=atr)
            else:
                delta = target_size - self.position.size
                if abs(delta) >= self.min_trade_qty:
                    if (self.position.size > 0 and delta < 0) or (self.position.size < 0 and delta > 0):
                        # 減倉
                        self._reduce_position(delta_size=delta, price=current_price)
                    else:
                        # 加倉
                        self._increase_position(delta_size=delta, price=current_price, atr=atr)

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
        return abs(notional) * (self.fee_rate /100)# 手續費 = 名目金額 * 手續費率(%)

    def _required_margin(self, size: float, price: float) -> float:
        return abs(size) * price / self.leverage

    # 加倉
    def _increase_position(self, *, delta_size: float, price: float, atr: float = 0.0) -> None:
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
            
            # 正確修正 delta_size 方向
            if delta_size > 0:
                delta_size = max_size_add
            else:
                delta_size = -max_size_add
                
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

        # 設定/更新止損價 (固定 ATR 倍數)
        if atr > 0 and self.stop_loss_atr > 0:
            stop_distance = atr * self.stop_loss_atr
            if self.position.size > 0:
                # 多單止損 = 新均價 - ATR距離
                new_sl = self.position.entry_price - stop_distance
                # 若已有止損，加倉時通常不希望止損變寬（往下移），取較高者（更緊）
                # 但若均價大幅上移，也許可以接受新的寬止損？
                # 這裡採用：加倉後一律重算 ATR 止損，因為風險基礎（均價）變了
                self.position.stop_loss_price = new_sl
            else:
                # 空單止損 = 新均價 + ATR距離
                new_sl = self.position.entry_price + stop_distance
                self.position.stop_loss_price = new_sl
        else:
            if self.position.stop_loss_price == 0.0: # 只有未設定時才歸零，否則保留舊值？不，若無 ATR 則無法計算
                 self.position.stop_loss_price = 0.0

        # 若原先為空倉，視為進場（記一次）
        if was_flat and abs(delta_size) > 0.0:
            if delta_size > 0:
                self.long_entry_count += 1
            else:
                self.short_entry_count += 1

    def _update_max_loss(self, realized_pnl: float, close_size: float, entry_price: float) -> None:
        """Update max trade loss percentage (ROI based)."""
        if close_size <= 0 or entry_price <= 0:
            return
        
        # ROI Calculation: PnL / Initial Margin
        # Initial Margin = (Size * Entry Price) / Leverage
        initial_margin = (close_size * entry_price) / self.leverage
        if initial_margin > 0:
            roi_pct = (realized_pnl / initial_margin) * 100.0
            if roi_pct < self.max_trade_loss_pct:
                self.max_trade_loss_pct = roi_pct

    # 減倉
    def _reduce_position(self, *, delta_size: float, price: float) -> None:
        # delta_size 與當前持倉方向相反；以下計算實際平倉數量
        close_size = -delta_size  # 平倉數量（與現有持倉同號）
        current_entry_price = self.position.entry_price # Capture entry price
        
        was_long = self.position.size > 0
        if was_long:
            close_size = min(abs(close_size), abs(self.position.size))
            realized_pnl = (price - current_entry_price) * close_size
            new_size = self.position.size - close_size
        else:
            close_size = min(abs(close_size), abs(self.position.size))
            realized_pnl = (current_entry_price - price) * close_size
            new_size = - (abs(self.position.size) - close_size)

        fee = self._fee(close_size * price)
        margin_release = self._required_margin(close_size, current_entry_price)

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
        
        # 更新最大虧損統計
        self._update_max_loss(realized_pnl, close_size, current_entry_price)

    # 平倉
    def _close_position(self, price: float) -> None:
        if self.position.size == 0.0:
            return
        size_to_close = abs(self.position.size)
        current_entry_price = self.position.entry_price # Capture entry price
        
        was_long = self.position.size > 0
        if was_long:
            realized_pnl = (price - current_entry_price) * size_to_close
        else:
            realized_pnl = (current_entry_price - price) * size_to_close

        fee = self._fee(size_to_close * price)
        self.wallet_balance += realized_pnl - fee
        self.used_margin = 0.0
        self.position = PositionState()  # 重置（包含 stop_loss_price）
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
        
        # 更新最大虧損統計
        self._update_max_loss(realized_pnl, size_to_close, current_entry_price)
        
    # ---------- 風險控制與強平 ----------
    def _calc_liquidation_price(self, *, current_price: float) -> float | None:
        """
        以「當前價格的權益」作為全倉抵押，避免只用 wallet_balance 導致過早/過晚觸發。
        清算條件：equity(p) <= maintenance_margin(p)
        """
        if self.position.size == 0.0 or self.position.entry_price == 0.0:
            return None

        m = self.maintenance_margin_rate  # 維持保證金率
        s = self.position.size
        e = self.position.entry_price

        # 使用當前價格計算權益，避免忽略未實現損益
        equity_now = self.equity(current_price)

        # Cross：用全部 equity 作抵押；Isolated：用已用保證金 + 正向的未實現盈餘
        if self.margin_mode == "cross":
            collateral = equity_now
        else:
            collateral = self.used_margin + max(self.unrealized_pnl(current_price), 0.0)

        collateral = max(0.0, collateral)

        if s > 0:
            denom = s * (1.0 - m)
            if denom <= 0:
                return None
            price = (s * e - collateral) / denom
        else:
            u = abs(s)
            denom = u * (1.0 + m)
            if denom <= 0:
                return None
            price = (collateral + u * e) / denom

        return max(0.0, price)

