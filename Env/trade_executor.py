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
            trade_id: 交易 ID（用於 RUDDER 回填）
    '''
    size: float = 0.0  # 正數為多單，負數為空單（合約大小，資產單位）
    entry_price: float = 0.0
    stop_loss_price: float = 0.0  # 止損價格
    trade_id: int = -1  # 交易 ID（-1 表示無倉位）

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
    ) -> None:
        self.fee_rate = float(fee_rate)# 交易手續費
        self.leverage = float(leverage)# 槓桿倍數
        self.min_trade_qty = float(min_trade_qty)# 最低交易數量(BTC)
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
        # 進場統計（本集）：同向連續進場追蹤
        self.entry_happened: bool = False
        self.entry_streak_side: int = 0  # -1 空 / 0 無 / +1 多
        self.entry_streak_count: int = 0
        # 交易 ID 追蹤（用於 RUDDER 回填）
        self._next_trade_id: int = 0
        self.last_exit_info: Dict = {}  # 記錄最後一次出場信息（供環境讀取）
        # 停損冷靜期狀態（由環境配置默認值）
        self.cooldown_until_step: int = -1
        self.stop_loss_steps: List[int] = []
        self.stop_loss_cooldown_window: int = 6
        self.stop_loss_cooldown_length: int = 12
        self.stop_loss_cooldown_limit: int = 2
        self.cooldown_hold_ratio: float = 0.0

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
        self.entry_happened = False
        self.entry_streak_side = 0
        self.entry_streak_count = 0
        self._next_trade_id = 0
        self.last_exit_info = {}
        self.cooldown_until_step = -1
        self.stop_loss_steps = []

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
    ) -> None:
        # 重置本步觸發標記與出場信息
        self.stop_loss_triggered = False
        self.entry_happened = False
        self.last_exit_info = {}  # 清空上一步的出場信息
        
        # 1. 優先檢查止損（先於清算，保護倉位）
        if self.position.size != 0.0 and self.position.stop_loss_price > 0.0:
            if (self.position.size > 0 and low <= self.position.stop_loss_price) or \
               (self.position.size < 0 and high >= self.position.stop_loss_price):
                # 觸發止損，強制平倉
                self._close_position(self.position.stop_loss_price, exit_reason='stop_loss')
                self.stop_loss_triggered = True
                return  # 止損後不再執行其他邏輯
        
        # 2. 檢查清算（極端情況兜底）
        liq_price = self._calc_liquidation_price()
        if liq_price is not None and liq_price > 0.0:
            if (self.position.size > 0 and low <= liq_price) or (self.position.size < 0 and high >= liq_price):
                self._close_position(liq_price, exit_reason='liq')
                self.liq_triggered = True
                return

        # 以錢包餘額為風險基準計算目標倉位數量（避免未實現損益造成漂移）
        risk_base = self.wallet_balance
        target_size = (risk_base * position_percent * self.leverage) / current_price if current_price > 0 else 0.0

        # 3. 若無持倉，則依目標倉位數量開倉
        if self.position.size == 0.0:
            if abs(target_size) >= self.min_trade_qty:
                self._increase_position(delta_size=target_size, price=current_price, atr=atr)
        else:
            # 若方向反轉，先平舊倉再依新方向開倉（反手）
            if self.position.size * target_size < 0:
                self._close_position(price=current_price, exit_reason='close')
                # 平倉後依錢包餘額重算目標數量（已實現手續費/損益已反映在餘額）
                risk_base = self.wallet_balance
                target_size = (risk_base * position_percent * self.leverage) / current_price if current_price > 0 else 0.0
                # 新方向倉位若達最小交易量，才開倉
                if abs(target_size) >= self.min_trade_qty:
                    self._increase_position(delta_size=target_size, price=current_price, atr=atr)
            else:
                delta = target_size - self.position.size
                if abs(delta) >= self.min_trade_qty:
                    if (self.position.size > 0 and delta < 0) or (self.position.size < 0 and delta > 0):
                        # 減倉
                        self._reduce_position(delta_size=delta, price=current_price, exit_reason='reduce')
                    else:
                        # 加倉
                        self._increase_position(delta_size=delta, price=current_price, atr=atr)

    
    def forced_close_position(self, price: float) -> None:
        """強制平倉（回合結束時，用於 RUDDER 收斂）"""
        if self.position.size != 0.0:
            self._close_position(price, exit_reason='forced_close_on_done')

    # ---------- 內部操作 ----------
    def _fee(self, notional: float) -> float:
        return abs(notional) * (self.fee_rate /100)# 手續費 = 名目金額 * 手續費率(%)

    def _required_margin(self, size: float, price: float) -> float:
        return abs(size) * price / self.leverage

    # 加倉（進場事件）
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

        # 本步進場事件與同向連續進場追蹤（任何加倉皆視為一次進場事件）
        if abs(delta_size) > 0.0:
            side = 1 if delta_size > 0 else -1
            if self.entry_streak_side == side:
                self.entry_streak_count += 1
            else:
                self.entry_streak_side = side
                self.entry_streak_count = 1
            self.entry_happened = True

        # 若原先為空倉，視為首次進場（僅統計長短方向次數）並設定止損價與交易 ID
        if was_flat and abs(delta_size) > 0.0:
            if delta_size > 0:
                self.long_entry_count += 1
            else:
                self.short_entry_count += 1
            
            # 分配新的交易 ID（用於 RUDDER 回填）
            self.position.trade_id = self._next_trade_id
            self._next_trade_id += 1
            
            # 設定固定止損價（以 ATR 倍數計算）
            if atr > 0 and self.stop_loss_atr > 0:
                stop_distance = atr * self.stop_loss_atr
                if self.position.size > 0:
                    self.position.stop_loss_price = self.position.entry_price - stop_distance
                else:
                    self.position.stop_loss_price = self.position.entry_price + stop_distance
            else:
                self.position.stop_loss_price = 0.0

    # 減倉（出場事件：部分平倉）
    def _reduce_position(self, *, delta_size: float, price: float, exit_reason: str = 'reduce') -> None:
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
        # 記錄交易與出場信息（供環境讀取）
        trade_info = {
            'side': 'long' if was_long else 'short',
            'size': float(close_size),
            'price': float(price),
            'fee': float(fee),
            'realized_pnl': float(realized_pnl),
            'type': exit_reason,
            'trade_id': self.position.trade_id,  # 當前倉位的交易 ID
            'exit_reason': exit_reason,
        }
        self.closed_trades.append(trade_info)
        self.last_exit_info = trade_info  # 保存最後一次出場信息供環境讀取

    # 平倉（出場事件：完全平倉）
    def _close_position(self, price: float, exit_reason: str = 'close') -> None:
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
        
        # 記錄出場信息（在重置倉位前保存交易 ID）
        exited_trade_id = self.position.trade_id
        trade_info = {
            'side': 'long' if was_long else 'short',
            'size': float(size_to_close),
            'price': float(price),
            'fee': float(fee),
            'realized_pnl': float(realized_pnl),
            'type': exit_reason,
            'trade_id': exited_trade_id,
            'exit_reason': exit_reason,
        }
        self.closed_trades.append(trade_info)
        self.last_exit_info = trade_info  # 保存最後一次出場信息供環境讀取
        
        self.position = PositionState()  # 重置（包含 stop_loss_price 與 trade_id）
        # 重置同向連續進場追蹤
        if exit_reason == 'stop_loss':
            # 若為止損出場，保留方向與累積次數，下一次同向再進視為連續行為
            was_long_side = 1 if was_long else -1
            self.entry_streak_side = was_long_side
            # 維持現有 entry_streak_count（若為 0，下一次仍從 1 開始）
        else:
            self.entry_streak_side = 0
            self.entry_streak_count = 0
        self.entry_happened = False
        # 統計
        self.total_fees += fee
        if was_long:
            self.long_close_count += 1
        else:
            self.short_close_count += 1

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

