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
        stop_loss_liq_buffer_pct: float = 0.0, # 止損相對強平價的安全緩衝（比例）
        min_position_change: float = 0.0, # 最小調倉幅度 (0.0 ~ 1.0)
    ) -> None:
        self.fee_rate = float(fee_rate)# 交易手續費
        self.leverage = float(leverage)# 槓桿倍數
        self.min_trade_qty = float(min_trade_qty)# 最低交易數量(BTC)
        self.min_position_change = float(min_position_change) # 最小調倉幅度
        self.liq_triggered = False# 強平觸發
        self.stop_loss_triggered = False# 止損觸發
        self.stop_loss_atr = float(stop_loss_atr)# 止損距離（ATR 倍數）
        self.stop_loss_liq_buffer_pct = float(stop_loss_liq_buffer_pct)
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
        # Max stop-loss distance percentage (unleveraged price distance relative to entry).
        # Example (long): entry=100, stop=95 => 5% => 0.05
        self.max_stop_loss_distance_pct: float = 0.0

        # --- Intrabar / trailing-stop helpers ---
        # 為了避免同一根 K 線使用 high/low 推進 trailing stop 造成「前視偏差」：
        # trailing stop 只使用「上一根已完成 K」的 high/low 來推進，並在本根以當前 low/high 判斷是否觸發。
        self._prev_bar_high: float | None = None
        self._prev_bar_low: float | None = None

    def _clamp_stop_loss_before_liquidation(self, *, current_price: float) -> None:
        """
        確保止損價一定「先於強平價」觸發，避免 stop_loss_price 設得比強平更遠。

        為了保留既有 ATR-stop 的語義，本函式只在「止損比強平更遠」時才進行收斂(clamp)。

        Args:
            current_price: 當前價格（僅用於數值穩定與極端情況保護）。
        """
        if self.position.size == 0.0 or self.position.entry_price <= 0.0:
            return
        if self.position.stop_loss_price <= 0.0:
            return

        liq_price = self._calc_liquidation_price(current_price=current_price)
        if liq_price is None or liq_price <= 0.0:
            return

        # stop_loss_liq_buffer_pct 的語意：
        # - 不是「liq_price * (1+buffer)」這種會在高槓桿下變得不可能的比例，
        # - 而是「在 liq_price 與 current_price 之間保留 buffer_pct 的安全緩衝」。
        #   例：long，liq=90, current=100, buffer=0.2 => min_sl = 90 + 0.2*(100-90)=92
        buffer_pct = max(0.0, float(self.stop_loss_liq_buffer_pct))
        eps = max(1e-8, float(abs(current_price)) * 1e-9)

        if self.position.size > 0.0:
            # 多單：止損必須在強平價之上（較早觸發）
            # clamp 到 (liq_price, current_price) 之間的緩衝位置，避免 buffer_pct 太大導致 min_sl > current_price
            min_sl = float(liq_price) + buffer_pct * (float(current_price) - float(liq_price)) + eps
            if self.position.stop_loss_price < min_sl:
                # 避免 stop 反而高於現價造成立即「邏輯不一致」；若接近爆倉，讓止損貼近現價即可。
                cap = float(current_price) - eps
                self.position.stop_loss_price = float(min(min_sl, cap)) if cap > 0.0 else float(min_sl)
                self._update_max_stop_loss_distance_metric()
        else:
            # 空單：止損必須在強平價之下（較早觸發）
            max_sl = float(liq_price) - buffer_pct * (float(liq_price) - float(current_price)) - eps
            if self.position.stop_loss_price > max_sl:
                cap = float(current_price) + eps
                self.position.stop_loss_price = float(max(max_sl, cap))
                self._update_max_stop_loss_distance_metric()

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
        self.max_stop_loss_distance_pct = 0.0
        self._prev_bar_high = None
        self._prev_bar_low = None

    def set_fee_rate(self, fee_rate: float) -> None:
        """
        動態更新手續費率（供訓練課程式學習使用）。

        注意：本專案 fee_rate 單位沿用既有設計：
        - fee_rate=0.005 代表 0.005%（在 _fee() 會除以 100）

        Args:
            fee_rate: 新的 fee_rate（必須 >= 0）
        """
        fee_rate = float(fee_rate)
        if fee_rate < 0.0:
            raise ValueError("fee_rate must be >= 0")
        self.fee_rate = fee_rate

    def get_fee_rate(self) -> float:
        """取得目前手續費率（fee_rate）。"""
        return float(self.fee_rate)

    def get_liquidation_price(self, current_price: float) -> float:
        """
        計算並快取當前持倉的強平價。
        
        Args:
            current_price: 最新標記價格
        
        Returns:
            最新強平價，若無倉位則回傳 0.0。
        """
        # 強平價為封閉式推導（依 wallet_balance/used_margin 與 entry/size/mmr），不依賴 close(current_price)。
        # 保留 current_price 參數僅為接口兼容（供外部呼叫）。
        liq_price = self._calc_liquidation_price(current_price=current_price)
        self.position.liq_price = float(liq_price) if liq_price is not None else 0.0
        return self.position.liq_price

    def _cache_prev_bar(self, *, high: float, low: float) -> None:
        """快取本次輸入的 K 線 high/low，供下一步 trailing stop 更新使用。"""
        self._prev_bar_high = float(high)
        self._prev_bar_low = float(low)

    def _update_max_stop_loss_distance_metric(self) -> None:
        """
        Update max stop-loss distance percentage (relative to entry price).

        Uses current position entry and stop_loss_price. No-op if data missing.
        """
        if self.position.size == 0.0:
            return
        entry = float(self.position.entry_price)
        stop = float(self.position.stop_loss_price)
        if entry <= 0.0 or stop <= 0.0:
            return
        dist_pct = abs(entry - stop) / entry
        if dist_pct > self.max_stop_loss_distance_pct:
            self.max_stop_loss_distance_pct = float(dist_pct)

    def _maybe_update_trailing_stop_from_prev_bar(self, *, atr: float) -> None:
        """
        用「上一根已完成 K」的高低點更新 trailing stop，避免同 K 線前視。

        Args:
            atr: 用於計算 trailing 距離（atr * stop_loss_atr）。
        """
        if self.position.size == 0.0 or self.position.stop_loss_price <= 0.0:
            return
        if atr <= 0.0 or self.stop_loss_atr <= 0.0:
            return
        if self._prev_bar_high is None or self._prev_bar_low is None:
            return

        trail_dist = float(atr * self.stop_loss_atr)
        if trail_dist <= 0.0:
            return

        if self.position.size > 0:
            # 多單：用上一根 high 推進止損（只會變緊）
            potential_new_sl = float(self._prev_bar_high - trail_dist)
            if potential_new_sl > self.position.stop_loss_price:
                self.position.stop_loss_price = potential_new_sl
                self._update_max_stop_loss_distance_metric()
        else:
            # 空單：用上一根 low 推進止損（只會變緊）
            potential_new_sl = float(self._prev_bar_low + trail_dist)
            if potential_new_sl < self.position.stop_loss_price:
                self.position.stop_loss_price = potential_new_sl
                self._update_max_stop_loss_distance_metric()

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

        # 不論本步是否因為 deadband/資金不足而提早 return，都要快取本根 K 線 high/low
        # 供下一步 trailing stop 使用，避免 _prev_bar_* 停留在更舊的 K 線。
        try:
            # 0) 先用「上一根已完成 K」更新 trailing stop（避免同 K 線 high/low 前視）
            self._maybe_update_trailing_stop_from_prev_bar(atr=atr)

            # 1) 優先檢查止損（先於清算）
            if self.position.size != 0.0 and self.position.stop_loss_price > 0.0:
                if (self.position.size > 0 and low <= self.position.stop_loss_price) or \
                   (self.position.size < 0 and high >= self.position.stop_loss_price):
                    # 觸發止損，強制平倉（回測假設：以 stop_loss_price 成交）
                    self._close_position(self.position.stop_loss_price)
                    self.stop_loss_triggered = True
                    return  # 止損後不再執行其他邏輯

            # 2) 檢查清算（極端情況兜底）
            liq_price = self._calc_liquidation_price(current_price=current_price)
            if liq_price is not None and liq_price > 0.0:
                if (self.position.size > 0 and low <= liq_price) or (self.position.size < 0 and high >= liq_price):
                    # Debug: 強平發生時，印出當下止損價（注意：平倉後 position 會被 reset，所以一定要在 close 前抓）
                    try:
                        sl = float(self.position.stop_loss_price)
                        size = float(self.position.size)
                        entry = float(self.position.entry_price)
                        required_margin_now = (abs(size) * entry / float(self.leverage)) if float(self.leverage) > 0.0 else 0.0
                       # print(
                       #     "[LIQ_TRIGGERED] "
                       #     f"stop_loss_price={sl:.8f} "
                       #     f"liq_price={float(liq_price):.8f} "
                       #     f"current_price={float(current_price):.8f} "
                       #     f"high={float(high):.8f} low={float(low):.8f} "
                       #     f"size={size:.8f} entry={entry:.8f} "
                       #     f"margin_mode={self.margin_mode} leverage={float(self.leverage):.2f} "
                       #     f"wallet_balance={float(self.wallet_balance):.4f} used_margin={float(self.used_margin):.4f} "
                       #     f"required_margin_now={float(required_margin_now):.6f}"
                       # )
                    except (TypeError, ValueError):
                        # debug print 不應中斷訓練流程
                        pass
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
            # 若目標倉位小於交易所最小下單量，視為 0（避免產生「永遠無法成交/平掉」的小倉位）
            if abs(target_size) < self.min_trade_qty:
                target_size = 0.0

            # 檢查最小調倉幅度 (避免微小變動刷手續費)
            # 計算最大可持倉數量 (Max Capacity) based on base_amount
            max_capacity_size = (base_amount * self.leverage) / current_price if current_price > 0 else 1.0
            # 計算變動比例 (相对于总容量)
            change_ratio = abs(target_size - self.position.size) / max_capacity_size if max_capacity_size > 0 else 0.0

            if change_ratio < self.min_position_change:
                # 變動幅度太小，這裡簡單處理：只要不是為了平倉(target=0)，就忽略
                if abs(target_size) > 1e-8:  # 不是要全平
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
                    target_size = (base_amount * position_percent * self.leverage) / current_price if current_price > 0 else 0.0
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
        finally:
            # 將本根 K 線 high/low 快取到下一步使用（trailing stop 只吃上一根）
            self._cache_prev_bar(high=high, low=low)

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
        """
        同向加倉（或空倉 -> 開倉）。

        重要（修正點）：
        - 舊版用 `required_margin(desired_size, current_price) - used_margin` 估算 additional_margin。
          在價格大幅波動時（尤其是「在更低價位加多 / 更高價位加空」），
          可能出現 required_margin 反而變小，導致 additional_margin=0，
          使 `used_margin` 嚴重低估，進而讓 isolated 的強平價計算失真。
        - 新版改為：以「加倉前/後」的倉位（size + 平均 entry）計算 required margin，
          並同步 `used_margin = required_margin_after`，確保帳務一致。
        """
        if abs(delta_size) < self.min_trade_qty:
            return
        if not (price > 0.0) or not (self.leverage > 0.0):
            return

        # 確保可用資金足以覆蓋新增保證金與手續費
        was_flat = (self.position.size == 0.0)
        old_size = float(self.position.size)
        old_entry = float(self.position.entry_price)

        # 先把 used_margin 同步到「目前倉位」應有的初始保證金（避免 drift）
        required_before = 0.0
        if abs(old_size) > 1e-12 and old_entry > 0.0:
            required_before = float(self._required_margin(old_size, old_entry))
        self.used_margin = float(max(0.0, required_before))

        def _weighted_entry(*, prev_size: float, prev_entry: float, add_size: float, add_price: float) -> float:
            if abs(prev_size) <= 1e-12:
                return float(add_price)
            total = abs(prev_size + add_size)
            if total <= 0.0:
                return 0.0
            old_notional = abs(prev_size) * float(prev_entry)
            new_notional = abs(add_size) * float(add_price)
            return float((old_notional + new_notional) / total)

        desired_size = float(old_size + float(delta_size))
        new_entry = _weighted_entry(prev_size=old_size, prev_entry=old_entry, add_size=float(delta_size), add_price=float(price))
        required_after = float(self._required_margin(desired_size, new_entry)) if (abs(desired_size) > 1e-12 and new_entry > 0.0) else 0.0
        additional_margin = float(max(0.0, required_after - required_before))

        trade_notional = abs(float(delta_size)) * float(price)
        fee = float(self._fee(trade_notional))

        if self.available_balance() < additional_margin + fee:
            # 依可用資金上限縮小加倉數量（margin + fee 都要算進去）
            avail = float(self.available_balance())
            # 每 1 單位 size（資產單位）的「保證金+手續費」消耗（以本次成交價估算，線性上界）
            cost_per_size = float(price) * ((1.0 / float(self.leverage)) + (float(self.fee_rate) / 100.0))
            if cost_per_size <= 0.0 or avail <= 0.0:
                return
            max_size_add = float(avail / cost_per_size)

            # 正確修正 delta_size 方向（同向加碼）
            sign = 1.0 if float(delta_size) > 0.0 else -1.0
            delta_size = float(sign * min(abs(float(delta_size)), max_size_add))
            if abs(delta_size) < self.min_trade_qty:
                return

            desired_size = float(old_size + float(delta_size))
            new_entry = _weighted_entry(prev_size=old_size, prev_entry=old_entry, add_size=float(delta_size), add_price=float(price))
            required_after = float(self._required_margin(desired_size, new_entry)) if (abs(desired_size) > 1e-12 and new_entry > 0.0) else 0.0
            additional_margin = float(max(0.0, required_after - required_before))
            trade_notional = abs(float(delta_size)) * float(price)
            fee = float(self._fee(trade_notional))
            if self.available_balance() < additional_margin + fee:
                # 仍不足就直接放棄（避免負資金/數值炸裂）
                return

        # 執行加倉
        self.position.entry_price = float(new_entry) if float(new_entry) > 0.0 else float(self.position.entry_price)
        self.position.size = float(desired_size)
        # 同步 used_margin（保證 isolated 強平價不會因低估 collateral 而失真）
        self.used_margin = float(max(0.0, required_after))
        self.wallet_balance = float(self.wallet_balance) - float(fee)
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
            self._update_max_stop_loss_distance_metric()
        else:
            if self.position.stop_loss_price == 0.0: # 只有未設定時才歸零，否則保留舊值？不，若無 ATR 則無法計算
                 self.position.stop_loss_price = 0.0

        # 保證止損一定先於強平（避免 SL 比 LIQ 更遠而永遠觸發不到）
        if self.position.size != 0.0 and self.position.stop_loss_price > 0.0:
            self._clamp_stop_loss_before_liquidation(current_price=price)

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
        # 以 entry_price 為基準釋放初始保證金（並在最後同步 used_margin）
        margin_release = self._required_margin(close_size, current_entry_price)

        self.wallet_balance += realized_pnl - fee
        self.used_margin = max(0.0, self.used_margin - margin_release)
        self.position.size = new_size
        if self.position.size == 0.0:
            self.position.entry_price = 0.0
            self.position.stop_loss_price = 0.0
        else:
            if self.position.stop_loss_price > 0.0:
                self._clamp_stop_loss_before_liquidation(current_price=price)
            # 同步 used_margin，避免因歷史 drift/釋放過量導致保證金失真
            self.used_margin = float(max(0.0, self._required_margin(self.position.size, current_entry_price)))
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
        以封閉式解推導強平價（避免「用 close 計算強平價、但用 high/low 觸發」的不一致）。

        清算條件（簡化模型）：equity(p) <= maintenance_margin(p)
        - equity(p) = collateral + unrealized_pnl(p)
        - maintenance_margin(p) = abs(size) * p * maintenance_margin_rate

        cross：collateral = wallet_balance
        isolated：collateral = used_margin

        注意：此處保留 current_price 參數僅為接口兼容，不作為計算輸入。
        """
        if self.position.size == 0.0 or self.position.entry_price == 0.0:
            return None

        m = self.maintenance_margin_rate  # 維持保證金率
        s = self.position.size
        e = self.position.entry_price

        if self.margin_mode == "cross":
            collateral = float(self.wallet_balance)
        else:
            collateral = float(self.used_margin)

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

