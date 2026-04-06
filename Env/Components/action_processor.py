import numpy as np

class ActionProcessor:
    """
    負責動作處理邏輯：
    1. 將 raw action clip 至 [-max_position_pct, +max_position_pct]（目標倉位比例）
    2. 翻倉偵測 (Flip Detection)（僅標記 is_flip；不做預算限制）
    2.1 no-trade 雙門檻（hysteresis）：讓 0 倉位更穩定（避免 action 0 附近抖動造成反覆成交）
    3. 單步倉位變化限制 (Max Step Position Change)
    4. 最小調倉幅度過濾 (Deadband)
    """
    def __init__(self, 
                 leverage: float, 
                 max_step_pos_change_pct: float,
                 min_position_change: float,
                 no_trade_entry_threshold: float = 0.0,
                 no_trade_exit_threshold: float = 0.0,
                 max_position_pct: float = 1.0):
        """
        Args:
            leverage: 槓桿倍數。
            max_step_pos_change_pct: 單步相對 risk_base 的名義變化上限比例。
            min_position_change: 最小調倉 deadband（相對名義容量）。
            no_trade_entry_threshold: 空倉時 |action| 低於此值則目標強制為 0。
            no_trade_exit_threshold: 有倉時 |action| 低於此值則目標強制為 0（應 <= entry）。
            max_position_pct: 目標持倉比例絕對值上限，須落在 (0, 1]；預設 1.0 與舊版 [-1,1] clip 相容。

        Raises:
            ValueError: 門檻順序不合法或 max_position_pct 不在 (0, 1]。
        """
        self.leverage = float(leverage)
        self.max_step_pos_change_pct = float(max_step_pos_change_pct)
        self.min_position_change = float(min_position_change)
        self.no_trade_entry_threshold = float(no_trade_entry_threshold)
        self.no_trade_exit_threshold = float(no_trade_exit_threshold)
        self.max_position_pct = float(max_position_pct)
        if self.max_position_pct <= 0.0 or self.max_position_pct > 1.0:
            raise ValueError("max_position_pct must be in (0, 1]")

        # 允許關閉 hysteresis：兩者皆為 0 即不生效
        if self.no_trade_entry_threshold < 0.0:
            raise ValueError("no_trade_entry_threshold must be >= 0")
        if self.no_trade_exit_threshold < 0.0:
            raise ValueError("no_trade_exit_threshold must be >= 0")
        if self.no_trade_entry_threshold < self.no_trade_exit_threshold:
            raise ValueError("no_trade_entry_threshold must be >= no_trade_exit_threshold")

    def process_action(self, 
                       action_raw: np.ndarray, 
                       executor, 
                       current_price: float) -> tuple:
        """
        處理原始動作，並偵測是否為翻倉（不再做 flip 預算限制）。
        
        Returns:
            (target_pos_pct, is_flip)
        """
        # 1. Clip raw action
        action_val = float(np.clip(action_raw[0], -self.max_position_pct, self.max_position_pct))
        target_pos_pct = action_val

        # 2. Current Position Pct
        current_pos_pct = 0.0
        last_equity = executor.equity(current_price)
        max_nominal = last_equity * self.leverage
        if max_nominal > 0:
            current_pos_val = executor.position.size * current_price
            current_pos_pct = current_pos_val / max_nominal

        # 2.1 No-trade hysteresis (方案2)
        # - 空倉時：|action| < entry_threshold => 強制 0（避免被小噪音推著開小倉）
        # - 有倉時：|action| < exit_threshold  => 強制 0（允許更敏感地回到空倉）
        # 注意：閾值單位是「目標倉位比例」(position pct)，並且 action 可能已被 ActionClipWrapper 先 clip。
        entry_th = float(self.no_trade_entry_threshold)
        exit_th = float(self.no_trade_exit_threshold)
        if (entry_th > 0.0) or (exit_th > 0.0):
            has_pos = abs(float(getattr(executor.position, "size", 0.0))) > 1e-8
            th = exit_th if has_pos else entry_th
            if abs(float(target_pos_pct)) < th:
                target_pos_pct = 0.0

        # 3. Flip Detect（只標記，不限制）
        is_flip = (target_pos_pct * current_pos_pct < -0.01)  # Crossing zero significantly
        return target_pos_pct, bool(is_flip)

    def calculate_effective_action(self, 
                                   target_pos_pct: float, 
                                   executor, 
                                   current_price: float, 
                                   risk_base: float) -> float:
        """
        計算考量了步幅限制與 Deadband 後的實際執行目標比例。
        """
        last_equity = executor.equity(current_price)
        
        # 1. Desired Size (BTC)
        desired_notional = target_pos_pct * last_equity * self.leverage
        desired_size = desired_notional / current_price if current_price > 0 else 0.0
        
        current_size = float(executor.position.size)

        # 2. Step Change Limit
        # Max change in BTC based on risk_base (daily baseline equity)
        max_change_qty = (risk_base * self.leverage * self.max_step_pos_change_pct) / current_price if current_price > 0 else 0.0
        
        # Relax limit if closing
        is_closing = (abs(desired_size) < abs(current_size)) and (desired_size * current_size >= 0)
        
        change = desired_size - current_size
        
        if not is_closing and max_change_qty > 0:
            if abs(change) > max_change_qty:
                change = np.sign(change) * max_change_qty
                desired_size = current_size + change
                
        # 3. Min Position Change (Deadband) handled inside Executor mostly, but can preempt here
        # If change is tiny and not closing to zero, we might skip. 
        # But Executor checks min_trade_qty. Here we check min_position_change relative to capacity.
        # Logic kept simple: pass desired_size converted back to pct.
        
        final_action_pct = 0.0
        max_cap = last_equity * self.leverage
        if max_cap > 0:
            final_action_pct = (desired_size * current_price) / max_cap
            
        return float(final_action_pct)

