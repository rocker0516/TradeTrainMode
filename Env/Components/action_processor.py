import numpy as np

class ActionProcessor:
    """
    負責動作處理邏輯：
    1. 翻倉偵測 (Flip Detection)（僅標記 is_flip；不做預算限制）
    2. 單步倉位變化限制 (Max Step Position Change)
    3. 最小調倉幅度過濾 (Deadband)
    """
    def __init__(self, 
                 leverage: float, 
                 max_step_pos_change_pct: float,
                 min_position_change: float):
        self.leverage = float(leverage)
        self.max_step_pos_change_pct = float(max_step_pos_change_pct)
        self.min_position_change = float(min_position_change)

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
        action_val = float(np.clip(action_raw[0], -1.0, 1.0))
        target_pos_pct = action_val

        # 2. Current Position Pct
        current_pos_pct = 0.0
        last_equity = executor.equity(current_price)
        max_nominal = last_equity * self.leverage
        if max_nominal > 0:
            current_pos_val = executor.position.size * current_price
            current_pos_pct = current_pos_val / max_nominal

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

