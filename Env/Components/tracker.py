import numpy as np
import os
import time
import json
from collections import deque

class Tracker:
    """
    負責追蹤環境狀態歷史與日誌記錄。
    1. Account Series (用於繪圖或分析)
    2. Step Logging (JSONL)
    3. Fee History (Rolling Window)
    """
    def __init__(self, 
                 step_log_enabled: bool, 
                 step_log_dir: str, 
                 step_log_every_n: int,
                 env_id: int, 
                 initial_balance: float, 
                 data_len: int,
                 fee_rolling_window: int):
        
        self.step_log_enabled = step_log_enabled
        self.step_log_dir = step_log_dir
        self.step_log_every_n = step_log_every_n
        self.env_id = int(env_id)
        self.initial_balance = float(initial_balance)
        
        # Log File Setup
        self._step_log_path = None
        if self.step_log_enabled:
            env_dir = os.path.join(self.step_log_dir, f"env_{self.env_id:03d}")
            os.makedirs(env_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self._step_log_path = os.path.join(env_dir, f"steps_{timestamp}.jsonl")
            
        # Account Series History
        self.account_series = {
            'position': np.zeros(data_len),
            'position_value': np.zeros(data_len),
            'equity': np.zeros(data_len),
            'wallet': np.zeros(data_len)
        }
        
        # Fee Tracking
        self.fee_history = deque() # (step_index, fee_amount)
        self.rolling_fee_sum = 0.0
        self.fee_rolling_window = fee_rolling_window
        self.prev_total_fees = 0.0
        self.last_step_fee = 0.0

    def update_account_series(self, step_idx: int, executor, current_price: float):
        """更新帳戶狀態序列"""
        if step_idx >= len(self.account_series['position']):
            return

        pos_norm = executor.position.size / (self.initial_balance / current_price) if self.initial_balance > 0 and current_price > 0 else 0.0
        pos_value_norm = (executor.position.size * current_price) / self.initial_balance if self.initial_balance > 0 else 0.0
        equity_norm = executor.equity(current_price) / self.initial_balance if self.initial_balance > 0 else 0.0
        wallet_norm = executor.wallet_balance / self.initial_balance if self.initial_balance > 0 else 0.0
        
        self.account_series['position'][step_idx] = float(pos_norm)
        self.account_series['position_value'][step_idx] = float(pos_value_norm)
        self.account_series['equity'][step_idx] = float(equity_norm)
        self.account_series['wallet'][step_idx] = float(wallet_norm)

    def update_fee_tracking(self, step_idx: int, current_total_fees: float):
        """更新滾動手續費統計"""
        step_fee = current_total_fees - self.prev_total_fees
        self.last_step_fee = step_fee
        self.prev_total_fees = current_total_fees
        
        # Add new
        self.fee_history.append((step_idx, step_fee))
        self.rolling_fee_sum += step_fee
        
        # Pop old
        while self.fee_history and (step_idx - self.fee_history[0][0]) > self.fee_rolling_window:
            _, old_fee = self.fee_history.popleft()
            self.rolling_fee_sum -= old_fee

    def log_step(self, payload: dict, episode_steps: int, force: bool = False):
        """寫入日誌"""
        if not (self.step_log_enabled and self._step_log_path):
            return
            
        should_log = force or (episode_steps % self.step_log_every_n == 0)
        
        if should_log:
            try:
                with open(self._step_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False))
                    f.write("\n")
            except Exception:
                pass

