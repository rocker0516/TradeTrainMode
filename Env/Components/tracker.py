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
        #
        # 設計說明：
        # - 既有 key（position/position_value/equity/wallet）維持「相對 initial_balance 的正規化序列」，
        #   方便與舊版分析/繪圖保持相容。
        # - 新增 raw key（*_raw / position_size）保存「原始數值」，供 render / debug 直接畫曲線與事件標記使用。
        #   raw 預設用 NaN，避免未走到的未來區段被誤畫成 0（符合「done 後曲線留白」需求）。
        self.account_series = {
            # --- normalized series (backward compatible) ---
            "position": np.zeros(data_len, dtype=np.float64),
            "position_value": np.zeros(data_len, dtype=np.float64),
            "equity": np.zeros(data_len, dtype=np.float64),
            "wallet": np.zeros(data_len, dtype=np.float64),
            # --- raw series (for render/debug) ---
            "position_size": np.full(data_len, np.nan, dtype=np.float64),  # asset units (e.g., BTC)
            "position_value_raw": np.full(data_len, np.nan, dtype=np.float64),  # notional (asset*price)
            "equity_raw": np.full(data_len, np.nan, dtype=np.float64),
            "wallet_raw": np.full(data_len, np.nan, dtype=np.float64),
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

        # Raw series (for render/debug)
        try:
            pos_size = float(getattr(executor.position, "size", 0.0))
            pos_value = float(pos_size) * float(current_price)
            equity_raw = float(executor.equity(current_price))
            wallet_raw = float(getattr(executor, "wallet_balance", 0.0))
        except (TypeError, ValueError):
            # Do not break training due to plotting helpers
            return

        self.account_series["position_size"][step_idx] = pos_size
        self.account_series["position_value_raw"][step_idx] = pos_value
        self.account_series["equity_raw"][step_idx] = equity_raw
        self.account_series["wallet_raw"][step_idx] = wallet_raw

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

