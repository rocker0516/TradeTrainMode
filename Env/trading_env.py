import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import random
import os
import json
import time
from collections import deque
from .trade_executor import TradeExecutor
from .reward import create_default_calculator
from .features import build_all_features, compute_market_shape_features

'''
    交易環境
    
    Args:
        df: 交易數據
        initial_balance: 初始資金
        transaction_fee: 交易手續費
        window_size: 窗口大小(K線數量)
        leverage: 槓桿倍數
        min_balance: 最小資金(資金不足時強制結束)
        min_trade_qty: 最低交易數量(BTC)
        margin_mode: 保證金模式 (cross: 全倉, isolated: 逐倉)
        reward_weights: 獎勵權重
        reward_calculator: 獎勵計算器
        random_start: 是否隨機起點
        max_step_pos_change_pct: 單步最大倉位變化限制 (0.0 ~ 1.0, 相對 Max Capacity)
        turnover_penalty: 換手獎勵懲罰係數
'''
class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=10_000, transaction_fee=0.001, window_size=288, leverage=10, min_balance=100, min_trade_qty=0.001,
                 reward_weights=None, margin_mode: str = 'isolated', reward_calculator=None, random_start: bool = False, min_episode_steps: int = 1000,
                 min_position_change: float = 0.0, max_step_pos_change_pct: float = 1.0, turnover_penalty: float = 0.0,
                 dd_penalty_coef: float = 0.0, hold_bonus: float = 0.0, fee_limit_ratio: float = 0.08, fee_rolling_window: int = 3000,
                 fee_budget_penalty: float = 0.0,
                 flip_budget_max: float = 1.0, flip_cost: float = 0.25, flip_threshold: float = 0.2,
                 flip_recovery_rate: float = 0.01, flip_profit_recovery_rate: float = 0.1, stop_loss_atr: float = 2.5,
                 env_id: int = 0, step_log_enabled: bool = False, step_log_dir: str = "step_logs", step_log_every_n: int = 1):
        super(TradingEnvironment, self).__init__()
        
        # 只保留數值列，並確保包含必要的OHLCV列
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume', 'volume_ratio', 'long_short_ratio', 'trades', 'quote_volume']
        
        # 檢查必要列是否存在
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"數據缺少必要列: {missing_columns}")
        
        # 過濾數值列
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.df = df[numeric_columns].copy()  # 只保留數值列
        # 嘗試保留 datetime index 若存在
        if isinstance(df.index, pd.DatetimeIndex):
            self.df.index = df.index
        
        self.initial_balance = initial_balance  # 初始資金
        self.transaction_fee = transaction_fee  # 交易手續費
        self.window_size = window_size # 窗口大小(K線數量)
        self.leverage = leverage    #槓桿倍數
        self.min_balance = min_balance  # 最小資金(資金不足時強制結束)
        self.min_trade_qty = min_trade_qty  # 最低交易數量(BTC)
        self.margin_mode = str(margin_mode).lower()  # 保證金模式
        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.min_position_change = float(min_position_change)
        self.max_step_pos_change_pct = float(max_step_pos_change_pct)
        self.turnover_penalty = float(turnover_penalty)
        self.dd_penalty_coef = float(dd_penalty_coef)
        self.hold_bonus = float(hold_bonus)
        self.fee_limit_ratio = float(fee_limit_ratio)
        self.fee_rolling_window = int(fee_rolling_window)
        self.flip_budget_max = float(flip_budget_max)
        self.flip_cost = float(flip_cost)
        self.flip_threshold = float(flip_threshold)
        self.flip_recovery_rate = float(flip_recovery_rate)
        self.flip_profit_recovery_rate = float(flip_profit_recovery_rate)
        self.fee_budget_penalty = float(fee_budget_penalty)
        # Step logging (debug)
        self.env_id = int(env_id)
        self.step_log_enabled = bool(step_log_enabled)
        self.step_log_every_n = max(1, int(step_log_every_n))
        self.step_log_dir = step_log_dir
        self._step_log_path = None
        if self.step_log_enabled:
            env_dir = os.path.join(self.step_log_dir, f"env_{self.env_id:03d}")
            os.makedirs(env_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self._step_log_path = os.path.join(env_dir, f"steps_{timestamp}.jsonl")
        
        # 定義動作空間
        # 目標持倉比例 (-1.0 ~ 1.0)，限制小數位為1位
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # 1. 準備內部特徵（主要用於 step 邏輯，如 ATR 止損）
        self.feature_lookback = int(max(288, self.window_size))
        df_num = self.df
        
        close = df_num['close'].astype(np.float64)
        high = df_num['high'].astype(np.float64)
        low = df_num['low'].astype(np.float64)
        prev_close = close.shift(1)
        
        # ATR 計算 (用於止損)
        true_range = np.maximum.reduce([
            (high - low).values,
            np.abs(high - prev_close).fillna(0.0).values,
            np.abs(low - prev_close).fillna(0.0).values,
        ])
        atr = pd.Series(true_range, index=df_num.index).rolling(14, min_periods=5).mean().fillna(0.0)
        atr_ratio = (atr / np.clip(close, 1e-12, None)).astype(np.float32)
        
        # 保存內部使用的特徵 (供 step 使用)
        self.internal_features = pd.DataFrame({
            'atr_ratio': atr_ratio
        }, index=df_num.index)
        
        # 1.5 Pre-calculate All Features for State Vector to avoid slow lookup
        # This ensures all columns required by _get_observation (dollar_volume_log_z, amihud_z, etc.) are present.
        all_feats = build_all_features(self.df, lookback=self.feature_lookback)
        self.internal_features = pd.concat([self.internal_features, all_feats], axis=1)
        
        # Remove duplicate columns if any (keep first occurrence)
        self.internal_features = self.internal_features.loc[:, ~self.internal_features.columns.duplicated()]

        # 2. 準備觀測特徵 (price_seq)
        self.market_shape_df = compute_market_shape_features(self.df)
        self.price_seq_features = self.market_shape_df.shape[1]
        
        # 3. 定義觀察空間
        # price_seq: [window_size, F]
        # Split state_vector into 5 semantic groups:
        # 1. Account Status (11)
        # 2. Time Features (2)
        # 3. Market Rhythm (2)
        # 4. Cost/Risk State (6)
        # 5. Market State (6) -> Total 27
        state_dim = 27
        
        self.observation_space = spaces.Dict({
            'price_seq': spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.price_seq_features), dtype=np.float32),
            'account_state': spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32),
            'time_state': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'rhythm_state': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'cost_state': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            'market_state': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        })
        
        self.executor = TradeExecutor(
            initial_balance=self.initial_balance,
            fee_rate=self.transaction_fee,
            leverage=self.leverage,
            min_trade_qty=self.min_trade_qty,
            margin_mode=self.margin_mode,
            min_position_change=self.min_position_change,
            stop_loss_atr=stop_loss_atr
        )
        
        self.reward_calculator = reward_calculator or create_default_calculator(
            turnover_penalty=self.turnover_penalty,
            dd_penalty_coef=self.dd_penalty_coef,
            hold_bonus=self.hold_bonus,
            fee_budget_penalty=self.fee_budget_penalty
        )
        
        self.account_series = {
            'position': np.zeros(len(df)),
            'position_value': np.zeros(len(df)),
            'equity': np.zeros(len(df)),
            'wallet': np.zeros(len(df))
        }
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        # 隨機選擇起始點，但要保留窗口數據
        if self.random_start:
            # 預留 min_episode_steps 給 agent 跑
            max_start_index = len(self.df) - self.min_episode_steps - 2
            self.current_step = int(random.randint(self.window_size, max_start_index))
        else:
            self.current_step = self.window_size 

        self.executor.reset(self.initial_balance)
        self.balance = self.initial_balance
        self.btc_held = 0.0
        self.total_value = self.balance
        self.done = False
        self.position_holding_time = 0
        self._last_position_size = 0.0
        self.last_total_value = self.initial_balance
        self.episode_steps = 0
        
        # 重置追蹤變數
        self.max_equity_so_far = self.initial_balance
        self.last_trade_step = -999999
        self.trade_steps_buffer = []
        
        self.open_trades = []
        self.closed_trades = []
        self.last_position = 0
        self.avg_entry_price = 0
        
        self.episode_stop_loss_count = 0
        self.episode_liq_count = 0
        
        self.prev_total_fees = 0.0
        self.last_step_fee = 0.0
        
        # 初始化滾動手續費計算
        self.fee_history = deque() # (step_index, fee_amount)
        self.rolling_fee_sum = 0.0
        self.fee_limit_hit = False
        
        self.risk_budget = self.flip_budget_max
        self.last_equity_for_budget = self.initial_balance
        
        self.episode_start_step = int(self.current_step)
        self.episode_max_steps = max(0, (len(self.df) - 1) - self.episode_start_step)
        
        # 初始化 risk_base (每日更新一次的基準資金)
        self.daily_risk_base = self.initial_balance
        self.last_risk_base_update_step = self.current_step

        # 初始化帳戶序列 (填入初始值)
        current_price = float(self.df.iloc[self.current_step]['close'])
        self._update_account_series(current_price)

        return self._get_observation(), {}
    
    def _update_account_series(self, current_price):
        # 輔助函數：更新歷史序列
        pos_norm = self.executor.position.size / (self.initial_balance / current_price) if self.initial_balance > 0 and current_price > 0 else 0.0
        pos_value_norm = (self.executor.position.size * current_price) / self.initial_balance if self.initial_balance > 0 else 0.0
        equity_norm = self.executor.equity(current_price) / self.initial_balance if self.initial_balance > 0 else 0.0
        wallet_norm = self.executor.wallet_balance / self.initial_balance if self.initial_balance > 0 else 0.0
        
        if self.current_step < len(self.df):
            self.account_series['position'][self.current_step] = float(pos_norm)
            self.account_series['position_value'][self.current_step] = float(pos_value_norm)
            self.account_series['equity'][self.current_step] = float(equity_norm)
            self.account_series['wallet'][self.current_step] = float(wallet_norm)

    def _log_step(self, payload: dict):
        """將本步資訊以 JSONL 方式寫入檔案；僅除錯用。"""
        if not (self.step_log_enabled and self._step_log_path):
            return
        try:
            with open(self._step_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False))
                f.write("\n")
        except Exception:
            # 除錯log失敗不應影響訓練流程
            pass

    def _get_observation(self):
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
            
        # 1. Price Sequence (CNN Input)
        # Shape: [window_size, F]
        price_seq = self.market_shape_df.iloc[self.current_step - self.window_size : self.current_step].values.astype(np.float32)
        
        # 2. State Features Extraction
        current_price = float(self.df.iloc[self.current_step]['close'])
        equity = self.executor.equity(current_price)
        
        # --- Account Status (10) ---
        size = self.executor.position.size
        pos_side_oh = np.zeros(3, dtype=np.float32)
        if size > 0: pos_side_oh[0] = 1.0
        elif size < 0: pos_side_oh[1] = 1.0
        else: pos_side_oh[2] = 1.0
        
        pos_notional = abs(size) * current_price
        max_notional = equity * self.leverage
        pos_size_norm = pos_notional / max_notional if max_notional > 0 else 0.0
        pos_size_norm = np.clip(pos_size_norm, 0.0, 1.0)
        
        upnl = self.executor.unrealized_pnl(current_price)
        unreal_pnl_ratio = upnl / self.initial_balance if self.initial_balance > 0 else 0.0
        unreal_pnl_ratio = np.clip(unreal_pnl_ratio, -2.0, 2.0)
        
        equity_ratio = equity / self.initial_balance if self.initial_balance > 0 else 0.0
        equity_ratio = np.clip(equity_ratio, 0.0, 5.0)
        
        max_equity_ratio = self.max_equity_so_far / self.initial_balance if self.initial_balance > 0 else 0.0
        max_equity_ratio = np.clip(max_equity_ratio, 0.0, 5.0)
        
        dd = (self.max_equity_so_far - equity) / self.max_equity_so_far if self.max_equity_so_far > 0 else 0.0
        dd = np.clip(dd, 0.0, 1.0)
        
        # Maintenance Margin Ratio (Replaces Margin Ratio)
        # Maint Margin = Position Value * MMR
        # Ratio = Maint Margin / Equity
        # If Ratio >= 1.0 -> Liquidation
        maint_margin_ratio = 0.0
        leverage_ratio = 0.0
        if equity > 0 and abs(size) > 0:
            # Default MMR = 0.005 (0.5%) if not available
            mmr = getattr(self.executor, 'maintenance_margin_rate', 0.005)
            maint_margin = pos_notional * mmr
            maint_margin_ratio = maint_margin / equity
            leverage_ratio = pos_notional / equity
        elif equity <= 0:
            maint_margin_ratio = 1.1 # Already bust

        maint_margin_ratio = np.clip(maint_margin_ratio, 0.0, 1.1)
        leverage_ratio = np.clip(leverage_ratio, 0.0, 10.0)

        # Recent Performance
        profit_rate = (equity - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0.0
        profit_rate = np.clip(profit_rate, -1.0, 5.0)

        account_state = np.array([
            pos_size_norm,
            unreal_pnl_ratio,
            equity_ratio,
            max_equity_ratio,
            dd,
            maint_margin_ratio,
            profit_rate,
            self.executor.long_entry_count * 0.01,
            self.executor.short_entry_count * 0.01,
            self.episode_stop_loss_count * 0.1,
            self.episode_liq_count * 1.0
        ], dtype=np.float32)
        
        # --- Time Features (2) ---
        time_state = np.zeros(2, dtype=np.float32)
        if 'timestamp' in self.df.columns:
            ts = self.df.iloc[self.current_step]['timestamp']
            # Encode Hour of Day (Cyclical)
            hour = ts.hour
            time_state[0] = np.sin(2 * np.pi * hour / 24.0)
            time_state[1] = np.cos(2 * np.pi * hour / 24.0)
            
        # --- Market Rhythm (2) ---
        # Use pre-calculated internal features for speed
        rhythm_state = np.zeros(2, dtype=np.float32)
        try:
            idx = self.current_step
            rhythm_state[0] = self.internal_features['atr_ratio'].iloc[idx] if 'atr_ratio' in self.internal_features else 0.0
            # Add more rhythm features if needed (e.g. volume trend)
        except:
            pass
            
        # --- Cost/Risk State (2) ---
        cost_state = np.zeros(6, dtype=np.float32)
        step_fee_ratio_stable = np.clip((self.last_step_fee / self.initial_balance if self.initial_balance > 0 else 0.0), 0.0, 0.1)
        # Use rolling fee ratio instead of cumulative
        rolling_fee_ratio = 0.0
        if equity > 0:
            rolling_fee_ratio = self.rolling_fee_sum / equity
        rolling_fee_ratio = np.clip(rolling_fee_ratio, 0.0, 1.0)
        # Remaining fee budget ratio (1 means full budget; 0 means exceeded)
        safe_equity = max(equity, self.initial_balance * 0.5)
        limit_amount = max(1e-8, safe_equity * self.fee_limit_ratio)
        remaining_fee_budget_ratio = 1.0 - (self.rolling_fee_sum / limit_amount)
        remaining_fee_budget_ratio = float(np.clip(remaining_fee_budget_ratio, 0.0, 1.0))

        cost_state[0] = step_fee_ratio_stable
        cost_state[1] = rolling_fee_ratio
        cost_state[2] = maint_margin_ratio
        cost_state[3] = dd
        cost_state[4] = leverage_ratio
        cost_state[5] = remaining_fee_budget_ratio
        
        # --- Market State (6) ---
        # [dollar_volume_log_z, amihud_z, parkinson_vol_z, kyle_lambda_z, vpin_z, trade_entropy_z]
        market_cols = ['dollar_volume_log_z', 'amihud_z', 'parkinson_vol_z', 'kyle_lambda_z', 'vpin_z', 'trade_entropy_z']
        market_vals = []
        for c in market_cols:
            val = 0.0
            if c in self.internal_features.columns:
                 val = self.internal_features[c].iloc[self.current_step]
            market_vals.append(val)
        market_state = np.array(market_vals, dtype=np.float32)
        
        return {
            'price_seq': price_seq,
            'account_state': account_state,
            'time_state': time_state,
            'rhythm_state': rhythm_state,
            'cost_state': cost_state,
            'market_state': market_state
        }

    def step(self, action):
        # 限制動作範圍
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 取得當前市場數據
        current_data = self.df.iloc[self.current_step]
        current_price = float(current_data['close'])
        current_high = float(current_data['high'])
        current_low = float(current_data['low'])
        
        # 更新 risk_base（每日/每一定期間更新一次基準，用於動態風控）
        risk_base = self.daily_risk_base

        # 計算當前權益
        last_equity = self.executor.equity(current_price)
        
        # Flip Strategy Check (反手限制邏輯)
        target_pos_pct = float(action[0])
        current_pos_pct = 0.0
        max_nominal = last_equity * self.leverage
        if max_nominal > 0:
            current_pos_val = self.executor.position.size * current_price
            current_pos_pct = current_pos_val / max_nominal
            
        # Check for sign flip (Long <-> Short)
        is_flip = (target_pos_pct * current_pos_pct < -0.01) # Crossing zero significantly
        flip_blocked = False
        flip_budget_spent = 0.0
        
        if is_flip:
             # Check budget
             cost = self.flip_cost
             if abs(target_pos_pct - current_pos_pct) > self.flip_threshold: # Large flip
                 pass # standard cost
             
             if self.risk_budget >= cost:
                 self.risk_budget -= cost
                 flip_budget_spent = cost
             else:
                 # Budget exhausted, try partial flip or reduce size
                 # Allow flipping but cap the target size to what remaining budget allows?
                 # Simplification: If budget is low, force a smaller target in the new direction 
                 # instead of blocking completely.
                 
                 # Assume cost is proportional to flip size? 
                 # Current model: Fixed cost "flip_cost" per flip event.
                 # If we want to allow partial flip, we need to define "partial cost" or allow flip
                 # but with penalty.
                 
                 # New Logic: Allow flip, but limit the TARGET SIZE in the new direction
                 # to be very small (e.g. 10% of intended) if budget is low?
                 # Or just force target to 0.0 (Close) as before?
                 
                 # Let's try: Force target to be sign(target) * min(abs(target), 0.1)
                 # Effectively capping the new position size to 10% leverage equivalent if budget empty.
                 # This allows direction change but prevents aggressive betting when "tired".
                 
                 # BUT, for simplicity and robustness based on request #2:
                 # We want to avoid "Deadlock" (Forced to 0 forever).
                 # Let's allow a small flip.
                 
                 target_sign = np.sign(target_pos_pct)
                 # Allow at least 0.05 (5%) position in new direction
                 safe_flip_limit = 0.05 
                 
                 if abs(target_pos_pct) > safe_flip_limit:
                     target_pos_pct = target_sign * safe_flip_limit
                     flip_blocked = True # Mark as blocked/limited
                 
                 # Still consume whatever small budget or set to 0?
                 # Let's just say no cost charged if we are forced to limit?
                 # Or charge what we have.
                 self.risk_budget = max(0.0, self.risk_budget - 0.1 * cost) # Penalize slightly
        
        # Convert target percent to concrete position change
        # Action is target position % (-1.0 ~ 1.0) of Max Capacity (Equity * Leverage)
        # But we need to apply min_position_change and max_step_pos_change
        
        # 1. Calculate Desired Position Size (BTC)
        desired_notional = target_pos_pct * last_equity * self.leverage
        desired_size = desired_notional / current_price if current_price > 0 else 0.0
        
        current_size = float(self.executor.position.size)
        
        # 2. Apply Step Change Limit
        # Max change in BTC
        max_change_qty = (risk_base * self.leverage * self.max_step_pos_change_pct) / current_price
        
        # 2.1 Check if we are reducing risk (Closing position)
        # If abs(desired_size) < abs(current_size) and sign matches or zero, we are closing.
        # Relax limit for closing to allow panic exit.
        is_closing = (abs(desired_size) < abs(current_size)) and (desired_size * current_size >= 0)
        
        change = desired_size - current_size
        
        # Apply limit only if NOT closing (or if opening direction)
        if not is_closing:
            if abs(change) > max_change_qty:
                change = np.sign(change) * max_change_qty
                desired_size = current_size + change
            
        # 3. Apply Min Position Change (Deadband)
        # If change is too small relative to capacity, ignore it (hold)
        # EXCEPTION: If closing to zero, allow it regardless of size to clear dust.
        is_close_to_zero = (abs(desired_size) < 1e-8)
        
        if not is_close_to_zero:
             # Check deadband for normal adjustments
             pass # TradeExecutor handles basic min_qty, but we can filter here too if needed.
             # Currently relying on TradeExecutor's min_trade_qty or min_position_change if set there.
             # But wait, logic for ignoring small updates usually happens here to save fees.
             # Let's keep simple: if change is tiny and not closing to zero, ignore.
             pass
        
        # Convert back to % for final command
        final_action = 0.0
        max_cap = last_equity * self.leverage
        if max_cap > 0:
            final_action = (desired_size * current_price) / max_cap
        
        position_percent = float(final_action)
        # ----------------------------------------------

        # 估算當前 ATR（用於止損計算）
        atr_ratio = float(self.internal_features['atr_ratio'].iloc[self.current_step - 1]) if self.current_step > 0 else 0.02
        atr_est = atr_ratio * current_price

        # 檢查是否需要更新 daily_risk_base
        if (self.current_step - self.last_risk_base_update_step) >= self.window_size:
            self.daily_risk_base = float(self.executor.wallet_balance)
            self.last_risk_base_update_step = self.current_step

        # 執行交易
        prev_wallet_balance = float(self.executor.wallet_balance)
        prev_fees = float(self.executor.total_fees)
        
        self.executor.execute(
            position_percent=position_percent,
            current_price=current_price,
            high=current_high,
            low=current_low,
            equity=last_equity,
            atr=atr_est,
            risk_base=self.daily_risk_base
        )

        # Update fee tracking (Rolling Window)
        current_fees = float(self.executor.total_fees)
        step_fee = current_fees - prev_fees
        self.last_step_fee = step_fee
        self.prev_total_fees = current_fees
        
        # Maintain rolling window
        self.fee_history.append((self.current_step, step_fee))
        self.rolling_fee_sum += step_fee
        
        # Pop old fees
        while self.fee_history and (self.current_step - self.fee_history[0][0]) > self.fee_rolling_window:
            _, old_fee = self.fee_history.popleft()
            self.rolling_fee_sum -= old_fee
            
        # Check Fee Limit against Current Equity
        # Use max(equity, initial_balance) to avoid instant death if equity drops slightly, 
        # or just equity to enforce survival? 
        # Let's use current equity to prevent churning when broke.
        # FIX #4: Use Floor to prevent fee limit trap at low equity
        safe_equity = max(self.executor.equity(current_price), self.initial_balance * 0.5)
        limit_amount = safe_equity * self.fee_limit_ratio
        
        self.fee_limit_hit = self.rolling_fee_sum >= limit_amount

        # 同步帳戶狀態
        new_equity = self.executor.equity(current_price)
        self.balance = self.executor.wallet_balance
        self.btc_held = self.executor.position.size
        self.total_value = new_equity
        
        # Risk budget recovery (time-based + profit-based)
        self.risk_budget = min(self.flip_budget_max, self.risk_budget + self.flip_recovery_rate)
        if new_equity > self.last_equity_for_budget:
            gain_ratio = (new_equity - self.last_equity_for_budget) / max(1.0, self.initial_balance)
            self.risk_budget = min(self.flip_budget_max, self.risk_budget + gain_ratio * self.flip_profit_recovery_rate)
        self.last_equity_for_budget = new_equity
        
        # Update Max Equity
        if new_equity > self.max_equity_so_far:
            self.max_equity_so_far = new_equity
            
        realized_pnl_step = float(self.executor.wallet_balance - prev_wallet_balance)

        # Current drawdown after updating max_equity_so_far
        current_dd = 0.0
        if self.max_equity_so_far > 0:
            current_dd = (self.max_equity_so_far - new_equity) / self.max_equity_so_far
            current_dd = float(np.clip(current_dd, 0.0, 1.0))

        # 計算保證金緩衝
        margin_buffer = 1.0
        leverage_ratio = 0.0
        try:
            position_value = abs(float(self.executor.position.size * current_price))
            if new_equity > 0:
                leverage_ratio = position_value / new_equity
                safe_leverage = float(self.leverage) * 0.8
                if leverage_ratio >= safe_leverage:
                    margin_buffer = 0.0
                else:
                    margin_buffer = 1.0 - (leverage_ratio / safe_leverage)
                margin_buffer = float(np.clip(margin_buffer, 0.0, 1.0))
        except Exception:
            margin_buffer = 1.0
        
        # 計算倉位變動（換手） 
        position_change = abs(float(self.executor.position.size - self._last_position_size))
        traded = position_change > 1e-8
        
        is_risk_reducing = False
        if abs(self.executor.position.size) < abs(self._last_position_size) - 1e-8:
            is_risk_reducing = True
        
        if traded:
            self.last_trade_step = self.current_step
            self.trade_steps_buffer.append(self.current_step)
            
        turnover_ratio = 0.0
        try:
            notional_change = position_change * current_price
            if new_equity > 0:
                turnover_ratio = float(abs(notional_change) / new_equity)
        except Exception:
            turnover_ratio = 0.0
            
        self._last_position_size = float(self.executor.position.size)

        # 計算未實現損益
        unrealized_pnl = float(self.executor.unrealized_pnl(current_price))
        has_position = abs(self.executor.position.size) > 1e-8
        
        stop_loss_triggered = self.executor.stop_loss_triggered
        if stop_loss_triggered:
            self.episode_stop_loss_count += 1

        # 計算結構性指標 (for reward)
        dist_to_extreme_atr = None
        mae_atr = None
        try:
            start_idx = max(0, self.current_step - self.window_size)
            window_high = float(np.max(self.df['high'].iloc[start_idx:self.current_step]))
            window_low = float(np.min(self.df['low'].iloc[start_idx:self.current_step]))
            
            atr_est = max(1e-8, atr_est)
            if has_position:
                if self.executor.position.size > 0:
                    dist = max(0.0, window_high - current_price)
                    adverse_move = max(0.0, float(self.executor.position.entry_price) - current_low)
                else:
                    dist = max(0.0, current_price - window_low)
                    adverse_move = max(0.0, current_high - float(self.executor.position.entry_price))
                dist_to_extreme_atr = float(dist / atr_est)
                mae_atr = float(adverse_move / atr_est)
                position_value = abs(float(self.executor.position.size * current_price))
                leverage_ratio = float(position_value / new_equity) if new_equity > 0 else 0.0
        except Exception:
            pass
        
        # Calculate Position Change Norm for Reward/Cost
        max_capacity_qty = (risk_base * self.leverage) / current_price if current_price > 0 else 1.0
        position_change_norm = abs(position_change) / max_capacity_qty if max_capacity_qty > 0 else 0.0

        # 檢查結束條件 (先算好以便 reward 塑形)
        data_exhausted = self.current_step >= len(self.df) - 1
        balance_insufficient = new_equity <= self.min_balance
        liq_triggered = self.executor.liq_triggered
        stop_loss_hit = stop_loss_triggered

        if liq_triggered:
            self.episode_liq_count += 1

        self.done = data_exhausted or balance_insufficient or liq_triggered or self.fee_limit_hit

        termination_reason = None
        if self.fee_limit_hit:
            termination_reason = 'fee_limit'
        elif data_exhausted:
            termination_reason = 'data_exhausted'
        elif liq_triggered:
            termination_reason = 'liq_triggered'
        elif balance_insufficient:
            termination_reason = 'balance_insufficient'

        # Remaining fee budget ratio for reward shaping
        safe_equity_for_limit = max(new_equity, self.initial_balance * 0.5)
        limit_amount = max(1e-8, safe_equity_for_limit * self.fee_limit_ratio)
        remaining_fee_budget_ratio = 1.0 - (self.rolling_fee_sum / limit_amount)
        remaining_fee_budget_ratio = float(np.clip(remaining_fee_budget_ratio, 0.0, 1.0))

        step_fee_ratio = step_fee / self.initial_balance if self.initial_balance > 0 else 0.0

        # 回饋
        reward = self.reward_calculator.compute(
            last_equity=last_equity,
            new_equity=new_equity,
            margin_buffer=margin_buffer,
            position_change=position_change,
            position_change_norm=position_change_norm,
            turnover_ratio=turnover_ratio,
            dist_to_extreme_atr=dist_to_extreme_atr,
            mae_atr=mae_atr,
            leverage_ratio=leverage_ratio,
            has_position=has_position,
            unrealized_pnl=unrealized_pnl,
            traded=traded,
            realized_pnl_step=realized_pnl_step,
            episode_steps=self.episode_steps,
            episode_max_steps=self.episode_max_steps,
            stop_loss_triggered=stop_loss_triggered,
            done=self.done,
            termination_reason=termination_reason,
            step_fee_ratio=step_fee_ratio,
            current_dd=current_dd,
            fee_budget_ratio=remaining_fee_budget_ratio
        )

        # 更新帳戶狀態時間序列
        self._update_account_series(current_price)

        # 更新步驟
        self.current_step += 1
        self.episode_steps += 1

        info = {}
        if stop_loss_hit:
            info['stop_loss_triggered'] = True

        # Cost Calculation Helper Info
        info['equity'] = float(new_equity)
        info['maintenance_margin'] = 0.0
        if abs(self.executor.position.size) > 0:
             mmr = self.executor.maintenance_margin_rate
             pos_val = abs(self.executor.position.size * current_price)
             info['maintenance_margin'] = pos_val * mmr
        info['liq_triggered'] = liq_triggered
        
        # Cost 3 Info: Fee Risk
        info['step_fee_ratio'] = step_fee_ratio
        
        # Other info for debug/analysis
        info['position_change_norm'] = position_change_norm
        info['is_risk_reducing'] = is_risk_reducing
        info['fee_limit_hit'] = bool(self.fee_limit_hit)
        info['flip_blocked'] = flip_blocked
        info['flip_budget_spent'] = flip_budget_spent
        info['risk_budget'] = float(self.risk_budget)
        info['current_dd'] = current_dd
        info['remaining_fee_budget_ratio'] = remaining_fee_budget_ratio

        maint_margin_ratio_log = 0.0
        if abs(self.executor.position.size) > 0:
            mmr = self.executor.maintenance_margin_rate
            pos_val = abs(self.executor.position.size * current_price)
            if new_equity > 0:
                maint_margin_ratio_log = (pos_val * mmr) / new_equity
        maint_margin_ratio_log = float(np.clip(maint_margin_ratio_log, 0.0, 2.0))

        # 逐步記錄（可依頻率或關鍵事件寫入）
        should_log_step = self.step_log_enabled and (
            (self.episode_steps % self.step_log_every_n == 0) or
            self.done or stop_loss_triggered or liq_triggered
        )
        if should_log_step:
            step_payload = {
                "env_id": self.env_id,
                "global_step": int(self.current_step),
                "episode_step": int(self.episode_steps),
                "price": current_price,
                "action_raw": float(action[0]),
                "target_pos_pct": float(target_pos_pct),
                "final_pos_pct": float(position_percent),
                "position_size": float(self.executor.position.size),
                "equity_before": float(last_equity),
                "equity_after": float(new_equity),
                "wallet": float(self.executor.wallet_balance),
                "unrealized_pnl": float(unrealized_pnl),
                "realized_pnl_step": float(realized_pnl_step),
                "turnover_ratio": float(turnover_ratio),
                "position_change_norm": float(position_change_norm),
                "fees_step": float(step_fee),
                "fees_total": float(self.executor.total_fees),
                "rolling_fee_sum": float(self.rolling_fee_sum),
                "fee_limit_hit": bool(self.fee_limit_hit),
                "maint_margin_ratio": maint_margin_ratio_log,
                "leverage_ratio": float(leverage_ratio),
                "current_dd": float(current_dd),
                "stop_loss_triggered": bool(stop_loss_triggered),
                "liq_triggered": bool(liq_triggered),
                "risk_budget": float(self.risk_budget),
                "flip_blocked": bool(flip_blocked),
                "reward": float(reward),
                "done": bool(self.done),
                "termination_reason": termination_reason if termination_reason else "",
            }
            self._log_step(step_payload)

        if self.done:
            if termination_reason:
                info['termination_reason'] = termination_reason
            else:
                info['termination_reason'] = 'other'
            
            info['final_balance'] = float(new_equity)
            info['profit'] = float(new_equity - self.initial_balance)
            info['profit_rate'] = float((info['profit'] / self.initial_balance) * 100) if self.initial_balance > 0 else 0.0
            
            try:
                info['long_close_count'] = int(self.executor.long_close_count)
                info['short_close_count'] = int(self.executor.short_close_count)
                info['total_fees'] = float(self.executor.total_fees)
                info['episode_steps'] = int(self.episode_steps)
                info['long_entry_count'] = int(self.executor.long_entry_count)
                info['short_entry_count'] = int(self.executor.short_entry_count)
                info['episode_max_steps'] = int(self.episode_max_steps)
                info['data_len'] = int(len(self.df))
                info['window_size'] = int(self.window_size)
                info['episode_stop_loss_count'] = int(self.episode_stop_loss_count)
                info['episode_liq_count'] = int(self.episode_liq_count)
                info['max_single_trade_loss_pct'] = float(self.executor.max_trade_loss_pct)
            except Exception:
                pass
            
        return self._get_observation(), reward, self.done, False, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass
