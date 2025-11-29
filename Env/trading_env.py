import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import random
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
'''
class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=10_000, transaction_fee=0.001, window_size=288, leverage=10, min_balance=100, min_trade_qty=0.001,
                 reward_weights=None, margin_mode: str = 'isolated', reward_calculator=None, random_start: bool = False, min_episode_steps: int = 1000):
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

        # 2. 準備觀測特徵 (price_seq)
        self.market_shape_df = compute_market_shape_features(self.df)
        self.price_seq_features = self.market_shape_df.shape[1]
        
        # 3. 定義觀察空間
        # price_seq: [window_size, F]
        # state_vector: [D]
        self.state_dim = 20  # 固定長度
        
        self.observation_space = spaces.Dict({
            "price_seq": spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.window_size, self.price_seq_features), 
                dtype=np.float32
            ),
            "state_vector": spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.state_dim,), 
                dtype=np.float32
            )
        })
        
        # 建立交易執行器
        self.executor = TradeExecutor(
            initial_balance=self.initial_balance,
            fee_rate=self.transaction_fee,
            leverage=self.leverage,
            min_trade_qty=self.min_trade_qty,
            margin_mode=self.margin_mode,
            stop_loss_atr=2.5,
        )

        # 帳戶狀態時間序列（逐筆滾動保存，保留用於後續分析或 debug）
        series_len = len(self.df)
        self.account_series = {
            'position': np.zeros(series_len, dtype=np.float32),
            'position_value': np.zeros(series_len, dtype=np.float32),
            'equity': np.zeros(series_len, dtype=np.float32),
            'wallet': np.zeros(series_len, dtype=np.float32),
        }

        self.reward_calculator = reward_calculator or create_default_calculator()
        self._last_position_size = 0.0
        
        # State tracking variables
        self.max_equity_so_far = self.initial_balance
        self.last_trade_step = -999999
        self.trade_steps_buffer = [] # 存儲最近交易的 step

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.random_start and len(self.df) > (self.window_size + self.min_episode_steps + 2):
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
        
        self.episode_start_step = int(self.current_step)
        self.episode_max_steps = max(0, (len(self.df) - 1) - self.episode_start_step)
        
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

    def _get_observation(self):
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
            
        # 1. Price Sequence (CNN Input)
        # Shape: [window_size, F]
        price_seq = self.market_shape_df.iloc[self.current_step - self.window_size : self.current_step].values.astype(np.float32)
        
        # 2. State Vector (MLP Input)
        # Gather features
        current_price = float(self.df.iloc[self.current_step]['close'])
        equity = self.executor.equity(current_price)
        
        # Pos Side One-Hot (3)
        size = self.executor.position.size
        pos_side_oh = np.zeros(3, dtype=np.float32)
        if size > 0: pos_side_oh[0] = 1.0
        elif size < 0: pos_side_oh[1] = 1.0
        else: pos_side_oh[2] = 1.0
        
        # Pos Size Norm (1) - Normalized by Max Leverage
        # size / (equity / current_price * leverage) approximately
        # Simply: current leverage usage / max leverage
        # 1.0 means full leverage used
        pos_notional = abs(size) * current_price
        max_notional = equity * self.leverage
        pos_size_norm = pos_notional / max_notional if max_notional > 0 else 0.0
        pos_size_norm = np.clip(pos_size_norm, 0.0, 1.0)
        
        # Unreal PnL Ratio (1)
        upnl = self.executor.unrealized_pnl(current_price)
        unreal_pnl_ratio = upnl / self.initial_balance if self.initial_balance > 0 else 0.0
        # Clip to reasonable range [-1, 1] or larger? Let's do [-2, 2]
        unreal_pnl_ratio = np.clip(unreal_pnl_ratio, -2.0, 2.0)
        
        # Equity Ratio (1)
        equity_ratio = equity / self.initial_balance if self.initial_balance > 0 else 0.0
        equity_ratio = np.clip(equity_ratio, 0.0, 5.0) # Cap at 5x
        
        # Max Equity Ratio (1)
        max_equity_ratio = self.max_equity_so_far / self.initial_balance if self.initial_balance > 0 else 0.0
        max_equity_ratio = np.clip(max_equity_ratio, 0.0, 5.0)
        
        # Drawdown (1)
        dd = (self.max_equity_so_far - equity) / self.max_equity_so_far if self.max_equity_so_far > 0 else 0.0
        dd = np.clip(dd, 0.0, 1.0)
        
        # Margin Ratio (1) - Normalized by Max Leverage
        # effective_leverage = pos_notional / equity
        # margin_ratio = effective_leverage / self.leverage
        # This is actually same as pos_size_norm mathematically if we use equity as base
        # But let's keep it distinct if the previous logic was different.
        # Previous: position_value / equity. This is effective leverage (e.g., 5.0)
        # Normalized: 5.0 / 10.0 = 0.5
        if equity > 0:
            effective_leverage = pos_notional / equity
            margin_ratio = effective_leverage / self.leverage
        else:
            margin_ratio = 1.0 # Max danger
        margin_ratio = np.clip(margin_ratio, 0.0, 1.0)
        
        # Steps Since Last Trade Norm (1)
        steps_since = self.current_step - self.last_trade_step
        steps_since_norm = steps_since / self.window_size # Normalize by window
        steps_since_norm = np.clip(steps_since_norm, 0.0, 1.0) # Clip to 1 window length
        
        # Trade Count Recent Norm (1)
        # Count trades in last window_size steps
        recent_threshold = self.current_step - self.window_size
        recent_trades = [t for t in self.trade_steps_buffer if t > recent_threshold]
        trade_count_norm = len(recent_trades) / 20.0 # Assume 20 is high activity
        trade_count_norm = np.clip(trade_count_norm, 0.0, 1.0)
        
        # Est Cost Per Unit (1) - Log deviation
        # log(current / entry)
        entry_price = self.executor.position.entry_price
        if size != 0 and entry_price > 0:
            est_cost_norm = np.log(current_price / entry_price)
            est_cost_norm = np.clip(est_cost_norm, -0.5, 0.5) # +/- 50% move
        else:
            est_cost_norm = 0.0
        
        # Time of Day (1)
        # Assuming 5m candles -> 288 per day
        time_of_day = (self.current_step % 288) / 288.0
        
        # Day of Week One-Hot (7)
        day_of_week_oh = np.zeros(7, dtype=np.float32)
        if isinstance(self.df.index, pd.DatetimeIndex):
            day_idx = self.df.index[self.current_step].dayofweek
            day_of_week_oh[day_idx] = 1.0
        else:
            # Fallback: approximate if not datetime index
            approx_day = (self.current_step // 288) % 7
            day_of_week_oh[approx_day] = 1.0
            
        # Concatenate all
        state_vector = np.concatenate([
            pos_side_oh,                    # 3
            [pos_size_norm],                # 1
            [unreal_pnl_ratio],             # 1
            [equity_ratio],                 # 1
            [max_equity_ratio],             # 1
            [dd],                           # 1
            [margin_ratio],                 # 1
            [steps_since_norm],             # 1
            [trade_count_norm],             # 1
            [est_cost_norm],                # 1
            [time_of_day],                  # 1
            day_of_week_oh                  # 7
        ]).astype(np.float32)
        
        # Ensure size matches self.state_dim (20)
        if len(state_vector) != self.state_dim:
             # Padding or Truncating if logic changes, but currently it sums to 20.
             pass

        obs = {
            "price_seq": np.nan_to_num(price_seq, nan=0.0),
            "state_vector": np.nan_to_num(state_vector, nan=0.0)
        }
        
        return obs
    
    def step(self, action):
        '''
        執行交易步驟
        '''
        # 取得當前K線
        candle = self.df.iloc[self.current_step]
        current_price = float(candle['close'])
        current_high = float(candle['high'])
        current_low = float(candle['low'])
        last_equity = self.executor.equity(current_price)
        position_percent = float(action)

        # 估算當前 ATR（用於止損計算） - 使用 internal_features
        atr_ratio = float(self.internal_features['atr_ratio'].iloc[self.current_step - 1]) if self.current_step > 0 else 0.02
        atr_est = atr_ratio * current_price

        # 執行交易
        prev_wallet_balance = float(self.executor.wallet_balance)
        self.executor.execute(
            position_percent=position_percent,
            current_price=current_price,
            high=current_high,
            low=current_low,
            equity=last_equity,
            atr=atr_est,
        )

        # 同步帳戶狀態
        new_equity = self.executor.equity(current_price)
        self.balance = self.executor.wallet_balance
        self.btc_held = self.executor.position.size
        self.total_value = new_equity
        
        # Update Max Equity
        if new_equity > self.max_equity_so_far:
            self.max_equity_so_far = new_equity
            
        realized_pnl_step = float(self.executor.wallet_balance - prev_wallet_balance)

        # 計算保證金緩衝
        margin_buffer = 1.0
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
        
        # Update Trade History
        if traded:
            self.last_trade_step = self.current_step
            self.trade_steps_buffer.append(self.current_step)
        
        # Clean buffer (keep only recent for observation calculation efficiency, though obs re-filters)
        # To be safe, we can clean up very old ones occasionally or just keep them if episode is not infinite.
        # Given episode length max is data length, list append is fine.
        
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
        
        # 取得止損觸發狀態
        stop_loss_triggered = self.executor.stop_loss_triggered
        if stop_loss_triggered:
            self.episode_stop_loss_count += 1

        # 計算結構性指標 (for reward)
        dist_to_extreme_atr = None
        mae_atr = None
        leverage_ratio = None
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

        # 回饋
        reward = self.reward_calculator.compute(
            last_equity=last_equity,
            new_equity=new_equity,
            margin_buffer=margin_buffer,
            position_change=position_change,
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
        )

        # 更新帳戶狀態時間序列
        self._update_account_series(current_price)

        # 更新步驟
        self.current_step += 1
        self.episode_steps += 1

        # 檢查結束條件
        data_exhausted = self.current_step >= len(self.df) - 1
        balance_insufficient = new_equity <= self.min_balance
        liq_triggered = self.executor.liq_triggered
        stop_loss_hit = stop_loss_triggered
        
        if liq_triggered:
            self.episode_liq_count += 1

        self.done = data_exhausted or balance_insufficient or liq_triggered
        
        info = {}
        if stop_loss_hit:
            info['stop_loss_triggered'] = True

        # Cost Calculation Helper Info
        # Margin Ratio: Equity / Maintenance Margin
        # Maintenance Margin = |size| * price * mmr
        info['equity'] = float(new_equity)
        info['maintenance_margin'] = 0.0
        if abs(self.executor.position.size) > 0:
             mmr = self.executor.maintenance_margin_rate
             pos_val = abs(self.executor.position.size * current_price)
             info['maintenance_margin'] = pos_val * mmr
        info['liq_triggered'] = liq_triggered
        
        if self.done:
            if data_exhausted:
                info['termination_reason'] = 'data_exhausted'
                #print(f"Episode結束：數據用完 (step={self.current_step}, data_len={len(self.df)})")
            elif liq_triggered:
                info['termination_reason'] = 'liq_triggered'
                #print(f"Episode結束：強平 (balance={self.balance:.2f})")
            elif balance_insufficient:
                info['termination_reason'] = 'balance_insufficient'
                #print(f"Episode結束：資金不足 (balance={self.balance:.2f}, min={self.min_balance})")
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
            except Exception:
                pass
            
            reward = self.reward_calculator.compute(
                last_equity=last_equity,
                new_equity=new_equity,
                done=True,
                termination_reason=info['termination_reason'],
                realized_pnl_step=realized_pnl_step,
                episode_steps=self.episode_steps,
                episode_max_steps=self.episode_max_steps,
                margin_buffer=margin_buffer,
                dist_to_extreme_atr=dist_to_extreme_atr,
                mae_atr=mae_atr,
                stop_loss_triggered=stop_loss_triggered,
            )
        
        return self._get_observation(), reward, self.done, False, info
