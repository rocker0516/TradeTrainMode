import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import talib

class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_fee=0.001, window_size= 24 * 60 //5, leverage = 10, min_balance=0):
        super(TradingEnvironment, self).__init__()
        
        self.df = df    #資料集
        self.initial_balance = initial_balance  # 初始資金
        self.transaction_fee = transaction_fee  # 交易手續費
        self.window_size = window_size # 窗口大小(K線數量)
        self.leverage = leverage    #槓桿倍數
        self.min_balance = min_balance  # 最小資金(資金不足時強制結束)
        self.stop_loss_price = 0.0    # 止損價格
        self.take_profit_price = 0.0    # 止盈價格
        self.entry_price = 0.0      # 平均進場價格
        self.margin_used = 0.0      # 已用保證金
        self.long_trades = 0
        self.short_trades = 0

        # 定義動作空間
        # [交易方向(-1:做空, 1:做多), 倉位百分比(0-1), 止盈比例(0.2-10%), 止損比例(0.1-30%)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.2, 0.1]),  
            high=np.array([1.0, 1.0, 10.0, 30.0]), #(動作方向, 倉位大小, 止盈%, 止損%)
            dtype=np.float32
        )
        
        # 計算特徵數量
        # 原始價格特徵 + 賬戶狀態(5個) + 時間特徵(5個) + 市場情緒特徵(8個)
        self.n_features = len(df.columns) + 5 + 5 + 8
        
        # 定義觀察空間
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.n_features, window_size),  # 所有特徵 + 賬戶狀態
            dtype=np.float32
        )
        
        self.reset()
    
    def _calculate_market_sentiment_features(self, window_data):
        """計算市場情緒特徵"""
        close_prices = window_data['close'].values.astype(np.float64)
        high_prices = window_data['high'].values.astype(np.float64)
        low_prices = window_data['low'].values.astype(np.float64)
        volume = window_data['volume'].values.astype(np.float64) if 'volume' in window_data.columns else np.ones_like(close_prices)
        
        # 創建特徵數組
        sentiment_features = np.zeros((8, len(close_prices)), dtype=np.float32)
        
        # 1. RSI (相對強弱指數) - 14期
        if len(close_prices) >= 14:
            rsi = talib.RSI(close_prices, timeperiod=14)
            sentiment_features[0] = np.nan_to_num(rsi, nan=50.0) / 100.0  # 標準化到0-1
        
        # 2. MACD 信號
        if len(close_prices) >= 26:
            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            # 將 macd_hist 標準化，使其成為與價格相關的相對值
            safe_close_prices = np.copy(close_prices)
            safe_close_prices[safe_close_prices == 0] = 1 # 避免除以零
            normalized_macd_hist = macd_hist / safe_close_prices
            sentiment_features[1] = np.nan_to_num(normalized_macd_hist, nan=0.0)
        
        # 3. 布林帶位置 (價格在布林帶中的相對位置)
        if len(close_prices) >= 20:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20)
            bb_range = bb_upper - bb_lower
            
            # 建立一個預設值為0.5的數組 (代表價格在中間)
            bb_position = np.full_like(bb_range, 0.5, dtype=np.float64)
            
            # 僅在布林帶寬度不為零時，才進行除法運算
            non_zero_mask = bb_range != 0
            bb_position[non_zero_mask] = (close_prices[non_zero_mask] - bb_lower[non_zero_mask]) / bb_range[non_zero_mask]
            
            sentiment_features[2] = np.nan_to_num(bb_position, nan=0.5)
        
        # 4. 價格變化率 (ROC) - 10期
        if len(close_prices) >= 10:
            roc = talib.ROC(close_prices, timeperiod=10)
            sentiment_features[3] = np.nan_to_num(roc, nan=0.0) / 100.0  # 標準化
        
        # 5. 成交量變化率
        if len(volume) >= 10:
            volume_roc = talib.ROC(volume, timeperiod=10)
            sentiment_features[4] = np.nan_to_num(volume_roc, nan=0.0) / 100.0
        
        # 6. ATR (平均真實波動幅度) - 標準化
        if len(close_prices) >= 14:
            atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            atr_pct = atr / close_prices  # 轉換為百分比
            sentiment_features[5] = np.nan_to_num(atr_pct, nan=0.01)
        
        # 7. 威廉姆斯%R
        if len(close_prices) >= 14:
            williams_r = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
            sentiment_features[6] = np.nan_to_num(williams_r, nan=-50.0) / -100.0  # 標準化到0-1
        
        # 8. 隨機指標 %K
        if len(close_prices) >= 14:
            slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices, 
                                     fastk_period=14, slowk_period=3, slowd_period=3)
            sentiment_features[7] = np.nan_to_num(slowk, nan=50.0) / 100.0  # 標準化到0-1
        
        return sentiment_features
    
    def _calculate_time_features(self, window_data):
        """計算時間特徵"""
        # 確保索引是 datetime 類型
        if not isinstance(window_data.index, pd.DatetimeIndex):
            # 如果沒有時間索引，創建一個虛擬的時間序列
            timestamps = pd.date_range(start='2023-01-01', periods=len(window_data), freq='5T')
        else:
            timestamps = window_data.index
        
        time_features = np.zeros((5, len(window_data)), dtype=np.float32)
        
        # 1. 小時 (0-23) -> 標準化到 0-1
        time_features[0] = timestamps.hour / 23.0
        
        # 2. 分鐘 (0-59) -> 標準化到 0-1  
        time_features[1] = timestamps.minute / 59.0
        
        # 3. 星期幾 (0-6) -> 標準化到 0-1
        time_features[2] = timestamps.dayofweek / 6.0
        
        # 4. 月份中的第幾天 (1-31) -> 標準化到 0-1
        time_features[3] = (timestamps.day - 1) / 30.0
        
        # 5. 月份 (1-12) -> 標準化到 0-1
        time_features[4] = (timestamps.month - 1) / 11.0
        
        return time_features
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = self.window_size  # 從window_size開始，確保有足夠的歷史數據
        self.balance = self.initial_balance
        self.btc_held = 0
        self.total_value = self.balance
        self.done = False
        self.last_action = 0  # 記錄上一次的動作
        self.position_holding_time = 0  # 記錄持倉時間
        self.last_total_value = self.initial_balance  # 記錄上一次的總資產
        self.entry_price = 0.0 # 重置進場價格
        self.margin_used = 0.0 # 重置已用保證金
        self.long_trades = 0
        self.short_trades = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        # 獲取歷史數據窗口
        window_data = self.df.iloc[self.current_step - self.window_size:self.current_step]
        
        # 計算特徵矩陣
        obs = np.zeros((self.n_features, self.window_size), dtype=np.float32)
        
        # 1. 一次性獲取所有價格特徵
        price_features = window_data.values.T  # 轉置以匹配形狀
        obs[:len(self.df.columns)] = price_features
        
        feature_idx = len(self.df.columns)
        
        # 2. 填充賬戶狀態（重複最後的值）
        current_data = self.df.iloc[self.current_step]
        
        obs[feature_idx] = np.linspace(0, 1, self.window_size) * (self.current_step / len(self.df)) # 步驟進度
        obs[feature_idx + 1] = np.full(self.window_size, self.btc_held)         # 持倉
        obs[feature_idx + 2] = np.full(self.window_size, self.btc_held * current_data['close']) # 持倉價值
        obs[feature_idx + 3] = np.full(self.window_size, self.total_value)      # 總資產(持倉價值+資金)
        obs[feature_idx + 4] = np.full(self.window_size, self.balance)          # 資金
        
        feature_idx += 5
        
        # 3. 添加時間特徵
        time_features = self._calculate_time_features(window_data)
        obs[feature_idx:feature_idx + 5] = time_features
        feature_idx += 5
        
        # 4. 添加市場情緒特徵
        sentiment_features = self._calculate_market_sentiment_features(window_data)
        obs[feature_idx:feature_idx + 8] = sentiment_features
        
        return obs
    
    # 計算獎勵
    def _calculate_reward(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        
        # 基礎獎勵：資產變化
        value_change = self.total_value - self.last_total_value
        total_reward = value_change / self.initial_balance
        
        # 更新上一次的總資產
        self.last_total_value = self.total_value
        
        return total_reward
    # 平倉
    def _close_position(self, price):
        if self.btc_held == 0:
            return

        # 1. 計算已實現損益
        pnl = self.btc_held * (price - self.entry_price)
        
        # 2. 計算平倉手續費
        closing_fee = abs(self.btc_held) * price * self.transaction_fee
        
        # 3. 更新餘額：返還保證金 + 已實現損益 - 平倉手續費
        self.balance += self.margin_used + pnl - closing_fee
        
        # 4. 重置倉位資訊
        self.btc_held = 0
        self.entry_price = 0
        self.margin_used = 0

    # 開倉
    def _open_position(self, target_position, price):
        # 1. 計算開倉費用和保證金
        fee = abs(target_position) * price * self.transaction_fee
        self.margin_used = abs(target_position) * price / self.leverage
        
        # 2. 檢查資金是否充足
        if self.balance < (self.margin_used + fee):
            # print("資金不足，無法開倉") # 可選的除錯信息
            self.done = True
            return

        # 3. 更新餘額
        self.balance -= (self.margin_used + fee)
        
        # 4. 設定倉位
        self.btc_held = target_position
        self.entry_price = price
        if target_position > 0:
            self.long_trades += 1
        elif target_position < 0:
            self.short_trades += 1

    # 執行交易
    def _execute_trade(self, action):
        current_price = self.df.iloc[self.current_step]['close']    # 當前價格
        current_high = self.df.iloc[self.current_step]['high']    # 當前最高價
        current_low = self.df.iloc[self.current_step]['low']    # 當前最低價
        
        # --- 止盈止損檢查 (應在所有交易決策之前) ---
        if self.btc_held > 0:  # 多倉止盈止損
            if self.stop_loss_price > 0 and current_low <= self.stop_loss_price:
                self._close_position(self.stop_loss_price)
                # 觸發後，重置倉位相關目標，避免在本步做出錯誤決策
                self.btc_held = 0 
            elif self.take_profit_price > 0 and current_high >= self.take_profit_price:
                self._close_position(self.take_profit_price)
                self.btc_held = 0
        elif self.btc_held < 0: # 空倉止盈止損
            if self.stop_loss_price > 0 and current_high >= self.stop_loss_price:
                self._close_position(self.stop_loss_price)
                self.btc_held = 0
            elif self.take_profit_price > 0 and current_low <= self.take_profit_price:
                self._close_position(self.take_profit_price)
                self.btc_held = 0

        # --- 根據 Action 計算目標倉位 ---
        position_direction = action[0]  # -1 (空) to 1 (多)
        position_percent = action[1]    # 0 to 1
        take_profit_percent = action[2] # e.g., 5 for 5%
        stop_loss_percent = action[3]   # e.g., 2 for 2%
        
        signed_position_percent = position_direction * position_percent
        
        # 根據當前淨值計算目標倉位
        equity = self.balance + self.margin_used # 在交易決策前的總資產
        target_position = equity * signed_position_percent * self.leverage / current_price
        
        # --- 執行倉位調整 ---
        # 只有在目標倉位與現有倉位不同時，才執行平倉和開倉
        if not np.isclose(target_position, self.btc_held):
            # 1. 如果有現有倉位，先平倉
            if self.btc_held != 0:
                self._close_position(current_price)
            
            # 2. 如果目標倉位不為零，開新倉
            if not np.isclose(target_position, 0):
                self._open_position(target_position, current_price)

        # --- 更新止盈止損價格 (基於新的開倉價) ---
        if self.btc_held != 0 and self.entry_price > 0:
            if self.btc_held > 0:  # 做多倉位
                self.take_profit_price = self.entry_price * (1 + take_profit_percent * 0.01)
                self.stop_loss_price = self.entry_price * (1 - stop_loss_percent * 0.01)
            else:  # 做空倉位
                self.take_profit_price = self.entry_price * (1 - take_profit_percent * 0.01)
                self.stop_loss_price = self.entry_price * (1 + stop_loss_percent * 0.01)
        else:  # 無倉位
            self.take_profit_price = 0
            self.stop_loss_price = 0

        # --- 核心修改：修正 total_value 的計算 ---
        # 總資產(淨值) = 可用餘額 + 已用保證金 + 未實現損益
        unrealized_pnl = 0
        if self.btc_held != 0:
            unrealized_pnl = self.btc_held * (current_price - self.entry_price)
        
        self.total_value = self.balance + self.margin_used + unrealized_pnl
        
        return current_price

    def step(self, action):
        # 執行交易
        current_price = self._execute_trade(action)
        
        # 計算獎勵
        reward = self._calculate_reward(action)
        
        # 更新步驟
        self.current_step += 1

        self.done = (self.current_step >= len(self.df) - 1
                     or self.balance <= self.min_balance)  # 資金不足
        
        info = {}
        if self.done:
            info['long_trades'] = self.long_trades
            info['short_trades'] = self.short_trades
            
        return self._get_observation(), reward, self.done, False, info 