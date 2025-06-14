import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

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

        # 定義動作空間
        # [交易方向(0:不動作, 1:做多, -1:做空), 止盈比例(0-1000%), 止損比例(10-30%)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.2, 0.1]),  
            high=np.array([1.0, 10, 0.3]), #(動作方向，止盈，止損)
            dtype=np.float32
        )
        
        # 計算特徵數量
        self.n_features = len(df.columns) + 5  # 價格特徵 + 賬戶狀態(5個)
        
        # 定義觀察空間
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.n_features, window_size),  # 所有特徵 + 賬戶狀態
            dtype=np.float32
        )
        
        self.reset()
    
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
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        # 獲取歷史數據窗口
        window_data = self.df.iloc[self.current_step - self.window_size:self.current_step]
        
        # 計算特徵矩陣
        obs = np.zeros((self.n_features, self.window_size), dtype=np.float32)
        
        # 一次性獲取所有價格特徵
        price_features = window_data.values.T  # 轉置以匹配形狀
        obs[:len(self.df.columns)] = price_features
        
        # 填充賬戶狀態（重複最後的值）
        current_data = self.df.iloc[self.current_step]
        account_features_start = len(self.df.columns)
        
        obs[account_features_start] = np.linspace(0, 1, self.window_size) * (self.current_step / len(self.df)) # 步驟進度
        obs[account_features_start + 1] = np.full(self.window_size, self.btc_held)         # 持倉
        obs[account_features_start + 2] = np.full(self.window_size, self.btc_held * current_data['close']) # 持倉價值
        obs[account_features_start + 3] = np.full(self.window_size, self.total_value)      # 總資產(持倉價值+資金)
        obs[account_features_start + 4] = np.full(self.window_size, self.balance)          # 資金
        
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
    def _close_position(self, current_price):
        if self.btc_held == 0:
            return
        # 返回保證金（考慮手續費）
        self.balance += abs(self.btc_held) * current_price * (1 - self.transaction_fee) / self.leverage
        self.btc_held = 0
    # 更新倉位(加倉或減倉)
    def _update_position(self, diff_position, current_price):
        self.btc_held += diff_position
        self.balance -= abs(diff_position) * current_price * (1 + self.transaction_fee) / self.leverage

    # 執行交易
    def _execute_trade(self, action):
        current_price = self.df.iloc[self.current_step]['close']    # 當前價格
        current_high = self.df.iloc[self.current_step]['high']    # 當前最高價
        current_low = self.df.iloc[self.current_step]['low']    # 當前最低價
        position_percent = action[0]    # 交易方向（正負）和倉位百分比
        take_profit_percent = action[1]    # 止盈比例
        stop_loss_percent = action[2]    # 止損比例
        
        # 計算目標倉位
        target_position = self.total_value * position_percent * self.leverage / current_price
        
        
        if position_percent > 0:  # 做多
                
            # 如果達到止盈或止損價格，平倉
            if  current_low <= self.stop_loss_price:
                self._close_position(self.stop_loss_price)
            elif current_high >= self.take_profit_price:
                self._close_position(self.take_profit_price)
            if self.btc_held >= 0:  # 當前是多倉或無倉位
                # 計算需要調整的數量
                position_diff = target_position - self.btc_held
                if position_diff != 0:
                    self._update_position(position_diff, current_price)
            else:  # 當前是空倉，需要平倉後開多
                self._close_position(current_price)
                self._update_position(target_position, current_price)
                
        elif position_percent < 0:  # 做空

            # 如果達到止盈或止損價格，平倉
            if  current_high >= self.stop_loss_price:
                self._close_position(self.stop_loss_price)
            elif current_low <= self.take_profit_price:
                self._close_position(self.take_profit_price)

            if self.btc_held <= 0:  # 當前是空倉或無倉位
                # 計算需要調整的數量
                position_diff = target_position - self.btc_held
                if position_diff != 0:
                    self._update_position(position_diff, current_price)
            else:  # 當前是多倉，需要平倉後開空
                self._close_position(current_price)
                self._update_position(target_position, current_price)
        
        # 計算止盈止損價格（根據實際持倉方向）
        if self.btc_held > 0:  # 做多倉位
            self.take_profit_price = current_price * (1 + take_profit_percent * 0.01)   # 止盈價格 = 當前價格 * (1 + 止盈比例)
            self.stop_loss_price = current_price * (1 - stop_loss_percent * 0.01)       # 止損價格 = 當前價格 * (1 - 止損比例)
        elif self.btc_held < 0:  # 做空倉位
            self.take_profit_price = current_price * (1 - take_profit_percent * 0.01)   # 止盈價格 = 當前價格 * (1 - 止盈比例)
            self.stop_loss_price = current_price * (1 + stop_loss_percent * 0.01)       # 止損價格 = 當前價格 * (1 + 止損比例)
        else:  # 無倉位
            self.take_profit_price = 0
            self.stop_loss_price = 0

        # 更新總價值
        self.total_value = self.balance + (self.btc_held * current_price)
        
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
        
        return self._get_observation(), reward, self.done, False, {} 