import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import talib

class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_fee=0.001, window_size= 24 * 60 //5, leverage = 10, min_balance=0, min_trade_amount=10):
        super(TradingEnvironment, self).__init__()
        
        self.df = df    #資料集
        self.initial_balance = initial_balance  # 初始資金
        self.transaction_fee = transaction_fee  # 交易手續費
        self.window_size = window_size # 窗口大小(K線數量)
        self.leverage = leverage    #槓桿倍數
        self.min_balance = min_balance  # 最小資金(資金不足時強制結束)
        self.min_trade_amount = min_trade_amount  # 最低交易金額(USDT)
        self.stop_loss_price = 0.0    # 止損價格
        self.take_profit_price = 0.0    # 止盈價格

        # 定義動作空間
        # [交易方向和倉位百分比(-1~1), 止盈比例(0~10%), 止損比例(0~10%)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),  # 最小值：[倉位比例, 止盈%, 止損%]
            high=np.array([1.0, 50.0, 20.0]), # 最大值：[倉位比例, 止盈%, 止損%]
            shape=(3,),
            dtype=np.float32
        )
        
        # 計算特徵數量
        # 原始價格特徵 + 賬戶狀態(5個)
        self.n_features = len(df.columns) + 5 
        
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
        
        # 1. 一次性獲取所有價格特徵（正規化）
        price_features = window_data.values.T  # 轉置以匹配形狀
        
        # 正規化價格特徵（使用窗口內的最大最小值）
        for i in range(len(self.df.columns)):
            feature_data = price_features[i]
            if feature_data.max() != feature_data.min():
                price_features[i] = (feature_data - feature_data.min()) / (feature_data.max() - feature_data.min())
            else:
                price_features[i] = np.zeros_like(feature_data)
        
        obs[:len(self.df.columns)] = price_features
        
        feature_idx = len(self.df.columns)
        
        # 2. 填充賬戶狀態（重複最後的值）
        current_data = self.df.iloc[self.current_step]
        
        obs[feature_idx] = np.linspace(0, 1, self.window_size) * (self.current_step / len(self.df)) # 步驟進度
        obs[feature_idx + 1] = np.full(self.window_size, self.btc_held / (self.initial_balance / current_data['close']))  # 持倉（正規化）
        obs[feature_idx + 2] = np.full(self.window_size, (self.btc_held * current_data['close']) / self.initial_balance) # 持倉價值（正規化）
        obs[feature_idx + 3] = np.full(self.window_size, self.total_value / self.initial_balance)      # 總資產（正規化）
        obs[feature_idx + 4] = np.full(self.window_size, self.balance / self.initial_balance)          # 資金（正規化）
        
        return obs
    
    # 計算獎勵
    def _calculate_reward(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        
        # 基礎獎勵：資產變化百分比
        value_change = self.total_value - self.last_total_value
        base_reward = value_change / self.initial_balance
        
        
        # 交易頻率懲罰（避免過度交易）
        trade_penalty = 0
        if abs(action[0]) > 0.05:  # 有實際交易行為
            trade_penalty = -self.transaction_fee * 0.5  # 部分手續費懲罰
        
        total_reward = base_reward  + trade_penalty
        
        # 更新上一次的總資產
        self.last_total_value = self.total_value
        
        return total_reward

    # 平倉（平倉不受最低交易金額限制）
    def _close_position(self, current_price):
        if self.btc_held == 0:
            return
        # 返回保證金（考慮手續費）
        self.balance += abs(self.btc_held) * current_price * (1 - self.transaction_fee) / self.leverage
        self.btc_held = 0

    # 更新倉位(加倉或減倉)
    def _update_position(self, diff_position, real_price):
        # 計算交易金額（使用槓桿後的實際金額）
        trade_amount = abs(diff_position) * real_price
        
        # 檢查是否達到最低交易金額
        if trade_amount < self.min_trade_amount:
            return  # 不執行交易
            
        self.btc_held += diff_position
        self.balance -= abs(diff_position) * real_price 
    
    # 檢查交易是否滿足最低金額要求
    def _is_trade_valid(self, diff_position, current_price):
        trade_amount = abs(diff_position) * current_price / self.leverage
        return trade_amount >= self.min_trade_amount

    # 執行交易
    def _execute_trade(self, action):
        current_price = self.df.iloc[self.current_step]['close']    # 當前價格
        current_high = self.df.iloc[self.current_step]['high']    # 當前最高價
        current_low = self.df.iloc[self.current_step]['low']    # 當前最低價
        real_price = current_price * (1 + self.transaction_fee) / self.leverage # 實際價格(考慮手續費和槓桿)

        position_percent = action[0]    # 交易方向（正負）和倉位百分比
        take_profit_percent = action[1]    # 止盈比例
        stop_loss_percent = action[2]    # 止損比例
        
        # 計算目標倉位(倉位比例*總資產*槓桿倍數/當前價格)
        target_position = self.total_value * position_percent / real_price
        diff_position = target_position - self.btc_held

        # 檢查前根K線是否達到止盈或止損價格，平倉
        if self.btc_held != 0:
            if self.btc_held > 0:  # 做多
                if current_low <= self.stop_loss_price:
                    self._close_position(self.stop_loss_price)
                elif current_high >= self.take_profit_price:
                    self._close_position(self.take_profit_price)
            else:  # 做空
                if current_high >= self.stop_loss_price:
                    self._close_position(self.stop_loss_price)
                elif current_low <= self.take_profit_price:
                    self._close_position(self.take_profit_price)
        
        if position_percent > 0:  # 做多

            if self.btc_held >= 0:  # 當前是多倉或無倉位
                # 計算需要調整的數量
                if diff_position != 0 and self._is_trade_valid(diff_position, current_price):
                    self._update_position(diff_position, real_price)
            else:  # 當前是空倉，需要平倉後開多
                self._close_position(current_price)
                self._update_position(target_position, real_price)
                
        elif position_percent < 0:  # 做空

            if self.btc_held <= 0:  # 當前是空倉或無倉位
                # 計算需要調整的數量
                if diff_position != 0 and self._is_trade_valid(diff_position, current_price):
                    self._update_position(diff_position, real_price)
            else:  # 當前是多倉，需要平倉後開空
                self._close_position(current_price)
                self._update_position(target_position, real_price)
        
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
        
        return 

    def step(self, action):
        # 執行交易
        self._execute_trade(action)
        
        # 計算獎勵
        reward = self._calculate_reward(action)
        
        # 更新步驟
        self.current_step += 1

        self.done = (self.current_step >= len(self.df) - 1
                     or self.balance <= self.min_balance)  # 資金不足
        
        return self._get_observation(), reward, self.done, False, {} 