import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_fee=0.001):
        super(TradingEnvironment, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        # 定義動作空間 (0: 持有, 1: 買入, 2: 賣出)
        self.action_space = spaces.Discrete(3)
        
        # 定義觀察空間
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(10,),  # 價格特徵 + 賬戶狀態
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.btc_held = 0
        self.total_value = self.balance
        self.done = False
        self.last_action = 0  # 記錄上一次的動作
        self.position_holding_time = 0  # 記錄持倉時間
        self.last_total_value = self.initial_balance  # 記錄上一次的總資產
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        # 獲取當前步驟的數據
        current_data = self.df.iloc[self.current_step]
        
        # 計算特徵
        obs = np.array([
            current_data['close'],
            current_data['open'],
            current_data['high'],
            current_data['low'],
            current_data['volume'],
            self.balance,
            self.btc_held,
            self.btc_held * current_data['close'],
            self.total_value,
            self.current_step / len(self.df)
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_reward(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        previous_price = self.df.iloc[self.current_step - 1]['close'] if self.current_step > 0 else current_price
        
        # 1. 基礎獎勵：資產變化
        value_change = self.total_value - self.last_total_value
        base_reward = value_change / self.initial_balance
        
        # 2. 交易頻率懲罰
        trading_penalty = 0
        if action != 0:  # 如果不是持有
            trading_penalty = -0.001  # 交易成本懲罰
        
        # 3. 持倉時間獎勵/懲罰
        holding_reward = 0
        if action == 0:  # 如果是持有
            self.position_holding_time += 1
            # 如果持倉時間過長，給予懲罰
            if self.position_holding_time > 10:
                holding_reward = -0.0001 * (self.position_holding_time - 10)
        else:
            self.position_holding_time = 0
        
        # 4. 趨勢獎勵
        trend_reward = 0
        price_change = (current_price - previous_price) / previous_price
        if (action == 1 and price_change > 0) or (action == 2 and price_change < 0):
            trend_reward = abs(price_change) * 0.1
        
        # 5. 風險懲罰
        risk_penalty = 0
        if self.btc_held > 0:
            # 如果持倉比例過高，給予懲罰
            position_ratio = (self.btc_held * current_price) / self.total_value
            if position_ratio > 0.8:
                risk_penalty = -0.001 * (position_ratio - 0.8)
        
        # 總獎勵
        total_reward = base_reward + trading_penalty + holding_reward + trend_reward + risk_penalty
        
        # 更新上一次的總資產
        self.last_total_value = self.total_value
        
        return total_reward
    
    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        
        # 執行交易
        if action == 1:  # 買入
            if self.balance > 0:
                btc_to_buy = self.balance / current_price
                fee = btc_to_buy * self.transaction_fee
                self.btc_held += btc_to_buy - fee
                self.balance = 0
        elif action == 2:  # 賣出
            if self.btc_held > 0:
                balance_to_add = self.btc_held * current_price
                fee = balance_to_add * self.transaction_fee
                self.balance += balance_to_add - fee
                self.btc_held = 0
        
        # 更新總價值
        self.total_value = self.balance + (self.btc_held * current_price)
        
        # 計算獎勵
        reward = self._calculate_reward(action)
        
        # 更新步驟
        self.current_step += 1
        self.done = self.current_step >= len(self.df) - 1
        
        return self._get_observation(), reward, self.done, False, {} 