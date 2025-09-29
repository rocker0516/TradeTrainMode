import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_fee=0.001, window_size= 24 * 60 //5, leverage = 10, min_balance=0, min_trade_amount=10, 
                 reward_weights=None):
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
        
        # 獎勵權重配置
        default_weights = {
            'pnl': 0.6,         # PnL獎勵權重
            'entry': 0.2,       # 進場質量權重  
            'stop': 0.1,        # 止盈止損權重
            'holding': 0.05,    # 持倉管理權重
            'penalty': 0.05     # 交易懲罰權重
        }
        self.reward_weights = reward_weights if reward_weights is not None else default_weights
        
        # 獎勵正規化參數
        self.reward_normalization = {
            'pnl_scale': 20.0,      # PnL正規化縮放係數
            'entry_scale': 100.0,   # 進場獎勵放大係數
            'stop_scale': 500.0,    # 止盈止損獎勵放大係數
            'holding_scale': 10.0,  # 持倉獎勵放大係數
            'penalty_scale': 1000.0, # 懲罰放大係數
            'clip_range': 1.0       # 最終獎勵裁剪範圍
        }

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
        
        # 驗證獎勵權重配置
        self._validate_weights()
        
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
        
        # 交易追蹤
        self.open_trades = []  # 記錄開倉信息
        self.closed_trades = []  # 記錄平倉信息
        self.last_position = 0  # 記錄上一次的持倉量
        self.avg_entry_price = 0  # 平均進場價格
        self.stop_triggered_this_step = None  # 記錄本步是否觸發止盈止損
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        # 確保 current_step 在有效範圍內
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
            
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
    
    # 獎勵正規化方法
    def _normalize_pnl_reward(self, pnl_ratio):
        """正規化PnL獎勵到 [-1, 1] 範圍"""
        scaled = pnl_ratio * self.reward_normalization['pnl_scale']
        return np.tanh(scaled)  # tanh函數將結果壓縮到 [-1, 1]
    
    def _normalize_entry_reward(self, entry_score):
        """正規化進場質量獎勵"""
        scaled = entry_score * self.reward_normalization['entry_scale']
        return np.clip(scaled, -0.5, 0.5)  # 限制在 [-0.5, 0.5]
    
    def _normalize_stop_reward(self, stop_score):
        """正規化止盈止損獎勵"""
        scaled = stop_score * self.reward_normalization['stop_scale']
        return np.clip(scaled, 0, 1.0)  # 止盈止損總是正獎勵
    
    def _normalize_holding_reward(self, holding_score):
        """正規化持倉管理獎勵"""
        scaled = holding_score * self.reward_normalization['holding_scale']
        return np.clip(scaled, -0.3, 0.3)  # 限制在 [-0.3, 0.3]
    
    def _normalize_penalty(self, penalty_score):
        """正規化交易懲罰"""
        scaled = penalty_score * self.reward_normalization['penalty_scale']
        return np.clip(scaled, -0.5, 0)  # 懲罰總是負值
    
    def _validate_weights(self):
        """驗證權重總和是否為1"""
        total_weight = sum(self.reward_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            print(f"警告：獎勵權重總和為 {total_weight:.6f}，建議調整為 1.0")
        return total_weight

    # 計算獎勵
    def _calculate_reward(self, action):
        # 初始化各組件獎勵
        reward_components = {
            'pnl': 0.0,
            'entry': 0.0,
            'stop': 0.0,
            'holding': 0.0,
            'penalty': 0.0
        }
        
        # 1. PnL獎勵（平倉時）
        if self._position_closed_this_step():
            pnl_ratio = self._calculate_closed_trade_pnl() / self.initial_balance
            reward_components['pnl'] = self._normalize_pnl_reward(pnl_ratio)
        
        # 2. 進場質量獎勵
        if self._just_opened_position():
            entry_score = self._evaluate_entry_timing()
            reward_components['entry'] = self._normalize_entry_reward(entry_score)
        
        # 3. 止盈止損執行獎勵
        if self.stop_triggered_this_step is not None:
            stop_score = self._evaluate_stop_execution()
            reward_components['stop'] = self._normalize_stop_reward(stop_score)
        
        # 4. 持倉管理獎勵
        if self.btc_held != 0:
            holding_score = self._evaluate_holding_decision()
            reward_components['holding'] = self._normalize_holding_reward(holding_score)
        
        # 5. 交易頻率懲罰
        if abs(action[0]) > 0.05:  # 有交易行為
            penalty_score = -self.transaction_fee * 0.5
            reward_components['penalty'] = self._normalize_penalty(penalty_score)
        
        # 6. 加權合成最終獎勵
        total_reward = 0.0
        for component, weight in self.reward_weights.items():
            total_reward += reward_components[component] * weight
        
        # 7. 最終裁剪到合理範圍
        total_reward = np.clip(total_reward, 
                              -self.reward_normalization['clip_range'], 
                              self.reward_normalization['clip_range'])
        
        # 重置止盈止損觸發標記
        self.stop_triggered_this_step = None
        
        return total_reward

    # 平倉（平倉不受最低交易金額限制）
    def _close_position(self, current_price):
        if self.btc_held == 0:
            return
        
        # 記錄平倉交易信息
        self._track_trade_close(current_price)
        
        # 返回保證金（考慮手續費）
        self.balance += abs(self.btc_held) * current_price * (1 - self.transaction_fee) / self.leverage
        self.btc_held = 0
        self.avg_entry_price = 0

    # 更新倉位(加倉或減倉)
    def _update_position(self, diff_position, real_price):
        # 計算交易金額（使用槓桿後的實際金額）
        trade_amount = abs(diff_position) * real_price
        
        # 檢查是否達到最低交易金額
        if trade_amount < self.min_trade_amount:
            return  # 不執行交易
        
        # 記錄開倉信息（在更新持倉前）
        current_price = self.df.iloc[self.current_step]['close']
        old_position = self.btc_held
        self._track_trade_open(current_price, diff_position, old_position)
            
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
                    self.stop_triggered_this_step = 'stop_loss'
                    self._close_position(self.stop_loss_price)
                elif current_high >= self.take_profit_price:
                    self.stop_triggered_this_step = 'take_profit'
                    self._close_position(self.take_profit_price)
            else:  # 做空
                if current_high >= self.stop_loss_price:
                    self.stop_triggered_this_step = 'stop_loss'
                    self._close_position(self.stop_loss_price)
                elif current_low <= self.take_profit_price:
                    self.stop_triggered_this_step = 'take_profit'
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
        
        else:  # position_percent == 0，平倉
            if self.btc_held != 0:
                self._close_position(current_price)
        
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
        # 記錄執行前的狀態
        self.last_position = self.btc_held
        self.last_total_value = self.total_value
        
        # 執行交易
        self._execute_trade(action)
        
        # 更新持倉時間
        if self.btc_held != 0:
            self.position_holding_time += 1
        else:
            self.position_holding_time = 0
        
        # 計算獎勵
        reward = self._calculate_reward(action)
        
        # 更新步驟
        self.current_step += 1

        self.done = (self.current_step >= len(self.df) - 1
                     or self.balance <= self.min_balance)  # 資金不足
        
        # 確保不會超出數據範圍
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
            self.done = True
        
        return self._get_observation(), reward, self.done, False, {}
    
    # 交易追蹤方法
    def _track_trade_open(self, entry_price, position_size, old_position):
        """記錄開倉信息"""
        if old_position == 0:  # 新開倉
            self.avg_entry_price = entry_price
        else:  # 加倉
            total_cost = abs(old_position) * self.avg_entry_price + abs(position_size) * entry_price
            total_size = abs(old_position) + abs(position_size)
            self.avg_entry_price = total_cost / total_size if total_size > 0 else entry_price
    
    def _track_trade_close(self, exit_price):
        """記錄平倉交易信息"""
        if self.btc_held == 0 or self.avg_entry_price == 0:
            return
        
        # 計算交易PnL
        if self.btc_held > 0:  # 多倉
            pnl = (exit_price - self.avg_entry_price) * abs(self.btc_held)
        else:  # 空倉
            pnl = (self.avg_entry_price - exit_price) * abs(self.btc_held)
        
        # 扣除手續費
        entry_fee = abs(self.btc_held) * self.avg_entry_price * self.transaction_fee / self.leverage
        exit_fee = abs(self.btc_held) * exit_price * self.transaction_fee / self.leverage
        net_pnl = pnl - entry_fee - exit_fee
        
        # 記錄已完成交易
        trade_record = {
            'pnl': net_pnl,
            'hold_time': self.position_holding_time,
            'entry_price': self.avg_entry_price,
            'exit_price': exit_price,
            'position_size': self.btc_held,
            'return_rate': net_pnl / (abs(self.btc_held) * self.avg_entry_price) if self.avg_entry_price > 0 else 0
        }
        self.closed_trades.append(trade_record)
    
    # 交易質量評估方法
    def _just_opened_position(self):
        """檢查是否剛剛開倉"""
        return self.last_position == 0 and self.btc_held != 0
    
    def _position_closed_this_step(self):
        """檢查本步是否平倉"""
        return self.last_position != 0 and self.btc_held == 0
    
    def _calculate_short_term_momentum(self, periods=5):
        """計算短期動量"""
        if self.current_step < periods:
            return 0
        recent_prices = self.df.iloc[self.current_step-periods:self.current_step]['close']
        if len(recent_prices) < 2:
            return 0
        return (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
    
    def _check_volume_confirmation(self, multiplier=1.2):
        """檢查成交量確認"""
        if self.current_step < 10:
            return False
        current_volume = self.df.iloc[self.current_step]['volume']
        avg_volume = self.df.iloc[self.current_step-10:self.current_step]['volume'].mean()
        return current_volume > avg_volume * multiplier
    
    def _evaluate_entry_timing(self):
        """評估進場時機質量"""
        if not self._just_opened_position():
            return 0
        
        entry_score = 0
        momentum = self._calculate_short_term_momentum()
        volume_confirm = self._check_volume_confirmation()
        
        # 順勢開倉獎勵
        if (self.btc_held > 0 and momentum > 0) or (self.btc_held < 0 and momentum < 0):
            entry_score += 0.001
        
        # 成交量確認獎勵
        if volume_confirm:
            entry_score += 0.0005
        
        # 逆勢開倉小懲罰
        if (self.btc_held > 0 and momentum < -0.005) or (self.btc_held < 0 and momentum > 0.005):
            entry_score -= 0.0005
            
        return entry_score
    
    def _evaluate_stop_execution(self):
        """評估止盈止損執行質量"""
        if self.stop_triggered_this_step is None:
            return 0
        
        if self.stop_triggered_this_step == 'take_profit':
            return 0.002  # 止盈執行獎勵
        elif self.stop_triggered_this_step == 'stop_loss':
            return 0.001  # 止損執行獎勵（避免更大虧損）
        
        return 0
    
    def _calculate_closed_trade_pnl(self):
        """獲取最近平倉交易的PnL"""
        if not self.closed_trades:
            return 0
        return self.closed_trades[-1]['pnl']
    
    def _evaluate_holding_decision(self):
        """評估持倉決策"""
        if self.btc_held == 0:
            return 0
        
        current_price = self.df.iloc[self.current_step]['close']
        
        # 未實現損益
        if self.avg_entry_price > 0:
            if self.btc_held > 0:  # 多倉
                unrealized_pnl = (current_price - self.avg_entry_price) * abs(self.btc_held)
            else:  # 空倉
                unrealized_pnl = (self.avg_entry_price - current_price) * abs(self.btc_held)
            
            holding_reward = unrealized_pnl / self.initial_balance * 0.1
        else:
            holding_reward = 0
        
        # 持倉時間懲罰（鼓勵日內交易）
        if self.position_holding_time > 100:  # 超過100個time steps
            holding_reward -= 0.0001 * (self.position_holding_time - 100)
        
        return holding_reward 