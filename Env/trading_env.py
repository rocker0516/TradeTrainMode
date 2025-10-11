import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from .trade_executor import TradeExecutor

class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_fee=0.001, window_size= 24 * 60 //5, leverage = 10, min_balance=0, min_trade_qty=0.001, max_stop_loss_percent=1000, max_take_profit_percent=50,
                 reward_weights=None):
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
        
        print(f"使用的數據列: {list(self.df.columns)}")
        self.initial_balance = initial_balance  # 初始資金
        self.transaction_fee = transaction_fee  # 交易手續費
        self.window_size = window_size # 窗口大小(K線數量)
        self.leverage = leverage    #槓桿倍數
        self.min_balance = min_balance  # 最小資金(資金不足時強制結束)
        self.min_trade_qty = min_trade_qty  # 最低交易數量(BTC)
        self.max_stop_loss_percent = max_stop_loss_percent  # 最大止損比例
        self.max_take_profit_percent = max_take_profit_percent  # 最大止盈比例
        self.stop_loss_price = 0.0    # 止損價格
        self.take_profit_price = 0.0    # 止盈價格
        

        # 定義動作空間
        # [交易方向和倉位百分比(-1~1), 止盈比例(0~10%), 止損比例(0~10%)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),  # 最小值：[倉位比例, 止盈%, 止損%]
            high=np.array([1.0, 1.0, 1.0]), # 最大值：[倉位比例, 止盈%, 止損%]
            shape=(3,),
            dtype=np.float32
        )
        
        # 計算特徵數量（包含帳戶狀態 4 項：持倉、持倉價值、總資產、資金）
        self.n_features = len(df.columns) + 4
        
        # 定義觀察空間
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.n_features, window_size),  # 所有特徵
            dtype=np.float32
        )
        
        # 建立交易執行器（槓桿、手續費、最小交易量）
        self.executor = TradeExecutor(
            initial_balance=self.initial_balance,
            fee_rate=self.transaction_fee,
            leverage=self.leverage,
            min_trade_qty=self.min_trade_qty,
            max_stop_loss_percent=self.max_stop_loss_percent,
            max_take_profit_percent=self.max_take_profit_percent,
        )

        # 帳戶狀態時間序列（逐筆滾動保存）
        series_len = len(self.df)
        self.account_series = {
            'position': np.zeros(series_len, dtype=np.float32),
            'position_value': np.zeros(series_len, dtype=np.float32),
            'equity': np.zeros(series_len, dtype=np.float32),
            'wallet': np.zeros(series_len, dtype=np.float32),
        }

        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = self.window_size  # 從window_size開始，確保有足夠的歷史數據
        self.executor.reset(self.initial_balance)
        self.balance = self.initial_balance
        self.btc_held = 0.0
        self.total_value = self.balance
        self.done = False
        self.position_holding_time = 0  # 記錄持倉時間
        self.last_total_value = self.initial_balance  # 記錄上一次的總資產
        
        # 交易追蹤
        self.open_trades = []  # 記錄開倉信息
        self.closed_trades = []  # 記錄平倉信息
        self.last_position = 0  # 記錄上一次的持倉量
        self.avg_entry_price = 0  # 平均進場價格
        
        # 初始化帳戶狀態時間序列（用當前值填滿至 current_step 作為初始歷史）
        current_price = float(self.df.iloc[self.current_step]['close'])
        pos_norm = self.executor.position.size / (self.initial_balance / current_price)
        pos_value_norm = (self.executor.position.size * current_price) / self.initial_balance
        equity_norm = self.executor.equity(current_price) / self.initial_balance
        wallet_norm = self.executor.wallet_balance / self.initial_balance

        # 初始化帳戶狀態時間序列（用當前值填滿至 current_step 作為初始歷史） 
        for key, value in (
            ('position', pos_norm),
            ('position_value', pos_value_norm),
            ('equity', equity_norm),
            ('wallet', wallet_norm),
        ):
            self.account_series[key][:self.current_step] = value

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
        
        # 正規化價格特徵（現在所有列都是數值列）
        for i in range(len(self.df.columns)):
            feature_data = price_features[i]
            if len(feature_data) > 0 and feature_data.max() != feature_data.min():
                price_features[i] = (feature_data - feature_data.min()) / (feature_data.max() - feature_data.min())
            else:
                price_features[i] = np.zeros_like(feature_data)
        
        obs[:len(self.df.columns)] = price_features
        
        feature_idx = len(self.df.columns)

        # 2. 帳戶狀態（逐筆滾動歷史）
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        obs[feature_idx + 0] = self.account_series['position'][start_idx:end_idx]           # 持倉（正規化）
        obs[feature_idx + 1] = self.account_series['position_value'][start_idx:end_idx]     # 持倉價值（正規化）
        obs[feature_idx + 2] = self.account_series['equity'][start_idx:end_idx]             # 總資產（正規化）
        obs[feature_idx + 3] = self.account_series['wallet'][start_idx:end_idx]             # 資金（正規化）
        
        return obs
    
    # 已移除複雜獎勵計算：獎勵改為總資產變化的簡單歸一化

    # 執行交易改由 TradeExecutor 處理

    def step(self, action):
        # 取得當前K線
        candle = self.df.iloc[self.current_step]
        current_price = float(candle['close'])
        current_high = float(candle['high'])
        current_low = float(candle['low'])

        last_equity = self.executor.equity(current_price)

        position_percent = float(action[0])
        take_profit_percent = float(action[1])
        stop_loss_percent = float(action[2])

        # 執行交易
        self.executor.execute(
            position_percent=position_percent,
            take_profit_percent=take_profit_percent,
            stop_loss_percent=stop_loss_percent,
            current_price=current_price,
            high=current_high,
            low=current_low,
            equity=last_equity,
        )

        # 同步帳戶狀態
        new_equity = self.executor.equity(current_price)
        self.balance = self.executor.wallet_balance
        self.btc_held = self.executor.position.size
        self.total_value = new_equity

        # 獎勵：本步權益變化
        reward = (new_equity - last_equity) / self.initial_balance
        
        # 更新帳戶狀態時間序列（在當前索引寫入，之後再遞增步驟）
        pos_norm = self.executor.position.size / (self.initial_balance / current_price) if current_price > 0 else 0.0
        pos_value_norm = (self.executor.position.size * current_price) / self.initial_balance
        equity_norm = new_equity / self.initial_balance
        wallet_norm = self.executor.wallet_balance / self.initial_balance

        if 0 <= self.current_step < len(self.df):
            self.account_series['position'][self.current_step] = float(pos_norm)
            self.account_series['position_value'][self.current_step] = float(pos_value_norm)
            self.account_series['equity'][self.current_step] = float(equity_norm)
            self.account_series['wallet'][self.current_step] = float(wallet_norm)

        # 更新步驟
        self.current_step += 1

        # 檢查結束條件並記錄原因
        data_exhausted = self.current_step >= len(self.df) - 1
        balance_insufficient = new_equity <= self.min_balance
        
        self.done = data_exhausted or balance_insufficient
        
        # 調試：記錄結束原因
        if self.done:
            if data_exhausted:
                print(f"Episode結束：數據用完 (step={self.current_step}, data_len={len(self.df)})")
            if balance_insufficient:
                print(f"Episode結束：資金不足 (balance={self.balance:.2f}, min={self.min_balance})")
        
        # 確保不會超出數據範圍
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
            self.done = True
        
        return self._get_observation(), reward, self.done, False, {}
    
    # 已移除：與獎勵計算相關的輔助函式（進場質量、風險管理等）