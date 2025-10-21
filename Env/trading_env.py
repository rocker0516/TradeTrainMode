import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import random
from .trade_executor import TradeExecutor
from .reward import RewardCalculator

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
'''
class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_fee=0.001, window_size= 24 * 60 //5, leverage = 10, min_balance=100, min_trade_qty=0.001,
                 reward_weights=None, margin_mode: str = 'isolated', reward_calculator: RewardCalculator | None = None, random_start: bool = False):
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
        self.margin_mode = str(margin_mode).lower()  # 保證金模式
        self.random_start = bool(random_start)
        
        # 定義動作空間
        # 目標持倉比例 (-1.0 ~ 1.0)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        
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
            margin_mode=self.margin_mode,
        )

        # 帳戶狀態時間序列（逐筆滾動保存）
        series_len = len(self.df)
        self.account_series = {
            'position': np.zeros(series_len, dtype=np.float32),
            'position_value': np.zeros(series_len, dtype=np.float32),
            'equity': np.zeros(series_len, dtype=np.float32),
            'wallet': np.zeros(series_len, dtype=np.float32),
        }

        # 獎勵計算器
        self.reward_calculator = reward_calculator or RewardCalculator(mode='log', scale=1.0)
        # 日績效追蹤
        self._bars_per_day = 24 * 60 // 5
        self._day_start_equity = None
        # 倉位追蹤（用於計算換手）
        self._last_position_size = 0.0

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 若啟用隨機起點，從 [window_size, len(df)-2] 範圍中抽樣起始步
        if self.random_start and len(self.df) > (self.window_size + 2):
            self.current_step = int(random.randint(self.window_size, len(self.df) - 2))
        else:
            self.current_step = self.window_size  # 從window_size開始，確保有足夠的歷史數據
        self.executor.reset(self.initial_balance)
        self.balance = self.initial_balance
        self.btc_held = 0.0
        self.total_value = self.balance
        self.done = False
        self.position_holding_time = 0  # 記錄持倉時間
        self._day_start_equity = self.executor.equity(float(self.df.iloc[self.current_step]['close']))
        self._last_position_size = 0.0
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
        
        # 安全：確保觀察不含 NaN/Inf
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    
    def step(self, action):
        # 取得當前K線
        candle = self.df.iloc[self.current_step]# 當前K線
        current_price = float(candle['close']) # 當前價格
        current_high = float(candle['high']) # 當前最高價
        current_low = float(candle['low']) # 當前最低價
        last_equity = self.executor.equity(current_price) # 上一步的權益
        position_percent = float(action) # 目標持倉比例 (-1.0 ~ 1.0)

        # 執行交易
        self.executor.execute(
            position_percent=position_percent, # 目標持倉比例 (-1.0 ~ 1.0)
            current_price=current_price, # 當前價格
            high=current_high, # 當前最高價
            low=current_low, # 當前最低價
            equity=last_equity, # 上一步的權益
        )

        # 同步帳戶狀態
        new_equity = self.executor.equity(current_price) # 當前權益
        self.balance = self.executor.wallet_balance # 當前資金
        self.btc_held = self.executor.position.size # 當前持倉量
        self.total_value = new_equity # 當前總資產
        # 計算保證金緩衝（距離強平的安全空間）
        margin_buffer = None
        try:
            # 使用淨槓桿 proxy：position_value / equity
            position_value = abs(float(self.executor.position.size * current_price))
            if new_equity > 0:
                leverage_ratio = position_value / new_equity
                # 將槓桿比轉為緩衝比：槓桿越高緩衝越低
                # 假設安全槓桿上限為 self.leverage * 0.8，超過此值緩衝降為 0
                safe_leverage = float(self.leverage) * 0.8
                if leverage_ratio >= safe_leverage:
                    margin_buffer = 0.0
                else:
                    margin_buffer = 1.0 - (leverage_ratio / safe_leverage)
                margin_buffer = float(np.clip(margin_buffer, 0.0, 1.0))
        except Exception:
            margin_buffer = 1.0  # 異常時視為安全
        
        # 計算倉位變動（換手）
        position_change = abs(float(self.executor.position.size - self._last_position_size))
        self._last_position_size = float(self.executor.position.size)

        # 回饋：使用獨立獎勵計算器（含每日結算、緩衝、換手）
        steps_since_window = self.current_step - self.window_size
        is_day_end = (steps_since_window > 0 and (steps_since_window % self._bars_per_day) == 0)
        day_return = None
        if is_day_end and self._day_start_equity and self._day_start_equity > 0:
            day_return = float(new_equity / self._day_start_equity - 1.0)
            self._day_start_equity = new_equity  # 下一日起點
        reward = self.reward_calculator.compute(
            last_equity=last_equity,
            new_equity=new_equity,
            is_day_end=is_day_end,
            day_return=day_return,
            margin_buffer=margin_buffer,
            position_change=position_change,
        )

        # 更新帳戶狀態時間序列
        if 0 <= self.current_step < len(self.df):
            # 與 reset 一致：使用初始資金做正規化，避免尺度飄移
            pos_norm = self.executor.position.size / (self.initial_balance / current_price) if self.initial_balance > 0 and current_price > 0 else 0.0
            pos_value_norm = (self.executor.position.size * current_price) / self.initial_balance if self.initial_balance > 0 else 0.0
            equity_norm = new_equity / self.initial_balance if self.initial_balance > 0 else 0.0
            wallet_norm = self.executor.wallet_balance / self.initial_balance if self.initial_balance > 0 else 0.0
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
        
        # 構造 info，提供結束原因
        info = {}
        if self.done:
            if data_exhausted:
                info['termination_reason'] = 'data_exhausted'
            elif balance_insufficient:
                info['termination_reason'] = 'balance_insufficient'
            else:
                info['termination_reason'] = 'other'
            # 附加結算資訊供回調使用
            info['final_balance'] = float(new_equity)
            info['profit'] = float(new_equity - self.initial_balance)
            info['profit_rate'] = float((info['profit'] / self.initial_balance) * 100) if self.initial_balance > 0 else 0.0
            # 以終止資訊補充最後一步的獎勵（失敗懲罰）
            reward = self.reward_calculator.compute(
                last_equity=last_equity,
                new_equity=new_equity,
                done=True,
                termination_reason=info['termination_reason']
            )

        # 調試：記錄結束原因
        if self.done:
            if data_exhausted:
                print(f"Episode結束：數據用完 (step={self.current_step}, data_len={len(self.df)})")
            if balance_insufficient:
                print(f"Episode結束：資金不足 (balance={self.balance:.2f}, min={self.min_balance})")
            else:
                print(f"Episode結束：強平 (balance={self.balance:.2f})")
        
        return self._get_observation(), reward, self.done, False, info