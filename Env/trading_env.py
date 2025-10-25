import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import random
from .trade_executor import TradeExecutor
from .reward import RewardCalculator
from .features import build_all_features

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
    def __init__(self, df, initial_balance=10_000, transaction_fee=0.001, window_size= 24 * 60 //5, leverage = 10, min_balance=100, min_trade_qty=0.001,
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
        
        self.initial_balance = initial_balance  # 初始資金
        self.transaction_fee = transaction_fee  # 交易手續費
        self.window_size = window_size # 窗口大小(K線數量)
        self.leverage = leverage    #槓桿倍數
        self.min_balance = min_balance  # 最小資金(資金不足時強制結束)
        self.min_trade_qty = min_trade_qty  # 最低交易數量(BTC)
        self.margin_mode = str(margin_mode).lower()  # 保證金模式
        self.random_start = bool(random_start)
        
        # 定義動作空間
        # 目標持倉比例 (-1.0 ~ 1.0)，限制小數位為1位
        # ex: -1.0, -0.7, -0.5, -0.2, 0.0, 0.2, 0.5, 0.7, 1.0 
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
            rounding_decimals=1
        )
        
        # 構建穩定的觀測特徵：報酬/比例/滾動z-score（只用過去資料）
        self.feature_lookback = int(max(288, self.window_size))
        df_num = self.df  # shorthand

        close = df_num['close'].astype(np.float64)
        high = df_num['high'].astype(np.float64)
        low = df_num['low'].astype(np.float64)
        volume = df_num['volume'].astype(np.float64)
        buy_vol = df_num.get('buy_volume', pd.Series(0.0, index=df_num.index)).astype(np.float64)
        sell_vol = df_num.get('sell_volume', pd.Series(0.0, index=df_num.index)).astype(np.float64)

        # 價格動能（log returns）
        log_close = np.log(np.clip(close, 1e-12, None))
        log_ret_1 = log_close.diff().fillna(0.0)
        log_ret_5 = (log_close.diff(5) / 5.0).fillna(0.0)

        # 波動度（ATR 比例、當根高低幅度）
        prev_close = close.shift(1)
        true_range = np.maximum.reduce([
            (high - low).values,
            np.abs(high - prev_close).fillna(0.0).values,
            np.abs(low - prev_close).fillna(0.0).values,
        ])
        atr = pd.Series(true_range, index=df_num.index).rolling(14, min_periods=5).mean().fillna(0.0)
        atr_ratio = (atr / np.clip(close, 1e-12, None)).astype(np.float32)
        hl_range = ((high - low) / np.clip(close, 1e-12, None)).fillna(0.0)

        # 滾動 z-score（僅使用過去樣本）
        def _rolling_z(series: pd.Series, w: int) -> pd.Series:
            m = series.rolling(w, min_periods=20).mean()
            s = series.rolling(w, min_periods=20).std()
            return ((series - m) / s.replace(0.0, np.nan)).fillna(0.0)

        log_vol = np.log1p(np.clip(volume, 0.0, None))
        vol_z = _rolling_z(log_vol, self.feature_lookback)

        flow_ratio = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-12)
        flow_z = _rolling_z(flow_ratio, self.feature_lookback)

        lsr = df_num.get('long_short_ratio', pd.Series(0.0, index=df_num.index)).astype(np.float64)
        lsr_log = np.log(np.clip(lsr, 1e-6, None))
        lsr_z = _rolling_z(lsr_log, self.feature_lookback)

        # 組裝最終觀測特徵（基礎特徵）
        base_features = pd.DataFrame({
            'log_ret_1': log_ret_1.astype(np.float32),
            'log_ret_5': log_ret_5.astype(np.float32),
            'hl_range': hl_range.astype(np.float32),
            'atr_ratio': atr_ratio.astype(np.float32),
            'vol_z': vol_z.astype(np.float32),
            'flow_z': flow_z.astype(np.float32),
            'lsr_z': lsr_z.astype(np.float32),
        }, index=df_num.index)

        # 擴充：流動性、MACD、SMC 特徵（使用外部模組，僅用過去資訊）
        extra_features = build_all_features(df_num, lookback=self.feature_lookback)
        self.obs_features = pd.concat([base_features, extra_features], axis=1)

        # 特徵數量：預計算特徵（含擴充）+ 帳戶 4 項
        self.base_feature_count = int(self.obs_features.shape[1])
        self.n_features = self.base_feature_count + 4

        # 觀察空間
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features, self.window_size),
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
        self.reward_calculator = reward_calculator or RewardCalculator(mode='log', scale=1000.0)
        # 倉位追蹤（用於計算換手）
        self._last_position_size = 0.0

        # 延後打印使用的數據列，待環境完成初始化後輸出
        print(f"使用的數據列: {list(self.df.columns)}")

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
        self._last_position_size = 0.0
        self.last_total_value = self.initial_balance  # 記錄上一次的總資產
        self.episode_steps = 0  # 回合步數統計
        
        # 交易追蹤
        self.open_trades = []  # 記錄開倉信息
        self.closed_trades = []  # 記錄平倉信息
        self.last_position = 0  # 記錄上一次的持倉量
        self.avg_entry_price = 0  # 平均進場價格
        
        # 本回合最大步數（受資料長度限制）
        self.episode_start_step = int(self.current_step)
        self.episode_max_steps = max(0, (len(self.df) - 1) - self.episode_start_step)
        
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
        
        # 1. 使用預先計算的穩定特徵窗口
        features_window = self.obs_features.iloc[self.current_step - self.window_size:self.current_step]
        obs[:self.base_feature_count] = features_window.values.T.astype(np.float32)
        
        feature_idx = self.base_feature_count

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
        '''
        執行交易步驟
        執行交易後，更新帳戶狀態，計算保證金緩衝，計算倉位變動，計算未實現損益，計算獎勵
        最後更新帳戶狀態時間序列，更新步驟，檢查結束條件並記錄原因，構造 info，提供結束原因

        Args:
            action: 目標持倉比例 (-1.0 ~ 1.0)

        Returns:
            observation: 觀測
            reward: 獎勵
            done: 是否結束
            truncated: 是否截斷
            info: 信息
        '''
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

        # 計算保證金緩衝（距離強平的安全空間）= 1 - (槓桿比 / 安全槓桿比) <= reward 使用
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
        position_change = abs(float(self.executor.position.size - self._last_position_size))# 倉位變動 = 當前持倉量 - 上一次持倉量
        traded = position_change > 1e-8  # 是否發生交易 pos
        self._last_position_size = float(self.executor.position.size)

        # 計算未實現損益（用於持倉獎勵）
        unrealized_pnl = float(self.executor.unrealized_pnl(current_price))
        has_position = abs(self.executor.position.size) > 1e-8  # 是否持有倉位

        # 回饋：使用獨立獎勵計算器（簡化版）
        reward = self.reward_calculator.compute(
            last_equity=last_equity,
            new_equity=new_equity,
            margin_buffer=margin_buffer,
            position_change=position_change,
            has_position=has_position,
            unrealized_pnl=unrealized_pnl,
            traded=traded,
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
        self.episode_steps += 1

        # 檢查結束條件並記錄原因
        data_exhausted = self.current_step >= len(self.df) - 1
        balance_insufficient = new_equity <= self.min_balance# 資金不足
        liq_triggered = self.executor.liq_triggered# 強平觸發

        self.done = data_exhausted or balance_insufficient or liq_triggered
        
        # 構造 info，提供結束原因
        info = {}
        if self.done:
            if data_exhausted:
                info['termination_reason'] = 'data_exhausted'
                print(f"Episode結束：數據用完 (step={self.current_step}, data_len={len(self.df)})")
            elif balance_insufficient:
                info['termination_reason'] = 'balance_insufficient'
                print(f"Episode結束：資金不足 (balance={self.balance:.2f}, min={self.min_balance})")
            elif liq_triggered:
                info['termination_reason'] = 'liq_triggered'
                print(f"Episode結束：強平 (balance={self.balance:.2f})")
            else:
                info['termination_reason'] = 'other'
            # 附加結算資訊供回調使用
            info['final_balance'] = float(new_equity)
            info['profit'] = float(new_equity - self.initial_balance)
            info['profit_rate'] = float((info['profit'] / self.initial_balance) * 100) if self.initial_balance > 0 else 0.0
            # 交易統計：平倉次數與手續費
            try:
                info['long_close_count'] = int(self.executor.long_close_count)
                info['short_close_count'] = int(self.executor.short_close_count)
                info['total_fees'] = float(self.executor.total_fees)
                info['episode_steps'] = int(self.episode_steps)
                # 進場次數（多/空）
                info['long_entry_count'] = int(self.executor.long_entry_count)
                info['short_entry_count'] = int(self.executor.short_entry_count)
                # 預計最大步數與資料長度
                info['episode_max_steps'] = int(self.episode_max_steps)
                info['data_len'] = int(len(self.df))
                info['window_size'] = int(self.window_size)
            except Exception:
                pass
            # 以終止資訊補充最後一步的獎勵（失敗懲罰）
            reward = self.reward_calculator.compute(
                last_equity=last_equity,
                new_equity=new_equity,
                done=True,
                termination_reason=info['termination_reason']
            )
        
        return self._get_observation(), reward, self.done, False, info