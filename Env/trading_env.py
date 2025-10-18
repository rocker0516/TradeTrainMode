import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from .trade_executor import TradeExecutor
from .reward import RewardCalculator, RewardConfig

class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_fee=0.001, window_size= 24 * 60 //5, leverage = 10, min_balance=0, min_trade_qty=0.001, max_stop_loss_percent=1000, max_take_profit_percent=50,
                 reward_weights=None, verbose: bool = True, **reward_kwargs):
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
        # 清理資料中的 NaN / Inf，避免觀察為 NaN 造成策略產生 NaN 動作
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # 使用新推薦 API，避免 FutureWarning
        self.df.ffill(inplace=True)
        self.df.bfill(inplace=True)
        
        self.verbose = bool(verbose)
        if self.verbose:
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
        # 構建獎勵計算器（從 config 注入）
        reward_cfg = RewardConfig(
            reward_scale=reward_kwargs.get('reward_scale', 100.0),      #對數權益報酬縮放
            reward_clip_min=reward_kwargs.get('reward_clip_min', -5.0), #獎勵裁剪最小值
            reward_clip_max=reward_kwargs.get('reward_clip_max', 5.0), #獎勵裁剪最大值
            use_vol_norm=reward_kwargs.get('use_vol_norm', False), #波動正規化
            vol_window=reward_kwargs.get('vol_window', 200), #波動窗口
            safety_threshold=reward_kwargs.get('safety_threshold', 0.02), #安全門檻
            safety_lambda=reward_kwargs.get('safety_lambda', 0.7), #安全門檻
            dd_lambda=reward_kwargs.get('dd_lambda', 0.5), #回撤增量懲罰權重
            friction_eta=reward_kwargs.get('friction_eta', 0.08), #摩擦力
            terminal_penalty=reward_kwargs.get('terminal_penalty', -10.0), #終局懲罰
        )
        self.reward_calc = RewardCalculator(reward_cfg)
        # 其他控制參數
        self.action_deadband = reward_kwargs.get('action_deadband', 0.0) #動作死區
        self.action_quantum = reward_kwargs.get('action_quantum', 0.0) #動作量化
        

        # 定義動作空間
        # [交易方向和倉位百分比(-1~1), 止盈比例(0~10%), 止損比例(0~10%)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # 最小值：[倉位比例, 止盈%, 止損%]
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32), # 最大值：[倉位比例, 止盈%, 止損%]
            shape=(3,),
            dtype=np.float32
        )
        
        # 計算特徵數量（包含帳戶狀態 4 項：持倉、持倉價值、總資產、資金）
        self.n_features = len(df.columns) + 4
        
        # 定義觀察空間
        # 使用與 dtype 一致的 low/high，避免 Gym 的精度降級警告
        obs_low = np.full((self.n_features, window_size), -np.inf, dtype=np.float32)#觀察空間最小值
        obs_high = np.full((self.n_features, window_size), np.inf, dtype=np.float32)#觀察空間最大值
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(self.n_features, window_size),  # 所有特徵
            dtype=np.float32
        )
        
        # 建立交易執行器（槓桿、手續費、最小交易量）
        self.executor = TradeExecutor(
            initial_balance=self.initial_balance,#初始資金
            fee_rate=self.transaction_fee, #交易手續費
            leverage=self.leverage, #槓桿倍數
            min_trade_qty=self.min_trade_qty, #最低交易數量
            max_stop_loss_percent=self.max_stop_loss_percent, #最大止損比例
            max_take_profit_percent=self.max_take_profit_percent, #最大止盈比例
        )
        # 將名目/冷卻注入 executor（需在 executor 建立之後）
        self.executor.min_notional_usd = reward_kwargs.get('min_notional_usd', 0.0) #最小名目金額
        self.executor.rebalance_cooldown_steps = int(reward_kwargs.get('rebalance_cooldown_steps', 0)) #重平衡冷卻步數

        # 帳戶狀態時間序列（逐筆滾動保存）
        series_len = len(self.df)
        self.account_series = {
            'position': np.zeros(series_len, dtype=np.float32),#持倉
            'position_value': np.zeros(series_len, dtype=np.float32),#持倉價值
            'equity': np.zeros(series_len, dtype=np.float32),#總資產
            'wallet': np.zeros(series_len, dtype=np.float32),#資金
        }

        self.reset()
    
    def reset(self, seed=None):
        '''
        重置環境
        '''
        super().reset(seed=seed)
        self.current_step = self.window_size  # 從window_size開始，確保有足夠的歷史數據
        self.executor.reset(self.initial_balance)#重置交易執行器
        self.balance = self.initial_balance#重置資金 = 初始資金
        self.btc_held = 0.0 #重置持倉 = 0.0
        self.total_value = self.balance#重置總資產 = 資金
        self.done = False#重置done = False
        self.position_holding_time = 0  # 記錄持倉時間
        self.last_total_value = self.initial_balance  # 記錄上一次的總資產 = 初始資金
        
        # 交易追蹤
        self.open_trades = []  # 記錄開倉信息
        self.closed_trades = []  # 記錄平倉信息
        self.last_position = 0  # 記錄上一次的持倉量
        self.avg_entry_price = 0  # 平均進場價格 
        self.reward_calc.reset(self.initial_balance)#重置獎勵計算器
        
        # 初始化帳戶狀態時間序列（用當前值填滿至 current_step 作為初始歷史）
        current_price = float(self.df.iloc[self.current_step]['close'])#重置當前價格 = 當前K線收盤價
        pos_norm = self.executor.position.size / (self.initial_balance / current_price)#重置持倉 = 持倉 / (初始資金 / 當前價格)
        pos_value_norm = (self.executor.position.size * current_price) / self.initial_balance#重置持倉價值 = 持倉 * 當前價格 / 初始資金
        equity_norm = self.executor.equity(current_price) / self.initial_balance#重置總資產 = 總資產 / 初始資金
        wallet_norm = self.executor.wallet_balance / self.initial_balance#重置資金 = 資金 / 初始資金

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
        # 保底：將 NaN 轉為 0，避免後續正規化產生 NaN
        price_features = np.nan_to_num(price_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 正規化價格特徵（現在所有列都是數值列）
        for i in range(len(self.df.columns)):
            feature_data = price_features[i]
            if len(feature_data) > 0 and feature_data.max() != feature_data.min():
                price_features[i] = (feature_data - feature_data.min()) / (feature_data.max() - feature_data.min())
            else:
                price_features[i] = np.zeros_like(feature_data)
        
        obs[:len(self.df.columns)] = price_features
        
        # 帳戶狀態特徵索引
        feature_idx = len(self.df.columns)

        # 2. 帳戶狀態（逐筆滾動歷史）
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        obs[feature_idx + 0] = self.account_series['position'][start_idx:end_idx]           # 持倉（正規化）
        obs[feature_idx + 1] = self.account_series['position_value'][start_idx:end_idx]     # 持倉價值（正規化）
        obs[feature_idx + 2] = self.account_series['equity'][start_idx:end_idx]             # 總資產（正規化）
        obs[feature_idx + 3] = self.account_series['wallet'][start_idx:end_idx]             # 資金（正規化）
        
        return obs

    # 執行交易改由 TradeExecutor 處理

    def step(self, action):
        '''
        執行交易
        1. 取得當前K線
        2. 取得當前價格
        3. 取得當前高價
        4. 取得當前低價
        5. 取得上一次的總資產
        6. 動作護欄：將非法/NaN 動作轉為安全值並裁剪到空間範圍
        7. 動作死區與量化（僅作用於倉位方向百分比）
        8. 估計上一倉位百分比（已是歸一化相對 initial_balance）→ 近似用於死區
        9. 執行交易
        10. 同步帳戶狀態
        11. 計算整合獎勵（正規化與 shaping 內聚於 RewardCalculator）
        12. 更新帳戶狀態時間序列（在當前索引寫入，之後再遞增步驟）
        13. 若觸發強平，立即結束回合（一次車禍語義）
        14. 更新步驟
        15. 檢查結束條件並記錄原因
        16. 確保不會超出數據範圍
        17. 返回觀察、獎勵、done、info
        '''
        # 取得當前K線
        candle = self.df.iloc[self.current_step]
        current_price = float(candle['close'])
        current_high = float(candle['high'])
        current_low = float(candle['low'])

        last_equity = self.executor.equity(current_price)#重置上一次的總資產 = 總資產

        # 動作護欄：將非法/NaN 動作轉為安全值並裁剪到空間範圍
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,):#如果動作形狀不為(3,)，則設為0
            action = np.zeros((3,), dtype=np.float32)
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)#將非法/NaN 動作轉為安全值
        action = np.clip(action, self.action_space.low, self.action_space.high)#將動作裁剪到空間範圍
        
        # 動作死區與量化（僅作用於倉位方向百分比）
        raw_pos = float(action[0])
        # 量化
        quantum = getattr(self, 'action_quantum', 0.0)#動作量化
        if quantum and quantum > 0.0:#如果動作量化大於0，則將動作量化
            raw_pos = float(np.round(raw_pos / quantum) * quantum)
        # 死區：對相對於上一倉位百分比的變化做死區
        last_pos = self.account_series['position'][self.current_step - 1] if self.current_step > 0 else 0.0#上一倉位百分比
        # 估計上一倉位百分比（已是歸一化相對 initial_balance）→ 近似用於死區
        deadband = getattr(self, 'action_deadband', 0.0)#動作死區
        if deadband and abs(raw_pos - last_pos) < deadband:#如果動作死區大於0，且相對於上一倉位百分比的變化小於動作死區，則將動作設為上一倉位百分比
            raw_pos = last_pos
        position_percent = raw_pos#倉位方向百分比
        take_profit_percent = float(action[1])#止盈百分比
        stop_loss_percent = float(action[2])#止損百分比

        # 執行交易
        self.executor.execute(
            position_percent=position_percent,
            take_profit_percent=take_profit_percent,
            stop_loss_percent=stop_loss_percent,
            current_price=current_price,
            high=current_high,
            low=current_low,
            equity=last_equity,
            step_index=self.current_step,
        )

        # 同步帳戶狀態
        new_equity = self.executor.equity(current_price)
        self.balance = self.executor.wallet_balance
        self.btc_held = self.executor.position.size
        self.total_value = new_equity

        # 計算整合獎勵（正規化與 shaping 內聚於 RewardCalculator）
        liq_price = None
        try:
            liq_price = self.executor._calc_liquidation_price()
        except Exception:
            liq_price = None
        reward = self.reward_calc.compute_step_reward(
            last_equity=last_equity,
            new_equity=new_equity,
            position_size=self.executor.position.size,
            current_price=current_price,
            liquidation_price=liq_price,
        )
        
        # 更新帳戶狀態時間序列（在當前索引寫入，之後再遞增步驟）
        denom = (self.initial_balance / current_price) if current_price > 0 else 0.0
        pos_norm = (self.executor.position.size / denom) if denom > 0 else 0.0
        pos_value_norm = (self.executor.position.size * current_price) / self.initial_balance
        equity_norm = new_equity / self.initial_balance
        wallet_norm = self.executor.wallet_balance / self.initial_balance

        if 0 <= self.current_step < len(self.df):
            self.account_series['position'][self.current_step] = float(pos_norm)
            self.account_series['position_value'][self.current_step] = float(pos_value_norm)
            self.account_series['equity'][self.current_step] = float(equity_norm)
            self.account_series['wallet'][self.current_step] = float(wallet_norm)

        # 若觸發強平，立即結束回合（一次車禍語義）
        if getattr(self.executor, 'was_liquidated', False):
            self.done = True
            # 強化終局懲罰
            reward += self.reward_calc.terminal_fail_penalty()
            if self.verbose:
                print(f"Episode結束：觸發強平 liquidation_price={self.executor.last_liquidation_price}")
            return self._get_observation(), reward, self.done, False, {}

        # 更新步驟
        self.current_step += 1

        # 檢查結束條件並記錄原因
        data_exhausted = self.current_step >= len(self.df) - 1
        balance_insufficient = new_equity <= self.min_balance
        
        self.done = data_exhausted or balance_insufficient
        
        # 調試：記錄結束原因
        if self.done:
            if data_exhausted:
                # 數據用完視為完整存活到數據終點
                final_price = float(self.df.iloc[self.current_step - 1]['close']) if self.current_step > 0 else float(self.df.iloc[self.current_step]['close'])
                final_equity = self.executor.equity(final_price)
                return_pct = (final_equity / self.initial_balance - 1.0) * 100.0
                trade_count = getattr(self.executor, 'trade_count', 0)
                total_fee = getattr(self.executor, 'total_fee_spent', 0.0)
                total_notional = getattr(self.executor, 'total_notional_traded', 0.0)
                orders_attempted = getattr(self.executor, 'orders_attempted', 0)
                orders_executed = getattr(self.executor, 'orders_executed', 0)
                avg_notional = (total_notional / orders_executed) if orders_executed > 0 else 0.0
                if self.verbose:
                    print(f"Episode結束：數據用完 此回合收益率= {return_pct:.2f}% 交易次數= {trade_count} 總手續費= {total_fee:.4f} 平均名目= {avg_notional:.2f} (step={self.current_step}, data_len={len(self.df)})")
            if balance_insufficient:
                # 同樣視為失敗終局：附加強懲罰
                reward += self.reward_calc.terminal_fail_penalty()
                total_fee = getattr(self.executor, 'total_fee_spent', 0.0)
                total_notional = getattr(self.executor, 'total_notional_traded', 0.0)
                orders_attempted = getattr(self.executor, 'orders_attempted', 0)
                orders_executed = getattr(self.executor, 'orders_executed', 0)
                avg_notional = (total_notional / orders_executed) if orders_executed > 0 else 0.0
                if self.verbose:
                    print(f"Episode結束：資金不足 (balance={self.balance:.2f}, min={self.min_balance}) 交易次數= {getattr(self.executor, 'trade_count', 0)} 總手續費= {total_fee:.4f} 平均名目= {avg_notional:.2f}")
        
        # 確保不會超出數據範圍
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
            self.done = True
        
        return self._get_observation(), reward, self.done, False, {}