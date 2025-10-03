import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from .reward import (
    RewardCalculator,
    RewardWeights,
    RewardNormalizer,
    RewardNormalizationConfig,
    RewardContext,
)
from .execution import (
    TradingExecutor,
    TradeContext,
)

class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_fee=0.001, window_size= 24 * 60 //5, leverage = 10, min_balance=0, min_trade_amount=10, 
                 reward_weights=None, episode_length: int | None = None, random_start: bool = True):
        super(TradingEnvironment, self).__init__()
        
        # 只保留數值列，並確保包含必要的OHLCV列
        # OHLCV + 成交量 + 成交量比 + 多空比 + 交易量 + 交易額
        required_columns = ['open', 'high', 'low', 'close', 'volume','buy_volume','sell_volume','volume_ratio','long_short_ratio','trades','quote_volume']
        
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
        self.min_balance = max(min_balance, 0)  # 最小資金(資金不足時強制結束，至少 100)
        self.min_trade_amount = min_trade_amount  # 最低交易金額(USDT)
        self.stop_loss_price = 0.0    # 止損價格
        self.take_profit_price = 0.0    # 止盈價格
        
        # 獎勵權重配置（已降低風險管理比重，去除頻率類訊號）
        default_weights = {
            'pnl': 0.4,              # PnL獎勵權重（提高，收益率優先！）
            'entry': 0.05,           # 進場質量權重  
            'stop': 0.05,            # 止盈止損權重
            'holding': 0.03,         # 持倉管理權重
            'penalty': 0.02,         # 交易懲罰權重（降低以鼓勵交易）
            'risk_management': 0.45  # 風險管理權重（仍重要但讓位給 PnL）
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

        # 初始化獎勵計算器（抽離為模組）
        weights = RewardWeights(**self.reward_weights)
        normalization_config = RewardNormalizationConfig(
            pnl_scale=self.reward_normalization['pnl_scale'],
            entry_scale=self.reward_normalization['entry_scale'],
            stop_scale=self.reward_normalization['stop_scale'],
            holding_scale=self.reward_normalization['holding_scale'],
            penalty_scale=self.reward_normalization['penalty_scale'],
            clip_range=self.reward_normalization['clip_range'],
        )
        normalizer = RewardNormalizer(normalization_config)
        self.reward_calculator = RewardCalculator(weights=weights, normalizer=normalizer)
        self.executor = TradingExecutor()

        # 定義動作空間：
        # [倉位比例(-1~1), 止盈控制(0~1 → 0~100%), 止損控制(0~1 → 0~20%)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            shape=(3,),
            dtype=np.float32
        )
        
        # 計算特徵數量
        # 原始價格特徵 + 賬戶狀態(5個)
        # 改用稀疏帳戶通道：僅在當步通道填入帳戶狀態（最後一列），減少鋪滿窗口造成的冗餘
        self.n_features = len(df.columns) + 5 

        # Episode 配置
        self.episode_length = episode_length if episode_length is not None else (window_size * 5)
        self.random_start = random_start
        self.steps_in_episode = 0
        
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
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start:
            start = np.random.randint(self.window_size, len(self.df) - self.episode_length - 1)
            self.current_step = int(start)
        else:
            self.current_step = self.window_size
        self.balance = self.initial_balance
        self.btc_held = 0
        self.total_value = self.balance
        self.done = False
        self.last_action = 0  # 記錄上一次的動作
        self.position_holding_time = 0  # 記錄持倉時間
        self.last_total_value = self.initial_balance  # 記錄上一次的總資產
        self.steps_in_episode = 0
        
        # 交易追蹤
        self.open_trades = []  # 記錄開倉信息
        self.closed_trades = []  # 記錄平倉信息
        self.last_position = 0  # 記錄上一次的持倉量
        self.avg_entry_price = 0  # 平均進場價格
        self.stop_triggered_this_step = None  # 記錄本步是否觸發止盈止損
        # 交易與風險統計
        self.trades_long_count = 0
        self.trades_short_count = 0
        self.trades_take_profit_count = 0
        self.trades_stop_loss_count = 0
        self.trades_close_count = 0
        self.trades_forced_liquidation_count = 0
        # 回撤統計
        self.peak_value = float(self.initial_balance)
        self.max_drawdown = 0.0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        # 確保 current_step 在有效範圍內
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
            
        # 獲取歷史數據窗口
        window_data = self.df.iloc[self.current_step - self.window_size:self.current_step]
        
        # 計算特徵矩陣
        obs = np.zeros((self.n_features, self.window_size), dtype=np.float32)
        
        # 1. 價格特徵：改用 z-score（以視窗內統計，或可改為離線統計）
        price_features = window_data.values.T.astype(np.float32)
        mean = price_features.mean(axis=1, keepdims=True)
        std = price_features.std(axis=1, keepdims=True) + 1e-8
        price_z = (price_features - mean) / std
        obs[:len(self.df.columns)] = price_z
        
        feature_idx = len(self.df.columns)
        
        # 2. 帳戶狀態：僅在最後一列填入（稀疏化），其他時間步為 0
        current_data = self.df.iloc[self.current_step]
        close_price = float(current_data['close']) if 'close' in self.df.columns else 0.0
        max_position = (self.initial_balance / close_price) if close_price > 0 else 1.0
        last_col = self.window_size - 1
        obs[feature_idx, last_col] = np.clip(self.current_step / len(self.df), 0, 1)  # 步驟進度（比例）
        obs[feature_idx + 1, last_col] = np.clip(self.btc_held / max_position, -10, 10)  # 持倉
        obs[feature_idx + 2, last_col] = np.clip((self.btc_held * close_price) / max(self.initial_balance, 1e-8), -10, 10) # 持倉價值
        obs[feature_idx + 3, last_col] = np.clip(self.total_value / max(self.initial_balance, 1e-8), -10, 10)      # 總資產
        obs[feature_idx + 4, last_col] = np.clip(self.balance / max(self.initial_balance, 1e-8), 0, 10)          # 資金
        
        # 最終 NaN 檢查與修正
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return obs
    
    def _validate_weights(self):
        """驗證權重總和是否為1"""
        total_weight = sum(self.reward_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            print(f"警告：獎勵權重總和為 {total_weight:.6f}，建議調整為 1.0")
        return total_weight

    # 計算獎勵
    def _calculate_reward(self, action):
        # 構建上下文並完全委派至 reward 模組
        ctx = RewardContext(
            df=self.df,
            current_step=self.current_step,
            initial_balance=self.initial_balance,
            transaction_fee=self.transaction_fee,
            leverage=self.leverage,
            min_balance=self.min_balance,
            total_value=self.total_value,
            balance=self.balance,
            btc_held=self.btc_held,
            avg_entry_price=self.avg_entry_price,
            position_holding_time=self.position_holding_time,
            last_position=self.last_position,
            stop_triggered_this_step=self.stop_triggered_this_step,
            closed_trades=self.closed_trades,
        )
        total_reward = self.reward_calculator.compute_reward(ctx, np.asarray(action))

        # 重置止盈止損觸發標記
        self.stop_triggered_this_step = None

        return total_reward

    # 交易執行與資金檢查已委派至 Env/execution.py 的 TradingExecutor

    # 執行交易
    def _execute_trade(self, action):
        trade_ctx = TradeContext(
            df=self.df,
            current_step=self.current_step,
            balance=self.balance,
            total_value=self.total_value,
            btc_held=self.btc_held,
            avg_entry_price=self.avg_entry_price,
            take_profit_price=self.take_profit_price,
            stop_loss_price=self.stop_loss_price,
            leverage=self.leverage,
            fee_rate=self.transaction_fee,
            min_trade_amount=self.min_trade_amount,
            min_balance=self.min_balance,
            last_position=self.last_position,
            position_holding_time=self.position_holding_time,
            steps_in_episode=self.steps_in_episode,
            slippage_bps=5.0,
            warmup_steps=int(self.window_size * 1.5),
        )

        result = self.executor.execute(trade_ctx, np.asarray(action))

        # 回寫狀態
        self.balance = result.balance
        self.total_value = result.total_value
        self.btc_held = result.btc_held
        self.avg_entry_price = result.avg_entry_price
        self.take_profit_price = result.take_profit_price
        self.stop_loss_price = result.stop_loss_price
        self.stop_triggered_this_step = result.stop_triggered_this_step

        # 若有平倉交易，記錄
        if result.closed_trade is not None:
            self.closed_trades.append(result.closed_trade)

        return result

    def step(self, action):
        # 記錄執行前的狀態
        self.last_position = self.btc_held
        self.last_total_value = self.total_value
        
        # 執行交易
        result = self._execute_trade(action)
        
        # 記錄交易方向與事件（做多/做空/平倉/止盈/止損/強平）
        if self.last_position == 0 and self.btc_held != 0:
            if self.btc_held > 0:
                self.trades_long_count += 1
            else:
                self.trades_short_count += 1
        if result is not None:
            if result.closed_trade is not None:
                self.trades_close_count += 1
            if result.stop_triggered_this_step == 'take_profit':
                self.trades_take_profit_count += 1
            if result.stop_triggered_this_step == 'stop_loss':
                self.trades_stop_loss_count += 1
            if result.forced_liquidation:
                self.trades_forced_liquidation_count += 1
        
        # 更新持倉時間
        if self.btc_held != 0:
            self.position_holding_time += 1
        else:
            self.position_holding_time = 0
        
        # 當步 Reward 訊號（供 info 使用）
        ctx_for_reward = RewardContext(
            df=self.df,
            current_step=self.current_step,
            initial_balance=self.initial_balance,
            transaction_fee=self.transaction_fee,
            leverage=self.leverage,
            min_balance=self.min_balance,
            total_value=self.total_value,
            balance=self.balance,
            btc_held=self.btc_held,
            avg_entry_price=self.avg_entry_price,
            position_holding_time=self.position_holding_time,
            last_position=self.last_position,
            stop_triggered_this_step=self.stop_triggered_this_step,
            closed_trades=self.closed_trades,
        )
        signals = self.reward_calculator.compute_signals(ctx_for_reward, np.asarray(action))

        # 計算獎勵
        reward = self._calculate_reward(action)
        
        # 更新步驟
        self.current_step += 1
        self.steps_in_episode += 1

        # 檢查結束條件並記錄原因
        data_exhausted = self.steps_in_episode >= self.episode_length
        balance_insufficient = self.total_value <= self.min_balance
        
        self.done = data_exhausted or balance_insufficient
        # 標記結束原因供外部統計
        self.last_done_reason = None
        
        # 調試：記錄結束原因
        if self.done:
            if data_exhausted:
                print(f"Episode結束：數據用完 (step={self.current_step}, data_len={len(self.df)})")
                self.last_done_reason = 'data_exhausted'
            if balance_insufficient:
                print(f"Episode結束：資金不足 (balance={self.total_value:.2f}, min={self.min_balance})")
                self.last_done_reason = 'balance_insufficient'
        
        # 確保不會超出數據範圍
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
            self.done = True
        
        # 風險/績效衍生統計
        current_price = float(self.df.iloc[self.current_step - 1]['close']) if self.current_step - 1 < len(self.df) else float(self.df.iloc[-1]['close'])
        position_size = float(abs(self.btc_held))
        position_value = float(position_size * current_price)
        leverage_used = float(position_value / max(self.total_value, 1e-8)) if self.total_value > 0 else 0.0
        # 未實現損益（供 info 展示）
        if self.avg_entry_price > 0 and self.btc_held != 0:
            if self.btc_held > 0:
                unrealized_pnl = (current_price - float(self.avg_entry_price)) * position_size
            else:
                unrealized_pnl = (float(self.avg_entry_price) - current_price) * position_size
        else:
            unrealized_pnl = 0.0
        margin_used = float(position_size * float(self.avg_entry_price) / max(self.leverage, 1e-12)) if self.avg_entry_price > 0 else 0.0

        # 回撤統計更新
        self.peak_value = float(max(self.peak_value, self.total_value))
        drawdown = float(self.total_value / max(self.peak_value, 1e-12) - 1.0)
        self.max_drawdown = float(min(self.max_drawdown, drawdown))

        # Reward 組件（原始與正規化後貢獻，便於分析）
        try:
            normalizer = self.reward_calculator._normalizer  # 診斷用途
            weights = self.reward_calculator._weights
            penalty_norm = (
                normalizer.normalize_penalty(signals.penalty_score)
                if float(signals.penalty_score) < 0.0
                else float(np.clip(float(signals.penalty_score) * 50.0, 0.0, 0.1))
            )
            reward_components = {
                'signals': {
                    'pnl_ratio': float(signals.pnl_ratio),
                    'entry_score': float(signals.entry_score),
                    'stop_score': float(signals.stop_score),
                    'holding_score': float(signals.holding_score),
                    'penalty_score': float(signals.penalty_score),
                    'risk_score': float(signals.risk_score),
                },
                'normalized': {
                    'pnl': float(normalizer.normalize_pnl(signals.pnl_ratio)) * float(weights.pnl),
                    'entry': float(normalizer.normalize_entry(signals.entry_score)) * float(weights.entry),
                    'stop': float(normalizer.normalize_stop(signals.stop_score)) * float(weights.stop),
                    'holding': float(normalizer.normalize_holding(signals.holding_score)) * float(weights.holding),
                    'penalty': float(penalty_norm) * float(weights.penalty),
                    'risk_management': float(signals.risk_score) * float(weights.risk_management),
                },
            }
        except Exception:
            reward_components = {}

        realized_pnl_this_step = float(result.closed_trade.get('pnl')) if (result is not None and result.closed_trade is not None) else 0.0

        info = {
            'leverage_used': leverage_used,
            'fees_paid': float(self.transaction_fee),  # 簡化：本步費率指示；可改為累計
            'steps_in_episode': int(self.steps_in_episode),
            'done_reason': self.last_done_reason,
            'account': {
                'total_value': float(self.total_value),
                'balance': float(self.balance),
                'cum_return': float(self.total_value / max(self.initial_balance, 1e-12) - 1.0),
            },
            'position': {
                'size': float(self.btc_held),
                'avg_entry_price': float(self.avg_entry_price),
                'value': position_value,
                'leverage_used': leverage_used,
                'holding_time': int(self.position_holding_time),
            },
            'pnl': {
                'unrealized': float(unrealized_pnl),
                'realized_this_step': realized_pnl_this_step,
                'drawdown': float(drawdown),
                'max_drawdown': float(self.max_drawdown),
            },
            'risk': {
                'stop_triggered': self.stop_triggered_this_step,
                'forced_liquidation': bool(result.forced_liquidation) if result is not None else False,
                'take_profit_price': float(self.take_profit_price),
                'stop_loss_price': float(self.stop_loss_price),
                'margin_used': float(margin_used),
            },
            'trades': {
                'count': int(len(self.closed_trades)),
                'count_close': int(self.trades_close_count),
                'count_long_entries': int(self.trades_long_count),
                'count_short_entries': int(self.trades_short_count),
                'count_take_profit': int(self.trades_take_profit_count),
                'count_stop_loss': int(self.trades_stop_loss_count),
                'count_forced_liquidation': int(self.trades_forced_liquidation_count),
                'last': result.closed_trade if result is not None else None,
            },
            'reward': {
                'total': float(reward),
                'components': reward_components,
            },
        }
        return self._get_observation(), reward, self.done, False, info
    
    # 交易追蹤邏輯已完全委派至 Env/execution.py 的 TradingExecutor
    
    # 交易質量與風險相關的評估邏輯已搬至 Env/reward.py 的 RewardCalculator