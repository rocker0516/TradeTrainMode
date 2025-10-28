import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import random
from typing import Any
from .trade_executor import TradeExecutor
from .reward import create_default_calculator, create_default_reward_system
from .features import build_all_features, normalize_feature_frame
from .info_builder import StepContext, StepInfoBuilder
from .step_handlers.market import MarketSnapshotBuilder
from .step_handlers.execution import TradeExecutionProcessor
from .step_handlers.reward import RewardAdapter, PotentialState
from .step_handlers.account import AccountSeriesUpdater
from .step_handlers.termination import TerminationEvaluator

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
        stop_loss_cooldown_window: 停損冷靜期觸發時計算連續止損的滑動窗口（步數）
        stop_loss_cooldown_length: 停損冷靜期啟動後暫停進場的期間（步數）
        stop_loss_cooldown_limit: 在滑動窗口內允許的最大止損次數（達到即啟動冷靜期）
        cooldown_hold_ratio: 冷靜期維持的目標持倉比例（預設 0 代表保持空倉）
'''
class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=10_000, transaction_fee=0.001, window_size= 24 * 60 //5, leverage = 10, min_balance=100, min_trade_qty=0.001,
                 reward_weights=None, margin_mode: str = 'isolated', reward_calculator=None, random_start: bool = False, use_rudder: bool = True,
                 max_daily_trades: int | None = None, min_position_delta: float = 0.0,
                 stop_loss_cooldown_window: int = 6, stop_loss_cooldown_length: int = 12,
                 stop_loss_cooldown_limit: int = 2, cooldown_hold_ratio: float = 0.0):
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
        self.max_daily_trades = max_daily_trades
        self.min_position_delta = float(min_position_delta)

        cooldown_window = int(stop_loss_cooldown_window)
        cooldown_length = int(stop_loss_cooldown_length)
        cooldown_limit = int(stop_loss_cooldown_limit)
        hold_ratio = float(cooldown_hold_ratio)

        if cooldown_window <= 0:
            raise ValueError("stop_loss_cooldown_window must be positive")
        if cooldown_length <= 0:
            raise ValueError("stop_loss_cooldown_length must be positive")
        if cooldown_limit <= 0:
            raise ValueError("stop_loss_cooldown_limit must be positive")

        self.stop_loss_cooldown_window = cooldown_window
        self.stop_loss_cooldown_length = cooldown_length
        self.stop_loss_cooldown_limit = cooldown_limit
        self.cooldown_hold_ratio = hold_ratio
        
        # 定義動作空間
        # 目標持倉比例 (-1.0 ~ 1.0)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float16
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
        base_features = normalize_feature_frame(base_features, lookback=self.feature_lookback, preserve_binary=False)

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
        
        # 建立交易執行器（槓桿、手續費、最小交易量、止損規則）
        self.executor = TradeExecutor(
            initial_balance=self.initial_balance,
            fee_rate=self.transaction_fee,
            leverage=self.leverage,
            min_trade_qty=self.min_trade_qty,
            margin_mode=self.margin_mode,
            stop_loss_atr=2.5,  # 固定止損：進場後 ±2.5 ATR
        )

        self.executor.stop_loss_cooldown_window = self.stop_loss_cooldown_window
        self.executor.stop_loss_cooldown_length = self.stop_loss_cooldown_length
        self.executor.stop_loss_cooldown_limit = self.stop_loss_cooldown_limit
        self.executor.cooldown_hold_ratio = self.cooldown_hold_ratio

        self.episode_peak_equity = float(self.initial_balance)
        self.consecutive_stop_losses = 0

        # 帳戶狀態時間序列（逐筆滾動保存）
        series_len = len(self.df)
        self.account_series = {
            'position': np.zeros(series_len, dtype=np.float32),
            'position_value': np.zeros(series_len, dtype=np.float32),
            'equity': np.zeros(series_len, dtype=np.float32),
            'wallet': np.zeros(series_len, dtype=np.float32),
        }

        # 獎勵系統（支援 RUDDER + Lagrangian）
        self.use_rudder = bool(use_rudder)
        if self.use_rudder:
            # 新版：分離 shaping / outcome / cost
            reward_system = create_default_reward_system()
            self.shaping_calc = reward_system['shaping']
            self.outcome_calc = reward_system['outcome']
            self.cost_calc = reward_system['cost']
            self.potential_calc = reward_system['potential']
            self.reward_calculator = None  # 舊版不使用
        else:
            # 向後兼容：使用舊版整合 reward
            self.reward_calculator = reward_calculator or create_default_calculator()
            self.shaping_calc = None
            self.outcome_calc = None
            self.cost_calc = None
            self.potential_calc = None

        self.info_builder = StepInfoBuilder()
        self.market_builder = MarketSnapshotBuilder()
        self.execution_processor = TradeExecutionProcessor(
            stop_loss_cooldown_window=self.stop_loss_cooldown_window,
            stop_loss_cooldown_length=self.stop_loss_cooldown_length,
            stop_loss_cooldown_limit=self.stop_loss_cooldown_limit,
            cooldown_hold_ratio=self.cooldown_hold_ratio,
        )
        self.reward_adapter = RewardAdapter()
        self.account_updater = AccountSeriesUpdater()
        self.termination_evaluator = TerminationEvaluator()

        self._potential_state = PotentialState(
            margin_buffer=None,
            mae_atr=None,
            dist_to_extreme_atr=None,
            has_position=False,
        )
        
        # 倉位追蹤（用於計算換手）
        self._last_position_size = 0.0

        # 延後打印：原始數據列、特徵列與最終觀察欄位（包含帳戶4欄）
        try:
            account_feature_names = ['position', 'position_value', 'equity', 'wallet']
            #print(f"使用的原始數據列: {list(self.df.columns)}")
            #print(f"基礎特徵列: {list(self.obs_features.columns[:self.base_feature_count]) if hasattr(self, 'base_feature_count') else 'N/A'}")
            # 依據組裝順序，extra_features 已併入 obs_features，這裡直接打印全部觀察特徵名稱
            #print(f"觀察特徵列（不含帳戶4欄）: {list(self.obs_features.columns)}")
            #print(f"帳戶狀態欄位: {account_feature_names}")
            #print(f"最終觀察欄位（總數={self.n_features}）：{list(self.obs_features.columns) + account_feature_names}")
        except Exception:
            pass

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
        self.episode_peak_equity = float(self.initial_balance)
        self.consecutive_stop_losses = 0
        
        # 交易追蹤
        self.open_trades = []  # 記錄開倉信息
        self.closed_trades = []  # 記錄平倉信息
        self.last_position = 0  # 記錄上一次的持倉量
        self.avg_entry_price = 0  # 平均進場價格
        
        # PBRS 勢能追蹤（重置）
        self._potential_state = PotentialState(
            margin_buffer=None,
            mae_atr=None,
            dist_to_extreme_atr=None,
            has_position=False,
        )
        
        # 統計追蹤（本 episode）
        self.episode_stop_loss_count = 0  # 本集止損次數
        self.episode_liq_count = 0  # 本集清算次數
        self.episode_trade_count = 0
        
        # 本回合最大步數（受資料長度限制）
        self.episode_start_step = int(self.current_step)
        self.episode_max_steps = max(0, (len(self.df) - 1) - self.episode_start_step)
        
        # 初始化帳戶狀態時間序列（用當前值填滿至 current_step 作為初始歷史）
        current_price = float(self.df.iloc[self.current_step]['close'])
        pos_norm = self.executor.position.size / (self.initial_balance / current_price)#持倉比例正規化
        pos_value_norm = (self.executor.position.size * current_price) / self.initial_balance#持倉價值正規化
        equity_norm = self.executor.equity(current_price) / self.initial_balance#權益正規化
        wallet_norm = self.executor.wallet_balance / self.initial_balance#資金正規化

        # 初始化帳戶狀態時間序列（用當前值填滿至 current_step 作為初始歷史） 
        for key, value in (
            ('position', pos_norm),
            ('position_value', pos_value_norm),
            ('equity', equity_norm),
            ('wallet', wallet_norm),
        ):
            self.account_series[key][:self.current_step] = value

        return self._get_observation(), {}

    def _current_position_percent(self, *, snapshot) -> float:
        current_size = float(self.executor.position.size)
        if abs(current_size) <= 1e-8:
            return 0.0
        wallet_balance = float(getattr(self.executor, 'wallet_balance', 0.0))
        if wallet_balance <= 1e-8 or snapshot.current_price <= 0.0 or self.leverage <= 0.0:
            return 0.0
        return float((current_size * snapshot.current_price) / (wallet_balance * self.leverage))

    def _would_open_new_entry(self, *, position_percent: float, snapshot) -> bool:
        executor = self.executor
        current_price = float(snapshot.current_price)
        wallet_balance = float(getattr(executor, 'wallet_balance', 0.0))
        current_size = float(executor.position.size)
        leverage = float(self.leverage)
        min_qty = float(getattr(executor, 'min_trade_qty', 0.0))

        if current_price <= 0.0 or leverage <= 0.0 or wallet_balance <= 0.0:
            return False

        target_size = (wallet_balance * position_percent * leverage) / current_price

        # 無倉 → 任何達到最小交易量的目標都算新進場
        if abs(current_size) <= 1e-8:
            return abs(target_size) >= min_qty

        # 反手：關掉舊倉再開新倉
        if current_size * target_size < 0 and abs(target_size) >= min_qty:
            return True

        # 同向加碼（增加持倉）
        delta_size = target_size - current_size
        if current_size * target_size > 0 and abs(delta_size) >= min_qty:
            return True

        return False
    
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
        執行交易並透過模組化流程更新市場快照、交易統計、帳戶序列與終止條件。

        Args:
            action: 目標持倉比例 (-1.0 ~ 1.0)

        Returns:
            observation: 觀測
            reward: 獎勵
            done: 是否結束
            truncated: 是否截斷
            info: 信息
        '''
        position_percent = float(action)#目標持倉比例

        if self.min_position_delta > 0.0:
            target_percent = float(getattr(self.executor, 'target_position_percent', 0.0))#目標持倉比例 
            if abs(position_percent - target_percent) < self.min_position_delta:
                return self._get_observation(), 0.0, self.done, False, {}

        # 建立市場快照
        snapshot = self.market_builder.build(
            df=self.df,
            obs_features=self.obs_features,
            current_step=self.current_step,
            window_size=self.window_size,
            executor=self.executor,
        )

        trade_blocked = False#交易是否被阻塞
        if (
            self.max_daily_trades is not None#最大每日交易次數
            and self.episode_trade_count >= self.max_daily_trades#本回合交易次數是否達到最大每日交易次數
            and self._would_open_new_entry(position_percent=position_percent, snapshot=snapshot)#是否可以開新倉
        ):
            trade_blocked = True
            position_percent = self._current_position_percent(snapshot=snapshot)#目前持倉比例

        # 執行交易
        execution = self.execution_processor.execute(
            executor=self.executor,
            action=position_percent,
            snapshot=snapshot,
            leverage=self.leverage,
            last_position_size=self._last_position_size,
        )

        current_price = snapshot.current_price
        self.balance = float(self.executor.wallet_balance)
        self.btc_held = float(self.executor.position.size)
        self.total_value = execution.new_equity
        self._last_position_size = execution.updated_last_position_size

        current_equity = float(execution.new_equity)
        if current_equity > self.episode_peak_equity:
            self.episode_peak_equity = current_equity

        if execution.stop_loss_triggered:
            self.episode_stop_loss_count += 1
            self.consecutive_stop_losses += 1

        reward_result = self.reward_adapter.compute(
            execution=execution,
            use_rudder=self.use_rudder,
            shaping_calc=self.shaping_calc,
            outcome_calc=self.outcome_calc,
            cost_calc=self.cost_calc,
            reward_calculator=self.reward_calculator,
            last_equity=snapshot.last_equity,
            new_equity=execution.new_equity,
            realized_pnl_step=execution.realized_pnl_step,
            episode_steps=self.episode_steps,
            termination_reason=None,
            previous_state=self._potential_state,
            initial_equity=self.initial_balance,
        )
        reward = reward_result.reward
        self._potential_state = reward_result.updated_state

        self.account_updater.update(
            account_series=self.account_series,
            current_step=self.current_step,
            initial_balance=self.initial_balance,
            current_price=current_price,
            executor=self.executor,
        )

        self.current_step += 1
        self.episode_steps += 1

        liq_triggered_flag = bool(self.executor.liq_triggered)
        termination = self.termination_evaluator.evaluate(
            current_step=self.current_step,
            data_len=len(self.df),
            new_equity=execution.new_equity,
            min_balance=self.min_balance,
            liq_triggered=liq_triggered_flag,
        )
        self.done = termination.done

        if liq_triggered_flag:
            self.episode_liq_count += 1

        info = self.info_builder.build(
            StepContext(
                executor=self.executor,
                has_position=execution.has_position,
                stop_loss_triggered=execution.stop_loss_triggered,
                liq_triggered=liq_triggered_flag,
                current_price=current_price,
                use_rudder=self.use_rudder,
                outcome_calc=self.outcome_calc,
                cost_calc=self.cost_calc,
                margin_buffer=execution.margin_buffer,
                mae_atr=execution.mae_atr,
                dist_to_extreme_atr=execution.dist_to_extreme_atr,
                traded=execution.position_change > 1e-8,
                entry_happened=execution.entry_happened,
                entry_streak_count=execution.entry_streak_count,
                peak_equity=self.episode_peak_equity,
                current_equity=current_equity,
                consecutive_stop_losses=self.consecutive_stop_losses,
                cooldown_active=execution.cooldown_active,
                cooldown_remaining=execution.cooldown_remaining,
                cooldown_prevent_entry=execution.cooldown_prevent_entry,
            )
        )
        if execution.entry_happened:
            self.episode_trade_count += 1


        info.update(reward_result.details)

        if execution.stop_loss_triggered:
            info['stop_loss_triggered'] = True
        else:
            # 若出場且不是止損，重置連續止損計數
            if info.get('is_exit', False) and info.get('exit_reason') != 'stop_loss':
                self.consecutive_stop_losses = 0

        if trade_blocked:
            info['trade_limit_exceeded'] = True
            info['episode_trade_count'] = int(self.episode_trade_count)

        if self.done:
            reason = termination.reason or 'other'
            info['termination_reason'] = reason

            if reason == 'data_exhausted':
                print(f"Episode結束：數據用完 (step={self.current_step}, data_len={len(self.df)})")
            elif reason == 'liq_triggered':
                print(f"Episode結束：強平 (balance={self.balance:.2f})")
            elif reason == 'balance_insufficient':
                print(f"Episode結束：資金不足 (balance={self.balance:.2f}, min={self.min_balance})")

            info['final_balance'] = float(execution.new_equity)
            info['profit'] = float(execution.new_equity - self.initial_balance)
            info['profit_rate'] = float((info['profit'] / self.initial_balance) * 100) if self.initial_balance > 0 else 0.0

            try:
                info['long_close_count'] = int(self.executor.long_close_count)
                info['short_close_count'] = int(self.executor.short_close_count)
                info['total_fees'] = float(self.executor.total_fees)
                info['episode_steps'] = int(self.episode_steps)
                info['long_entry_count'] = int(self.executor.long_entry_count)
                info['short_entry_count'] = int(self.executor.short_entry_count)
                info['episode_max_steps'] = int(self.episode_max_steps)
                info['data_len'] = int(len(self.df))
                info['window_size'] = int(self.window_size)
                info['episode_stop_loss_count'] = int(self.episode_stop_loss_count)
                info['episode_liq_count'] = int(self.episode_liq_count)
                info['episode_trade_count'] = int(self.episode_trade_count)
            except Exception:
                pass

            if self.use_rudder and abs(self.executor.position.size) > 1e-8:
                self.executor.forced_close_position(current_price)
                if self.executor.last_exit_info:
                    exit_info = self.executor.last_exit_info
                    notional = abs(exit_info['size'] * exit_info['price'])
                    outcome_delta = self.outcome_calc.compute_outcome_delta(
                        realized_pnl=exit_info['realized_pnl'],
                        notional=notional,
                        exit_reason='forced_close_on_done',
                        atr_multiple=exit_info.get('mae_atr'),
                    )
                    info['outcome_delta_to_entry'] = float(outcome_delta)
                    info['exited_trade_id'] = int(exit_info['trade_id'])
                    info['exit_reason'] = 'forced_close_on_done'
                    info['is_exit'] = True

            if not self.use_rudder and self.reward_calculator is not None:
                reward = float(
                    self.reward_calculator.compute(
                        last_equity=snapshot.last_equity,
                        new_equity=execution.new_equity,
                        margin_buffer=execution.margin_buffer,
                        position_change=execution.position_change,
                        turnover_ratio=execution.turnover_ratio,
                        dist_to_extreme_atr=execution.dist_to_extreme_atr,
                        mae_atr=execution.mae_atr,
                        leverage_ratio=execution.leverage_ratio,
                        has_position=execution.has_position,
                        unrealized_pnl=execution.unrealized_pnl,
                        traded=execution.position_change > 1e-8,
                        entry_happened=execution.entry_happened,
                        entry_streak_count=execution.entry_streak_count,
                        entry_streak_side=execution.entry_streak_side,
                        realized_pnl_step=execution.realized_pnl_step,
                        episode_steps=self.episode_steps,
                        stop_loss_triggered=execution.stop_loss_triggered,
                        done=True,
                        termination_reason=reason,
                    )
                )

        return self._get_observation(), reward, self.done, False, info
    