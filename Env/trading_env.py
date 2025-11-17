"""
交易環境模組（重構版）

符合 SAC 拉格朗日 agent 訓練需求，專注於帳戶狀態管理與步進邏輯。
遵循 SOLID 原則，透過依賴注入實現高度可擴展性。

成功條件：資料耗盡
失敗條件：強平次數達上限 或 資金耗盡

擴展點：
1. AccountFeatureBuilder: 自定義帳戶特徵
2. RewardCalculator: 自定義獎勵函數
3. InfoCollector: 自定義資訊收集
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import random
from typing import Optional, Tuple, Dict, Any

from .trade_executor import TradeExecutor
from .reward import RewardCalculator, create_default_calculator
from .account_features import AccountFeatureBuilder, DefaultAccountFeatureBuilder
from .info_collector import InfoCollector, DefaultInfoCollector


class TradingEnvironment(gym.Env):
    """
    交易環境（重構版）
    
    專注於帳戶狀態管理與環境步進邏輯，技術特徵由外部傳入。
    
    Args:
        df: 交易數據 (必須包含 OHLCV 列: open, high, low, close, volume)
        initial_balance: 初始資金
        transaction_fee: 交易手續費比例(%)
        window_size: 觀測窗口大小（時間步數）
        leverage: 槓桿倍數
        min_balance: 最小資金（低於此值視為資金耗盡）
        min_trade_qty: 最低交易數量
        margin_mode: 保證金模式 ('cross': 全倉, 'isolated': 逐倉)
        random_start: 是否隨機起始位置
        account_feature_builder: 帳戶特徵構建器（可自定義）
        reward_calculator: 獎勵計算器（可自定義）
        info_collector: 資訊收集器（可自定義）
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10_000,
        transaction_fee: float = 0.001,
        window_size: int = 24 * 60 // 5,  # 24小時（5分K）
        leverage: float = 10,
        min_balance: float = 100,
        min_trade_qty: float = 0.001,
        margin_mode: str = 'isolated',
        random_start: bool = False,
        account_feature_builder: Optional[AccountFeatureBuilder] = None,
        reward_calculator: Optional[RewardCalculator] = None,
        info_collector: Optional[InfoCollector] = None
    ):
        """初始化交易環境"""
        super(TradingEnvironment, self).__init__()
        
        # 驗證必要數據列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"數據缺少必要列: {missing_columns}")
        
        # 只保留數值列
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.df = df[numeric_columns].copy()
        
        # 基礎參數
        self.initial_balance = float(initial_balance)
        self.transaction_fee = float(transaction_fee)
        self.window_size = int(window_size)
        self.leverage = float(leverage)
        self.min_balance = float(min_balance)#最小資金（低於此值視為資金耗盡）
        self.min_trade_qty = float(min_trade_qty)#最低交易數量
        self.margin_mode = str(margin_mode).lower()#保證金模式 ('cross': 全倉, 'isolated': 逐倉)
        self.random_start = bool(random_start)#是否隨機起始位置
        
        # 依賴注入：可擴展的模組
        self.account_feature_builder = account_feature_builder or DefaultAccountFeatureBuilder()
        self.reward_calculator = reward_calculator or create_default_calculator()
        self.info_collector = info_collector or DefaultInfoCollector()
        
        # 動作空間：連續動作 [-1.0, 1.0]（目標持倉比例）
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # 觀察空間：[數據特徵 + 帳戶特徵] × window_size
        # 數據特徵數量 = df 欄位數量
        self.data_feature_count = len(self.df.columns)
        self.account_feature_count = self.account_feature_builder.feature_count()
        self.n_features = self.data_feature_count + self.account_feature_count
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features, self.window_size),
            dtype=np.float32
        )
        
        # 交易執行器
        self.executor = TradeExecutor(
            initial_balance=self.initial_balance,
            fee_rate=self.transaction_fee,
            leverage=self.leverage,
            min_trade_qty=self.min_trade_qty,
            margin_mode=self.margin_mode,
            stop_loss_atr=3.0,  # 固定止損倍數
        )
        
        # 帳戶狀態時間序列（用於構建觀測窗口）
        series_len = len(self.df)
        self.account_series = {
            name: np.zeros(series_len, dtype=np.float32)
            for name in self.account_feature_builder.feature_names()
        }
        
        # 狀態追蹤變數
        self.current_step: int = 0
        self.episode_steps: int = 0
        self.episode_start_step: int = 0
        self._last_position_size: float = 0.0
        self.done: bool = False
        
        # 打印環境配置
        print(f"=== 交易環境初始化 ===")
        print(f"數據長度: {len(self.df)}")
        print(f"數據特徵數: {self.data_feature_count}")
        print(f"帳戶特徵數: {self.account_feature_count}")
        print(f"帳戶特徵名稱: {self.account_feature_builder.feature_names()}")
        print(f"總觀測特徵數: {self.n_features}")
        print(f"觀測空間形狀: {self.observation_space.shape}")
        print(f"=====================")
        
        self.reset()
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置環境
        
        Args:
            seed: 隨機種子
            options: 額外選項
            
        Returns:
            observation: 初始觀測
            info: 初始資訊
        """
        super().reset(seed=seed)
        
        # 確定起始步數
        if self.random_start and len(self.df) > (self.window_size + 2):
            self.current_step = int(random.randint(self.window_size, len(self.df) - 2))
        else:
            self.current_step = self.window_size
        
        # 重置執行器與狀態
        self.executor.reset(self.initial_balance)
        self.done = False
        self.episode_steps = 0
        self.episode_start_step = int(self.current_step)
        self._last_position_size = 0.0
        
        # 重置資訊收集器（如果支援）
        if hasattr(self.info_collector, 'reset'):
            self.info_collector.reset()
        
        # 初始化帳戶狀態時間序列
        current_price = float(self.df.iloc[self.current_step]['close'])
        self._update_account_series(self.current_step, current_price, fill_history=True)
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """
        構建觀測矩陣
        
        Returns:
            np.ndarray: 觀測矩陣 [n_features, window_size]
        """
        # 確保 current_step 在有效範圍內
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
        
        # 計算窗口範圍
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        # 初始化觀測矩陣
        obs = np.zeros((self.n_features, self.window_size), dtype=np.float32)
        
        # 1. 數據特徵（直接從 df 提取）
        data_window = self.df.iloc[start_idx:end_idx].values.T  # [data_features, window_size]
        obs[:self.data_feature_count] = data_window.astype(np.float32)
        
        # 2. 帳戶特徵（從時間序列提取）
        feature_idx = self.data_feature_count
        for i, name in enumerate(self.account_feature_builder.feature_names()):
            obs[feature_idx + i] = self.account_series[name][start_idx:end_idx]
        
        # 安全處理 NaN/Inf
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    
    def _update_account_series(
        self,
        step: int,
        current_price: float,
        fill_history: bool = False
    ) -> None:
        """
        更新帳戶狀態時間序列
        
        Args:
            step: 當前步數
            current_price: 當前價格
            fill_history: 是否填充歷史（用於 reset）
        """
        # 計算帳戶特徵
        position_size = self.executor.position.size
        position_value = abs(position_size * current_price)
        equity = self.executor.equity(current_price)
        wallet_balance = self.executor.wallet_balance
        
        features = self.account_feature_builder.compute_features(
            position_size=position_size,
            position_value=position_value,
            equity=equity,
            wallet_balance=wallet_balance,
            current_price=current_price,
            initial_balance=self.initial_balance,
            leverage=self.leverage  # 傳遞 leverage 參數供擴展特徵使用
        )
        
        # 更新時間序列
        for i, name in enumerate(self.account_feature_builder.feature_names()):
            if fill_history:
                # 填充整個歷史（用於初始化）
                self.account_series[name][:step] = features[i]
            else:
                # 只更新當前步
                if 0 <= step < len(self.df):
                    self.account_series[name][step] = features[i]
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        執行一步交易
        
        Args:
            action: 動作（目標持倉比例 [-1.0, 1.0]）
            
        Returns:
            observation: 觀測
            reward: 獎勵
            terminated: 是否終止（成功/失敗）
            truncated: 是否截斷（固定為 False）
            info: 資訊字典
        """
        # 取得當前 K 線數據
        candle = self.df.iloc[self.current_step]
        current_price = float(candle['close'])
        current_high = float(candle['high'])
        current_low = float(candle['low'])
        
        # 記錄執行前狀態
        last_equity = self.executor.equity(current_price)
        prev_wallet_balance = float(self.executor.wallet_balance)
        
        # 估算 ATR（用於止損計算）
        atr_est = self._estimate_atr(current_price)
        
        # 執行交易
        position_percent = float(action[0])
        self.executor.execute(
            position_percent=position_percent,
            current_price=current_price,
            high=current_high,
            low=current_low,
            equity=last_equity,
            atr=atr_est
        )
        
        # 同步帳戶狀態
        new_equity = self.executor.equity(current_price)
        realized_pnl_step = float(self.executor.wallet_balance - prev_wallet_balance)
        
        # 計算輔助指標（供 reward 使用）
        position_change = abs(float(self.executor.position.size - self._last_position_size))
        traded = position_change > 1e-8
        self._last_position_size = float(self.executor.position.size)
        
        unrealized_pnl = float(self.executor.unrealized_pnl(current_price))
        has_position = abs(self.executor.position.size) > 1e-8
        
        # 止損觸發處理（統計由 executor 管理）
        stop_loss_triggered = self.executor.stop_loss_triggered
        
        # 計算保證金緩衝
        margin_buffer = self._compute_margin_buffer(new_equity, current_price)
        
        # 計算獎勵
        reward = self.reward_calculator.compute(
            last_equity=last_equity,
            new_equity=new_equity,
            margin_buffer=margin_buffer,
            position_change=position_change,
            has_position=has_position,
            unrealized_pnl=unrealized_pnl,
            traded=traded,
            realized_pnl_step=realized_pnl_step,
            episode_steps=self.episode_steps,
            stop_loss_triggered=stop_loss_triggered,
            liq_triggered=self.executor.liq_triggered
        )
        
        # 更新帳戶狀態時間序列
        self._update_account_series(self.current_step, current_price)
        
        # 更新步數
        self.current_step += 1
        self.episode_steps += 1
        
        # 檢查終止條件
        terminated, termination_reason = self._check_termination(new_equity)
        
        # 收集資訊
        info = self.info_collector.collect_step_info(
            current_step=self.current_step,
            current_price=current_price,
            position_size=self.executor.position.size,
            equity=new_equity,
            stop_loss_triggered=stop_loss_triggered
        )
        
        # 如果終止，收集回合資訊並重新計算最終獎勵
        if terminated:
            episode_info = self.info_collector.collect_episode_info(
                termination_reason=termination_reason,
                final_equity=new_equity,
                initial_balance=self.initial_balance,
                episode_steps=self.episode_steps,
                executor=self.executor,
                data_len=len(self.df),
                window_size=self.window_size,
                episode_max_steps=(len(self.df) - 1) - self.episode_start_step
            )
            info.update(episode_info)
            
            # 重新計算終局獎勵
            reward = self.reward_calculator.compute(
                last_equity=last_equity,
                new_equity=new_equity,
                done=True,
                termination_reason=termination_reason,
                realized_pnl_step=realized_pnl_step,
                episode_steps=self.episode_steps,
                margin_buffer=margin_buffer,
                stop_loss_triggered=stop_loss_triggered,
                initial_balance=self.initial_balance,
                liq_triggered=self.executor.liq_triggered
            )
            
            # 打印終止訊息
            if termination_reason == 'data_exhausted':
                print(f"[SUCCESS] Episode成功：資料耗盡 (步數={self.episode_steps}, 權益={new_equity:.2f}, 報酬率={info['profit_rate']:.2f}%)")
            elif termination_reason == 'balance_insufficient':
                print(f"[FAIL] Episode失敗：資金不足 (權益={new_equity:.2f}, 最小={self.min_balance})")
        
        self.done = terminated
        
        return self._get_observation(), reward, terminated, False, info
    
    def _estimate_atr(self, current_price: float) -> float:
        """
        估算當前 ATR
        
        Args:
            current_price: 當前價格
            
        Returns:
            float: ATR 估算值
        """
        try:
            # 簡易 ATR 估算：過去 14 根的高低差均值
            lookback = min(14, self.current_step)
            if lookback > 0:
                start_idx = self.current_step - lookback
                high_low_range = (
                    self.df['high'].iloc[start_idx:self.current_step] -
                    self.df['low'].iloc[start_idx:self.current_step]
                )
                atr_est = float(high_low_range.mean())
                return max(atr_est, current_price * 0.001)  # 最小 0.1%
        except Exception:
            pass
        return current_price * 0.02  # 預設 2%
    
    def _compute_margin_buffer(self, equity: float, current_price: float) -> float:
        """
        計算保證金緩衝比例
        
        Args:
            equity: 當前權益
            current_price: 當前價格
            
        Returns:
            float: 緩衝比例 [0.0, 1.0]
        """
        try:
            position_value = abs(float(self.executor.position.size * current_price))
            if equity > 0:
                leverage_ratio = position_value / equity
                safe_leverage = float(self.leverage) * 0.8
                if leverage_ratio >= safe_leverage:
                    return 0.0
                else:
                    buffer = 1.0 - (leverage_ratio / safe_leverage)
                    return float(np.clip(buffer, 0.0, 1.0))
        except Exception:
            pass
        return 1.0
    
    def _check_termination(self, equity: float) -> Tuple[bool, str]:
        """
        檢查終止條件（簡化版）
        
        成功條件：資料耗盡
        失敗條件：資金不足
        止損/強平後允許繼續交易
        
        Args:
            equity: 當前權益
            
        Returns:
            (is_terminated, termination_reason)
        """
        # 成功條件：資料耗盡
        if self.current_step >= len(self.df) - 1:
            return True, 'data_exhausted'
        
        # 失敗條件：資金不足
        if equity <= self.min_balance:
            return True, 'balance_insufficient'
        
        return False, ''
