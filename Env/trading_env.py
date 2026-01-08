import gymnasium as gym
import numpy as np
import random
from dataclasses import dataclass
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any

from Env.config import Config
from Env.Executors.trade_executor import TradeExecutor
from Env.Rewards.reward import create_default_calculator
from Env.Components.market_data import MarketData
from Env.Components.observer import TradingObserver
from Env.Components.action_processor import ActionProcessor
from Env.Components.tracker import Tracker
from Env.load_file import load_data
from Env.Costs.cost import CostCalculator, CostWeights


@dataclass(frozen=True)
class _StepPrices:
    """step() 單步計算會用到的價格與波動資訊。"""

    current_price: float
    current_high: float
    current_low: float
    atr_est: float


class TradingEnvironment(gym.Env):
    """
    重構後的交易環境，採用組件化設計。
    - MarketData: 數據與特徵 (支援 5min/1day 雙週期)
    - TradeExecutor: 交易執行與帳務
    - TradingObserver: 觀察值生成
    - ActionProcessor: 動作處理與限制
    - Tracker: 狀態追蹤與日誌
    - RewardCalculator: 獎勵計算
    """
    def __init__(self, env_id: int = 0, **kwargs):
        """
        Args:
            df_5m: 5分鐘線數據 (主要執行時間軸)
            df_1d: 日線數據 (背景趨勢參考)
            env_id: 環境 ID (用於 Log)
            **kwargs: 覆寫 Config 的參數
        """
        super(TradingEnvironment, self).__init__()
        
        # 1. 配置參數載入 (優先使用 kwargs，後備 Config)
        self.env_id = int(env_id)
        self.initial_balance = float(kwargs.get("initial_balance", Config.INITIAL_BALANCE))
        self.transaction_fee = float(kwargs.get("transaction_fee", Config.TRANSACTION_FEE))
        self.window_size = int(kwargs.get("window_size", Config.WINDOW_SIZE))
        self.window_size_1d = int(kwargs.get("window_size_1d", Config.WINDOW_SIZE_1D))
        self.leverage = float(kwargs.get("leverage", Config.LEVERAGE))
        self.min_balance = float(kwargs.get("min_balance", Config.MIN_BALANCE))
        self.min_episode_steps = int(kwargs.get("min_episode_steps", Config.MIN_EPISODE_STEPS))
        self.max_episode_steps = getattr(Config, 'MAX_EPISODE_STEPS', 1000000)
        self.min_position_change = float(kwargs.get("min_position_change", Config.MIN_POSITION_CHANGE))
        self.random_start = kwargs.get('random_start', True)
        self.target_symbol = kwargs.get('target_symbol', 'BTCUSDT') # 預設交易對
        self.margin_mode = 'isolated'
        self.min_trade_qty = 0.001 

        # 載入數據
        self.df_5m, self.df_1d = load_data()

        # 2. 初始化組件
        # Market Data (傳入兩個 DataFrame，並指定目標交易對)
        self.market_data = MarketData(self.df_5m, self.df_1d, self.window_size, self.window_size_1d, target_symbol=self.target_symbol)
        
        # Observer
        self.observer = TradingObserver(self.window_size, self.window_size_1d, self.market_data)
        self.observation_space = self.observer.observation_space
        
        # Action Processor
        self.action_processor = ActionProcessor(
            leverage=self.leverage,
            flip_budget_max=Config.FLIP_BUDGET_MAX,
            flip_cost=Config.FLIP_COST,
            flip_threshold=Config.FLIP_THRESHOLD,
            max_step_pos_change_pct=float(kwargs.get("max_step_pos_change_pct", Config.MAX_STEP_POS_CHANGE_PCT)),
            min_position_change=self.min_position_change
        )
        # Action Space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Executor
        self.executor = TradeExecutor(
            initial_balance=self.initial_balance,
            fee_rate=self.transaction_fee,
            leverage=self.leverage,
            min_trade_qty=self.min_trade_qty,
            maintenance_margin_rate=Config.MAINTENANCE_MARGIN_RATE,
            margin_mode=self.margin_mode,
            min_position_change=self.min_position_change,
            stop_loss_atr=float(kwargs.get("stop_loss_atr", Config.STOP_LOSS_ATR)),
            stop_loss_liq_buffer_pct=getattr(Config, "STOP_LOSS_LIQ_BUFFER_PCT", 0.0),
        )
        
        # Reward Calculator
        self.reward_calculator = create_default_calculator(
            c_liq=0.0,
            base_log_ret_weight=1.0,
            conviction_trend_bonus_weight=0.0,
            conviction_trend_min_strength=0.8,
            conviction_min_abs_pos=0.15,
        )
        
        # Tracker
        self.tracker = Tracker(
            step_log_enabled=Config.STEP_LOG_ENABLED,
            step_log_dir=Config.STEP_LOG_DIR,
            step_log_every_n=Config.STEP_LOG_EVERY_N,
            env_id=self.env_id,
            initial_balance=self.initial_balance,
            data_len=len(self.df_5m),
            fee_rolling_window=int(kwargs.get("fee_rolling_window", Config.FEE_ROLLING_WINDOW))
        )
        
        # Fee Limit
        self.fee_limit_enabled = getattr(Config, "FEE_LIMIT_ENABLED", True)
        self.fee_limit_ratio = float(kwargs.get("fee_limit_ratio", Config.FEE_LIMIT_RATIO))

        # Cost / Constraint（供 Lagrangian-SAC 使用）
        # 注意：reward 與 cost 分離，cost 透過 info 回傳，方便訓練端做 λ 更新與解析。
        # REFACTORED: 只保留死亡懲罰 (Liq / Bankrupt) 與 摩擦成本 (Fee/Equity)
        # CostCalculator 現在不再需要 weights (已內建正規化公式)，這裡維持空建構
        self.cost_calculator = CostCalculator()

        # Runtime State
        self.current_step = 0
        self.episode_steps = 0
        self.done = False
        self.risk_budget = 0.0
        self.last_equity_for_budget = 0.0
        self.max_equity_so_far = 0.0
        # Episode-level max drawdown (0~1). Used for trade stats (e.g., last 100 episodes max DD).
        self.episode_max_dd = 0.0
        # Episode-level metrics (for Trade Stats)
        self.episode_turnover_notional = 0.0
        self.episode_holding_steps = 0
        self.episode_trade_count = 0
        # Episode-level event metrics (for Trade Stats)
        # 主動出場：由 agent 動作將持倉平到 0（排除 stop loss / liquidation 強制出場）
        self.episode_active_exit_count = 0
        
        # Action-conditioned effects cache (for next obs)
        self._last_action_effects = {}
        
        # Tracking helper
        self.last_trade_step = -999999
        self.position_entry_step = None
        self._last_position_size = 0.0
        self.episode_stop_loss_count = 0
        self.episode_liq_count = 0
        self.stop_loss_cooldown = 0
        self.daily_risk_base = 0.0
        self.last_risk_base_update_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        # 1. 決定起始點
        if self.random_start:
            max_start_index = len(self.market_data.df_5m) - self.min_episode_steps - 2
            # 若資料量不足以支援 min_episode_steps（常見於測試用合成資料），退化為固定起點，避免 randint 空範圍。
            if int(max_start_index) <= int(self.window_size):
                self.current_step = int(self.window_size)
            else:
                self.current_step = int(random.randint(self.window_size, max_start_index))
        else:
            self.current_step = self.window_size 
            
        self.episode_start_step = self.current_step
        self.episode_max_steps = min(
            max(0, (len(self.market_data.df_5m) - 1) - self.episode_start_step),
            self.max_episode_steps
        )
        self.episode_steps = 0
        self.done = False
        
        # 2. Reset Components
        self.executor.reset(self.initial_balance)
        self.tracker.fee_history.clear()
        self.tracker.rolling_fee_sum = 0.0
        self.tracker.prev_total_fees = 0.0
        
        # 3. Reset State Variables
        self.risk_budget = Config.FLIP_BUDGET_MAX
        self.last_equity_for_budget = self.initial_balance
        self.max_equity_so_far = self.initial_balance
        self.episode_max_dd = 0.0
        self.episode_turnover_notional = 0.0
        self.episode_holding_steps = 0
        self.episode_trade_count = 0
        self.episode_active_exit_count = 0
        
        self.last_trade_step = -999999
        self.position_entry_step = None
        self._last_position_size = 0.0
        self.episode_stop_loss_count = 0
        self.episode_liq_count = 0
        self.stop_loss_cooldown = 0
        self.daily_risk_base = self.initial_balance
        self.last_risk_base_update_step = self.current_step
        
        self._last_action_effects = {
            "expected_fee_if_trade": 0.0,
            "predicted_used_margin_after_action": 0.0,
            "predicted_available_balance_after_action": 0.0,
            "predicted_liq_distance_after_action": 0.0,
            "predicted_stop_distance_after_action": 0.0,
        }

        # 4. Initial Observation
        metrics = self.market_data.get_market_metrics(self.current_step)
        current_price = metrics['close']
        self.tracker.update_account_series(self.current_step, self.executor, current_price)
        
        return self._get_observation(), {}

    def _get_observation(self):
        # 準備 Observation 需要的各類 metrics
        metrics = self.market_data.get_market_metrics(self.current_step)
        current_price = metrics['close']
        
        risk_signals = self.observer.compute_risk_signals(
            self.executor, current_price, self.current_step, len(self.market_data.df_5m)
        )
        
        account_metrics = {
            'initial_balance': self.initial_balance,
            'max_equity_so_far': self.max_equity_so_far,
            'episode_stop_loss_count': self.episode_stop_loss_count,
            'episode_liq_count': self.episode_liq_count,
            'risk_budget': self.risk_budget,
            'steps_since_trade': float(self.current_step - self.last_trade_step) if self.last_trade_step > -1e8 else float(self.window_size),
            'holding_steps': float(self.current_step - self.position_entry_step) if self.position_entry_step is not None else 0.0,
            'last_step_fee': self.tracker.last_step_fee,
            'rolling_fee_sum': self.tracker.rolling_fee_sum,
            'fee_limit_ratio': self.fee_limit_ratio,
            'fee_limit_enabled': self.fee_limit_enabled
        }
        
        return self.observer.get_observation(
            step_idx=self.current_step,
            executor=self.executor,
            market_data=self.market_data,
            account_metrics=account_metrics,
            risk_signals=risk_signals,
            last_action_effects=self._last_action_effects
        )

    def _determine_termination(
        self,
        *,
        data_exhausted: bool,
        max_steps_reached: bool,
        balance_insufficient: bool,
        liq_triggered: bool,
    ) -> Tuple[bool, bool, Optional[str]]:
        """
        判斷回合是否結束，並以 Gymnasium 語意回傳 terminated/truncated。

        Args:
            data_exhausted: 數據走完（通常視為 truncated）
            max_steps_reached: 回合步數達上限（視為 truncated）
            balance_insufficient: 權益低於下限（視為 terminated）
            liq_triggered: 觸發爆倉（視為 terminated）

        Returns:
            (terminated, truncated, termination_reason)
        """
        terminated = bool(liq_triggered or balance_insufficient)
        truncated = bool((data_exhausted or max_steps_reached) and not terminated)

        termination_reason: Optional[str] = None
        if terminated:
            termination_reason = "liq_triggered" if liq_triggered else "balance_insufficient"
        elif truncated:
            termination_reason = "max_steps_reached" if max_steps_reached else "data_exhausted"

        return terminated, truncated, termination_reason

    def _build_step_info(
        self,
        *,
        new_equity: float,
        stop_loss_triggered: bool,
        liq_triggered: bool,
        step_fee: float,
        is_flip: bool,
        current_dd: float,
        episode_max_dd: float,
        episode_turnover_notional: float,
        episode_holding_steps: int,
        episode_trade_count: int,
        terminated: bool,
        truncated: bool,
        termination_reason: Optional[str],
    ) -> Dict[str, Any]:
        """
        組合 step() 要回傳的 info dict（抽離 step 內的大段組裝邏輯）。

        Args:
            new_equity: 本 step 後的權益（mark-to-market）
            stop_loss_triggered: 是否本 step 觸發停損
            liq_triggered: 是否本 step 觸發爆倉
            step_fee: 本 step 手續費
            is_flip: 是否發生翻倉/反手（由 ActionProcessor 回傳）
            current_dd: 當前回撤
            episode_max_dd: 本回合迄今最大回撤（0~1）
            episode_turnover_notional: 本回合累積換手名目（sum(abs(delta_size) * price)）
            episode_holding_steps: 本回合持倉步數（abs(position.size)>0 的 step 數）
            episode_trade_count: 本回合發生交易的 step 數（position_change > threshold）
            terminated: Gymnasium terminated（自然終止）
            truncated: Gymnasium truncated（時間/資料截斷）
            termination_reason: 終止原因（若結束回合）

        Returns:
            info dict
        """
        done = bool(terminated or truncated)
        info: Dict[str, Any] = {
            "equity": float(new_equity),
            "profit": float(new_equity - self.initial_balance),
            "stop_loss_triggered": bool(stop_loss_triggered),
            "liq_triggered": bool(liq_triggered),
            "step_fee_ratio": float(step_fee / self.initial_balance) if self.initial_balance > 0 else 0.0,
            "is_flip": bool(is_flip),
            "risk_budget": float(self.risk_budget),
            "current_dd": float(current_dd),
        }

        if done:
            info["termination_reason"] = termination_reason
            info["final_balance"] = float(new_equity)
            info["episode_max_dd"] = float(episode_max_dd)
            info["episode_turnover_notional"] = float(max(0.0, episode_turnover_notional))
            info["episode_holding_steps"] = int(max(0, int(episode_holding_steps)))
            info["episode_trade_count"] = int(max(0, int(episode_trade_count)))
            info["fees_to_equity_ratio"] = (
                float(getattr(self.executor, "total_fees", 0.0)) / float(max(1e-8, new_equity))
            )
            # 額外提供 Gymnasium 語意旗標，方便外部檢查（不影響既有 key）
            info["terminated"] = bool(terminated)
            info["truncated"] = bool(truncated)

            # ---- Episode summary（供訓練端每 N 回合統計/解析用）----
            # 注意：這些統計只在回合結束時提供，避免每步 info 過大造成效能負擔。
            # 1) 手續費（累積）
            info["total_fees"] = float(getattr(self.executor, "total_fees", 0.0))
            info["total_fees_ratio"] = (
                float(getattr(self.executor, "total_fees", 0.0)) / float(self.initial_balance)
                if self.initial_balance > 0
                else 0.0
            )
            # 2) 多空進場/平倉次數（累積）
            info["long_entry_count"] = int(getattr(self.executor, "long_entry_count", 0))
            info["short_entry_count"] = int(getattr(self.executor, "short_entry_count", 0))
            info["long_close_count"] = int(getattr(self.executor, "long_close_count", 0))
            info["short_close_count"] = int(getattr(self.executor, "short_close_count", 0))
            # 3) 結束時庫存（持倉 size）
            info["final_position_size"] = float(getattr(self.executor.position, "size", 0.0))
            info["final_position_notional"] = float(getattr(self.executor.position, "size", 0.0)) * float(
                new_equity
            )  # 粗略參考（不一定等於名目）
            # 4) 回合事件統計（累積）
            info["episode_stop_loss_count"] = int(getattr(self, "episode_stop_loss_count", 0))
            info["episode_liq_count"] = int(getattr(self, "episode_liq_count", 0))
            info["episode_active_exit_count"] = int(getattr(self, "episode_active_exit_count", 0))

        return info

    def _build_log_payload(
        self,
        *,
        step: int,
        new_equity: float,
        reward: float,
        action: np.ndarray,
        final_pos_pct: float,
    ) -> Dict[str, Any]:
        """
        組合 Tracker log payload（維持原本欄位命名，抽離 step 內的雜訊）。

        Args:
            step: 當前 step index（環境內部）
            new_equity: 權益
            reward: 獎勵
            action: 原始 action
            final_pos_pct: 最終執行倉位百分比

        Returns:
            log payload dict
        """
        return {
            "step": int(step),
            "equity": float(new_equity),
            "reward": float(reward),
            "action": float(action[0]) if hasattr(action, "__len__") else float(action),
            "final_pos": float(final_pos_pct),
        }

    def _maybe_update_daily_risk_base(self) -> None:
        """
        以固定間隔更新 daily_risk_base（原本 step() 內的邏輯抽離，行為不變）。
        """
        if (self.current_step - self.last_risk_base_update_step) >= self.window_size:
            self.daily_risk_base = float(self.executor.wallet_balance)
            self.last_risk_base_update_step = self.current_step

    def _prepare_step_prices(self, metrics: Dict[str, Any]) -> _StepPrices:
        """
        從 market metrics 萃取 step() 會用到的價格資訊。

        Args:
            metrics: MarketData.get_market_metrics 回傳 dict

        Returns:
            _StepPrices
        """
        current_price = float(metrics["close"])
        current_high = float(metrics["high"])
        current_low = float(metrics["low"])
        atr_est = float(metrics["atr_ratio"]) * current_price
        return _StepPrices(
            current_price=current_price,
            current_high=current_high,
            current_low=current_low,
            atr_est=atr_est,
        )

    def _apply_stop_loss_cooldown(self, action: np.ndarray) -> np.ndarray:
        """
        若處於停損冷卻期，強制本 step 動作為 0。

        Args:
            action: 原始 action

        Returns:
            action（可能被覆寫為 0）
        """
        if self.stop_loss_cooldown > 0:
            self.stop_loss_cooldown -= 1
            return np.zeros_like(action)
        return action

    def _process_action_and_execute(
        self,
        *,
        action: np.ndarray,
        last_equity: float,
        prices: _StepPrices,
    ) -> Tuple[float, float, float, bool, np.ndarray]:
        """
        動作處理（含限制/翻倉預算）+ 手續費預估 + 實際下單執行。

        Args:
            action: 原始 action
            last_equity: 以 current_price 計算的上一刻權益
            prices: 本 step 價格資訊

        Returns:
            (final_pos_pct, expected_fee, prev_wallet, is_flip, action_used)
        """
        action_used = self._apply_stop_loss_cooldown(action)

        target_pos_pct, new_risk_budget, _flip_blocked, _flip_budget_spent, is_flip = (
            self.action_processor.process_action(
                action_used, self.executor, prices.current_price, self.risk_budget
            )
        )
        self.risk_budget = new_risk_budget

        final_pos_pct = self.action_processor.calculate_effective_action(
            target_pos_pct, self.executor, prices.current_price, self.daily_risk_base
        )

        expected_fee = self._estimate_expected_fee(
            final_pos_pct=final_pos_pct,
            last_equity=last_equity,
            current_price=prices.current_price,
        )

        prev_wallet = float(self.executor.wallet_balance)
        self.executor.execute(
            position_percent=final_pos_pct,
            current_price=prices.current_price,
            high=prices.current_high,
            low=prices.current_low,
            equity=last_equity,
            atr=prices.atr_est,
            risk_base=self.daily_risk_base,
        )

        return float(final_pos_pct), float(expected_fee), float(prev_wallet), bool(is_flip), action_used

    def _estimate_expected_fee(
        self,
        *,
        final_pos_pct: float,
        last_equity: float,
        current_price: float,
    ) -> float:
        """
        預估本次動作若成交可能產生的手續費（供 next obs 參考）。

        注意：此為近似值，維持原本 step() 內的估算方式。
        """
        current_size = float(self.executor.position.size)
        desired_notional = float(final_pos_pct) * float(last_equity) * float(self.leverage)
        desired_size = desired_notional / float(current_price) if current_price > 0 else 0.0
        fee_rate_pct = float(self.executor.get_fee_rate())
        return abs(desired_size - current_size) * float(current_price) * (fee_rate_pct / 100.0)

    def _update_action_effects_cache(self, *, expected_fee: float, current_price: float) -> None:
        """
        更新 last_action_effects（供下一個 observation 使用）。
        """
        try:
            used_margin_after = float(getattr(self.executor, "used_margin", 0.0))
            available_after = float(self.executor.available_balance())
            liq_after = float(self.executor.get_liquidation_price(current_price))
            liq_dist_after = float(abs(current_price - liq_after) / current_price) if liq_after > 0 else 0.0

            entry_after = float(self.executor.position.entry_price)
            stop_after = float(self.executor.position.stop_loss_price)
            stop_dist_after = (
                float(abs(entry_after - stop_after) / entry_after)
                if (entry_after > 0 and stop_after > 0)
                else 0.0
            )

            self._last_action_effects = {
                "expected_fee_if_trade": float(expected_fee),
                "predicted_used_margin_after_action": used_margin_after,
                "predicted_available_balance_after_action": available_after,
                "predicted_liq_distance_after_action": np.clip(liq_dist_after, 0.0, 5.0),
                "predicted_stop_distance_after_action": np.clip(stop_dist_after, 0.0, 5.0),
            }
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            # 預測值僅供 next obs 參考，不應因偶發資料/狀態異常中斷訓練流程
            pass

    def _update_position_entry(self, *, new_size: float) -> None:
        """
        更新 position_entry_step（用於 holding_steps 等觀測特徵）。
        """
        if abs(new_size) <= 1e-12:
            self.position_entry_step = None
        elif (self._last_position_size * new_size < 0) or (abs(self._last_position_size) <= 1e-12):
            self.position_entry_step = self.current_step

    def _update_fee_tracking(self, *, current_price: float) -> Tuple[float, float]:
        """
        更新 rolling fee tracking，並回傳 step_fee 與 safe_equity（供 fee_budget_ratio 使用）。

        Returns:
            (step_fee, safe_equity)
        """
        self.tracker.update_fee_tracking(self.current_step, self.executor.total_fees)
        step_fee = float(self.tracker.last_step_fee)

        # Fee Limit Check（維持原邏輯：使用 current_price 當下估 equity）
        safe_equity = max(float(self.executor.equity(current_price)), self.initial_balance * 0.5)
        if self.fee_limit_enabled:
            _limit_amount = safe_equity * self.fee_limit_ratio
            _fee_limit_hit = self.tracker.rolling_fee_sum >= _limit_amount
        return step_fee, float(safe_equity)

    def _mark_to_market(self) -> Tuple[float, float]:
        """
        以 next close 做 mark-to-market（原本 step() 的做法），並更新 max_equity_so_far。

        Returns:
            (mark_price, new_equity)
        """
        next_step_idx = min(self.current_step + 1, len(self.market_data.df_5m) - 1)
        next_metrics = self.market_data.get_market_metrics(next_step_idx)
        mark_price = float(next_metrics["close"])
        new_equity = float(self.executor.equity(mark_price))

        if new_equity > self.max_equity_so_far:
            self.max_equity_so_far = new_equity

        return mark_price, new_equity

    def _recover_risk_budget(self, *, new_equity: float) -> None:
        """
        風險預算恢復邏輯（抽離 step 內的細節，行為不變）。
        """
        self.risk_budget = min(Config.FLIP_BUDGET_MAX, self.risk_budget + Config.FLIP_RECOVERY_RATE)
        if new_equity > self.last_equity_for_budget:
            gain_ratio = (new_equity - self.last_equity_for_budget) / max(1.0, self.initial_balance)
            self.risk_budget = min(
                Config.FLIP_BUDGET_MAX, self.risk_budget + gain_ratio * Config.FLIP_PROFIT_RECOVERY_RATE
            )
        self.last_equity_for_budget = float(new_equity)

    def _compute_reward_features(
        self,
        *,
        current_price: float,
        last_equity: float,
        new_equity: float,
        new_size: float,
    ) -> Tuple[float, bool, float, float, float]:
        """
        計算 reward 會用到的中間特徵。

        Returns:
            (position_change, traded, position_change_norm, turnover_ratio, current_dd)
        """
        position_change = float(abs(new_size - self._last_position_size))
        traded = bool(position_change > 1e-8)
        if traded:
            self.last_trade_step = self.current_step

        max_capacity_qty = (self.daily_risk_base * self.leverage) / current_price if current_price > 0 else 1.0
        position_change_norm = position_change / max_capacity_qty if max_capacity_qty > 0 else 0.0

        # Turnover ratio（只罰「加碼/加曝險」，不罰「減碼/平倉」）
        # c_to ∝ max(0, |pos_{t+1}| - |pos_t|)
        exposure_increase_qty = max(0.0, abs(float(new_size)) - abs(float(self._last_position_size)))
        turnover_notional_change = float(exposure_increase_qty) * float(current_price)
        turnover_scale = max(last_equity * self.leverage, 1e-8)
        turnover_ratio = turnover_notional_change / turnover_scale

        current_dd = (self.max_equity_so_far - new_equity) / self.max_equity_so_far if self.max_equity_so_far > 0 else 0.0

        return position_change, traded, position_change_norm, turnover_ratio, float(current_dd)

    def _update_episode_event_counters(self) -> Tuple[bool, bool]:
        """
        更新 stop loss / liquidation 事件統計與 cooldown。

        Returns:
            (stop_loss_triggered, liq_triggered)
        """
        stop_loss_triggered = bool(self.executor.stop_loss_triggered)
        if stop_loss_triggered:
            self.episode_stop_loss_count += 1
            if getattr(Config, "STOP_LOSS_COOLDOWN_STEPS", 0) > 0:
                self.stop_loss_cooldown = int(Config.STOP_LOSS_COOLDOWN_STEPS)

        liq_triggered = bool(self.executor.liq_triggered)
        if liq_triggered:
            self.episode_liq_count += 1

        return stop_loss_triggered, liq_triggered

    def step(self, action):
        # 1. Prepare market inputs
        metrics = self.market_data.get_market_metrics(self.current_step)
        prices = self._prepare_step_prices(metrics)
        self._maybe_update_daily_risk_base()
        last_equity = float(self.executor.equity(prices.current_price))
        prev_size = float(self._last_position_size)

        # 2. Process action + execute
        final_pos_pct, expected_fee, prev_wallet, is_flip, action_used = self._process_action_and_execute(
            action=action,
            last_equity=last_equity,
            prices=prices,
        )

        # 3. Post execution updates (for next obs + accounting)
        self._update_action_effects_cache(expected_fee=expected_fee, current_price=prices.current_price)
        new_size = float(self.executor.position.size)
        self._update_position_entry(new_size=new_size)
        step_fee, safe_equity = self._update_fee_tracking(current_price=prices.current_price)

        # 4. Mark-to-market (use NEXT close)
        mark_price, new_equity = self._mark_to_market()

        # 5. Risk budget recovery
        self._recover_risk_budget(new_equity=new_equity)

        # 6. Reward features + episode events
        position_change, traded, position_change_norm, turnover_ratio, current_dd = self._compute_reward_features(
            current_price=prices.current_price,
            last_equity=last_equity,
            new_equity=new_equity,
            new_size=new_size,
        )
        # Episode metrics: turnover / holding / trade count
        try:
            turnover_notional_change = float(position_change) * float(prices.current_price)
            if np.isfinite(turnover_notional_change) and turnover_notional_change > 0.0:
                self.episode_turnover_notional = float(self.episode_turnover_notional) + float(turnover_notional_change)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
        if bool(traded):
            self.episode_trade_count += 1
        if abs(float(new_size)) > 1e-8:
            self.episode_holding_steps += 1
        # Track episode-level max drawdown (0~1)
        try:
            dd_clamped = float(np.clip(float(current_dd), 0.0, 1.0))
            self.episode_max_dd = float(max(float(self.episode_max_dd), dd_clamped))
        except (TypeError, ValueError):
            # Should never break training due to a stats field
            pass
        stop_loss_triggered, liq_triggered = self._update_episode_event_counters()
        # Episode metrics: 主動出場次數（平倉到 0 且非 stop loss / liq）
        try:
            closed_to_flat = (abs(prev_size) > 1e-8) and (abs(float(new_size)) <= 1e-8)
            if closed_to_flat and (not bool(stop_loss_triggered)) and (not bool(liq_triggered)):
                self.episode_active_exit_count += 1
        except (TypeError, ValueError):
            pass
            
        # 8. Check Done
        data_exhausted = (self.current_step >= len(self.market_data.df_5m) - 1)
        max_steps_reached = (self.episode_steps + 1) >= self.episode_max_steps
        balance_insufficient = new_equity <= self.min_balance
        
        terminated, truncated, termination_reason = self._determine_termination(
            data_exhausted=data_exhausted,
            max_steps_reached=max_steps_reached,
            balance_insufficient=balance_insufficient,
            liq_triggered=liq_triggered,
        )
        self.done = bool(terminated or truncated)
        
        # 9. Reward Calculation
        # Prepare params
        pos_notional_reward = new_size * mark_price
        max_cap_reward = max(new_equity, 1e-12) * self.leverage
        pos_pct_reward = np.clip(pos_notional_reward / max_cap_reward, -1.0, 1.0)
        
        reward = self.reward_calculator.compute(
            last_equity=last_equity,
            new_equity=new_equity,
            margin_buffer=1.0, # Simplified
            position_change=position_change,
            position_change_norm=position_change_norm,
            turnover_ratio=turnover_ratio,
            dist_to_extreme_atr=0.0, # Simplified/Removed heavy calc
            mae_atr=0.0, # Simplified
            leverage_ratio= abs(new_size * mark_price) / new_equity if new_equity > 0 else 0.0,
            has_position=abs(new_size) > 1e-8,
            unrealized_pnl=self.executor.unrealized_pnl(mark_price),
            traded=traded,
            realized_pnl_step=self.executor.wallet_balance - prev_wallet,
            episode_steps=self.episode_steps,
            episode_max_steps=self.episode_max_steps,
            stop_loss_triggered=stop_loss_triggered,
            done=self.done,
            termination_reason=termination_reason,
            step_fee_ratio=step_fee / self.initial_balance if self.initial_balance > 0 else 0.0,
            current_dd=current_dd,
            fee_budget_ratio=1.0 - (self.tracker.rolling_fee_sum / (safe_equity * self.fee_limit_ratio)) if self.fee_limit_enabled else 1.0,
            position_pct=pos_pct_reward,
            abs_position_pct=abs(pos_pct_reward),
            trend_score=metrics['trend_score']
        )
        
        # 9. Cost / Constraint（成本線）
        # 我們使用「當下價格」計算風險訊號（含 stop_loss_missing / 距離爆倉 / margin_ratio 等），
        # 並把總 cost 與分項寫入 info，方便訓練端做 Lagrangian 更新與 debug。
        risk_post = self.observer.compute_risk_signals(
            self.executor, prices.current_price, self.current_step, len(self.market_data.df_5m)
        )
        step_fee_ratio = float(step_fee / self.initial_balance) if self.initial_balance > 0 else 0.0
        
        # REFACTORED: 僅傳遞必要參數 (liq_triggered, equity, min_balance, step_fee)
        cost_out = self.cost_calculator.compute(
            liq_triggered=bool(liq_triggered),
            equity=float(new_equity),
            min_balance=float(self.min_balance),
            step_fee=float(step_fee),
            # kwargs 傳遞以保留擴充性，但目前 cost.py 主要只用上述四個
            step_fee_ratio=step_fee_ratio,
            turnover_ratio=float(turnover_ratio),
            traded=bool(traded),
            current_dd=float(current_dd),
            risk_signals=risk_post,
            stop_loss_triggered=bool(stop_loss_triggered),
            # Stop-Buffer Cost inputs
            has_position=bool(abs(float(new_size)) > 1e-8),
            current_price=float(prices.current_price),
            stop_loss_price=float(getattr(self.executor.position, "stop_loss_price", 0.0) or 0.0),
            atr=float(prices.atr_est),
            stop_buffer_d_min=float(getattr(Config, "STOP_BUFFER_D_MIN", 0.3)),
            stop_buffer_d_scale=float(getattr(Config, "STOP_BUFFER_D_SCALE", 0.3)),
        )

        # 10. Update Step
        self.current_step += 1
        self.episode_steps += 1
        self._last_position_size = float(new_size)
        
        # 11. Tracker Log
        info = self._build_step_info(
            new_equity=float(new_equity),
            stop_loss_triggered=bool(stop_loss_triggered),
            liq_triggered=bool(liq_triggered),
            step_fee=float(step_fee),
            is_flip=bool(is_flip),
            current_dd=float(current_dd),
            episode_max_dd=float(self.episode_max_dd),
            episode_turnover_notional=float(self.episode_turnover_notional),
            episode_holding_steps=int(self.episode_holding_steps),
            episode_trade_count=int(self.episode_trade_count),
            terminated=bool(terminated),
            truncated=bool(truncated),
            termination_reason=termination_reason,
        )

        # 將 cost 與分項加入 info（不破壞既有 key）
        info["cost"] = float(cost_out["cost"])
        # 新增：雙路徑成本（供雙 λ 使用）；舊訓練端若不認得也不會壞
        if "cost_risk" in cost_out:
            info["cost_risk"] = float(cost_out["cost_risk"])
        if "cost_fric" in cost_out:
            info["cost_fric"] = float(cost_out["cost_fric"])
        if "cost_sl_buf" in cost_out:
            info["cost_sl_buf"] = float(cost_out["cost_sl_buf"])
        info["cost_breakdown"] = dict(cost_out["cost_breakdown"])

        log_payload = self._build_log_payload(
            step=int(self.current_step),
            new_equity=float(new_equity),
            reward=float(reward),
            action=action_used,
            final_pos_pct=float(final_pos_pct),
        )
        self.tracker.log_step(log_payload, self.episode_steps, force=self.done)
        
        # Update Series
        try:
             self.tracker.update_account_series(self.current_step, self.executor, mark_price)
        except (AttributeError, TypeError, ValueError):
            pass

        # Gymnasium: (obs, reward, terminated, truncated, info)
        return self._get_observation(), reward, bool(terminated), bool(truncated), info

    def render(self, mode='human'):
        pass
    
    def close(self):
        pass
    
    def set_fee_rate(self, fee_rate: float):
        self.transaction_fee = float(fee_rate)
        if hasattr(self, "executor"):
            self.executor.set_fee_rate(fee_rate)
            
    def get_fee_rate(self) -> float:
        if hasattr(self, "executor"):
            return self.executor.get_fee_rate()
        return self.transaction_fee
