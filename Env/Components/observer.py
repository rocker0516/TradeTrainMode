import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Env.config import Config
from Env.Executors.trade_executor import TradeExecutor
from Env.Components.market_data import MarketData

class TradingObserver:
    """
    負責構建觀察空間 (Observation Space) 與生成觀察值 (Observation)。
    包含風險指標計算 (Risk Signals)。
    """
    def __init__(self, window_size: int, window_size_1d: int, market_data: MarketData):
        self.window_size = window_size
        self.window_size_1d = window_size_1d
        self.price_seq_features_dim = market_data.price_seq_features_dim
        self.features_1d_dim = market_data.features_1d_dim
        
        # 建立各個子空間
        market_space = self._build_market_space()
        account_space = self._build_account_space()
        context_space = self._build_context_space()
        
        # 合併所有空間
        self.observation_space = spaces.Dict({
            **market_space,
            **account_space,
            **context_space
        })

    def _build_market_space(self) -> dict:
        """定義市場數據相關的觀察空間 (5m & 1d 序列)"""
        return {
            'price_seq': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.window_size, self.price_seq_features_dim), 
                dtype=np.float32
            ),
            'price_seq_1d': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.window_size_1d, self.features_1d_dim), 
                dtype=np.float32
            )
        }

    def _build_account_space(self) -> dict:
        """定義帳戶狀態相關的觀察空間"""
        return {
            'account_state': spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)
        }

    def _build_context_space(self) -> dict:
        """定義環境狀態、時間與成本風險相關的觀察空間"""
        return {
            'time_state': spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
            'rhythm_state': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'cost_state': spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)
        }

    def compute_risk_signals(self, executor: TradeExecutor, current_price: float, step_idx: int, total_steps: int) -> dict:
        """
        計算即時風險指標，包含強平價、距離、保證金率與止損距離。
        """
        size = float(executor.position.size)
        if step_idx >= total_steps or abs(size) < 1e-12 or current_price <= 0.0:
            return {
                'liq_price': 0.0,
                'price_gap': 0.0,
                'gap_pct': 0.0,
                'abs_gap_pct': 0.0,
                'margin_ratio': 0.0,
                'sl_gap_pct': 0.0,
                'stop_loss_missing': 0.0,
                'near_liq': False,
                'near_margin': False,
                'near_stop': False,
            }

        liq_price = float(executor.get_liquidation_price(current_price))
        price_gap = current_price - liq_price if liq_price > 0 else 0.0
        gap_pct = price_gap / current_price
        abs_gap_pct = abs(price_gap) / current_price

        equity = float(executor.equity(current_price))
        upnl = float(executor.unrealized_pnl(current_price))
        unrealized_loss = max(0.0, -upnl)
        pos_notional = abs(size) * current_price
        margin_ratio = (equity - unrealized_loss) / pos_notional if pos_notional > 0 else 0.0

        mmr = float(getattr(executor, "maintenance_margin_rate", 0.0))
        liq_warn_pct = getattr(Config, "LIQUIDATION_WARN_PCT", 0.005)
        stop_warn_pct = getattr(Config, "STOP_LOSS_WARN_PCT", 0.002)
        
        near_liq = abs_gap_pct <= liq_warn_pct
        near_margin = (pos_notional > 0) and (margin_ratio <= mmr if mmr > 0 else False)

        stop_price = float(executor.position.stop_loss_price)
        has_stop = stop_price > 0.0
        sl_gap_pct = 0.0
        near_stop = False
        if has_stop:
            sl_gap_pct = (current_price - stop_price) / current_price
            near_stop = abs(sl_gap_pct) <= stop_warn_pct

        stop_loss_missing = float(1.0 if (pos_notional > 0 and not has_stop) else 0.0)

        return {
            'liq_price': liq_price,
            'price_gap': price_gap,
            'gap_pct': gap_pct,
            'abs_gap_pct': abs_gap_pct,
            'margin_ratio': margin_ratio,
            'sl_gap_pct': sl_gap_pct,
            'stop_loss_missing': stop_loss_missing,
            'near_liq': bool(near_liq),
            'near_margin': bool(near_margin),
            'near_stop': bool(near_stop),
        }

    def get_observation(self, 
                        step_idx: int, 
                        executor: TradeExecutor, 
                        market_data: MarketData,
                        account_metrics: dict,
                        risk_signals: dict,
                        last_action_effects: dict
                        ) -> dict:
        
        market_obs = self._get_market_obs(step_idx, market_data)
        account_obs = self._get_account_obs(step_idx, executor, market_data, account_metrics, risk_signals)
        context_obs = self._get_context_obs(step_idx, executor, market_data, account_metrics, risk_signals, last_action_effects)
        
        return {
            **market_obs,
            **account_obs,
            **context_obs
        }

    def _get_market_obs(self, step_idx: int, market_data: MarketData) -> dict:
        """生成市場數據觀察值"""
        return {
            'price_seq': market_data.get_price_seq(step_idx),
            'price_seq_1d': market_data.get_1d_seq(step_idx, self.window_size_1d)
        }

    def _get_account_obs(self, step_idx: int, executor: TradeExecutor, market_data: MarketData, account_metrics: dict, risk_signals: dict) -> dict:
        """生成帳戶狀態觀察值"""
        metrics = market_data.get_market_metrics(step_idx)
        current_price = metrics['close']
        atr_ratio = metrics['atr_ratio']
        
        initial_balance = account_metrics['initial_balance']
        max_equity_so_far = account_metrics['max_equity_so_far']
        episode_stop_loss_count = account_metrics['episode_stop_loss_count']
        episode_liq_count = account_metrics['episode_liq_count']
        risk_budget = account_metrics['risk_budget']
        
        equity = executor.equity(current_price)
        size = executor.position.size
        
        # --- Account Status (27) ---
        pos_side_oh = np.zeros(3, dtype=np.float32)
        if size > 0: pos_side_oh[0] = 1.0
        elif size < 0: pos_side_oh[1] = 1.0
        else: pos_side_oh[2] = 1.0
        
        pos_notional = abs(size) * current_price
        max_notional = equity * executor.leverage
        pos_size_norm = pos_notional / max_notional if max_notional > 0 else 0.0
        pos_size_norm = np.clip(pos_size_norm, 0.0, 1.0)
        
        upnl = executor.unrealized_pnl(current_price)
        unreal_pnl_ratio = upnl / initial_balance if initial_balance > 0 else 0.0
        unreal_pnl_ratio = np.clip(unreal_pnl_ratio, -2.0, 2.0)
        
        equity_ratio = equity / initial_balance if initial_balance > 0 else 0.0
        equity_ratio = np.clip(equity_ratio, 0.0, 5.0)
        
        max_equity_ratio = max_equity_so_far / initial_balance if initial_balance > 0 else 0.0
        max_equity_ratio = np.clip(max_equity_ratio, 0.0, 5.0)
        
        dd = (max_equity_so_far - equity) / max_equity_so_far if max_equity_so_far > 0 else 0.0
        dd = np.clip(dd, 0.0, 1.0)
        
        maint_margin_ratio = 0.0
        if equity > 0 and abs(size) > 0:
            mmr = getattr(executor, 'maintenance_margin_rate', 0.005)
            maint_margin = pos_notional * mmr
            maint_margin_ratio = maint_margin / equity
        elif equity <= 0:
            maint_margin_ratio = 1.1
        maint_margin_ratio = np.clip(maint_margin_ratio, 0.0, 1.1)
        
        profit_rate = (equity - initial_balance) / initial_balance if initial_balance > 0 else 0.0
        profit_rate = np.clip(profit_rate, -1.0, 5.0)
        
        dist_to_sl_norm = 0.0
        if abs(size) > 0 and executor.position.stop_loss_price > 0:
            sl_price = executor.position.stop_loss_price
            dist = abs(current_price - sl_price)
            if current_price > 0:
                dist_to_sl_norm = np.clip((dist / current_price) * 10.0, 0.0, 5.0)

        wallet_balance = float(executor.wallet_balance)
        used_margin = float(getattr(executor, "used_margin", 0.0))
        available_balance = float(executor.available_balance())
        fee_rate_pct = float(executor.get_fee_rate())
        
        wallet_balance_ratio = np.clip(wallet_balance / initial_balance if initial_balance > 0 else 0.0, 0.0, 5.0)
        used_margin_ratio = np.clip(used_margin / max(1e-8, equity) if equity > 0 else 0.0, 0.0, 5.0)
        available_balance_ratio = np.clip(available_balance / max(1e-8, equity) if equity > 0 else 0.0, -5.0, 5.0)
        equity_to_position_notional = np.clip(equity / max(1e-8, pos_notional) if pos_notional > 0 else 0.0, 0.0, 5.0)
        
        liq_price = float(risk_signals.get("liq_price", 0.0))
        liq_distance_pct = abs(current_price - liq_price) / current_price if (current_price > 0 and liq_price > 0) else 0.0
        liq_distance_pct = np.clip(liq_distance_pct, 0.0, 5.0)
        
        entry_price = float(executor.position.entry_price)
        stop_price = float(executor.position.stop_loss_price)
        stop_distance_pct = abs(entry_price - stop_price) / entry_price if (abs(size) > 1e-12 and entry_price > 0 and stop_price > 0) else 0.0
        stop_distance_pct = np.clip(stop_distance_pct, 0.0, 5.0)
        fee_rate_pct = np.clip(fee_rate_pct, 0.0, 1.0)
        
        atr_est = max(1e-8, atr_ratio * max(current_price, 1e-8))
        fee_frac = fee_rate_pct / 100.0
        fee_frac = np.clip(fee_frac, 0.0, 0.05)
        
        entry_gap_atr = 0.0
        breakeven_gap_atr = 0.0
        if abs(size) > 1e-12 and entry_price > 0.0:
            signed = 1.0 if size > 0 else -1.0
            entry_gap_atr = signed * ((current_price - entry_price) / atr_est)
            if size > 0:
                be_price = entry_price * (1.0 + 2.0 * fee_frac)
            else:
                be_price = entry_price * (1.0 - 2.0 * fee_frac)
            breakeven_gap_atr = signed * ((current_price - be_price) / atr_est)
        entry_gap_atr = np.clip(entry_gap_atr, -10.0, 10.0)
        breakeven_gap_atr = np.clip(breakeven_gap_atr, -10.0, 10.0)
        
        steps_since_trade_norm = np.clip(account_metrics['steps_since_trade'] / max(1.0, float(self.window_size)), 0.0, 5.0)
        holding_time_norm = np.clip(account_metrics['holding_steps'] / max(1.0, float(self.window_size)), 0.0, 5.0)

        account_state = np.array([
            pos_size_norm,
            unreal_pnl_ratio,
            equity_ratio,
            max_equity_ratio,
            dd,
            maint_margin_ratio,
            profit_rate,
            executor.long_entry_count * 0.01,
            executor.short_entry_count * 0.01,
            episode_stop_loss_count * 0.1,
            episode_liq_count * 1.0,
            dist_to_sl_norm,
            float(risk_budget),
            pos_side_oh[0],
            pos_side_oh[1],
            pos_side_oh[2],
            entry_gap_atr,
            breakeven_gap_atr,
            steps_since_trade_norm,
            holding_time_norm,
            wallet_balance_ratio,
            used_margin_ratio,
            available_balance_ratio,
            equity_to_position_notional,
            liq_distance_pct,
            stop_distance_pct,
            fee_rate_pct,
        ], dtype=np.float32)
        
        return {'account_state': account_state}

    def _get_context_obs(self, step_idx: int, executor: TradeExecutor, market_data: MarketData, account_metrics: dict, risk_signals: dict, last_action_effects: dict) -> dict:
        """生成環境與成本狀態觀察值"""
        metrics = market_data.get_market_metrics(step_idx)
        atr_ratio = metrics['atr_ratio']
        
        # --- Time State (7) ---
        time_state = np.zeros(7, dtype=np.float32)
        hour = market_data.hour_arr[step_idx]
        dow = market_data.dow_arr[step_idx]
        is_weekend = market_data.is_weekend_arr[step_idx]
        
        time_state[0] = np.sin(2 * np.pi * hour / 24.0)
        time_state[1] = np.cos(2 * np.pi * hour / 24.0)
        time_state[2] = np.sin(2 * np.pi * dow / 7.0)
        time_state[3] = np.cos(2 * np.pi * dow / 7.0)
        phase = (hour % 8.0) / 8.0
        time_state[4] = np.sin(2 * np.pi * phase)
        time_state[5] = np.cos(2 * np.pi * phase)
        time_state[6] = is_weekend
        
        # --- Rhythm State (2) ---
        rhythm_state = np.zeros(2, dtype=np.float32)
        rhythm_state[0] = float(atr_ratio)
        rhythm_state[1] = float(np.clip(metrics['rv_ratio'], 0.0, 10.0))
        
        # --- Cost State (19) ---
        cost_state = np.zeros(19, dtype=np.float32)
        last_step_fee = account_metrics.get('last_step_fee', 0.0)
        rolling_fee_sum = account_metrics.get('rolling_fee_sum', 0.0)
        fee_limit_ratio = account_metrics.get('fee_limit_ratio', 1.0)
        initial_balance = account_metrics['initial_balance']
        
        # Need equity and pos_notional again here, or pass it? 
        # For simplicity recalculate (cheap)
        metrics = market_data.get_market_metrics(step_idx)
        current_price = metrics['close']
        equity = executor.equity(current_price)
        size = executor.position.size
        pos_notional = abs(size) * current_price
        
        # Maint margin ratio again
        maint_margin_ratio = 0.0
        if equity > 0 and abs(size) > 0:
            mmr = getattr(executor, 'maintenance_margin_rate', 0.005)
            maint_margin = pos_notional * mmr
            maint_margin_ratio = maint_margin / equity
        elif equity <= 0:
            maint_margin_ratio = 1.1
        maint_margin_ratio = np.clip(maint_margin_ratio, 0.0, 1.1)
        
        max_equity_so_far = account_metrics['max_equity_so_far']
        dd = (max_equity_so_far - equity) / max_equity_so_far if max_equity_so_far > 0 else 0.0
        dd = np.clip(dd, 0.0, 1.0)

        step_fee_ratio_stable = np.clip((last_step_fee / initial_balance if initial_balance > 0 else 0.0), 0.0, 0.1)
        rolling_fee_ratio = np.clip(rolling_fee_sum / equity if equity > 0 else 0.0, 0.0, 1.0)
        
        remaining_fee_budget_ratio = 1.0
        if account_metrics.get('fee_limit_enabled', False):
             safe_equity = max(equity, initial_balance * 0.5)
             limit_amount = max(1e-8, safe_equity * fee_limit_ratio)
             remaining_fee_budget_ratio = np.clip(1.0 - (rolling_fee_sum / limit_amount), 0.0, 1.0)
        
        leverage_ratio = pos_notional / equity if equity > 0 else 0.0
        
        cost_state[0] = step_fee_ratio_stable
        cost_state[1] = rolling_fee_ratio
        cost_state[2] = maint_margin_ratio
        cost_state[3] = dd
        cost_state[4] = np.clip(leverage_ratio, 0.0, 10.0)
        cost_state[5] = remaining_fee_budget_ratio
        
        cost_state[6] = np.clip(risk_signals['gap_pct'], -5.0, 5.0)
        cost_state[7] = np.clip(risk_signals['abs_gap_pct'], 0.0, 5.0)
        cost_state[8] = np.clip(risk_signals['margin_ratio'], 0.0, 5.0)
        cost_state[9] = np.clip(risk_signals['sl_gap_pct'], -5.0, 5.0)
        cost_state[10] = float(risk_signals['stop_loss_missing'])
        cost_state[11] = float(risk_signals['near_liq'])
        cost_state[12] = float(risk_signals['near_margin'])
        cost_state[13] = float(risk_signals['near_stop'])
        
        effects = last_action_effects
        cost_state[14] = float(effects.get("expected_fee_if_trade", 0.0))
        cost_state[15] = float(effects.get("predicted_used_margin_after_action", 0.0))
        cost_state[16] = float(effects.get("predicted_available_balance_after_action", 0.0))
        cost_state[17] = float(effects.get("predicted_liq_distance_after_action", 0.0))
        cost_state[18] = float(effects.get("predicted_stop_distance_after_action", 0.0))
        
        return {
            'time_state': time_state,
            'rhythm_state': rhythm_state,
            'cost_state': cost_state
        }
