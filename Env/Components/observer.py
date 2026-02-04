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
    def __init__(self, window_size: int, window_size_1d: int, market_data: MarketData, *, obs_dtype: str | np.dtype | None = None):
        self.window_size = window_size
        self.window_size_1d = window_size_1d
        # 新的分离特征维度
        self.price_seq_target_features_dim = market_data.price_seq_target_features_dim
        self.price_seq_others_features_dim = market_data.price_seq_others_features_dim
        self.price_seq_1d_target_features_dim = market_data.price_seq_1d_target_features_dim
        self.price_seq_1d_others_features_dim = market_data.price_seq_1d_others_features_dim
        # 兼容性：保留旧接口
        self.price_seq_features_dim = market_data.price_seq_features_dim
        self.features_1d_dim = market_data.features_1d_dim

        # obs dtype（預設採用 Config.OBS_DTYPE）
        if obs_dtype is None:
            obs_dtype = getattr(Config, "OBS_DTYPE", "float32")
        if isinstance(obs_dtype, str):
            obs_dtype = obs_dtype.lower().strip()
            if obs_dtype == "float16":
                self.obs_dtype = np.float16
            elif obs_dtype == "float32":
                self.obs_dtype = np.float32
            else:
                raise ValueError(f"Unsupported obs_dtype: {obs_dtype}")
        else:
            self.obs_dtype = np.dtype(obs_dtype)
        
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
        """定義市場數據相關的觀察空間 (5m & 1d 序列，分离的 target 和 others)"""
        return {
            'price_seq_target': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.window_size, self.price_seq_target_features_dim), 
                dtype=self.obs_dtype
            ),
            'price_seq_others': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.window_size, self.price_seq_others_features_dim), 
                dtype=self.obs_dtype
            ),
            'price_seq_1d_target': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.window_size_1d, self.price_seq_1d_target_features_dim), 
                dtype=self.obs_dtype
            ),
            'price_seq_1d_others': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.window_size_1d, self.price_seq_1d_others_features_dim), 
                dtype=self.obs_dtype
            )
        }

    def _build_account_space(self) -> dict:
        """定義帳戶狀態相關的觀察空間"""
        return {
            'account_state': spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=self.obs_dtype)
        }

    def _build_context_space(self) -> dict:
        """定義環境狀態與成本風險相關的觀察空間"""
        return {
            # cost_state (27):
            # 0~18: 原有成本/風險/預測效果特徵
            # 19~26: 行為「偏差揭露」特徵（讓 agent 知道 raw action 是否被覆寫/限幅/未成交）
            'cost_state': spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=self.obs_dtype)
        }

    def compute_risk_signals(
        self,
        executor: TradeExecutor,
        current_price: float,
        atr_est: float,
        step_idx: int,
        total_steps: int,
    ) -> dict:
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
                # sl_gap_atr：以 ATR 正規化的止損距離（signed；同向為正，越大越安全）
                'sl_gap_atr': 0.0,
                'abs_sl_gap_atr': 0.0,
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

        # --- ATR-normalized stop distance (aligns with sl_buf semantics) ---
        # d_t = |P - SL| / ATR
        # Here we provide a signed variant so that "safe side" is always positive:
        # - long:  (P - SL)/ATR
        # - short: (SL - P)/ATR  (implemented via signed = sign(position))
        sl_gap_atr = 0.0
        abs_sl_gap_atr = 0.0
        atr_est = float(atr_est) if atr_est is not None else 0.0
        if has_stop and atr_est > 1e-12:
            signed = 1.0 if size > 0.0 else -1.0
            sl_gap_atr = signed * ((current_price - stop_price) / atr_est)
            abs_sl_gap_atr = abs(sl_gap_atr)

        stop_loss_missing = float(1.0 if (pos_notional > 0 and not has_stop) else 0.0)

        return {
            'liq_price': liq_price,
            'price_gap': price_gap,
            'gap_pct': gap_pct,
            'abs_gap_pct': abs_gap_pct,
            'margin_ratio': margin_ratio,
            'sl_gap_pct': sl_gap_pct,
            'sl_gap_atr': float(sl_gap_atr),
            'abs_sl_gap_atr': float(abs_sl_gap_atr),
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
        # 重要：避免同一步重複呼叫 get_market_metrics（以前 account/context 各呼叫一次，context 甚至呼叫兩次）
        metrics = market_data.get_market_metrics(step_idx)
        current_price = float(metrics["close"])
        atr_ratio = float(metrics.get("atr_ratio", 0.0))

        market_obs = self._get_market_obs(step_idx, market_data)
        account_obs = self._get_account_obs(
            step_idx, executor, market_data, account_metrics, risk_signals, current_price=current_price, atr_ratio=atr_ratio
        )
        context_obs = self._get_context_obs(
            step_idx,
            executor,
            market_data,
            account_metrics,
            risk_signals,
            last_action_effects,
            current_price=current_price,
            atr_ratio=atr_ratio,
        )
        
        out = {**market_obs, **account_obs, **context_obs}
        # 確保所有 key dtype 與 observation_space 一致（預設 float16 以省 replay buffer RAM）
        if self.obs_dtype != np.float32:
            for k, v in list(out.items()):
                if isinstance(v, np.ndarray) and v.dtype != self.obs_dtype:
                    out[k] = v.astype(self.obs_dtype, copy=False)

        # 防呆：任何觀測出現 NaN/Inf 都可能讓 policy 輸出 NaN 而直接炸訓練（你 terminal 的錯誤即屬此類）。
        # 這裡做「最後一道」清洗，不改變 shape，只確保數值是 finite。
        for k, v in list(out.items()):
            if isinstance(v, np.ndarray):
                # inplace 轉換：inf/-inf/nan -> 0
                # 注意：此步驟應避免產生額外 copy（copy=False）。
                out[k] = np.nan_to_num(v, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def _get_market_obs(self, step_idx: int, market_data: MarketData) -> dict:
        """生成市場數據觀察值（分离的 target 和 others）"""
        seq_target, seq_others = market_data.get_price_seq(step_idx)
        seq_1d_target, seq_1d_others = market_data.get_1d_seq(step_idx, self.window_size_1d)
        return {
            'price_seq_target': seq_target.astype(self.obs_dtype, copy=False),
            'price_seq_others': seq_others.astype(self.obs_dtype, copy=False),
            'price_seq_1d_target': seq_1d_target.astype(self.obs_dtype, copy=False),
            'price_seq_1d_others': seq_1d_others.astype(self.obs_dtype, copy=False),
        }

    def _get_account_obs(
        self,
        step_idx: int,
        executor: TradeExecutor,
        market_data: MarketData,
        account_metrics: dict,
        risk_signals: dict,
        *,
        current_price: float,
        atr_ratio: float,
    ) -> dict:
        """生成帳戶狀態觀察值"""
        
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
        ], dtype=self.obs_dtype)
        
        return {'account_state': account_state}

    def _get_context_obs(
        self,
        step_idx: int,
        executor: TradeExecutor,
        market_data: MarketData,
        account_metrics: dict,
        risk_signals: dict,
        last_action_effects: dict,
        *,
        current_price: float,
        atr_ratio: float,
    ) -> dict:
        """生成環境與成本狀態觀察值"""
        
        # --- Cost State (27) ---
        cost_state = np.zeros(27, dtype=self.obs_dtype)
        last_step_fee = account_metrics.get('last_step_fee', 0.0)
        rolling_fee_sum = account_metrics.get('rolling_fee_sum', 0.0)
        fee_limit_ratio = account_metrics.get('fee_limit_ratio', 1.0)
        initial_balance = account_metrics['initial_balance']
        
        # Need equity and pos_notional again here, or pass it? 
        # For simplicity recalculate (cheap)
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
        # 止損距離：改用 ATR-normalized（讓 agent 能直接對齊 sl_buf / 波動 regime）
        # sl_gap_atr > 0 表示在「安全側」且距離止損越遠；接近 0 表示貼近止損。
        cost_state[9] = np.clip(float(risk_signals.get('sl_gap_atr', 0.0)), -10.0, 10.0)
        cost_state[10] = float(risk_signals['stop_loss_missing'])
        cost_state[11] = float(risk_signals['near_liq'])
        cost_state[12] = float(risk_signals['near_margin'])
        cost_state[13] = float(risk_signals['near_stop'])
        
        effects = last_action_effects

        # ---- Action Effects (5) ----
        # 重要：這些欄位若直接用「USDT 絕對值」在 float16 下很容易 overflow -> inf，
        # 進而讓 policy / replay buffer 出現 NaN，導致 SB3 actor 直接崩潰。
        # 因此改用「相對權益比例」表示，並做有限值/clip 保護。
        safe_equity = float(max(float(equity), 1e-8))

        def _safe_ratio(x: object, *, denom: float, low: float, high: float) -> float:
            """將任意輸入轉成 (x/denom) 並 clip，若不可轉或非有限值則回傳 0。"""
            try:
                v = float(x) / float(denom)
            except (TypeError, ValueError, ZeroDivisionError):
                return 0.0
            if not np.isfinite(v):
                return 0.0
            return float(np.clip(v, float(low), float(high)))

        # 14) expected_fee_if_trade_ratio: 預估手續費 / equity（0~0.2）
        cost_state[14] = _safe_ratio(effects.get("expected_fee_if_trade", 0.0), denom=safe_equity, low=0.0, high=0.2)
        # 15) predicted_used_margin_ratio: used_margin / equity（0~10）
        cost_state[15] = _safe_ratio(effects.get("predicted_used_margin_after_action", 0.0), denom=safe_equity, low=0.0, high=10.0)
        # 16) predicted_available_balance_ratio: available_balance / equity（-10~10）
        cost_state[16] = _safe_ratio(effects.get("predicted_available_balance_after_action", 0.0), denom=safe_equity, low=-10.0, high=10.0)

        # 17) predicted_liq_distance_after_action: 已在 env 端做過 clip，但仍做 finite/clip 防呆（0~5）
        cost_state[17] = _safe_ratio(effects.get("predicted_liq_distance_after_action", 0.0), denom=1.0, low=0.0, high=5.0)
        # 18) predicted_stop_distance_after_action: 已在 env 端做過 clip，但仍做 finite/clip 防呆（0~5）
        cost_state[18] = _safe_ratio(effects.get("predicted_stop_distance_after_action", 0.0), denom=1.0, low=0.0, high=5.0)

        # ---- Action vs Execution discrepancy (8) ----
        # 19) cooldown_remaining_norm: 下一步是否會強制 action=0（0~1）
        cost_state[19] = float(effects.get("cooldown_remaining_norm", 0.0))
        # 20) action_overridden_flag: 上一步 action 是否被 env 覆寫（0/1）
        cost_state[20] = float(effects.get("action_overridden_flag", 0.0))
        # 21) last_action_raw: policy 原始輸出（-1~1）
        cost_state[21] = float(effects.get("last_action_raw", 0.0))
        # 22) last_action_used: 實際送入 processor/executor 的 action（-1~1；例如 cooldown 會變 0）
        cost_state[22] = float(effects.get("last_action_used", 0.0))
        # 23) last_target_pos_pct: processor 的 target_pos_pct（-1~1）
        cost_state[23] = float(effects.get("last_target_pos_pct", 0.0))
        # 24) last_final_pos_pct: 經 max_step_pos_change 等限制後的最終執行目標（-1~1）
        cost_state[24] = float(effects.get("last_final_pos_pct", 0.0))
        # 25) executed_pos_pct: 由實際持倉 size 反推的 signed exposure pct（-1~1）
        max_cap = max(float(equity) * float(executor.leverage), 1e-12)
        executed_pos_pct = (float(size) * float(current_price)) / max_cap
        cost_state[25] = float(np.clip(executed_pos_pct, -1.0, 1.0))
        # 26) trade_executed_flag: 本步是否真的成交/改變持倉（0/1；由 env 計算後注入）
        cost_state[26] = float(effects.get("trade_executed_flag", 0.0))
        
        return {
            'cost_state': cost_state
        }
