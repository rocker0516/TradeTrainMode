import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Env.config import Config
from Env.Executors.trade_executor import TradeExecutor
from Env.Components.market_data import MarketData

def _resolve_col_indices(full_cols: list[str], want_cols: tuple[str, ...]) -> list[int]:
    """依 Config 指定欄位解析索引；want_cols 為空則使用全部。"""
    if not want_cols:
        return list(range(len(full_cols)))
    full = list(full_cols)
    indices = []
    for c in want_cols:
        if c not in full:
            raise ValueError(f"OBS 特徵欄位 '{c}' 不在該 state 的可用欄位中: {full[:15]}...")
        indices.append(full.index(c))
    return indices


def _resolve_col_indices_others(full_cols: list[str], want_cols: tuple[str, ...]) -> list[int]:
    """Others 用：full_cols 為 {SYMBOL}_xxx 或 macro 名；want_cols 為 base 名稱，空 = 全部。"""
    if not want_cols:
        return list(range(len(full_cols)))
    want_set = set(want_cols)
    return [
        i
        for i, col in enumerate(full_cols)
        if col in want_set or ("_" in col and col.split("_", 1)[1] in want_set)
    ]


class TradingObserver:
    """
    負責構建觀察空間 (Observation Space) 與生成觀察值 (Observation)。
    包含風險指標計算 (Risk Signals)。
    """
    def __init__(self, window_size: int, window_size_1d: int, market_data: MarketData, *, obs_dtype: str | np.dtype | None = None):
        self.window_size = window_size
        self.window_size_1d = window_size_1d
        # 新的分离特征维度（完整維度，供內部切片前使用）
        self.price_seq_target_features_dim = market_data.price_seq_target_features_dim
        self.price_seq_others_features_dim = market_data.price_seq_others_features_dim
        self.price_seq_1d_target_features_dim = market_data.price_seq_1d_target_features_dim
        self.price_seq_1d_others_features_dim = market_data.price_seq_1d_others_features_dim
        # 兼容性：保留旧接口
        self.price_seq_features_dim = market_data.price_seq_features_dim
        self.features_1d_dim = market_data.features_1d_dim

        # 5 個 state 內部特徵欄位：依 Config 解析索引，空 = 全部
        cols_5m_t = getattr(market_data, "cols_5m_target", [])
        cols_5m_o = getattr(market_data, "cols_5m_others", [])
        cols_1d_t = getattr(market_data, "cols_1d_target", [])
        cols_1d_o = getattr(market_data, "cols_1d_others", [])
        want_5m_t = getattr(Config, "OBS_PRICE_SEQ_TARGET_COLS", ()) or ()
        want_5m_o = getattr(Config, "OBS_PRICE_SEQ_OTHERS_COLS", ()) or ()
        want_1d_t = getattr(Config, "OBS_PRICE_SEQ_1D_TARGET_COLS", ()) or ()
        want_1d_o = getattr(Config, "OBS_PRICE_SEQ_1D_OTHERS_COLS", ()) or ()
        self._obs_price_seq_target_idx = _resolve_col_indices(cols_5m_t, want_5m_t)
        self._obs_price_seq_others_idx = _resolve_col_indices_others(cols_5m_o, want_5m_o)
        self._obs_price_seq_1d_target_idx = _resolve_col_indices(cols_1d_t, want_1d_t)
        self._obs_price_seq_1d_others_idx = _resolve_col_indices_others(cols_1d_o, want_1d_o)
        account_names = getattr(Config, "OBS_ACCOUNT_STATE_NAMES", ())
        account_cols = getattr(Config, "OBS_ACCOUNT_STATE_COLS", ()) or ()
        context_names = getattr(Config, "OBS_CONTEXT_STATE_NAMES", ())
        context_cols = getattr(Config, "OBS_CONTEXT_STATE_COLS", ()) or ()
        self._obs_account_state_idx = _resolve_col_indices(list(account_names), account_cols) if account_names else list(range(22))
        self._obs_context_state_idx = _resolve_col_indices(list(context_names), context_cols) if context_names else list(range(8))
        # 納入 obs 的實際維度（用於 observation_space）
        self._eff_price_seq_target_dim = len(self._obs_price_seq_target_idx)
        self._eff_price_seq_others_dim = len(self._obs_price_seq_others_idx)
        self._eff_price_seq_1d_target_dim = len(self._obs_price_seq_1d_target_idx)
        self._eff_price_seq_1d_others_dim = len(self._obs_price_seq_1d_others_idx)
        self._eff_account_state_dim = len(self._obs_account_state_idx)
        self._eff_context_state_dim = len(self._obs_context_state_idx)

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
        all_space = {**market_space, **account_space, **context_space}

        # 依 Config.OBS_STATE_KEYS 篩選納入 obs 的 state（預設全部）
        obs_state_keys = getattr(Config, "OBS_STATE_KEYS", None)
        if obs_state_keys is not None and len(obs_state_keys) > 0:
            self._obs_state_keys = tuple(k for k in obs_state_keys if k in all_space)
            if len(self._obs_state_keys) < len(obs_state_keys):
                missing = set(obs_state_keys) - set(all_space.keys())
                if missing:
                    raise ValueError(f"OBS_STATE_KEYS 含未知鍵: {missing}；可用: {list(all_space.keys())}")
        else:
            self._obs_state_keys = tuple(all_space.keys())

        self.observation_space = spaces.Dict({k: all_space[k] for k in self._obs_state_keys})

    def _build_market_space(self) -> dict:
        """定義市場數據相關的觀察空間 (5m & 1d 序列)，維度依 Config 各 state 特徵欄位。"""
        market_dtype = np.float32
        return {
            'price_seq_target': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.window_size, self._eff_price_seq_target_dim),
                dtype=market_dtype,
            ),
            'price_seq_others': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.window_size, self._eff_price_seq_others_dim),
                dtype=market_dtype,
            ),
            'price_seq_1d_target': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.window_size_1d, self._eff_price_seq_1d_target_dim),
                dtype=market_dtype,
            ),
            'price_seq_1d_others': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.window_size_1d, self._eff_price_seq_1d_others_dim),
                dtype=market_dtype,
            ),
        }

    def _build_account_space(self) -> dict:
        """定義帳戶狀態相關的觀察空間，維度依 Config.OBS_ACCOUNT_STATE_COLS。"""
        return {
            'account_state': spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._eff_account_state_dim,), dtype=self.obs_dtype
            )
        }

    def _build_context_space(self) -> dict:
        """
        定義環境狀態與「上一動執行結果」相關的觀察空間。
        供 agent 觀察：action 是否被覆寫、raw/used/target/final、是否成交、預測強平距離與可用餘額。
        """
        return {
            'context_state': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._eff_context_state_dim,),
                dtype=self.obs_dtype,
            )
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
                        last_action_effects: dict,
                        *,
                        precomputed_metrics: dict | None = None,
                        ) -> dict:
        """生成單步觀察。若提供 precomputed_metrics 則不再呼叫 get_market_metrics（it/s 優化）。"""
        if precomputed_metrics is not None:
            metrics = precomputed_metrics
        else:
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
        # 僅保留 Config 啟用的 state keys（與 observation_space 一致）
        obs_keys = getattr(self, "_obs_state_keys", None)
        if obs_keys is not None:
            out = {k: out[k] for k in obs_keys if k in out}
        # 取代 inf 的有限值（避免極端訊號被壓成 0 而消失）
        posinf_val = float(getattr(Config, "OBS_INF_CLIP_HIGH", 10.0))
        neginf_val = float(getattr(Config, "OBS_INF_CLIP_LOW", -10.0))
        # 單一迴圈：dtype 轉換 + nan_to_num（nan→0 表缺失；inf→有限值保留極端訊號）
        # 5m/1d 市場序列不轉成 float16，維持 float32 以保留訊號
        for k, v in list(out.items()):
            if isinstance(v, np.ndarray):
                target_dtype = np.float32 if k in self._MARKET_SEQ_KEYS else self.obs_dtype
                if v.dtype != target_dtype:
                    v = v.astype(target_dtype, copy=False)
                v = np.nan_to_num(v, copy=False, nan=0.0, posinf=posinf_val, neginf=neginf_val)
                # 保證 contiguous，讓 CPU→GPU 傳輸時單次連續拷貝（減少 GPU 瓶頸卡在傳輸）
                out[k] = np.ascontiguousarray(v) if not v.flags.c_contiguous else v
        return out

    # 5m/1d 市場序列強制 float32，避免 float16 精度/underflow 把小幅訊號壓掉（「5m 都沒訊號」）
    _MARKET_SEQ_KEYS = frozenset({'price_seq_target', 'price_seq_others', 'price_seq_1d_target', 'price_seq_1d_others'})

    def _get_market_obs(self, step_idx: int, market_data: MarketData) -> dict:
        """生成市場數據觀察值（依 Config 各 state 特徵欄位切片）"""
        seq_target, seq_others = market_data.get_price_seq(step_idx)
        seq_1d_target, seq_1d_others = market_data.get_1d_seq(step_idx, self.window_size_1d)
        dt_market = np.float32
        return {
            'price_seq_target': seq_target[:, self._obs_price_seq_target_idx].astype(dt_market, copy=False),
            'price_seq_others': seq_others[:, self._obs_price_seq_others_idx].astype(dt_market, copy=False),
            'price_seq_1d_target': seq_1d_target[:, self._obs_price_seq_1d_target_idx].astype(dt_market, copy=False),
            'price_seq_1d_others': seq_1d_others[:, self._obs_price_seq_1d_others_idx].astype(dt_market, copy=False),
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
        """生成帳戶狀態觀察值（22 欄位：已移除 stop_loss_rate、fee_budget_remaining、realized_pnl_per_close_norm、episode_progress、trend_strength_last、chop_last）"""
        
        # 緩存常用計算值（優化效率）
        equity = float(executor.equity(current_price))
        atr_est = max(1e-8, atr_ratio * max(current_price, 1e-8))
        
        initial_balance = account_metrics['initial_balance']
        max_equity_so_far = account_metrics['max_equity_so_far']
        episode_stop_loss_count = account_metrics['episode_stop_loss_count']
        holding_steps = account_metrics['holding_steps']
        cooldown_remaining = account_metrics.get('cooldown_remaining', 0.0)
        rolling_fee_sum = account_metrics.get('rolling_fee_sum', 0.0)
        min_balance = float(account_metrics.get('min_balance', initial_balance * 0.5))
        episode_steps = int(account_metrics.get('episode_steps', 0))
        episode_max_steps = max(1, int(account_metrics.get('episode_max_steps', 1)))
        steps_since_trade = float(account_metrics.get('steps_since_trade', 0.0))

        size = executor.position.size
        
        # 1. position_side [-1, 0, 1]
        if size > 0:
            position_side = 1.0  # long
        elif size < 0:
            position_side = -1.0  # short
        else:
            position_side = 0.0  # flat
        
        # 2. position_size_norm [0, 1]
        pos_notional = abs(size) * current_price
        max_notional = equity * executor.leverage
        position_size_norm = pos_notional / max_notional if max_notional > 0 else 0.0
        position_size_norm = np.clip(position_size_norm, 0.0, 1.0)
        
        # 3. equity_ratio [0, 5]
        equity_ratio = equity / initial_balance if initial_balance > 0 else 0.0
        equity_ratio = np.clip(equity_ratio, 0.0, 5.0)
        
        # 4. realized_pnl_ratio [-1, 5]
        wallet_balance = float(executor.wallet_balance)
        realized_pnl = wallet_balance - initial_balance
        realized_pnl_ratio = realized_pnl / initial_balance if initial_balance > 0 else 0.0
        realized_pnl_ratio = np.clip(realized_pnl_ratio, -1.0, 5.0)
        
        # 5. unrealized_pnl_atr [-10, 10]
        upnl = executor.unrealized_pnl(current_price)
        unrealized_pnl_atr = upnl / (initial_balance * atr_ratio) if (initial_balance > 0 and atr_ratio > 0) else 0.0
        unrealized_pnl_atr = np.clip(unrealized_pnl_atr, -10.0, 10.0)
        
        # 6. drawdown [0, 1]
        drawdown = (max_equity_so_far - equity) / max_equity_so_far if max_equity_so_far > 0 else 0.0
        drawdown = np.clip(drawdown, 0.0, 1.0)
        
        # 7. liq_distance_atr [0, 10]
        liq_price = float(risk_signals.get("liq_price", 0.0))
        if liq_price > 0 and current_price > 0:
            liq_distance_atr = abs(current_price - liq_price) / atr_est
        else:
            liq_distance_atr = 10.0  # 無持倉或無強平價時設為最大值
        liq_distance_atr = np.clip(liq_distance_atr, 0.0, 10.0)
        
        # 8. stop_loss_distance_atr [-10, 10] (signed)
        stop_loss_distance_atr = 0.0
        if abs(size) > 1e-12 and executor.position.stop_loss_price > 0:
            sl_price = executor.position.stop_loss_price
            if size > 0:  # 多倉
                stop_loss_distance_atr = (current_price - sl_price) / atr_est
            else:  # 空倉
                stop_loss_distance_atr = (sl_price - current_price) / atr_est
        stop_loss_distance_atr = np.clip(stop_loss_distance_atr, -10.0, 10.0)
        
        # 9. margin_usage_ratio [0, 1.1]
        margin_usage_ratio = 0.0
        if equity > 0 and abs(size) > 0:
            mmr = getattr(executor, 'maintenance_margin_rate', 0.005)
            maint_margin = pos_notional * mmr
            margin_usage_ratio = maint_margin / equity
        elif equity <= 0:
            margin_usage_ratio = 1.1
        margin_usage_ratio = np.clip(margin_usage_ratio, 0.0, 1.1)
        
        # 10. cooldown_remaining_norm [0, 1]
        cooldown_max = max(1.0, float(getattr(Config, 'STOP_LOSS_COOLDOWN_STEPS', 0) or 0))
        cooldown_remaining_norm = np.clip(cooldown_remaining / cooldown_max, 0.0, 1.0)
        
        # 11. fee_rate [0, 0.05]
        fee_rate_pct = float(executor.get_fee_rate())
        fee_rate = fee_rate_pct / 100.0
        fee_rate = np.clip(fee_rate, 0.0, 0.05)
        
        # 12. rolling_fee_ratio [0, 1]
        rolling_fee_ratio = rolling_fee_sum / equity if equity > 0 else 0.0
        rolling_fee_ratio = np.clip(rolling_fee_ratio, 0.0, 1.0)

        # 13. trade_count_log [0, ∞)
        trade_count = executor.long_entry_count + executor.short_entry_count
        trade_count_log = np.log1p(float(trade_count))
        
        # 14. stop_loss_count_log [0, ∞)
        stop_loss_count_log = np.log1p(float(episode_stop_loss_count))
        
        # 15. holding_time_log [0, ∞)
        holding_time_log = np.log1p(max(0.0, holding_steps))

        # 16. buffer_to_min_balance_ratio [0, 1]：離 balance_insufficient 門檻的緩衝（0=碰到死亡線）
        buffer_to_min = (equity - min_balance) / initial_balance if initial_balance > 0 else 0.0
        buffer_to_min_balance_ratio = np.clip(buffer_to_min, 0.0, 1.0)

        # 17. steps_since_trade_norm [0, 1]：距上次成交步數正規化（供 trade_freq / flat cost 學習）
        steps_since_trade_norm = np.clip(
            np.log1p(steps_since_trade) / np.log1p(max(1.0, float(episode_max_steps))),
            0.0, 1.0,
        )

        # 18. trade_freq_remaining_ratio [0, 1]：交易頻率硬限制剩餘額度（視窗內還可交易步數/上限）
        trade_freq_remaining_ratio = float(account_metrics.get('trade_freq_remaining_ratio', 1.0))
        trade_freq_remaining_ratio = np.clip(trade_freq_remaining_ratio, 0.0, 1.0)

        # 19. trade_freq_blocked_last [0, 1]：上一步是否因額度滿被擋（1=被擋）
        trade_freq_blocked_last = float(account_metrics.get('trade_freq_blocked_last', 0.0))
        trade_freq_blocked_last = np.clip(trade_freq_blocked_last, 0.0, 1.0)

        # 20. entry_price_ratio [0.5, 1.5]：進場價 / 當前價，無倉位時 1.0（與當前價同）
        entry_price = float(executor.position.entry_price)
        if current_price > 0 and abs(size) > 1e-12 and entry_price > 0:
            entry_price_ratio = entry_price / current_price
        else:
            entry_price_ratio = 1.0
        entry_price_ratio = np.clip(entry_price_ratio, 0.5, 1.5)

        # 21. stop_loss_price_ratio [0.5, 1.5]：止損價 / 當前價，無止損時 1.0
        stop_loss_price = float(executor.position.stop_loss_price)
        if current_price > 0 and stop_loss_price > 0:
            stop_loss_price_ratio = stop_loss_price / current_price
        else:
            stop_loss_price_ratio = 1.0
        stop_loss_price_ratio = np.clip(stop_loss_price_ratio, 0.5, 1.5)

        # 22. recent_flat_ratio [0, 1]：最近 N 步空倉比例（與 cost_flat 同口徑；實盤可算）
        recent_flat_ratio = float(account_metrics.get('recent_flat_ratio', 0.5))
        recent_flat_ratio = np.clip(recent_flat_ratio, 0.0, 1.0)

        account_state_full = np.array([
            position_side,              # 0
            position_size_norm,         # 1
            equity_ratio,              # 2
            realized_pnl_ratio,         # 3
            unrealized_pnl_atr,         # 4
            drawdown,                   # 5
            liq_distance_atr,           # 6
            stop_loss_distance_atr,     # 7
            margin_usage_ratio,         # 8
            cooldown_remaining_norm,    # 9
            fee_rate,                   # 10
            rolling_fee_ratio,          # 11
            trade_count_log,            # 12
            stop_loss_count_log,        # 13
            holding_time_log,           # 14
            buffer_to_min_balance_ratio,  # 15
            steps_since_trade_norm,     # 17
            trade_freq_remaining_ratio, # 18
            trade_freq_blocked_last,    # 19
            entry_price_ratio,          # 20
            stop_loss_price_ratio,      # 21
            recent_flat_ratio,          # 22
        ], dtype=self.obs_dtype)
        account_state = account_state_full[self._obs_account_state_idx]
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
        """
        生成「上一動執行結果」觀察值，供 agent 得知 action 是否被強制調整、實際執行倉位與預測風險。
        維度順序：[0]action_overridden, [1]last_action_raw, [2]last_action_used,
                 [3]last_target_pos_pct, [4]last_final_pos_pct, [5]trade_executed_flag,
                 [6]predicted_liq_distance_after, [7]available_balance_after_norm
        """
        initial_balance = float(account_metrics.get('initial_balance', 1.0))
        if initial_balance <= 0:
            initial_balance = 1.0
        effects = last_action_effects or {}
        action_overridden = float(effects.get('action_overridden_flag', 0.0))
        last_action_raw = float(effects.get('last_action_raw', 0.0))
        last_action_used = float(effects.get('last_action_used', 0.0))
        last_target = float(effects.get('last_target_pos_pct', 0.0))
        last_final = float(effects.get('last_final_pos_pct', 0.0))
        trade_executed = float(effects.get('trade_executed_flag', 0.0))
        liq_dist = float(effects.get('predicted_liq_distance_after_action', 0.0))
        available_after = float(effects.get('predicted_available_balance_after_action', 0.0))
        available_norm = np.clip(available_after / initial_balance, 0.0, 2.0)
        context_state_full = np.array([
            action_overridden,
            np.clip(last_action_raw, -1.0, 1.0),
            np.clip(last_action_used, -1.0, 1.0),
            np.clip(last_target, -1.0, 1.0),
            np.clip(last_final, -1.0, 1.0),
            trade_executed,
            np.clip(liq_dist, 0.0, 5.0),
            available_norm,
        ], dtype=self.obs_dtype)
        context_state_full = np.nan_to_num(context_state_full, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        context_state = context_state_full[self._obs_context_state_idx]
        return {'context_state': context_state}
