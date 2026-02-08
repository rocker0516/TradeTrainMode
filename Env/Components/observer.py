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
            'account_state': spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=self.obs_dtype)
        }

    def _build_context_space(self) -> dict:
        """定義環境狀態與成本風險相關的觀察空間（目前無欄位；成本訊號改由 train 端以 wrapper 注入）"""
        return {}

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
        # 單一迴圈：dtype 轉換 + nan_to_num，減少對同一陣列的重複遍歷（it/s 優化）
        for k, v in list(out.items()):
            if isinstance(v, np.ndarray):
                if v.dtype != self.obs_dtype:
                    v = v.astype(self.obs_dtype, copy=False)
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
        """生成帳戶狀態觀察值（16個欄位，優化後）"""
        
        # 緩存常用計算值（優化效率）
        equity = executor.equity(current_price)
        atr_est = max(1e-8, atr_ratio * max(current_price, 1e-8))
        
        initial_balance = account_metrics['initial_balance']
        max_equity_so_far = account_metrics['max_equity_so_far']
        episode_stop_loss_count = account_metrics['episode_stop_loss_count']
        holding_steps = account_metrics['holding_steps']
        cooldown_remaining = account_metrics.get('cooldown_remaining', 0.0)
        rolling_fee_sum = account_metrics.get('rolling_fee_sum', 0.0)

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
        
        # 13. fee_budget_remaining [0, 1]（固定 1.0，Fee Limit 功能已移除）
        fee_budget_remaining = 1.0

        # 14. trade_count_log [0, ∞)
        trade_count = executor.long_entry_count + executor.short_entry_count
        trade_count_log = np.log1p(float(trade_count))
        
        # 15. stop_loss_count_log [0, ∞)
        stop_loss_count_log = np.log1p(float(episode_stop_loss_count))
        
        # 16. holding_time_log [0, ∞)
        holding_time_log = np.log1p(max(0.0, holding_steps))
        
        account_state = np.array([
            position_side,              # 1
            position_size_norm,         # 2
            equity_ratio,               # 3
            realized_pnl_ratio,         # 4
            unrealized_pnl_atr,          # 5
            drawdown,                   # 6
            liq_distance_atr,           # 7
            stop_loss_distance_atr,      # 8
            margin_usage_ratio,         # 9
            cooldown_remaining_norm,    # 10
            fee_rate,                   # 11
            rolling_fee_ratio,          # 12
            fee_budget_remaining,       # 13
            trade_count_log,            # 14
            stop_loss_count_log,         # 15
            holding_time_log,           # 16
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
        """生成環境與成本狀態觀察值（目前無欄位；成本訊號改由 train 端以 wrapper 注入）"""
        return {}
