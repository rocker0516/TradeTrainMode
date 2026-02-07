import gymnasium as gym
import numpy as np
import random
from dataclasses import dataclass
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any, Iterable
import pandas as pd

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
        # 允許外部（訓練/評估端）用 kwargs 覆寫 episode 上限，避免評估回合過長拖慢訓練。
        # 預設行為不變：若未提供 max_episode_steps，仍使用 Config.MAX_EPISODE_STEPS。
        self.max_episode_steps = int(kwargs.get("max_episode_steps", getattr(Config, "MAX_EPISODE_STEPS", 1000000)))
        self.min_position_change = float(kwargs.get("min_position_change", Config.MIN_POSITION_CHANGE))
        # daily_risk_base 更新頻率（用於單步倉位變化上限的基準）
        # 預設用 Config.RISK_BASE_UPDATE_STEPS；若未設定則回退到 window_size（維持舊語義）
        self.risk_base_update_steps = int(
            kwargs.get(
                "risk_base_update_steps",
                getattr(Config, "RISK_BASE_UPDATE_STEPS", self.window_size),
            )
        )
        self.random_start = kwargs.get('random_start', True)
        self.target_symbol = kwargs.get('target_symbol', 'BTCUSDT') # 預設交易對
        # 固定的特徵 symbols 清單（決定 5m 跨市場摘要的維度）
        # - 若不傳，維持相容：只用 target_symbol（但仍會包含主市場結構化/廣度特徵）
        self.feature_symbols = kwargs.get("feature_symbols", None)
        self.margin_mode = 'isolated'
        self.min_trade_qty = 0.001 

        # ---- Train/Eval split (optional; to prevent data leakage) ----
        # 語義：
        # - data_split_enabled=True 且 data_mode="train"：使用「非最近 N 個月」資料
        # - data_split_enabled=True 且 data_mode="eval" ：使用「最近 N 個月」資料
        # - 其他：維持舊行為（用全部資料）
        self.data_split_enabled = bool(kwargs.get("data_split_enabled", False))
        self.data_mode = str(kwargs.get("data_mode", "full")).lower().strip()
        self.holdout_months = int(kwargs.get("holdout_months", 3))
        # Eval obs 必須填滿：起點至少在 warmup_steps 之後
        # - 若使用 data_mode="eval"，預設強制啟用（除非你顯式傳 ensure_filled_obs=False）
        ensure_filled_default = bool(self.data_mode == "eval")
        self.ensure_filled_obs = bool(kwargs.get("ensure_filled_obs", ensure_filled_default))

        # ---- Render (episode end) ----
        # 說明：render 主要用於 eval/debug，不應在 step() 內做昂貴工作。
        # 預設：save + show（策略 A：能 show 就 show；無 GUI 自動只存檔）
        self.render_enabled = bool(kwargs.get("render_enabled", True))
        self.render_dir = str(kwargs.get("render_dir", "logs/renders"))
        self.render_save = bool(kwargs.get("render_save", True))
        self.render_show = bool(kwargs.get("render_show", True))
        # 重要：VecEnv（例如 SB3 DummyVecEnv/SubprocVecEnv）會在 step() 遇到 done 時「自動 reset」，
        # 因此 callback 在 episode 結束後再呼叫 env.render() 往往會失敗（env.done 已被 reset() 清掉）。
        # 解法：允許在「終止那一步」直接 render 並把路徑塞回 info（render_path），讓 callback 能拿到結果。
        # 預設關閉避免訓練時額外負擔；eval 建議開啟。
        self.render_on_done = bool(kwargs.get("render_on_done", False))
        self.render_dpi = int(kwargs.get("render_dpi", 140))
        self.render_figsize = tuple(kwargs.get("render_figsize", (14.0, 9.0)))
        self.render_export_events = bool(kwargs.get("render_export_events", False))

        # 載入數據（允許測試/外部注入 df，避免強耦合到檔案系統）
        df_5m_in = kwargs.get("df_5m", None)
        df_1d_in = kwargs.get("df_1d", None)
        if df_5m_in is not None and df_1d_in is not None:
            self.df_5m, self.df_1d = df_5m_in, df_1d_in
        else:
            self.df_5m, self.df_1d = load_data()

        # ---- Apply train/eval split by recent months ----
        if self.data_split_enabled and self.data_mode in {"train", "eval"}:
            self.df_5m, self.df_1d = self._split_train_eval_by_recent_months(
                df_5m=self.df_5m,
                df_1d=self.df_1d,
                holdout_months=int(self.holdout_months),
                mode=str(self.data_mode),
            )

        # 2. 初始化組件
        # Market Data (傳入兩個 DataFrame，並指定目標交易對)
        self.market_data = MarketData(
            self.df_5m,
            self.df_1d,
            self.window_size,
            self.window_size_1d,
            target_symbol=self.target_symbol,
            feature_symbols=self.feature_symbols,
        )
        
        # Observer
        obs_dtype = kwargs.get("obs_dtype", getattr(Config, "OBS_DTYPE", "float32"))
        self.observer = TradingObserver(self.window_size, self.window_size_1d, self.market_data, obs_dtype=obs_dtype)
        self.observation_space = self.observer.observation_space
        
        # Action Processor
        self.action_processor = ActionProcessor(
            leverage=self.leverage,
            max_step_pos_change_pct=float(kwargs.get("max_step_pos_change_pct", Config.MAX_STEP_POS_CHANGE_PCT)),
            min_position_change=self.min_position_change,
            # no-trade 雙門檻（hysteresis）：讓 0 倉位更穩定，避免 action 0 附近抖動造成反覆成交
            no_trade_entry_threshold=float(kwargs.get("no_trade_entry_threshold", getattr(Config, "NO_TRADE_ENTRY_THRESHOLD", 0.0))),
            no_trade_exit_threshold=float(kwargs.get("no_trade_exit_threshold", getattr(Config, "NO_TRADE_EXIT_THRESHOLD", 0.0))),
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

        # ---- Render runtime caches ----
        # 每步事件（供 episode 結束時 render 畫 entry/reduce/close/flip/SL/LIQ）
        self._episode_events: list[dict[str, Any]] = []
        self._last_info: Optional[Dict[str, Any]] = None
        self._last_executed_step_idx: Optional[int] = None
        self._rendered_this_episode: bool = False
        # Renderer（lazy import，避免在訓練時增加 import 成本）
        self._renderer = None
        if self.render_enabled:
            try:
                from Env.Renderers.mpl_episode_renderer import MplfinanceEpisodeRenderer, RenderOutput

                self._renderer = MplfinanceEpisodeRenderer(
                    output=RenderOutput(
                        save_dir=self.render_dir,
                        save=self.render_save,
                        show=self.render_show,
                        dpi=self.render_dpi,
                        figsize=self.render_figsize,
                        export_events=getattr(self, "render_export_events", False),
                    )
                )
            except Exception:
                self._renderer = None
        
        # Cost / Constraint（供 Lagrangian-SAC 使用）
        # 注意：reward 與 cost 分離，cost 透過 info 回傳，方便訓練端做 λ 更新與解析。
        # REFACTORED: 只保留死亡懲罰 (Liq / Bankrupt) 與 摩擦成本 (Fee/Equity)
        # CostCalculator 現在不再需要 weights (已內建正規化公式)，這裡維持空建構
        self.cost_calculator = CostCalculator()

        # Runtime State
        self.current_step = 0
        self.episode_steps = 0
        self.done = False
        # Flip budget 機制已移除；保留 risk_budget 欄位供觀測/相容性使用（固定為 1.0）
        self.risk_budget = 1.0
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

    def _record_step_events(
        self,
        *,
        step_idx: int,
        prev_size: float,
        new_size: float,
        current_price: float,
        current_high: float,
        current_low: float,
        stop_loss_triggered: bool,
        liq_triggered: bool,
    ) -> None:
        """記錄本 step 的交易事件（供 render 使用）。

        需求對應：
        - entry：0 -> 非 0
        - reduce：部分減倉（同方向、曝險變小、且未回到 0）
        - close：完全平倉（非 0 -> 0）
        - flip：翻倉（多<->空）同一步標兩點（close + entry）
        - SL：止損（獨立標記）
        - LIQ：爆倉/強平（獨立標記）

        備註：
        - delta_qty 單位為「資產單位」（例如 BTC）。
        - 事件位置以「本 step 使用的 K 線（step_idx）」對齊；時間由 df_5m.timestamp 決定。
        """
        try:
            step_idx = int(step_idx)
            prev_size = float(prev_size)
            new_size = float(new_size)
            delta = float(new_size - prev_size)
            if abs(delta) <= 1e-12 and (not stop_loss_triggered) and (not liq_triggered):
                return
            ts = None
            try:
                ts = self.market_data.df_5m["timestamp"].iloc[int(step_idx)]
            except Exception:
                ts = None

            base = {
                "step_idx": int(step_idx),
                "timestamp": ts,
                "price": float(current_price),
                "high": float(current_high),
                "low": float(current_low),
                "prev_size": float(prev_size),
                "new_size": float(new_size),
            }

            # 1) SL / LIQ 事件（獨立標記）
            if bool(stop_loss_triggered) and abs(prev_size) > 1e-12:
                self._episode_events.append({**base, "type": "sl", "delta_qty": float(-prev_size)})
            if bool(liq_triggered) and abs(prev_size) > 1e-12:
                self._episode_events.append({**base, "type": "liq", "delta_qty": float(-prev_size)})

            # 2) 交易事件（entry/reduce/close/flip）
            # flip：多<->空（同一步標兩點：close 舊方向 + entry 新方向）
            if (prev_size > 1e-12 and new_size < -1e-12) or (prev_size < -1e-12 and new_size > 1e-12):
                self._episode_events.append({**base, "type": "close", "delta_qty": float(-prev_size)})
                self._episode_events.append({**base, "type": "entry", "delta_qty": float(new_size)})
                return

            # entry
            if abs(prev_size) <= 1e-12 and abs(new_size) > 1e-12:
                self._episode_events.append({**base, "type": "entry", "delta_qty": float(new_size)})
                return

            # close
            if abs(prev_size) > 1e-12 and abs(new_size) <= 1e-12:
                self._episode_events.append({**base, "type": "close", "delta_qty": float(-prev_size)})
                return

            # reduce（部分減倉）
            if abs(prev_size) > 1e-12 and abs(new_size) > 1e-12:
                same_dir = (prev_size * new_size) > 0.0
                if same_dir and abs(new_size) < abs(prev_size) and abs(delta) > 1e-12:
                    self._episode_events.append({**base, "type": "reduce", "delta_qty": float(delta)})
        except Exception:
            # render helper must never break training
            return

    @staticmethod
    def _split_train_eval_by_recent_months(
        *,
        df_5m: "pd.DataFrame",
        df_1d: "pd.DataFrame",
        holdout_months: int,
        mode: str,
    ) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
        """
        依「最近 N 個月」切分資料，用於 Train/Eval 分離（避免資料洩漏）。

        - eval: 取 df_*.timestamp >= eval_start
        - train: 取 df_*.timestamp <  eval_start

        注意：
        - 這裡用 df_5m 的 max timestamp 決定 eval_start（避免 1d 某些來源晚開始造成切點偏移）
        """
        if "timestamp" not in df_5m.columns:
            raise ValueError("df_5m must contain 'timestamp' column for train/eval split")
        if "timestamp" not in df_1d.columns:
            raise ValueError("df_1d must contain 'timestamp' column for train/eval split")
        if int(holdout_months) <= 0:
            raise ValueError("holdout_months must be > 0")

        mode = str(mode).lower().strip()
        if mode not in {"train", "eval"}:
            raise ValueError("mode must be 'train' or 'eval'")

        # ensure datetime dtype
        if not pd.api.types.is_datetime64_any_dtype(df_5m["timestamp"]):
            df_5m = df_5m.copy()
            df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"])
        if not pd.api.types.is_datetime64_any_dtype(df_1d["timestamp"]):
            df_1d = df_1d.copy()
            df_1d["timestamp"] = pd.to_datetime(df_1d["timestamp"])

        max_ts = df_5m["timestamp"].max()
        if pd.isna(max_ts):
            raise ValueError("df_5m timestamp is empty; cannot split train/eval")

        eval_start = pd.Timestamp(max_ts) - pd.DateOffset(months=int(holdout_months))

        if mode == "eval":
            df_5m_out = df_5m[df_5m["timestamp"] >= eval_start].copy()
            df_1d_out = df_1d[df_1d["timestamp"] >= eval_start].copy()
        else:
            df_5m_out = df_5m[df_5m["timestamp"] < eval_start].copy()
            df_1d_out = df_1d[df_1d["timestamp"] < eval_start].copy()

        df_5m_out = df_5m_out.sort_values("timestamp").reset_index(drop=True)
        df_1d_out = df_1d_out.sort_values("timestamp").reset_index(drop=True)

        # Basic sanity: 5m must have enough rows to run at least one episode window.
        if len(df_5m_out) < 10:
            raise ValueError(
                f"Split produced too few 5m rows for mode={mode}. "
                f"holdout_months={holdout_months}, rows={len(df_5m_out)}"
            )

        return df_5m_out, df_1d_out

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        # 1. 決定起始點
        # Eval obs 必須填滿：起點至少在 max(window_size_5m, window_size_1d*288) 之後
        # - window_size_1d 用「天」計算，因此換算成 5m steps = window_size_1d * 288
        # - 若資料不足，會退化成可用的最小起點，避免 randint 空範圍
        warmup_steps = int(max(int(self.window_size), int(self.window_size_1d) * 288)) if bool(self.ensure_filled_obs) else int(self.window_size)
        warmup_steps = max(1, warmup_steps)

        if self.random_start:
            max_start_index = len(self.market_data.df_5m) - self.min_episode_steps - 2
            # 若資料量不足以支援 min_episode_steps（常見於測試用合成資料），退化為固定起點，避免 randint 空範圍。
            if int(max_start_index) <= int(warmup_steps):
                self.current_step = int(min(warmup_steps, max(1, len(self.market_data.df_5m) - 1)))
            else:
                self.current_step = int(random.randint(int(warmup_steps), int(max_start_index)))
        else:
            self.current_step = int(min(warmup_steps, max(1, len(self.market_data.df_5m) - 1)))
            
        self.episode_start_step = self.current_step
        # 供評估端追蹤「episode 起始時間」用（避免 callback 自己猜時間）
        # 注意：timestamp 取自 MarketData.df_5m（已確保是 datetime）
        self.episode_start_timestamp = self._get_timestamp_str(step_idx=int(self.episode_start_step))
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

        # Render caches
        self._episode_events = []
        self._last_info = None
        self._last_executed_step_idx = None
        self._rendered_this_episode = False
        
        # 3. Reset State Variables
        # Flip budget 機制已移除：risk_budget 固定為 1.0（僅供觀測/相容性）
        self.risk_budget = 1.0
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
            # --- action vs execution discrepancy (for next obs) ---
            "cooldown_remaining_norm": 0.0,
            "action_overridden_flag": 0.0,
            "last_action_raw": 0.0,
            "last_action_used": 0.0,
            "last_target_pos_pct": 0.0,
            "last_final_pos_pct": 0.0,
            "trade_executed_flag": 0.0,
        }

        # 4. Initial Observation
        metrics = self.market_data.get_market_metrics(self.current_step)
        current_price = metrics['close']
        self.tracker.update_account_series(self.current_step, self.executor, current_price)
        
        # Gymnasium reset() 允許回傳 info；我們把 episode 起始資訊放進去，方便評估端取用
        return self._get_observation(), {
            "episode_start_step": int(self.episode_start_step),
            "episode_start_timestamp": self.episode_start_timestamp,
        }

    # ---------------------------------------------------------------------
    # Public helpers (for EvalCallback / debugging)
    # ---------------------------------------------------------------------
    def get_current_step(self) -> int:
        """取得環境目前的 step index（用於 eval/debug）。"""
        return int(self.current_step)

    def get_episode_steps(self) -> int:
        """取得當前 episode 已走過的步數（環境內部步數）。"""
        return int(self.episode_steps)

    def get_episode_start_step(self) -> int:
        """取得當前 episode 的起始 step index。"""
        return int(getattr(self, "episode_start_step", 0))

    def get_episode_start_timestamp(self) -> Optional[str]:
        """取得當前 episode 的起始 timestamp（字串；若不可得則回傳 None）。"""
        return getattr(self, "episode_start_timestamp", None)

    def get_feature_distribution_stats(
        self,
        feature_names: Iterable[str],
        *,
        quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> Dict[str, Dict[str, float]]:
        """
        回傳指定 5m 特徵在「整段資料」上的分布摘要（mean/std + quantiles）。

        用途：
        - 快速檢查 train/eval 的 regime 是否漂移
        - 避免在 callback 端自己重算特徵或讀 CSV

        Returns:
            dict[str, dict[str, float]]，例如：
            {
              "atr_ratio_z": {"mean": 0.01, "std": 0.98, "q10": -1.2, "q50": 0.0, "q90": 1.3},
              ...
            }
        """
        names = [str(x) for x in feature_names]
        cols = list(getattr(self.market_data, "cols_5m", []) or [])
        if not cols or not hasattr(self.market_data, "features_5m_arr"):
            return {}

        idx_map = {c: i for i, c in enumerate(cols)}
        out: Dict[str, Dict[str, float]] = {}

        feats = np.asarray(self.market_data.features_5m_arr)
        if feats.ndim != 2 or feats.shape[0] <= 0:
            return {}

        qs = tuple(float(q) for q in quantiles)
        for name in names:
            i = idx_map.get(name)
            if i is None:
                continue
            x = feats[:, int(i)].astype(np.float64, copy=False)
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            d: Dict[str, float] = {
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
            }
            try:
                qv = np.quantile(x, qs)
                for q, v in zip(qs, qv):
                    key = f"q{int(round(q * 100)):02d}"
                    d[key] = float(v)
            except Exception:
                # quantile 非關鍵：失敗就只回 mean/std
                pass
            out[name] = d
        return out

    def get_episode_feature_stats(
        self,
        feature_names: Iterable[str],
        *,
        quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> Dict[str, Dict[str, float]]:
        """
        回傳指定 5m 特徵在「本 episode 區間」上的分布摘要。

        註：episode 區間使用 [episode_start_step, episode_start_step + episode_steps)。
        """
        start = int(getattr(self, "episode_start_step", 0))
        steps = int(getattr(self, "episode_steps", 0))
        end = int(max(start, start + max(1, steps)))

        cols = list(getattr(self.market_data, "cols_5m", []) or [])
        if not cols or not hasattr(self.market_data, "features_5m_arr"):
            return {}

        feats = np.asarray(self.market_data.features_5m_arr)
        n = int(feats.shape[0]) if feats.ndim == 2 else 0
        if n <= 0:
            return {}

        start = int(np.clip(start, 0, max(0, n - 1)))
        end = int(np.clip(end, start + 1, n))

        # 暫時切片成一個小矩陣（只用少數欄位）
        idx_map = {c: i for i, c in enumerate(cols)}
        names = [str(x) for x in feature_names]
        qs = tuple(float(q) for q in quantiles)
        out: Dict[str, Dict[str, float]] = {}
        for name in names:
            i = idx_map.get(name)
            if i is None:
                continue
            x = feats[start:end, int(i)].astype(np.float64, copy=False)
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            d: Dict[str, float] = {
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
            }
            try:
                qv = np.quantile(x, qs)
                for q, v in zip(qs, qv):
                    key = f"q{int(round(q * 100)):02d}"
                    d[key] = float(v)
            except Exception:
                pass
            out[name] = d
        return out

    def _get_timestamp_str(self, *, step_idx: int) -> Optional[str]:
        """安全取得 step_idx 對應的 timestamp 字串（YYYY-MM-DD HH:MM:SS）。"""
        try:
            df = getattr(self.market_data, "df_5m", None)
            if df is None or "timestamp" not in df.columns:
                return None
            idx = int(np.clip(int(step_idx), 0, max(0, len(df) - 1)))
            ts = df["timestamp"].iloc[idx]
            # pandas Timestamp / datetime
            return str(ts)[:19]
        except Exception:
            return None

    def _get_observation(self):
        # 準備 Observation 需要的各類 metrics
        metrics = self.market_data.get_market_metrics(self.current_step)
        current_price = metrics['close']
        atr_est = float(metrics.get("atr_ratio", 0.0)) * float(current_price)
        
        risk_signals = self.observer.compute_risk_signals(
            self.executor, current_price, atr_est, self.current_step, len(self.market_data.df_5m)
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
            'cooldown_remaining': float(self.stop_loss_cooldown)
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
            # 評估/統計常用：episode 起點與步數
            info["episode_start_step"] = int(getattr(self, "episode_start_step", 0))
            info["episode_start_timestamp"] = getattr(self, "episode_start_timestamp", None)
            info["episode_steps"] = int(getattr(self, "episode_steps", 0))
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
        update_every = int(max(1, self.risk_base_update_steps))
        if (self.current_step - self.last_risk_base_update_step) >= update_every:
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

    def _apply_stop_loss_cooldown(self, action: np.ndarray, current_price: float, last_equity: float) -> np.ndarray:
        """
        若處於停損冷卻期，保持當前倉位（目標倉位與當前倉位一致）。

        Args:
            action: 原始 action
            current_price: 當前價格（用於計算當前倉位百分比）
            last_equity: 當前權益（用於計算當前倉位百分比）

        Returns:
            action（可能被覆寫為當前倉位對應的百分比）
        """
        if self.stop_loss_cooldown > 0:
            self.stop_loss_cooldown -= 1
            # 獲取當前倉位大小
            current_size = float(self.executor.position.size)
            
            # 將當前倉位大小轉換為百分比
            # position_pct = (size * price) / (equity * leverage)
            if last_equity > 0 and current_price > 0:
                max_capacity = last_equity * self.leverage
                if max_capacity > 0:
                    current_pos_pct = (current_size * current_price) / max_capacity
                    # 確保在有效範圍內
                    current_pos_pct = np.clip(current_pos_pct, -1.0, 1.0)
                    return np.array([current_pos_pct], dtype=action.dtype)
            
            # 如果無法計算（例如權益為 0），則保持 action = 0（平倉）
            return np.zeros_like(action)
        return action

    def _process_action_and_execute(
        self,
        *,
        action: np.ndarray,
        last_equity: float,
        prices: _StepPrices,
    ) -> Tuple[float, float, float, bool, np.ndarray, float, bool]:
        """
        動作處理（含限制/翻倉預算）+ 手續費預估 + 實際下單執行。

        Args:
            action: 原始 action
            last_equity: 以 current_price 計算的上一刻權益
            prices: 本 step 價格資訊

        Returns:
            (final_pos_pct, expected_fee, prev_wallet, is_flip, action_used, target_pos_pct, action_overridden_flag)
        """
        action_used = self._apply_stop_loss_cooldown(action, prices.current_price, last_equity)
        # action 是否被 env 覆寫（目前主要是 cooldown）
        try:
            action_overridden_flag = bool(abs(float(action_used[0]) - float(action[0])) > 1e-8)
        except (TypeError, ValueError, IndexError):
            action_overridden_flag = False

        target_pos_pct, is_flip = self.action_processor.process_action(
            action_used, self.executor, prices.current_price
        )

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

        return (
            float(final_pos_pct),
            float(expected_fee),
            float(prev_wallet),
            bool(is_flip),
            action_used,
            float(target_pos_pct),
            bool(action_overridden_flag),
        )

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

    def _estimate_add_only_fee(
        self,
        *,
        prev_size: float,
        new_size: float,
        current_price: float,
    ) -> float:
        """
        估算「只計入加碼/加曝險」的手續費（排除減倉/平倉）。

        用途：
        - 供 cost_fric（摩擦成本線）使用：只在加碼/加曝險時才計入摩擦成本。

        定義：
        - 若同向（未翻倉）：add_qty = max(0, |new| - |prev|)
        - 若翻倉（跨 0 且新舊皆非 0）：只計入「新方向開倉」的部分 => add_qty = |new|

        Args:
            prev_size: 上一步的持倉 size
            new_size: 本步執行後的持倉 size
            current_price: 本步當下價格（用於估算名目）

        Returns:
            add_only_fee（>=0）
        """
        price = float(current_price)
        if not (price > 0.0):
            return 0.0

        prev = float(prev_size)
        new = float(new_size)
        prev_nz = abs(prev) > 1e-8
        new_nz = abs(new) > 1e-8

        # Flip：只計入新方向「開倉」的名目（排除關倉名目）
        if prev_nz and new_nz and (prev * new < 0.0):
            add_qty = abs(new)
        else:
            add_qty = max(0.0, abs(new) - abs(prev))

        add_notional = float(add_qty) * price
        fee_rate_pct = float(self.executor.get_fee_rate())
        return float(abs(add_notional) * (fee_rate_pct / 100.0))

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
        更新 rolling fee tracking，並回傳 step_fee 與 safe_equity。

        Returns:
            (step_fee, safe_equity)
        """
        self.tracker.update_fee_tracking(self.current_step, self.executor.total_fees)
        step_fee = float(self.tracker.last_step_fee)
        safe_equity = max(float(self.executor.equity(current_price)), self.initial_balance * 0.5)
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
        已停用：Flip budget 機制移除後，不再回復/消耗 risk_budget。
        保留函式僅為相容性（避免舊程式碼呼叫時出錯）。
        """
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
        # 記錄本次 action 對應的 bar index（render/事件對齊用）
        step_idx = int(self.current_step)

        # 1. Prepare market inputs
        metrics = self.market_data.get_market_metrics(self.current_step)
        prices = self._prepare_step_prices(metrics)
        self._maybe_update_daily_risk_base()
        last_equity = float(self.executor.equity(prices.current_price))
        prev_size = float(self._last_position_size)

        # 2. Process action + execute
        final_pos_pct, expected_fee, prev_wallet, is_flip, action_used, target_pos_pct, action_overridden_flag = self._process_action_and_execute(
            action=action,
            last_equity=last_equity,
            prices=prices,
        )

        # 3. Post execution updates (for next obs + accounting)
        self._update_action_effects_cache(expected_fee=expected_fee, current_price=prices.current_price)
        # --- store action discrepancy fields for next obs ---
        try:
            self._last_action_effects["last_action_raw"] = float(action[0]) if hasattr(action, "__len__") else float(action)
        except (TypeError, ValueError, IndexError):
            self._last_action_effects["last_action_raw"] = 0.0
        try:
            self._last_action_effects["last_action_used"] = float(action_used[0]) if hasattr(action_used, "__len__") else float(action_used)
        except (TypeError, ValueError, IndexError):
            self._last_action_effects["last_action_used"] = 0.0
        self._last_action_effects["last_target_pos_pct"] = float(target_pos_pct)
        self._last_action_effects["last_final_pos_pct"] = float(final_pos_pct)
        self._last_action_effects["action_overridden_flag"] = 1.0 if bool(action_overridden_flag) else 0.0
        new_size = float(self.executor.position.size)
        self._update_position_entry(new_size=new_size)
        step_fee, safe_equity = self._update_fee_tracking(current_price=prices.current_price)

        # 4. Mark-to-market (use NEXT close)
        mark_price, new_equity = self._mark_to_market()

        # 5. Risk budget recovery（已停用；保留呼叫不影響）
        self._recover_risk_budget(new_equity=new_equity)

        # 6. Reward features + episode events
        position_change, traded, position_change_norm, turnover_ratio, current_dd = self._compute_reward_features(
            current_price=prices.current_price,
            last_equity=last_equity,
            new_equity=new_equity,
            new_size=new_size,
        )
        # Add-only friction fee（排除減倉/平倉；翻倉只算新方向開倉）
        step_fee_add_only = self._estimate_add_only_fee(
            prev_size=float(prev_size),
            new_size=float(new_size),
            current_price=float(prices.current_price),
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
        # traded flag（供下一個 observation 使用）
        self._last_action_effects["trade_executed_flag"] = 1.0 if bool(traded) else 0.0
        # cooldown remaining norm（供下一個 observation 使用）
        try:
            cd = float(getattr(self, "stop_loss_cooldown", 0) or 0)
            cd_max = float(max(1, int(getattr(Config, "STOP_LOSS_COOLDOWN_STEPS", 0) or 0)))
            self._last_action_effects["cooldown_remaining_norm"] = float(np.clip(cd / cd_max, 0.0, 1.0))
        except (TypeError, ValueError):
            self._last_action_effects["cooldown_remaining_norm"] = 0.0
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
            fee_budget_ratio=1.0,
            position_pct=pos_pct_reward,
            abs_position_pct=abs(pos_pct_reward),
            trend_score=metrics['trend_score']
        )
        
        # 9. Cost / Constraint（成本線）
        # 我們使用「當下價格」計算風險訊號（含 stop_loss_missing / 距離爆倉 / margin_ratio 等），
        # 並把總 cost 與分項寫入 info，方便訓練端做 Lagrangian 更新與 debug。
        risk_post = self.observer.compute_risk_signals(
            self.executor, prices.current_price, prices.atr_est, self.current_step, len(self.market_data.df_5m)
        )
        step_fee_ratio = float(step_fee / self.initial_balance) if self.initial_balance > 0 else 0.0
        
        # 計算成本（僅保留 cost_risk 和 cost_fric）
        cost_out = self.cost_calculator.compute(
            liq_triggered=bool(liq_triggered),
            equity=float(new_equity),
            min_balance=float(self.min_balance),
            step_fee=float(step_fee),
        )

        # ---- Record render events (entry/reduce/close/flip/SL/LIQ) ----
        self._record_step_events(
            step_idx=step_idx,
            prev_size=float(prev_size),
            new_size=float(new_size),
            current_price=float(prices.current_price),
            current_high=float(prices.current_high),
            current_low=float(prices.current_low),
            stop_loss_triggered=bool(stop_loss_triggered),
            liq_triggered=bool(liq_triggered),
        )
        self._last_executed_step_idx = int(step_idx)

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
        # 雙通道成本（供多 λ 使用）
        if "cost_risk" in cost_out:
            info["cost_risk"] = float(cost_out["cost_risk"])
        if "cost_fric" in cost_out:
            info["cost_fric"] = float(cost_out["cost_fric"])

        # cache last info for render()
        self._last_info = dict(info)

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

        # ---- Auto render on done (VecEnv-safe) ----
        # 若啟用 render_on_done：在終止那一步直接 render，並把結果路徑塞回 info，
        # 讓 VecEnv 自動 reset 後仍可在 callback 端看到 render_path。
        if bool(self.done) and bool(self.render_enabled) and bool(getattr(self, "render_on_done", False)):
            if not bool(getattr(self, "_rendered_this_episode", False)):
                try:
                    out_path = self.render(mode="human")
                except Exception:
                    out_path = None
                if out_path is not None:
                    info["render_path"] = str(out_path)
                self._rendered_this_episode = True

        # Gymnasium: (obs, reward, terminated, truncated, info)
        return self._get_observation(), reward, bool(terminated), bool(truncated), info

    def render(self, mode='human'):
        # 只在 episode 結束時 render，避免訓練中每步繪圖造成效能負擔
        if not bool(getattr(self, "done", False)):
            return None
        renderer = getattr(self, "_renderer", None)
        if renderer is None:
            return None
        try:
            return renderer.render_episode(env=self, info=getattr(self, "_last_info", None))
        except Exception:
            return None
    
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
