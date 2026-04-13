"""
Env 專用設定檔。

目的：
- 將 `Env/trading_env.py` 依賴的參數集中在 `Env/` 下，方便環境側調整與維護。
- 其他模組可透過 kwargs 覆寫（例如 TradingEnvironment(..., **kwargs)）。
"""

from __future__ import annotations


class Config:
    """交易環境的預設參數集合（常數容器）。"""

    # -------------------------------------------------------------------------
    # 資金與交易成本
    # -------------------------------------------------------------------------
    INITIAL_BALANCE: float =300.0
    TRANSACTION_FEE: float = 0.01  # 手續費百分比（例：0.04 代表 0.04%）

    # 成交成本模型（以單一整數 bitmask 控制，0=關閉所有）
    # 1=spread（half-spread bps），2=min_notional（名目門檻），4=slippage（交易量比例滑點）
    EXECUTION_COST_MODE: int = 0
    # spread：half-spread（bps），買=price*(1+hs*1e-4)，賣=price*(1-hs*1e-4)
    SPREAD_HALF_BPS: float = 0.0
    # min_notional：單筆名目金額（abs(delta_size)*price）門檻；低於門檻即不成交（含減倉/平倉）
    MIN_NOTIONAL: float = 0.0
    # slippage（交易量比例）：
    # slip_bps_t = base + vol_coeff * vol_proxy + size_coeff * size_ratio
    # 其中：
    # - vol_proxy = abs(delta_notional) / max(bar_notional, 1e-8)   # 當前 bar 成交參與率
    # - size_ratio = abs(delta_notional) / max(ADV_notional, 1e-8) # 相對長期平均流動性
    SLIP_BASE_BPS: float = 0.0
    SLIP_VOL_COEFF: float = 0.0
    SLIP_SIZE_COEFF: float = 15.0
    # ADV 回顧天數（以 5m bar 計算名目之 rolling 平均）
    ADV_LOOKBACK_DAYS: int = 30

    # -------------------------------------------------------------------------
    # 視窗與步數
    # -------------------------------------------------------------------------
    WINDOW_SIZE: int = 12  # 5m 根數；目前預設 12 = 1 小時回看
    WINDOW_SIZE_1D: int = 6
    MIN_EPISODE_STEPS: int = 288 * 21 * 1 # 288 * 21 * 1 = 5760
    MAX_EPISODE_STEPS: int = 288 * 21 * 1 # 288 * 21 * 1 = 5760
    RISK_BASE_UPDATE_STEPS: int = 288  # 每 N steps 更新 daily_risk_base（用於單步倉位變化上限）

    # -------------------------------------------------------------------------
    # 槓桿、餘額與倉位限制
    # -------------------------------------------------------------------------
    LEVERAGE: float = 10.0
    MIN_BALANCE: float = INITIAL_BALANCE * 0.6  # 最小餘額比例（0.5 = 50%）

    # -------------------------------------------------------------------------
    # 調參指南：目標「評估集零 balance_insufficient」＋「平均正收益」
    # -------------------------------------------------------------------------
    # 死亡條件（見 trading_env）：new_equity <= min_balance → terminated，
    # termination_reason == "balance_insufficient"（另強平為 "liq_triggered"）。
    #
    # 在隨機起點 + 槓桿下，要同時「幾乎不死」又「平均賺錢」，通常要先**縮曝險**再談報酬：
    # 1) LEVERAGE：預設 10 偏高，可改 3～5 再訓練（最有效降低觸發 min_balance）。
    # 2) MAX_STEP_POS_CHANGE_PCT：Phase AB 常在 kwargs 設 0.5，可試 0.25～0.35 減少單步梭哈。
    # 3) NO_TRADE_ENTRY_THRESHOLD：提高（例如 0.35～0.45）可減少小訊號進出與手續費磨損。
    # 4) Phase B 訓練：`Train/run_sac_phase_ab.py` 預設 lambda_risk≈5、lambda_buffer≈0.001
    #    （事件型死亡懲罰 + 貼近 min_balance 的 dense 緩衝）；可依 eval 死亡率再調。
    # 5) 驗收：eval JSON 中 termination_reason_counts["balance_insufficient"]==0 且
    #    summary["profit"]["mean"]>0（或 log_return_sum mean>0）；勿只靠調低 MIN_BALANCE「假裝不死」。
    MIN_POSITION_CHANGE: float = 0.25 # 最小調倉幅度 deadband（0 = 不啟用）
    MAX_STEP_POS_CHANGE_PCT: float = 0.3  # 單步最大持倉比例變化（0.5 = 50%）
    MAX_POSITION_PCT: float = 0.8  # 最大目標持倉比例（供 ActionClipWrapper 等使用） 

    # No-trade 雙門檻（hysteresis）：空倉時 |action| < ENTRY 不進場；有倉時 |action| < EXIT 易回空倉
    NO_TRADE_ENTRY_THRESHOLD: float = 0.24
    NO_TRADE_EXIT_THRESHOLD: float = 0.12

    # -------------------------------------------------------------------------
    # 主線獎勵：順向交易獎勵（Conviction Trend Bonus）
    # -------------------------------------------------------------------------
    # 順向獎勵權重；>0 啟用「強訊號 + 大倉 + 同向」時加分；主線為 log return，此為輔助小權重＋退火
    CONVICTION_TREND_BONUS_WEIGHT: float = 0.1
    # trend_score 縮放倍數（(ma50-ma200)/ma200 為小數比，乘上此倍數後再 tanh 算 strength）
    # 例如 scale=10：trend_score=0.05 → strength≈0.46，易通過 min_strength 0.25
    CONVICTION_TREND_SCORE_SCALE: float = 10.0
    # 趨勢強度門檻 [0,1]；strength = |tanh(scale * trend_score)|，達此值才加分
    CONVICTION_TREND_MIN_STRENGTH: float = 0.35
    # 最小曝險門檻 [0,1]，僅當 abs(position_pct) >= 此值才加分，避免小倉刷分
    CONVICTION_MIN_ABS_POS: float = 0.5
    # Regime 對齊 bonus 權重：A 狀態多頭加分、C 狀態空頭加分，依 dir_strength 加權；0=不啟用；主線為 log return，此為輔助小權重＋退火
    REGIME_ALIGNMENT_BONUS_WEIGHT: float = 0.1
    # 中性區（無 Gate A/C）仍成交時，每步固定扣分（與 REGIME_ALIGNMENT 無關）；0=不啟用；建議小於典型單步 |log-return| 量級
    NEUTRAL_TRADE_PENALTY_WEIGHT: float = 0.0

    # -------------------------------------------------------------------------
    # 手續費與 Fee Limit
    # -------------------------------------------------------------------------
    FEE_ROLLING_WINDOW: int = 288

    # -------------------------------------------------------------------------
    # 止損與清算
    # -------------------------------------------------------------------------
    STOP_LOSS_ATR: float = 3  # 止損距離的 ATR 倍數
    STOP_LOSS_LIQ_BUFFER_PCT: float = 0.2  # 止損相對強平價的安全緩衝（比例）
    STOP_LOSS_COOLDOWN_STEPS: int = 0  # 止損後冷卻步數（30/5）
    STOP_LOSS_EVENT_COST: float = 0.02  # 觸發止損時的額外事件成本（比例）

    # Stop-Buffer Cost：罰「持倉接近止損卻不撤」（ATR 正規化）
    STOP_BUFFER_D_MIN: float = 0.4   # 距離止損 < d_min ATR 開始罰
    STOP_BUFFER_D_SCALE: float = 0.6 # 罰的尺度

    # 觀察用閾值（near_liq / near_stop 特徵），不參與 Cost 計算
    LIQUIDATION_WARN_PCT: float = 0.3
    STOP_LOSS_WARN_PCT: float = 0.6
    


    # -------------------------------------------------------------------------
    # 保證金與 Wrapper 預設
    # -------------------------------------------------------------------------
    MAINTENANCE_MARGIN_RATE: float = 0.005
    ACTION_REPEAT: int = 1  # ActionRepeatWrapper 預設（Frame Skip）

    # -------------------------------------------------------------------------
    # Episode 尾部風險統計（短窗累積 net log return；見 TradingEnvironment ep_tail_loss_count）
    # -------------------------------------------------------------------------
    # 向前 k 步 R_t(k)=sum r_{t..t+k-1}；若 R < -TAIL_LOSS_TAU 視為 tail；連續 t 僅在進入 tail 時計一次事件。
    TAIL_LOSS_WINDOW_K: int = 12
    TAIL_LOSS_TAU: float = 0.05

    # -------------------------------------------------------------------------
    # Step Log
    # -------------------------------------------------------------------------
    STEP_LOG_ENABLED: bool = False
    STEP_LOG_DIR: str = "step_logs"
    STEP_LOG_EVERY_N: int = 1

    # -------------------------------------------------------------------------
    # Observation 與相容性
    # -------------------------------------------------------------------------
    OBS_DTYPE: str = "float16"  # "float16" | "float32"（訓練 RAM 優化）
    # 取代 inf/neginf 的有限值，避免 nan_to_num(..., posinf=0, neginf=0) 把極端訊號壓成 0 而消失
    OBS_INF_CLIP_HIGH: float = 10.0   # +inf → 此值（與 z-score ±5 略大，保留「極端」訊號）
    OBS_INF_CLIP_LOW: float = -10.0   # -inf → 此值

    # -------------------------------------------------------------------------
    # Observation 使用特徵（obs state 開關）
    # -------------------------------------------------------------------------
    # 納入 observation 的 state 鍵名；預設為目前全部欄位。
    # 預計收斂為 5 個 state：4 個市場序列 (5m/1d × target/others) + 1 個帳戶/情境；
    # 可依實驗需求從此 list 關閉某幾項以縮減 obs 維度。
    OBS_STATE_KEYS: tuple[str, ...] = (
        "price_seq_target",      # 5m 目標標的序列
        "price_seq_1d_target",   # 1d 目標標的序列
        "price_seq_1d_others",   # 1d 其他標的序列
        "account_state",         # 帳戶狀態（含 actual_pos_pct）
        "gate_flags",            # Gate A/B/C multi-hot（1d up, 5m 流動性, 1d down）
        "regime_score",          # [p_up, p_down, dir_strength] 強度分數（與 gate 同頻率）
    )

    # -------------------------------------------------------------------------
    # 5 個 state 內部的詳細特徵欄位（空 tuple = 使用該 state 全部欄位）
    # -------------------------------------------------------------------------
    # 預設採用目前驗證較穩定的 hybrid_minimal_signal 組合。
    # 5m 目標（空 tuple = 使用 build_5m_features_split 產生的「全部工程化欄位」）
    OBS_PRICE_SEQ_TARGET_COLS: tuple[str, ...] = (
        "volume_ratio_z_long",
        "sell_volume_roc_12",
        "trades_z_short",
    )
    # 5m 其他（每標的為 {SYMBOL}_xxx；空 tuple = 每標的使用全部工程化 base 欄位）
    OBS_PRICE_SEQ_OTHERS_COLS: tuple[str, ...] = ()
    # 1d 目標（固定欄位）
    OBS_PRICE_SEQ_1D_TARGET_COLS: tuple[str, ...] = (
        "ret_7d_z_long",
        "body_1d_z",
        "funding_close_scale",
        "ls_account_ratio_z",
        "liq_long_log_scale",
        "volume_impact_scale",
        "ema_12_slope_z",
        "open_ema_12_48_spread",
        "high_z_long",
        "close_roc_12",
        "close_ema_12_48_spread",
        "quote_volume_z_long",
        "liq_long_ema_12_48_spread",
    )
    
    # 1d 其他（每標的為 {SYMBOL}_xxx；最後為 macro 固定名）
    OBS_PRICE_SEQ_1D_OTHERS_COLS: tuple[str, ...] = (
        "ret_1d_scale",
        "liq_long_log_z",
        "open_z_short",
        "trades_ema_12_48_spread",
        "liq_long_z_long",
        "sopr_ema_12_48_spread",
    )

    # 帳戶狀態：23 維的完整名稱（順序須與 observer 內建一致）；子集由 OBS_ACCOUNT_STATE_COLS 指定
    OBS_ACCOUNT_STATE_NAMES: tuple[str, ...] = (
        "position_side",
        "position_size_norm",
        "actual_pos_pct",        # 執行後真實倉位比例 [-1, 1]（與 last_final_pos_pct 同口徑）
        "equity_ratio",
        "realized_pnl_ratio",
        "unrealized_pnl_atr",
        "drawdown",
        "liq_distance_atr",
        "stop_loss_distance_atr",
        "margin_usage_ratio",
        "cooldown_remaining_norm",
        "fee_rate",
        "rolling_fee_ratio",
        "trade_count_log",
        "stop_loss_count_log",
        "holding_time_log",
        "buffer_to_min_balance_ratio",
        "steps_since_trade_norm",
        "trade_freq_remaining_ratio",
        "trade_freq_blocked_last",
        "entry_price_ratio",
        "stop_loss_price_ratio",
        "recent_flat_ratio",
    )
    OBS_ACCOUNT_STATE_COLS: tuple[str, ...] = ()  # 空 = 使用上列全部 23 欄

    # 情境狀態：8 維的完整名稱；子集由 OBS_CONTEXT_STATE_COLS 指定
    OBS_CONTEXT_STATE_NAMES: tuple[str, ...] = (
        "action_overridden",
        "last_action_raw",
        "last_action_used",
        "last_target_pos_pct",
        "last_final_pos_pct",
        "trade_executed_flag",
        "predicted_liq_distance_after",
        "available_balance_after_norm",
    )
    OBS_CONTEXT_STATE_COLS: tuple[str, ...] = ()  # 空 = 使用上列全部 8 欄

    # -------------------------------------------------------------------------
    # Data Fetch 服務排程設定
    # -------------------------------------------------------------------------
    # True: 啟用週期性補資料服務；False: 僅執行一次
    DATA_FETCH_SERVICE_ENABLED: bool = True
    # True: 服務啟動後立即執行一次
    DATA_FETCH_RUN_ON_STARTUP: bool = True
    # 每次循環間隔秒數（Binance 5m 資料）
    BINANCE_FETCH_INTERVAL_SECONDS: int = 300
    # 每次循環間隔秒數（CoinGlass 1d 資料）
    COINGLASS_FETCH_INTERVAL_SECONDS: int = 3600
    # 0 代表無限循環；>0 代表最多執行次數（含啟動時首次執行）
    DATA_FETCH_MAX_CYCLES: int = 0

    # Binance 抓取設定
    BINANCE_FETCH_TRADING_PAIRS: tuple[str, ...] = (
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "DOGEUSDT",
        "1000PEPEUSDT",
    )
    BINANCE_FETCH_INTERVAL: str = "5m"
    BINANCE_FETCH_LOOKBACK_DAYS: int = 2 * 365

    # CoinGlass 抓取設定
    COINGLASS_FETCH_TRADING_PAIRS: tuple[str, ...] = (
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "DOGEUSDT",
        "1000PEPEUSDT",
    )
    COINGLASS_EXCHANGE: str = "Binance"
    COINGLASS_FETCH_INTERVAL: str = "1d"
    COINGLASS_FETCH_LOOKBACK_DAYS: int = 365 * 6
