
import torch

class Config:
    """
    全域配置：交易強化學習環境
    
    本類整合所有超參數，包含：
    1. 數據載入
    2. 市場機理參數（槓桿、手續費、保證金）
    3. 強化學習訓練（超參數、網路等）
    4. 獎勵塑形與成本約束
    5. 策略限制（停損、風險預算）
    """
    
    # =========================
    # 1. 數據與環境設置
    # =========================
    DATA_PATH = "Data/BTCUSDT_futures_volume_5years_5min.csv" 
    # 歷史 OHLCV 數據路徑（建議 5 分 K 線）

    WINDOW_SIZE = 288 * 3
    # 觀察窗口長度（回看天數）：
    # 288 步/天 × 3 天 = 864 步。
    # 策略看到過去三天的行情做決策。

    INITIAL_BALANCE = 10_000     
    # 初始錢包資金（USDT）

    MIN_BALANCE = INITIAL_BALANCE * 0.01 
    # 強制清算閾值（初始資金 1%）： 
    # 餘額低於此值立即結束該輪實驗（模擬破產）。

    MIN_EPISODE_STEPS = 105_120 / 12
    # 每回合最小步數（保護期）：
    # 防止因軟性規則（如手續費限制）太早終止，  
    # 迫使 agent 經歷長期後果。

    MAX_EPISODE_STEPS = 105_120 / 12      # 105,120 steps = 1 year
    # 每回合最大步數（約 1 年）：
    # 超過此步數視為自然存活（資料耗盡），不給予死亡懲罰。
    
    # =========================
    # 2. 交易物理（交易所模擬）
    # =========================
    LEVERAGE = 10   
    # 槓桿倍數：
    # 放大盈虧風險，影響爆倉價。

    TRANSACTION_FEE = 0.005       
    # 交易手續費率（0.005%=Taker費率）：
    # 任意開倉／平倉需支付（倉位*價格*費率）

    # 手續費課程式（Fee Curriculum, stage-based by condition）
    # 目標：達標就升階，每次增加 10% 的 TRANSACTION_FEE，直到回到 TRANSACTION_FEE
    # 觸發條件（目前預設）：rolling win rate > 50%
    #
    # 注意：
    # - fee_rate 單位沿用專案既有設計：0.005 代表 0.005%（交易執行器內會 /100）
    # - 升階頻率用「episode 間隔」控制，避免同一段 win rate 長時間高於門檻時狂升
    FEE_CURRICULUM_ENABLED = True
    FEE_CURRICULUM_INITIAL_MULT = 0.05 #初始手續費倍數 0.05%
    FEE_CURRICULUM_STEP_MULT = 0.05 #每次升階增加的手續費倍數 5%
    FEE_CURRICULUM_WINRATE_THRESHOLD = 0.60 #30%勝率開始升階
    FEE_CURRICULUM_AVG_PROFIT_THRESHOLD_PCT = 20.0 #平均 Profit(%) 達到 50% 才升階（與 dashboard 顯示一致）
    FEE_CURRICULUM_MIN_EPISODES = 50 #50步開始啟用
    FEE_CURRICULUM_MIN_EPISODES_BETWEEN_ADVANCES = 50 #50步之間至少間隔50步才升階

    # （舊版、未接線）時間式 fee warmup/ramp 設定：保留以免舊文件/實驗引用
    FEE_MULT_WARMUP_STEPS = 500_000
    FEE_MULT_RAMP_STEPS = None

    # =========================
    # 2.1 Observation 設計（State Branch）
    # =========================
    # market_state：提供「較慢、更像 regime/結構」的訊號給 MLP state branch。
    # 注意：這些欄位來自 Env/features.build_all_features()（已做 NaN-safe & float32）
    # 若欄位不存在，環境會自動補 0.0（避免因資料集缺欄位而炸掉）。
    #
    # 若你「確定沒 edge」，擴充 market_state 不能憑空創造 edge；
    # 但常見用途是讓 agent 更容易學到「什麼時候應該少交易/不交易」，用於降低尾端損失。
    MARKET_STATE_COLS = [
        # Liquidity / microstructure proxies
        "dollar_volume_log_z",
        "amihud_z",
        "parkinson_vol_z",
        "kyle_lambda_z",
        "vpin_z",
        "trade_entropy_z",
        "vol_imbalance_z",
        "trades_z",
        "hl_spread_z",

        # Trend / momentum (normalized)
        "macd_z",
        "macd_hist_z",

        # SMC proxies (normalized)
        "smc_structure_trend",
        "smc_sweep_up",
        "smc_sweep_down",
        "smc_dist_to_swing_high",
        "smc_dist_to_swing_low",
        "smc_displacement_z",

        # Volume-profile density (longer window in build_all_features)
        "profile_density",

        # Long-term regime
        "lt_ret_5d_z",
        "lt_vol_regime",
        "lt_vol_5d_z",

        # Multi-timeframe bias
        "bias_15m",
        "bias_1h",
        "bias_1d",

        # Price position / ATR z
        "price_pos_in_range",
        "atr_z_score",

        # Tradability / no-trade gating (computed in Env; NaN-safe)
        "tradability_score",
    ]

    # =========================
    # 2.2 Observation 設計（Daily Macro CNN Branch）
    # =========================
    # 目的：提供「日線層級的市場體制/情緒/鏈上代理」給 daily CNN（long-term branch）。
    #
    # 注意（避免 look-ahead bias）：
    # - env 會在每個 5m step t 使用「昨天 (D-1)」的日線資料作為 daily_seq 的最後一筆。
    # - 因此 daily_seq 代表「過去已完成的 N 日」資訊，而不是當天未收盤的資訊。
    MACRO_ENABLED = True
    MACRO_DATA_DIR = "Data"
    DAILY_WINDOW_SIZE = 60  # 日線回看天數（long-term CNN）
    MACRO_ROLLING_Z_WINDOW = 365
    MACRO_ROLLING_Z_MIN_PERIODS = 30

    # 日線/宏觀資料檔案（由 Env/macro_daily.py 讀取合併）
    # 你在需求中列的檔案都在這裡（BTC coinglass 1d 重複列出一次，實際只需一份）
    MACRO_FILES = {
        "altcoin_season": "altcoin_season_index_history.csv",
        "bmo": "bitcoin_macro_oscillator_index_history.csv",
        "sopr": "bitcoin_sth_sopr_index_history.csv",
        "fear_greed": "fear_greed_index_history.csv",
        "coinglass_btc_1d": "BTCUSDT_futures_volume_coinglass_5years_1d.csv",
    }

    MAINTENANCE_MARGIN_RATE = 0.005   
    # 維持保證金比率（0.5%）： 0.005 = 0.5%
    # 當 (權益/持倉價值) < 此值時，觸發爆倉。
    LIQUIDATION_WARN_PCT = 0.005
    # 強平距離風險門檻（價格距離 / 現價），低於此值視為高風險。
    STOP_LOSS_WARN_PCT = 0.002
    # 止損距離風險門檻（價格距離 / 現價），低於此值視為即將觸發。ㄋ
    
    # =========================
    # 3. 策略限制（硬／軟約束）
    # =========================
    MIN_POSITION_CHANGE = 0.3
    # 最小動作死區（10%）： 0.1 = 10%
    # 小於 10% 倉位變動直接忽略（降低雜訊）。
    # (排除減倉平倉動作)

    MAX_STEP_POS_CHANGE_PCT = 0.75
    # 每步最大倉位變動（75%）： 0.75 = 75%
    # 除風險降低動作外，單步限最大增減30%倉位，防止瞬間滿倉。

    MAX_POSITION_PCT = 0.75
    # 持倉上限（75%）： 0.75 = 75%
    # ActionSmoothClipWrapper 用此做 soft-clip。

    STOP_LOSS_ATR = 4
    # 停損距離（以 ATR 倍數）： 5 = 5倍ATR
    # 動態停損 = 進場價 ± ATR×倍數
    # 觸及即強平（最近常用 6）

    STOP_LOSS_LIQ_BUFFER_PCT = 0.002
    # 止損相對強平的安全緩衝（百分比，0.002=0.2%）：
    # 目的：避免 stop_loss_price 設得比強平價更遠，導致「先被強平、止損永遠觸發不到」。
    # 多單：要求 stop_loss_price >= liq_price * (1 + buffer)
    # 空單：要求 stop_loss_price <= liq_price * (1 - buffer)

    STOP_LOSS_COOLDOWN_STEPS = 6
    # 停損後冷卻步數： 1 = 1步
    # 強迫 N 步內動作為 0，防止報復性交易。

    ACTION_SMOOTHING_ALPHA = 0.8
    # 動作 EMA 平滑係數： 0.3 = 30%
    # 0.2=極平滑, 0.3=平滑, 0.5=不平滑, 0.7=極不平滑, 1.0=極不平滑
    # 降低高頻振盪。
    #如何選擇平滑係數？ 動作變化 = 平滑係數 * 前一動作 + (1 - 平滑係數) * 當前動作

    # =========================
    # 4. 風險預算與手續費限制
    # =========================
    # 反多空 whipsaw 相關參數
    FLIP_BUDGET_MAX = 1.0 # 1.0 = 100% 
    FLIP_COST = 0.25 # 0.25 = 25%
    FLIP_THRESHOLD = 0.2 # 0.2 = 20%
    FLIP_RECOVERY_RATE = 0.01 # 0.01 = 1%
    FLIP_PROFIT_RECOVERY_RATE = 0.1 # 0.1 = 10%
    
    # 是否啟用手續費上限約束（若關閉，相關檢查與終止條件無效化）
    FEE_LIMIT_ENABLED = False
    FEE_LIMIT_RATIO = 0.20
    # 手續費滾動上限（30%）： 0.30 = 30%
    # 滾動窗口內收費累計超過權益50%則強制結束該回合。
    # （之前為0.35，現放寬至0.5方便學習）

    FEE_ROLLING_WINDOW = 2000
    # 手續費上限窗口（步數）： 2000 = 2000步
    # 僅統計最近 N 步手續費用作約束

    # =========================
    # 5. 獎勵塑形（"教師"設計）
    # =========================
    # 主線獎勵僅保留 log return；其他塑形移至成本線
    REWARD_LOG_RET_WEIGHT = 1  # 方案B：放大 log-return 影響力（預設 2x）

    # --- Optional: Conviction + Trend Alignment bonus (reward shaping) ---
    # 目的：鼓勵 agent 在「訊號夠強」時敢於用「較大倉位」承擔風險，而不是收斂到 0 倉位。
    # 設計：只加分、不扣分；且必須同時滿足
    # - |trend_dir| >= REWARD_CONVICTION_TREND_MIN_STRENGTH（trend_dir = tanh(macd_z)）
    # - abs_position_pct >= REWARD_CONVICTION_MIN_ABS_POS（以 equity*leverage 正規化）
    # 並且曝險方向需與 trend_dir 同向才給 bonus。
    #
    # 建議：先從 0.0 開始（維持純 log-return），若想要更「有信心壓大」再逐步調高。
    REWARD_CONVICTION_TREND_BONUS_WEIGHT = 0.3
    REWARD_CONVICTION_TREND_MIN_STRENGTH = 0.8
    REWARD_CONVICTION_MIN_ABS_POS = 0.3

    # =========================
    # 6. 強化學習訓練超參數（SAC）
    # =========================
    ACTION_REPEAT = 1
    # 幀跳（frame skip）：
    # 每 6 步 agent 再做一次決策（每30分鐘作一次決策）
    # 降低高頻投注造成不穩定

    TOTAL_TIMESTEPS = 10_000_000    
    # 訓練總步數（環境互動數）

    BATCH_SIZE = 64
    # 單次 mini-batch 訓練樣本數

    BUFFER_SIZE = 500_000                
    # Replay Buffer 最大容量
    # 更新比率 Batch_Size / Buffer_Size = 64 / 500,000 = 0.000256 = 0.0256%

    LEARNING_STARTS = int(BUFFER_SIZE / 5)       #  BUFFER_SIZE / 5 = 60,000
    # 預熱步數：前 N 步完全隨機探索

    GAMMA = 0.99                         
    # 折現因子（長期目標0.99）

    TAU = 0.005                          
    # Target 網路軟更新速率

    LR = 3e-4                            
    # 學習率

    SEED = 42                            
    # 隨機種子，方便實驗復現

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 運算設備（自動偵測是否用GPU）

    # =========================
    # 7. 安全層（Lagrangian約束）
    # =========================
    # 成本1：換手率約束 c_t(turnover) = |Δpos_notional| / notional_scale
    # 成本2：死亡約束 c_t(death) = 1 (爆倉/強平終局), 0 其他
    # 成本3：保證金安全約束 c_t(margin) = max(0, m_target - equity/maint_margin)
    # 成本4（原回撤約束）已停用：改以「止損接近度」作為路徑型安全成本（見 COST_SL_*）。
    
    # TURNOVER_TAU_MAX：允許的期望換手率（每步最大可接受的平均換手率），例如 0.003 代表「每步手續費相當於倉位規模的 0.3%」。此參數主要用於控制 agent 不要過度頻繁交易。
    TURNOVER_TAU_MAX = 0.5   # 允許的期望換手率（每步最大 50%）

    # C1 優化：針對「反覆翻多空 / churn」提高換手成本，避免策略靠 flip 賺到短期 reward 但長期不穩定。
    # - 當 is_flip=True 時，C1 會乘上 (1 + COST_TURNOVER_FLIP_MULT)。
    # - 當 is_risk_reducing=True（明顯減倉/去風險）時，C1 會乘上 COST_TURNOVER_RISK_REDUCING_MULT（<1 代表放寬）。
    COST_TURNOVER_FLIP_MULT = 1.0
    COST_TURNOVER_RISK_REDUCING_MULT = 0.1

    # DEATH_P_MAX：允許的爆倉（強平/資金耗盡）概率。這個值決定每一輪中爆倉事件的期望出現頻率（0.0005 代表千步僅允許 0.5 次），可用於約束風險管理，讓策略有強烈誘因避免爆倉。
    DEATH_P_MAX = 0.001       # 允許的爆倉概率（每步不超過 0.1%）
    
    # C3: 保證金安全限制相關參數說明
    # COST_MARGIN_TARGET：當「維持保證金倍數」（equity/maint_margin）低於此值時開始發生成本（懲罰），
    # 用於強制 agent 維持較高的保證金安全邊際。例如 1.5 表示需維持至少 1.5 倍的維持保證金。
    COST_MARGIN_TARGET = 1.5   # 目標維持保證金倍數 (m_target)

    # MARGIN_VIOLATION_MAX：可接受的平均違規程度（每步 step 可忍受的 margin cost 平均上限）。
    # 此值越小，agent 被要求越嚴格（通常 0.05 表示每 20 步內最多有 1 步顯著違規）。
    MARGIN_VIOLATION_MAX = 0.05 # 允許的平均違規程度 (per-step cost budget)

    # C6: 止損安全帶與無止損違規成本參數說明
    # COST_SL_SAFE_BAND：當目前價格距離止損線（stop loss）落入此比例內時開始產生成本（違規懲罰）。
    # 例如 0.05 代表在止損位置前的 5% 價格區間會被預警（越近違規成本越高），
    # 指在 agent 靠近止損時引導風險意識、警惕裸奔。
    COST_SL_SAFE_BAND = 0.02  # 2% 內開始產生成本

    # STOPLOSS_PROX_VIOLATION_MAX：允許的止損帶違規每步平均上限（cost budget），
    # 作用同上，值越低則允許越少次數或幅度違規。
    STOPLOSS_PROX_VIOLATION_MAX = 0.2  # 允許的平均違規程度 (per-step cost budget)

    # COST_SL_MISSING_ALPHA：當 position 沒有設置止損時，懲罰如何計算。
    # 該參數決定 missing stop loss 成本更偏向 maint_margin_ratio（權重 alpha 越高）或是槓桿率（1-alpha 越高）。
    # 用來對無止損裸奔行為給予嚴重處分（遷就不同情境的安全重要性）。
    COST_SL_MISSING_ALPHA = 0.8  # missing-stop 動態懲罰的權重，較偏 maint_margin 優先

    # COST_SL_MISSING_LCAP：missing-stop cost 中槓桿水平的最大歸一化範圍（與環境裁剪一致，設為10.0），
    # 避免不同環境下槓桿懲罰分佈不一致。
    COST_SL_MISSING_LCAP = LEVERAGE  # leverage_ratio 的歸一化上限（與 env clip 10 對齊）

    # COST_SL_CLIP：單步止損違規成本的最大值，用於防止 cost Q function 發散或極端值影響學習穩定性。
    COST_SL_CLIP = 1.0  # 單步 C6 上限裁剪，避免 Qc 發散

    # C4 優化（方向引導）：逆勢曝險成本（可控、保守）。
    # 定義：trend_dir = tanh(trend_score)，misalign = max(0, -position_pct * trend_dir)
    # 直覺：上升趨勢時做空、下降趨勢時做多，且曝險越大 => 成本越高。
    # 注意：這是「引導」而非硬規則，建議先用小權重觀察行為改變。
    COST_SL_TREND_WEIGHT = 0.10

    COST_LIMITS = [
        TURNOVER_TAU_MAX / (1 - GAMMA),
        DEATH_P_MAX / (1 - GAMMA),
        MARGIN_VIOLATION_MAX / (1 - GAMMA),
        STOPLOSS_PROX_VIOLATION_MAX / (1 - GAMMA),
    ]

    # =========================
    # 7.1 Lagrangian 訓練穩定化（不靠 penalty 上限，改用限制退火與數值穩定）
    # =========================
    # λ 的學習率：偏小避免早期因 cost critic 未收斂而讓 λ 飆升
    LAGRANGIAN_LR = 0.001
    # log(λ) clamp：僅做數值穩定（避免 NaN/爆炸），不是用來「調教策略」
    LAMBDA_LOG_CLAMP_MIN = -5.0
    LAMBDA_LOG_CLAMP_MAX = 5.0

    # Cost limits annealing：前期先放寬限制，避免 λ 在策略尚未學會時就爆炸
    # effective_limits = COST_LIMITS * mult(step)，mult 由大到小線性退火到 1.0
    COST_LIMIT_ANNEAL_STEPS = 1_000_000
    COST_LIMIT_MULT_START = 50.0
    COST_LIMIT_MULT_END = 1.0

    # --- λ 更新改用 EMA + warm-up/ramp ---
    # LAGRANGIAN_WARMUP_STEPS：前 N steps 不啟用任何成本約束（λ=0），讓策略先學基本互動/價差結構
    LAGRANGIAN_WARMUP_STEPS = 500_000
    # LAGRANGIAN_RAMP_STEPS：warm-up 後用此步數線性把 λ 從 0 拉到 1（逐步打開約束）
    LAGRANGIAN_RAMP_STEPS = 500_000
    # LAGRANGIAN_COST_EMA_BETA：λ 更新用的 EMA 平滑係數（越大越平滑）
    LAGRANGIAN_COST_EMA_BETA = 0.99
    
    # 成本計算相關參數
    TURNOVER_NOTIONAL_SCALE = None  # None 則使用 equity * leverage 動態尺度

    # =========================
    # 8. 系統＆日誌
    # =========================
    NUM_ENVS = 64
    # 並行環境數（開啟多進程）

    LOG_INTERVAL = MAX_EPISODE_STEPS * 25
    # Dashboard 印出間隔（步數）

    STEP_LOG_ENABLED = False           
    # DEBUG: 是否每步以 JSONL 紀錄 (極慢，訓練建議關閉)
    
    STEP_LOG_DIR = "step_logs"
    # 步驟日誌保存目錄
    
    STEP_LOG_EVERY_N = 1 * NUM_ENVS
    # =========================
    # 9. 經驗回放篩選參數 (Replay Buffer Filtering)
    # =========================
    FILTER_SMALL_ACTION_THRESHOLD = MIN_POSITION_CHANGE # 0.05 = 5%
    # 動作幅度閾值：小於此值視為"不動"
    
    FILTER_SMALL_REWARD_THRESHOLD = 0.005 # 0.005 = 0.5%
    # 獎勵回饋閾值：絕對值小於此值視為"無顯著後果"
    
    FILTER_DROP_PROBABILITY = 0.00 # 0.00 = 0%
    # 丟棄機率：對於無效動作樣本，有 0% 機率不寫入 Buffer
