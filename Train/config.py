
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

    WINDOW_SIZE = 288 * 7      
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

    MAX_EPISODE_STEPS = 105_120 / 12     # 105,120 steps = 1 year
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

    MAINTENANCE_MARGIN_RATE = 0.005   
    # 維持保證金比率（0.5%）： 0.005 = 0.5%
    # 當 (權益/持倉價值) < 此值時，觸發爆倉。
    LIQUIDATION_WARN_PCT = 0.005
    # 強平距離風險門檻（價格距離 / 現價），低於此值視為高風險。
    STOP_LOSS_WARN_PCT = 0.002
    # 止損距離風險門檻（價格距離 / 現價），低於此值視為即將觸發。
    
    # =========================
    # 3. 策略限制（硬／軟約束）
    # =========================
    MIN_POSITION_CHANGE = 0.03
    # 最小動作死區（5%）： 0.05 = 5%
    # 小於 5% 倉位變動直接忽略（降低雜訊）。

    MAX_STEP_POS_CHANGE_PCT = 0.5
    # 每步最大倉位變動（30%）： 0.3 = 30%
    # 除風險降低動作外，單步限最大增減30%倉位，防止瞬間滿倉。

    MAX_POSITION_PCT = 0.5
    # 持倉上限（50%）： 0.5 = 50%
    # ActionSmoothClipWrapper 用此做 soft-clip。

    STOP_LOSS_ATR = 5
    # 停損距離（以 ATR 倍數）： 5 = 5倍ATR
    # 動態停損 = 進場價 ± ATR×倍數
    # 觸及即強平（最近常用 6）

    STOP_LOSS_COOLDOWN_STEPS = 6
    # 停損後冷卻步數： 1 = 1步
    # 強迫 N 步內動作為 0，防止報復性交易。

    ACTION_SMOOTHING_ALPHA = 0.5
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

    BUFFER_SIZE = 300_000                
    # Replay Buffer 最大容量
    # 更新比率 Batch_Size / Buffer_Size = 128 / 300,000 = 0.0004266666666666667 = 0.04266666666666667%

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
    # 成本4：回撤約束 c_t(dd) = max(0, dd_t - dd_soft)
    
    TURNOVER_TAU_MAX = 0.003   # 允許的期望換手率 (per-step)
    DEATH_P_MAX = 0.0005       # 允許的爆倉概率
    
    # C3: Margin Safety
    COST_MARGIN_TARGET = 1.5   # 目標維持保證金倍數 (m_target)，低於此值開始產生成本
    MARGIN_VIOLATION_MAX = 0.05 # 允許的平均違規程度 (per-step cost budget)

    # C4: Drawdown Control
    COST_DD_SOFT_LIMIT = 0.20  # 軟性回撤限制 (dd_soft)，超過 20% 開始產生成本
    DD_VIOLATION_MAX = 0.05    # 允許的平均違規程度 (per-step cost budget)

    COST_LIMITS = [
        TURNOVER_TAU_MAX / (1 - GAMMA),
        DEATH_P_MAX / (1 - GAMMA),
        MARGIN_VIOLATION_MAX / (1 - GAMMA),
        DD_VIOLATION_MAX / (1 - GAMMA),
    ]
    
    # 成本計算相關參數
    TURNOVER_NOTIONAL_SCALE = None  # None 則使用 equity * leverage 動態尺度

    # =========================
    # 8. 系統＆日誌
    # =========================
    NUM_ENVS = 32
    # 並行環境數（開啟多進程）

    LOG_INTERVAL = MAX_EPISODE_STEPS * 10
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
    
    FILTER_DROP_PROBABILITY = 0.70 # 0.90 = 90%
    # 丟棄機率：對於無效動作樣本，有 90% 機率不寫入 Buffer
