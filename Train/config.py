
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

    MIN_EPISODE_STEPS = 10000    
    # 每回合最小步數（保護期）：
    # 防止因軟性規則（如手續費限制）太早終止，  
    # 迫使 agent 經歷長期後果。
    
    # =========================
    # 2. 交易物理（交易所模擬）
    # =========================
    LEVERAGE = 10                
    # 槓桿倍數：
    # 放大盈虧風險，影響爆倉價。

    TRANSACTION_FEE = 0.04       
    # 交易手續費率（0.04%=Taker費率）：
    # 任意開倉／平倉需支付（倉位*價格*費率）

    MAINTENANCE_MARGIN_RATE = 0.005   
    # 維持保證金比率（0.5%）： 0.005 = 0.5%
    # 當 (權益/持倉價值) < 此值時，觸發爆倉。
    
    # =========================
    # 3. 策略限制（硬／軟約束）
    # =========================
    MIN_POSITION_CHANGE = 0.05   
    # 最小動作死區（5%）： 0.05 = 5%
    # 小於 5% 倉位變動直接忽略（降低雜訊）。

    MAX_STEP_POS_CHANGE_PCT = 0.3
    # 每步最大倉位變動（30%）： 0.3 = 30%
    # 除風險降低動作外，單步限最大增減30%倉位，防止瞬間滿倉。

    MAX_POSITION_PCT = 0.5
    # 持倉上限（50%）： 0.5 = 50%
    # ActionSmoothClipWrapper 用此做 soft-clip。

    STOP_LOSS_ATR = 5
    # 停損距離（以 ATR 倍數）： 5 = 5倍ATR
    # 動態停損 = 進場價 ± ATR×倍數
    # 觸及即強平（最近常用 6）

    STOP_LOSS_COOLDOWN_STEPS = 4
    # 停損後冷卻步數： 1 = 1步
    # 強迫 N 步內動作為 0，防止報復性交易。

    ACTION_SMOOTHING_ALPHA = 0.2  
    # 動作 EMA 平滑係數： 0.2 = 20%
    # 0.2=極平滑, 1.0=不平滑
    # 降低高頻振盪。

    # =========================
    # 4. 風險預算與手續費限制
    # =========================
    # 反多空 whipsaw 相關參數
    FLIP_BUDGET_MAX = 1.0 # 1.0 = 100% 
    FLIP_COST = 0.25 # 0.25 = 25%
    FLIP_THRESHOLD = 0.2 # 0.2 = 20%
    FLIP_RECOVERY_RATE = 0.01 # 0.01 = 1%
    FLIP_PROFIT_RECOVERY_RATE = 0.1 # 0.1 = 10%
    
    FEE_LIMIT_RATIO = 0.30
    # 手續費滾動上限（30%）： 0.30 = 30%
    # 滾動窗口內收費累計超過權益50%則強制結束該回合。
    # （之前為0.35，現放寬至0.5方便學習）

    FEE_ROLLING_WINDOW = 2000
    # 手續費上限窗口（步數）： 2000 = 2000步
    # 僅統計最近 N 步手續費用作約束

    # =========================
    # 5. 獎勵塑形（"教師"設計）
    # =========================
    # 主線獎勵僅保留 log return + 終局懲罰；其他塑形移至成本線
    REWARD_FEE_LIMIT_PENALTY = 2.0

    # =========================
    # 6. 強化學習訓練超參數（SAC）
    # =========================
    ACTION_REPEAT = 6
    # 幀跳（frame skip）：
    # 每 6 步 agent 再做一次決策（每30分鐘作一次決策）
    # 降低高頻投注造成不穩定

    TOTAL_TIMESTEPS = 20_000_000    
    # 訓練總步數（環境互動數）

    BATCH_SIZE = 256                    
    # 單次 mini-batch 訓練樣本數

    BUFFER_SIZE = 100_000                
    # Replay Buffer 最大容量

    LEARNING_STARTS = 5_000             
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
    # 成本1：保證金風險（爆倉概率）
    # 成本2：回撤風險（大幅虧損概率）
    # 成本3：換手成本（position_change_norm）
    # 成本4：手續費預算成本（fee budget shortage）
    COST_LIMITS = [
        0.1 / (1 - GAMMA),   # Margin risk
        0.2 / (1 - GAMMA),   # Drawdown risk
        0.05 / (1 - GAMMA),  # Turnover cost (~0.05 均值)
        0.10 / (1 - GAMMA)   # Fee budget cost (~0.10 均值)
    ]
    # 成本斜率微調
    TURNOVER_COST_SCALE = 1.0
    FEE_BUDGET_COST_SCALE = 1.0

    MARGIN_SAFE = 1.2       
    # 安全保證金緩衝區比例

    COST_LIQ_PENALTY = 5.0  
    # 爆倉時的惡性成本懲罰

    DD_WARN = 0.1             
    DD_CRIT = 0.2             
    DD_MAX_PENALTY = 5.0      
    # 回撤成本曲線參數

    # =========================
    # 8. 系統＆日誌
    # =========================
    NUM_ENVS = 40
    # 並行環境數（開啟多進程）

    LOG_INTERVAL = NUM_ENVS * 1000  
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
    
    FILTER_SMALL_REWARD_THRESHOLD = 0.01 # 0.01 = 1%
    # 獎勵回饋閾值：絕對值小於此值視為"無顯著後果"
    
    FILTER_DROP_PROBABILITY = 0.60 # 0.90 = 90%
    # 丟棄機率：對於無效動作樣本，有 90% 機率不寫入 Buffer

