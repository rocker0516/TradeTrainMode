
import torch

class Config:
    # =========================
    # 資料設定
    # =========================
    DATA_PATH = "Data/BTCUSDT_futures_volume_5years_5min.csv" 
    # 歷史資料CSV路徑 (範例檔案為5年、5分鐘K線資料, 場景：BTCUSDT永續合約。請依自己的資料集調整路徑。)

    # =========================
    # 環境參數（定義每個episode和環境行為）
    # =========================
    WINDOW_SIZE = 288 * 3            
    # 狀態視窗長度：
    # 每一個step時，Agent會看到過去多少根K線(bar)的價格與其他屬性資料（如成交量）。
    # 1440相當於5天資料（5天*24小時*60分鐘/5分鐘=1440），滿足「3天以上」的觀察需求。
    LEVERAGE = 10                
    # 槓桿倍數：
    # 用於模擬期貨交易時持倉風險與可能損益乘數，影響風險與強平線。
    TRANSACTION_FEE = 0.04       
    # 交易手續費率 (單邊)：
    # 假設為萬分之4，每次交易收取，不論開倉或平倉，每次都需付。
    # 即買一次/賣一次各收一遍。
    INITIAL_BALANCE = 10_000     
    # 初始模擬資產餘額（USDT）
    MIN_BALANCE = INITIAL_BALANCE * 0.01 
    # 最低帳戶資產限制：
    # 若資產跌破此數值(預設1%)，則判定為破產，環境自動結束Episode。
    MIN_EPISODE_STEPS = 10000    
    # 每個Episode最少進行的步數：
    # 用於隨機起始點：強制一個回合至少走這麼多步才允許結束，避免太早結束造成不穩定的統計。
    MAINTENANCE_MARGIN_RATE = 0.005   
    # 強制平倉維持保證金率（0.5%）：
    # 期貨場景下只要保證金比跌破這條線即強制平倉（liquidation）。
    MIN_POSITION_CHANGE = 0.1   
    # 最小調倉幅度：
    # 單次下單動作可調整的最少倉位百分比。例如0.01等於1%，用於抑制微小動作與滑價。
    # 將其與 MIN_POSITION_CHANGE 同步
    
    MAX_STEP_POS_CHANGE_PCT = 0.5
    # 單步最大倉位調整幅度：
    # 每個step允許調整的最大多/空方向總幅度佔總倉位的比例（如0.5代表一次最多改變50%）。用於避免大跳或爆倉。

    # 目標持倉硬上限 (行為裁剪用)，同時配合平滑避免高頻翻倉
    MAX_POSITION_PCT = 0.5
    ACTION_SMOOTHING_ALPHA = 0.3  # 指數平滑係數，越小越平滑；0.3 建議值

    # =========================
    # 訓練設定（RL核心相關）
    # =========================
    ACTION_REPEAT = 4
    # 動作重複次數 (Frame Skip)：
    # Agent 每做一次決策，環境會持續執行該動作 4 個 Step (約20分鐘)。
    # 這能自然降低交易頻率，讓 Agent 學習更長期的趨勢，而非追逐短期雜訊。

    TOTAL_TIMESTEPS = 10_000_000    
    # 總訓練步數：
    # 指所有環境總共經過的決策次數，包含多進程並行後的總和，規模可調(如2千萬步)。
    BATCH_SIZE = 256                    
    # 批次訓練取樣大小：
    # 一次更新時，從經驗回放池(replay buffer)隨機抽取多少條 transition 資料訓練模型。
    # (調降至 256 以降低 GPU VRAM 負擔)
    BUFFER_SIZE = 100_000                
    # 經驗池最大容量：
    # 能儲存多少步transition，容量滿時會釋放最舊資料。
    # (大幅調降以防止 RAM 溢出導致 Swap 變慢。10萬步 * 80KB ≈ 8GB RAM，保留空間給系統)
    LEARNING_STARTS = 5_000             
    # 探索期步數：
    # 一開始預先隨機探索幾步（用來蒐集資料），之後再用學到的策略行為，避免冷啟動時的不穩定。
    GAMMA = 0.99                         
    # 折扣因子(γ)：
    # 強化學習用來折現未來獎勵的權重，越接近1代表更重視長期回報。
    TAU = 0.005                          
    # 軟更新參數(τ)：
    # 目標網路(target network)的參數更新比率，用於平滑更新提升穩定性。
    LR = 3e-4                            
    # 學習率：
    # 神經網路優化時的步進幅度，數值過大或過小會影響收斂速度與穩定性。
    SEED = 42                            
    # 隨機種子：
    # 控制所有隨機性來源（如torch、numpy等），確保實驗可重現（reproducibility）。
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 裝置設定：
    # 優先使用NVIDIA GPU運算，若無GPU則自動切換為CPU。
    # torch.cuda.is_available()可偵測CUDA裝置。

    # =========================
    # 並行環境數（加速取樣效率）
    # =========================
    NUM_ENVS = 64
    # 並行執行的環境數量：
    # 在取樣(training rollout)時，同時運行多組環境，能顯著加速資料生成和訓練速度。
    # 記憶體與CPU資源不足可調低本數。 (調降至 16)

    # =========================
    # 記錄設定（進度和訓練結果列印）
    # =========================
    LOG_INTERVAL = 50_000             
    # 記錄/顯示間隔：
    # 每當訓練總步數達到此間隔，就把目前訓練績效、損失、各種指標輸出到Console、Tensorboard或log檔。

    # =========================
    # 逐步Log設定（除錯用，預設關閉）
    # =========================
    STEP_LOG_ENABLED = False           # 是否記錄每一步（建議只在除錯時開啟）
    STEP_LOG_DIR = "step_logs"         # 根目錄；每個env_id會各自一個子資料夾
    STEP_LOG_EVERY_N = 1               # 每N步記錄一次（越大越省I/O）

    # =========================
    # 安全約束參數 (SAC-Lagrangian用)
    # =========================
    # 本專案支援多重成本(Cost)限制，也就是多目標安全強化學習。
    # 預設開啟兩項成本約束，分別如下：
    # Cost 1: 保證金/資產比風險 (Margin Risk, d1)
    #   - 控制極端槓桿帶來的爆倉(強平)或高風險時調降pos或採取防禦。
    # Cost 2: 回撤風險 (Drawdown Risk, d2) - 分段式
    #   - 控制大幅資金回撤，影響Agent風險曲線。
    # Cost 3: 手續費損耗風險 (Fee Risk, d3) - 不納入，避免干擾(預設移除)
    # COST_LIMITS:
    # 每種Cost的允許上限，[d1, d2, ...] 越低越嚴格，0.1意指最多允許10%的平均違規。
    # index 0 = 保證金風險（Margin Risk），違例率上限0.1
    # index 1 = 回撤風險（Drawdown Risk），違例率上限0.1
    COST_LIMITS = [0.1, 0.2]
    # index 0 = 保證金風險（Margin Risk），違例率上限0.1
    # index 1 = 回撤風險（Drawdown Risk），違例率上限0.2 (放寬回撤容忍度，避免過早的高懲罰干擾學習)

    # =========================
    # Cost損失計算相關參數
    # =========================
    MARGIN_SAFE = 1.2       
    # 保證金安全邊際比(M_safe)：
    # 下調至 1.2，避免在高槓桿下稍微波動就觸發懲罰。
    COST_LIQ_PENALTY = 5.0  
    # 強制平倉懲罰倍數(C_liq)：
    # 若資產被強平(liquidation)，則Cost會額外加重處罰，此係數可拉高罰則權重。
    
    # Drawdown Parameters (Segmented, 回撤風險分段參數)
    DD_WARN = 0.1             
    # 回撤警告水位(10%)：
    # 若最大回撤超過本水位，會開始計算第一級Cost懲罰。
    DD_CRIT = 0.2             
    # 回撤嚴重水位(20%)：
    # 若最大回撤超過此水位，懲罰Num會急劇提升，加速策略修正。
    DD_MAX_PENALTY = 5.0      
    # 最大回撤終局懲罰：
    # 當episode結束時，若最大回撤超標給予region最大Cost延伸懲罰(增強收益surface的懲罰效果)。

    # Reward Parameters (獎勵函數設定)
    REWARD_TURNOVER_PENALTY = 5.0      
    # 換手懲罰權重(alpha_turn)：
    # 降低至 5.0，因為我們透過 MIN_POSITION_CHANGE_FEE_PROTECT 來控制無效交易。
    
    REWARD_DD_PENALTY = 0.1          
    # 當前回撤懲罰權重(dd_penalty_coef)：
    # 每個step若有非零回撤會給予額外懲罰，鼓勵減少波動。
    REWARD_HOLD_BONUS = 0.01
    # 持倉不動獎勵(hold bonus)：
    # 微幅調升至 0.5，Action Repeat 模式下這會累積成可觀的獎勵，鼓勵"耐心"。

    # =========================
    # 交易執行參數
    # =========================
    STOP_LOSS_ATR = 5.0
    # 止損距離 (ATR倍數)：
    # 原預設 2.5 過於敏感，導致大部分交易被雜訊掃出場 (Avg StopLoss ~180/ep)。
    # 調寬至 5.0，給予交易更多呼吸空間，避免被隨機波動洗掉。

    FEE_LIMIT_RATIO = 0.35
    # 單回合累積手續費上限比例： 0.35
    # 降低上限並配合滾動窗口縮短，讓費用風控更即時。

    FEE_ROLLING_WINDOW = 2000
    # 手續費滾動計算窗口 (Steps)：
    # 縮短為約 2000 步，讓費用過熱時更快反映。

    # 費用預算塑形懲罰 (remaining budget 越低懲罰越高)
    REWARD_FEE_BUDGET_PENALTY = 2.0
