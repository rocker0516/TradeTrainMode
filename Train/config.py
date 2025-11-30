
import torch

class Config:
    # =========================
    # 資料設定
    # =========================
    DATA_PATH = "Data/BTCUSDT_futures_volume_5years_5min.csv" # 歷史資料CSV路徑(5年，5分鐘線)

    # =========================
    # 環境參數
    # =========================
    WINDOW_SIZE = 288                # 狀態視窗大小（每一個step需有多少bar的歷史價格資訊，約一天資料若1bar=5min）
    LEVERAGE = 10                    # 槓桿倍數，影響持倉風險
    TRANSACTION_FEE = 0.04           # 交易手續費（單邊，百分比形式：0.04 代表萬分之4）
    INITIAL_BALANCE = 10_000         # 初始帳戶資金
    MIN_BALANCE = INITIAL_BALANCE * 0.1 # 最小帳戶資金
    MIN_EPISODE_STEPS = 10000        # Episode最短步數，隨機起始點用
    MAINTENANCE_MARGIN_RATE = 0.005  # 強制平倉維持保證金率(0.5%)
    MIN_POSITION_CHANGE = 0.10       # 最小調倉幅度 (5% 總倉位)，避免微小變動造成手續費浪費
    MAX_STEP_POS_CHANGE_PCT = 0.2    # 單步最大倉位變化限制 (20% Max Capacity)

    # =========================
    # 訓練設定
    # =========================
    TOTAL_TIMESTEPS = 50_000_000      # 總訓練步數（例如 10 億步）
    BATCH_SIZE = 1024                    # 每次訓練批次大小
    BUFFER_SIZE = 100_000                # 經驗回放池最大容量
    LEARNING_STARTS = 5_000              # 先隨機探索幾步才開始學習
    GAMMA = 0.99                         # 折扣因子（回報折現率）
    TAU = 0.005                          # 軟更新參數（目標網路）
    LR = 3e-4                            # 學習率
    SEED = 42                            # 隨機種子
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 設備選擇，gpu優先

    # =========================
    # 並行環境數
    # =========================
    NUM_ENVS = 64                     # 並行強化環境數（效率為主）

    # =========================
    # 記錄設定
    # =========================
    LOG_INTERVAL = 10_000             # 每多少步顯示一次儀表板結果（console）

    # =========================
    # 安全約束參數(SAC-Lagrangian)
    # =========================
    # Cost 1: 保證金/資產比風險 (Margin Risk, d1)
    # Cost 2: 回撤風險 (Drawdown Risk, d2) - 分段式
    # Cost 3: 手續費損耗風險 (Fee Risk, d3) - 費率/初始資金
    # 可調成本參數：若想要嚴格/寬鬆限制，請修改此陣列。 
    # index 0 = margin risk，上限0.1
    # index 1 = drawdown risk，上限0.1
    # index 2 = fee risk (avg per step)，上限 2e-4 (0.02%)
    COST_LIMITS = [0.1, 0.1, 0.0002]

    # =========================
    # Cost損失相關參數
    # =========================
    MARGIN_SAFE = 1.5                 # 安全邊際比（M_safe, 保證金比低於此值就開始有懲罰）
    COST_LIQ_PENALTY = 5.0            # 強平發生懲罰倍數（C_liq）
    
    # Drawdown Parameters (Segmented)
    DD_WARN = 0.1                     # 警告回撤水位 (10%)
    DD_CRIT = 0.2                     # 嚴重回撤水位 (20%)
    DD_MAX_PENALTY = 5.0              # 終局最大回撤罰則（alpha, episode結束時大幅回撤的額外懲罰）
    
    # Reward Parameters
    REWARD_TURNOVER_PENALTY = 0.02    # 主線獎勵中對換手的懲罰權重 (alpha_turn)
