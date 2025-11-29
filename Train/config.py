
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
    MIN_EPISODE_STEPS = 10000        # Episode最短步數，隨機起始點用
    MAINTENANCE_MARGIN_RATE = 0.005  # 強制平倉維持保證金率(0.5%)

    # =========================
    # 訓練設定
    # =========================
    TOTAL_TIMESTEPS = 1_000_000_000      # 總訓練步數（例如 10 億步）
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
    NUM_ENVS = 32                     # 並行強化環境數（效率為主）

    # =========================
    # 記錄設定
    # =========================
    LOG_INTERVAL = 10_000             # 每多少步顯示一次儀表板結果（console）

    # =========================
    # 安全約束參數(SAC-Lagrangian)
    # =========================
    # Cost 1: 保證金/資產比風險 (Margin Risk, d1)
    # Cost 2: 回撤風險 (Drawdown Risk, d2)
    # 每一步允許的平均cost上限。大多數cost指標已正規化為[0,1]區間（除強平懲罰除外）。
    # 可調成本參數：若想要嚴格/寬鬆限制，請修改此陣列。 index 0 = margin risk，上限0.1， index 1 = drawdown risk，上限0.1
    COST_LIMITS = [0.1, 0.1]

    # =========================
    # Cost損失相關參數
    # =========================
    MARGIN_SAFE = 1.5                 # 安全邊際比（M_safe, 保證金比低於此值就開始有懲罰）
    COST_LIQ_PENALTY = 5.0            # 強平發生懲罰倍數（C_liq）
    DD_SOFT = 0.2                     # 軟性回撤門檻（例如0.2代表20%回撤會開始被懲罰）
    DD_MAX_PENALTY = 5.0              # 終局最大回撤罰則（alpha, episode結束時大幅回撤的額外懲罰）

