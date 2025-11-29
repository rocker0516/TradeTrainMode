
import numpy as np
import pandas as pd
import unittest
from Env.trading_env import TradingEnvironment

# Index Mapping Constants
IDX_SIDE_START = 0
IDX_SIZE = 3
IDX_UPNL = 4
IDX_EQ_RATIO = 5
IDX_MAX_EQ = 6
IDX_DD = 7
IDX_MARGIN = 8
IDX_STEPS = 9
IDX_TRADE_CNT = 10
IDX_COST = 11
IDX_TOD = 12
IDX_DOW_START = 13

class TestObsLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. 創建合成數據 (Sine Wave Price)
        # 長度 1000, Window 288
        length = 1000
        t = np.linspace(0, 4 * np.pi, length)
        
        # Price: 10000 base + sine wave * 1000
        close = 10000 + 1000 * np.sin(t)
        high = close + 50
        low = close - 50
        open_ = close # simple
        volume = np.random.rand(length) * 100 + 100
        
        # Create DataFrame with DatetimeIndex for DayOfWeek test
        dates = pd.date_range(start='2024-01-01', periods=length, freq='5min')
        
        df = pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'buy_volume': volume * 0.6,
            'sell_volume': volume * 0.4,
            'trades': volume / 10,
            'quote_volume': volume * close,
            'volume_ratio': np.ones(length), # Mock
            'long_short_ratio': np.ones(length) # Mock
        }, index=dates)
        
        cls.df = df
        cls.env = TradingEnvironment(
            df, 
            initial_balance=10000, 
            window_size=50, # 縮小 window 以便更快測試
            leverage=10,
            reward_weights={} 
        )
        
        print("\n=== 開始 Obs 邏輯驗證測試 (20+ Cases) ===\n")

    def check(self, case_id, desc, condition, actual_val=None):
        status = "PASS" if condition else "FAIL"
        val_str = f" | Actual: {actual_val}" if actual_val is not None else ""
        print(f"[{case_id:02d}] {desc:<60} : {status}{val_str}")
        if not condition:
            pass # Allow continue to see all results, but mark failure
            self.fail(f"Case {case_id} Failed: {desc} {val_str}")

    def test_observation_logic(self):
        # Reset Env
        obs, _ = self.env.reset()
        sv = obs['state_vector']
        
        # --- Case 1-4: 初始狀態檢查 ---
        self.check(1, "Obs結構: 包含 price_seq 和 state_vector", 
                   isinstance(obs, dict) and 'price_seq' in obs and 'state_vector' in obs)
        
        self.check(2, "Price Seq 形狀: (window_size, 6)", 
                   obs['price_seq'].shape == (50, 6), 
                   actual_val=obs['price_seq'].shape)
        
        self.check(3, "State Vector 維度: (20,)", 
                   sv.shape == (20,), 
                   actual_val=sv.shape)
        
        self.check(4, "初始倉位狀態: Flat (One-Hot [0,0,1])", 
                   np.array_equal(sv[IDX_SIDE_START:IDX_SIDE_START+3], [0, 0, 1]), 
                   actual_val=sv[IDX_SIDE_START:IDX_SIDE_START+3])

        # --- Case 5-9: 執行買入 (Long) ---
        # Action: 0.5 (50% Long)
        # Step
        obs, _, _, _, info = self.env.step(np.array([0.5]))
        sv = obs['state_vector']
        
        self.check(5, "開多後狀態: Long (One-Hot [1,0,0])", 
                   sv[IDX_SIDE_START] == 1.0, 
                   actual_val=sv[IDX_SIDE_START:IDX_SIDE_START+3])
        
        self.check(6, "倉位大小: Pos Size Norm > 0", 
                   sv[IDX_SIZE] > 0, 
                   actual_val=sv[IDX_SIZE])
        
        # Trade Count Index is 10
        self.check(7, "交易次數 (Recent Trade Count) > 0", 
                   sv[IDX_TRADE_CNT] > 0, 
                   actual_val=sv[IDX_TRADE_CNT])
        
        # Cost Index is 11
        self.check(8, "成本乖離 (Est Cost): 剛買入應接近 0", 
                   abs(sv[IDX_COST]) < 0.01, 
                   actual_val=sv[IDX_COST])

        # --- Case 9-12: 價格上漲 (Profit) ---
        # 讓環境多跑幾步
        for _ in range(5):
            obs, _, _, _, info = self.env.step(np.array([0.5])) # 保持持倉
        
        sv = obs['state_vector']
        current_price = self.env.df.iloc[self.env.current_step]['close']
        entry_price = self.env.executor.position.entry_price
        
        # 確保價格真的漲了
        self.check(9, "環境確認: 價格上漲中", 
                   current_price > entry_price, 
                   actual_val=f"Curr: {current_price:.2f} > Entry: {entry_price:.2f}")
        
        self.check(10, "未實現損益 (UPNL Ratio): 應為正值", 
                   sv[IDX_UPNL] > 0, 
                   actual_val=sv[IDX_UPNL])
        
        self.check(11, "淨值比例 (Equity Ratio): 應 > 1.0", 
                   sv[IDX_EQ_RATIO] > 1.0, 
                   actual_val=sv[IDX_EQ_RATIO])

        self.check(12, "成本乖離: 價格高於成本 (Log > 0)", 
                   sv[IDX_COST] > 0, 
                   actual_val=sv[IDX_COST])

        # --- Case 13-16: 平倉與反手 (Short) ---
        # Action: -0.5 (反手做空 50%)
        obs, _, _, _, info = self.env.step(np.array([-0.5]))
        sv = obs['state_vector']
        
        self.check(13, "反手做空: Short (One-Hot [0,1,0])", 
                   sv[IDX_SIDE_START+1] == 1.0, 
                   actual_val=sv[IDX_SIDE_START:IDX_SIDE_START+3])
        
        # 現在還在 Sine Wave 上漲段，做空應該會虧損
        for _ in range(5):
            obs, _, _, _, info = self.env.step(np.array([-0.5]))
            
        sv = obs['state_vector']
        
        self.check(14, "做空逆勢: UPNL Ratio 應為負值", 
                   sv[IDX_UPNL] < 0, 
                   actual_val=sv[IDX_UPNL])
        
        # Drawdown Check
        self.check(15, "回撤 (Drawdown): 應 > 0", 
                   sv[IDX_DD] > 0, 
                   actual_val=sv[IDX_DD])

        self.check(16, "Max Equity Ratio: 應保持在歷史高點 (>1.0)", 
                   sv[IDX_MAX_EQ] > 1.0, 
                   actual_val=sv[IDX_MAX_EQ])

        # --- Case 17-20: 特殊狀態與 Price Seq ---
        
        # 17. Time of Day (Index 12)
        prev_tod = sv[IDX_TOD]
        self.env.step(np.array([-0.5]))
        obs, _, _, _, _ = self.env.step(np.array([-0.5]))
        sv = obs['state_vector']
        curr_tod = sv[IDX_TOD]
        self.check(17, "時間特徵 (Time of Day): 隨 Step 變化", 
                   curr_tod != prev_tod, 
                   actual_val=f"{prev_tod} -> {curr_tod}")
        
        # 18. Steps Since Last Trade (Index 9)
        # Close position first
        obs, _, _, _, _ = self.env.step(np.array([0.0])) 
        
        # Wait one step
        obs, _, _, _, _ = self.env.step(np.array([0.0])) 
        
        sv_1 = obs['state_vector']
        steps_1 = sv_1[IDX_STEPS]
        
        # Wait another step
        obs, _, _, _, _ = self.env.step(np.array([0.0])) 
        sv_2 = obs['state_vector']
        steps_2 = sv_2[IDX_STEPS]
        
        # 更新 sv 變量供後續 Case 使用
        sv = sv_2

        self.check(18, "Steps Since Last Trade: 隨時間增加", 
                   steps_2 > steps_1, 
                   actual_val=f"{steps_2} > {steps_1}")

        # 19. Price Seq Normalization
        ps = obs['price_seq']
        last_feat = ps[-1]
        self.check(19, "Price Seq: Ret (Feature 0) 範圍合理 (-5 ~ 5)", 
                   -5.0 <= last_feat[0] <= 5.0, 
                   actual_val=last_feat[0])
        
        self.check(20, "Price Seq: Vol Regime (Feature 4) 在 [0,1]", 
                   0.0 <= last_feat[4] <= 1.0, 
                   actual_val=last_feat[4])
        
        # 21. Day of Week (Index 13-19)
        dow_slice = sv[IDX_DOW_START:IDX_DOW_START+7]
        self.check(21, "Day of Week: One-Hot Encoding 正確 (Sum=1)", 
                   abs(np.sum(dow_slice) - 1.0) < 1e-5, 
                   actual_val=dow_slice)
        
        # 22. Margin Ratio Safety (Index 8)
        self.check(22, "空手時 Margin Ratio 應為 0", 
                   sv[IDX_MARGIN] == 0.0, 
                   actual_val=sv[IDX_MARGIN])

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestObsLogic)
    unittest.TextTestRunner(verbosity=0).run(suite)
