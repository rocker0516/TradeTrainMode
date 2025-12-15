
import numpy as np
import pandas as pd
import unittest
from Env.trading_env import TradingEnvironment

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
        
        # Create timestamps (use 1h so time features change quickly in tests)
        dates = pd.date_range(start='2024-01-01', periods=length, freq='1h')
        
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

        # New env uses 'timestamp' column for time features (hour sin/cos)
        df["timestamp"] = dates
        
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

        # --- Case 1-3: 結構與形狀 ---
        self.check(
            1,
            "Obs 結構: 包含 price_seq/account_state/time_state/rhythm_state/cost_state/market_state",
            isinstance(obs, dict)
            and "price_seq" in obs
            and "account_state" in obs
            and "time_state" in obs
            and "rhythm_state" in obs
            and "cost_state" in obs
            and "market_state" in obs,
        )

        self.check(2, "Price Seq 第 0 維為 window_size", obs["price_seq"].shape[0] == 50, actual_val=obs["price_seq"].shape)
        self.check(3, "Account/Time/Rhythm/Cost/Market 維度正確",
                   obs["account_state"].shape == (13,)
                   and obs["time_state"].shape == (2,)
                   and obs["rhythm_state"].shape == (2,)
                   and obs["cost_state"].shape == (14,)
                   and obs["market_state"].shape == (6,),
                   actual_val=(obs["account_state"].shape, obs["time_state"].shape, obs["rhythm_state"].shape, obs["cost_state"].shape, obs["market_state"].shape))

        # --- Case 4: 初始狀態（空倉） ---
        acc0 = obs["account_state"]
        self.check(4, "初始空倉：pos_size_norm 應為 0", float(acc0[0]) == 0.0, actual_val=float(acc0[0]))

        # --- Case 5-7: 開多後，account_state 與 executor 的計算一致 ---
        obs, _, _, _, _ = self.env.step(np.array([0.5], dtype=np.float32))
        acc = obs["account_state"]

        self.check(5, "開多後：pos_size_norm > 0", float(acc[0]) > 0.0, actual_val=float(acc[0]))

        # obs 是 step 後的下一個狀態；用 env.current_step 對應的 close 去計算一致性
        current_price = float(self.env._close_arr[self.env.current_step])
        expected_upnl_ratio = float(self.env.executor.unrealized_pnl(current_price) / self.env.initial_balance)
        expected_equity_ratio = float(self.env.executor.equity(current_price) / self.env.initial_balance)

        self.check(6, "UPNL Ratio 與 executor 一致", np.isclose(float(acc[1]), expected_upnl_ratio, atol=1e-6),
                   actual_val=f"obs={float(acc[1]):.6f}, exp={expected_upnl_ratio:.6f}")
        self.check(7, "Equity Ratio 與 executor 一致", np.isclose(float(acc[2]), expected_equity_ratio, atol=1e-6),
                   actual_val=f"obs={float(acc[2]):.6f}, exp={expected_equity_ratio:.6f}")

        # --- Case 8: time_state 合法且會變化（timestamps 每步 +1h） ---
        t0 = obs["time_state"].copy()
        obs2, _, _, _, _ = self.env.step(np.array([0.5], dtype=np.float32))
        t1 = obs2["time_state"]
        self.check(8, "time_state 值域合理", np.all(np.abs(t1) <= 1.0 + 1e-6), actual_val=t1)
        self.check(9, "time_state 隨 step 變化", not np.allclose(t0, t1), actual_val=f"{t0} -> {t1}")

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestObsLogic)
    unittest.TextTestRunner(verbosity=0).run(suite)
