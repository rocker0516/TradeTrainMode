
import numpy as np
import pandas as pd
import unittest
from Env.trading_env import TradingEnvironment

class TestNaNLeak(unittest.TestCase):
    def setUp(self):
        # 1. 創建極端數據 (Zero, Negative, NaN, Inf)
        length = 500
        dates = pd.date_range(start='2024-01-01', periods=length, freq='5min')
        
        # Case 1: Normal but static
        self.df_zeros = pd.DataFrame({
            'open': np.zeros(length),
            'high': np.zeros(length),
            'low': np.zeros(length),
            'close': np.zeros(length),
            'volume': np.zeros(length),
            'buy_volume': np.zeros(length),
            'sell_volume': np.zeros(length),
            'trades': np.zeros(length),
            'quote_volume': np.zeros(length),
            'volume_ratio': np.zeros(length),
            'long_short_ratio': np.zeros(length)
        }, index=dates)
        
        # Case 2: Infs and NaNs in Input
        df_nan = self.df_zeros.copy()
        df_nan['close'] = np.nan
        df_nan['volume'] = np.inf
        self.df_nan = df_nan
        
    def check_obs_clean(self, obs, context=""):
        """檢查 obs 是否乾淨 (無 NaN, 無 Inf)"""
        for key, arr in obs.items():
            has_nan = np.isnan(arr).any()
            has_inf = np.isinf(arr).any()
            if has_nan or has_inf:
                print(f"FAIL [{context}] Key: {key} | NaN: {has_nan} | Inf: {has_inf}")
                print(f"Data: {arr}")
            self.assertFalse(has_nan, f"Obs[{key}] contains NaN in {context}")
            self.assertFalse(has_inf, f"Obs[{key}] contains Inf in {context}")

    def test_zeros_input(self):
        """測試全 0 輸入是否導致除以零產生的 NaN"""
        print("\nRunning test_zeros_input...")
        # 注意: close=0 可能導致某些計算 (如 log(close)) 出錯，看防護機制
        # Env init 可能會報錯如果 features 產生了問題，或者 features 處理了
        try:
            env = TradingEnvironment(self.df_zeros, window_size=50)
            obs, _ = env.reset()
            self.check_obs_clean(obs, "Zeros Init")
            
            for i in range(10):
                obs, _, done, _, _ = env.step(np.array([1.0])) # Try to trade
                self.check_obs_clean(obs, f"Zeros Step {i}")
                if done: break
        except Exception as e:
            print(f"Caught expected exception or error in Zeros test: {e}")
            # 有些 log 運算如果沒有 clip 1e-12 可能報錯，但我們想測的是如果沒報錯，出來的是否是 NaN

    def test_nan_inf_input(self):
        """測試輸入本身包含 NaN/Inf"""
        print("\nRunning test_nan_inf_input...")
        # features.py 應該處理這些輸入
        try:
            env = TradingEnvironment(self.df_nan, window_size=50)
            obs, _ = env.reset()
            self.check_obs_clean(obs, "NaN/Inf Init")
            
            for i in range(10):
                obs, _, done, _, _ = env.step(np.array([-1.0]))
                self.check_obs_clean(obs, f"NaN/Inf Step {i}")
                if done: break
        except ValueError as e:
            print(f"Caught expected validation error: {e}")
        except Exception as e:
            print(f"Caught unexpected error: {e}")

    def test_extreme_values(self):
        """測試極端大數值"""
        print("\nRunning test_extreme_values...")
        length = 500
        df_huge = self.df_zeros.copy()
        df_huge['close'] = 1e30
        df_huge['high'] = 1e30
        df_huge['low'] = 1e30
        df_huge['open'] = 1e30
        df_huge['volume'] = 1e30
        
        env = TradingEnvironment(df_huge, window_size=50)
        obs, _ = env.reset()
        self.check_obs_clean(obs, "Huge Values Init")
        
        obs, _, _, _, _ = env.step(np.array([1.0]))
        self.check_obs_clean(obs, "Huge Values Step")

if __name__ == '__main__':
    unittest.main()

