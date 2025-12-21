
import pandas as pd
import numpy as np
import gymnasium as gym
from Env.trading_env import TradingEnvironment
from Env.wrappers import ActionRepeatWrapper, ActionSmoothClipWrapper
from Train.config import Config

def run_baseline(env_name, policy_func):
    """在當前成本結構下，對單一策略跑一個 episode，回傳績效指標。"""
    # 載入完整資料，再只取「一年的長度」來做 baseline
    df = pd.read_csv(Config.DATA_PATH)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        start_ts = df['timestamp'].iloc[0]
        end_ts = start_ts + pd.Timedelta(days=365)
        df = df[df['timestamp'] < end_ts].reset_index(drop=True)
    else:
        # 若沒有時間欄，依步數切一年：約 105120 步（Config.MAX_EPISODE_STEPS）
        max_len = getattr(Config, "MAX_EPISODE_STEPS", 105_120)
        df = df.iloc[:max_len + Config.WINDOW_SIZE + 1].reset_index(drop=True)
    
    # 不再修改 Config.MAX_EPISODE_STEPS，保持環境設定為「最多一年」
    # 初始化環境（不隨機起點），並套用與訓練相同的 Wrapper
    base_env = TradingEnvironment(df, random_start=False)

    # Action Repeat
    env: gym.Env = base_env
    if getattr(Config, "ACTION_REPEAT", 1) > 1:
        env = ActionRepeatWrapper(env, repeat=Config.ACTION_REPEAT)

    # Clip + Smooth（確保 baseline 與 agent 有相同的動作限制與摩擦）
    env = ActionSmoothClipWrapper(
        env,
        max_position_pct=getattr(Config, "MAX_POSITION_PCT", 0.5),
        smoothing_alpha=getattr(Config, "ACTION_SMOOTHING_ALPHA", 0.3),
    )
    
    obs, info = env.reset()
    done = False
    truncated = False
    
    total_reward = 0.0
    max_drawdown = 0.0
    
    equity_curve = []
    
    step_count = 0
    
    print(f"--- Running {env_name} ---")
    
    while not (done or truncated):
        # Get action from policy
        action = policy_func(env, obs)
        
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        
        # Track MDD
        # info['current_dd'] comes from env
        current_dd = info.get('current_dd', 0.0)
        max_drawdown = max(max_drawdown, current_dd)
        
        equity_curve.append(info.get('equity', 0.0))
        
        step_count += 1
        if step_count % 10000 == 0:
            print(f"Step {step_count}: Equity={info.get('equity', 0.0):.2f}, DD={current_dd:.4f}")

    final_equity = info.get('final_balance', equity_curve[-1] if equity_curve else Config.INITIAL_BALANCE)
    profit_pct = ((final_equity - Config.INITIAL_BALANCE) / Config.INITIAL_BALANCE) * 100
    
    print(f"Result for {env_name}:")
    print(f"  Final Equity: {final_equity:.2f}")
    print(f"  Profit: {profit_pct:.2f}%")
    print(f"  Max Drawdown: {max_drawdown*100:.2f}%")
    print(f"  Total Steps: {step_count}")
    print(f"  Termination: {info.get('termination_reason', 'unknown')}")
    print("-" * 30)
    
    return {
        'name': env_name,
        'profit_pct': profit_pct,
        'mdd_pct': max_drawdown * 100,
        'final_equity': final_equity
    }

# === 基本策略 ===
def policy_no_trade(env, obs):
    return np.array([0.0], dtype=np.float32)

def policy_always_long(env, obs):
    return np.array([1.0], dtype=np.float32)

def policy_always_short(env, obs):
    return np.array([-1.0], dtype=np.float32)

def policy_sma_200(env, obs):
    # Calculate SMA on the fly or pre-calc?
    # Access env.df directly is cheating but this is baseline.
    # We need to look at current_step.
    # Note: env.current_step is the NEXT step index (after step increment).
    # But inside step(), it uses current_step.
    # When we are outside step(), env.current_step is the index of the observation we just got.
    # Wait, in reset(), current_step is set.
    # In step(), it uses current_step then increments it.
    # So if we are at step T, env.current_step is T.
    
    base_env: TradingEnvironment = env.unwrapped  # type: ignore[attr-defined]
    idx = base_env.current_step
    
    # We need history for SMA
    # Use env.df['close']
    # SMA 200
    if idx < 200:
        return np.array([0.0], dtype=np.float32)
        
    # Get closing prices
    # We can access env._close_arr
    # We need the previous 200 closes ending at idx-1? 
    # Or ending at idx? At step T, we make decision based on info up to T (inclusive or exclusive?)
    # Usually we see Open/High/Low/Close of T-1? Or T is the current candle forming?
    # In backtesting, usually we have data up to T.
    # Environment gives us observation at T.
    
    # Let's assume we can use data up to `idx`.
    # SMA calc:
    closes = base_env._close_arr[idx-200:idx]  # type: ignore[attr-defined]
    if len(closes) < 200:
        return np.array([0.0], dtype=np.float32)
        
    sma = np.mean(closes)
    current_price = base_env._close_arr[idx-1]  # type: ignore[attr-defined]
    # Or current_price at idx?
    # The environment executes at `current_price` (line 433 of trading_env.py: `current_price = self._close_arr[self.current_step]`)
    # So we are deciding for the current step.
    # BUT, technically we shouldn't know the CLOSE of the current step when making a decision if it's the candle we are trading IN.
    # However, this env seems to simulate "Next Open" or "Close to Close" trading?
    # It uses `current_price` (close of current step) for execution.
    # This implies we are trading AT the close.
    # So we can compare current_price (or previous close) to SMA.
    
    # To be "causal", if we execute at Close T, we know Close T?
    # Usually in backtesting with OHLC, if we trade at Close, we know Close.
    # If we trade at Next Open, we know Close T-1.
    
    # Let's assume we compare `current_price` (which is `_close_arr[idx]`) against SMA of previous data?
    # Or SMA including current price?
    # Standard: Price > SMA.
    # Let's use `_close_arr[idx-1]` for price and SMA up to `idx-1` to be safe/conservative (lagged).
    # Or use `_close_arr[idx]` if we assume we decide right at the close.
    
    # Let's use the price that execution uses: `env._close_arr[idx]`.
    # And SMA of the last 200 points ending at idx.
    
    closes = base_env._close_arr[idx-199 : idx+1]  # type: ignore[attr-defined]
    sma = np.mean(closes)
    price = closes[-1]
    
    if price > sma:
        return np.array([1.0], dtype=np.float32)
    else:
        return np.array([0.0], dtype=np.float32) # Flat if not bullish


# === 進階：尋找「簡單但不輸 No-Trade」的 baseline ===
def make_sma_atr_policy(position_pct: float, atr_threshold: float, sma_window: int):
    """
    建立一個簡單規則策略：
    - ATR 比例低於 atr_threshold（低波動）
    - 收盤價 > SMA(sma_window)
    則持有固定多頭倉位 position_pct，否則空倉。
    """
    position_pct = float(position_pct)
    atr_threshold = float(atr_threshold)
    sma_window = int(sma_window)

    def policy(env: gym.Env, obs):
        base_env: TradingEnvironment = env.unwrapped  # type: ignore[attr-defined]
        idx = base_env.current_step
        # 確保有足夠歷史資料
        if idx <= sma_window:
            return np.array([0.0], dtype=np.float32)

        # 使用環境內部的 close 與 ATR Ratio（與 reward/風控一致）
        closes = base_env._close_arr  # type: ignore[attr-defined]
        atr_ratio_arr = base_env._atr_ratio_arr  # type: ignore[attr-defined]

        atr_val = float(atr_ratio_arr[idx - 1])
        if atr_val > atr_threshold:
            return np.array([0.0], dtype=np.float32)

        window_closes = closes[idx - sma_window:idx]
        sma = float(np.mean(window_closes))
        price = float(closes[idx - 1])

        if price > sma:
            return np.array([position_pct], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)

    return policy


def search_simple_baselines():
    """
    嘗試多組簡單 SMA + ATR 規則，找出：
    - 報酬率 >= No-Trade（0%）
    - 且儘量避免過度複雜（固定倉位、多空方向單一）。
    """
    candidates = []

    # 參數網格：倉位大小、ATR 閾值、SMA 期間
    position_pcts = [0.1, 0.2, 0.3]           # 對應 1x / 2x / 3x 名目倍數
    atr_thresholds = [0.005, 0.01, 0.015]     # ATR 比例門檻（越小越少交易）
    sma_windows = [50, 100, 200]

    for p in position_pcts:
        for atr_th in atr_thresholds:
            for w in sma_windows:
                name = f"SMA{w}_ATR{atr_th}_P{p}"
                policy = make_sma_atr_policy(p, atr_th, w)
                result = run_baseline(name, policy)
                result["params"] = {"position_pct": p, "atr_threshold": atr_th, "sma_window": w}
                candidates.append(result)

    # 找出報酬率最高的一組
    best = max(candidates, key=lambda x: x["profit_pct"])
    print("\n=== BEST SIMPLE RULE (一年的資料) ===")
    print(f"Name: {best['name']}")
    print(f"Params: {best['params']}")
    print(f"Profit: {best['profit_pct']:.2f}%")
    print(f"MDD: {best['mdd_pct']:.2f}%")
    print(f"Final Equity: {best['final_equity']:.2f}")

    return best, candidates

if __name__ == "__main__":
    results = []
    
    # No Trade
    results.append(run_baseline("No-Trade", policy_no_trade))
    
    # Always Long
    results.append(run_baseline("Always-Long", policy_always_long))
    
    # Always Short
    results.append(run_baseline("Always-Short", policy_always_short))
    
    # SMA Rule
    results.append(run_baseline("SMA-200-LongOnly", policy_sma_200))
    
    print("\n=== SUMMARY ===")
    print(f"{'Strategy':<20} | {'Profit %':<10} | {'MDD %':<10} | {'Final Equity':<15}")
    print("-" * 65)
    for res in results:
        print(f"{res['name']:<20} | {res['profit_pct']:>9.2f}% | {res['mdd_pct']:>9.2f}% | {res['final_equity']:>14.2f}")

    # 搜尋簡單規則 baseline
    best_rule, _ = search_simple_baselines()
    print("\n=== BEST SIMPLE RULE VS NO-TRADE ===")
    print(f"No-Trade Profit: {results[0]['profit_pct']:.2f}% , MDD: {results[0]['mdd_pct']:.2f}%")
    print(f"Best Rule Profit: {best_rule['profit_pct']:.2f}% , MDD: {best_rule['mdd_pct']:.2f}%")

