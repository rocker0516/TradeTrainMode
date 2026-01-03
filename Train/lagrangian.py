from __future__ import annotations

import multiprocessing
import numpy as np
import gymnasium as gym
from collections import Counter, deque
from typing import Any, Dict, List, Tuple, Deque, Optional

from stable_baselines3.common.callbacks import BaseCallback


class SharedLagrangianController:
    """
    跨進程共享的 Lagrangian Multiplier (λ) 控制器。
    
    功能：
    - 維護全域 λ 值 (shared_lambda)。
    - 提供 update() 方法，根據 cost_violation (cost - limit) 調整 λ。
    - 實作 P-Control 與 Clamping 機制，防止 λ 暴衝導致模型放棄學習。
    """
    def __init__(
        self,
        cost_limit: float,
        kp: float = 0.1,  # P-gain
        lambda_init: float = 0.0,
        lambda_min: float = 0.0,
        lambda_max: float = 5.0,  # Clamp 上限，防躺平
    ) -> None:
        self.cost_limit = float(cost_limit)
        self.kp = float(kp)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        
        # 使用 multiprocessing.Value 讓並行環境能讀取同一份 λ
        self._lambda_val = multiprocessing.Value('d', float(lambda_init))
    
    @property
    def current_lambda(self) -> float:
        with self._lambda_val.get_lock():
            return self._lambda_val.value
            
    def update(self, avg_cost: float) -> float:
        """
        根據平均 Cost 更新 λ。
        公式：λ_new = clamp(λ_old + kp * (avg_cost - limit))
        注意：這裡使用簡單積分/累積概念的變體，或者是直接 P-control 調整。
        標準 Lagrangian 更新通常是梯度上升：λ = λ + lr * (J_C - Limit)。
        這裡 kp 扮演 lr 角色。
        """
        violation = avg_cost - self.cost_limit
        with self._lambda_val.get_lock():
            new_val = self._lambda_val.value + self.kp * violation
            new_val = max(self.lambda_min, min(self.lambda_max, new_val))
            self._lambda_val.value = new_val
            return new_val


class LagrangianRewardWrapper(gym.Wrapper):
    """
    環境包裝器：將原始 Reward 修正為 Lagrangian Reward。
    
    R' = (Reward * scale) - (λ * Cost)
    
    特點：
    - 引用 SharedLagrangianController，即時讀取最新 λ。
    - 支援 reward_scale，確保 Reward 與 λ*Cost 在可比較數量級。
    """
    def __init__(
        self, 
        env: gym.Env, 
        controller: SharedLagrangianController,
        reward_scale: float = 1.0
    ) -> None:
        super().__init__(env)
        self.controller = controller
        self.reward_scale = float(reward_scale)
        
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 讀取 Cost (由 trading_env.py 計算並放入 info)
        cost = float(info.get("cost", 0.0))
        
        # 讀取當前 λ
        lam = self.controller.current_lambda
        
        # 計算修正後獎勵
        # 這裡假設 reward 已經是 log-return 等相對數值。
        # 若 reward 極小 (e.g. 0.001)，建議 reward_scale 設大 (e.g. 10.0)。
        modified_reward = (reward * self.reward_scale) - (lam * cost)
        
        # 將 λ 與原始 reward 寫入 info 方便 debug
        info["lag_lambda"] = lam
        info["original_reward"] = reward
        info["modified_reward"] = modified_reward
        
        return obs, modified_reward, terminated, truncated, info


class LagrangianCallback(BaseCallback):
    """
    訓練回調：負責更新 λ 並記錄詳細交易統計。
    
    功能：
    1. 收集並行環境的 info。
    2. 每 N 步計算平均 Cost，呼叫 Controller 更新 λ。
    3. 統計交易狀態 (Win rate, Fees, Long/Short counts) 並寫入 Logger。
    """
    def __init__(
        self,
        controller: SharedLagrangianController,
        update_freq: int = 1000,  # 多少 global steps 更新一次 λ
        log_freq: int = 100,      # 多少 episodes 統計一次交易狀態
        verbose: int = 1
    ) -> None:
        super().__init__(verbose)
        self.controller = controller
        self.update_freq = update_freq
        self.log_freq = log_freq
        
        # 緩衝區：存 Cost 用於計算平均值
        self.cost_buffer: Deque[float] = deque(maxlen=int(update_freq))
        
        # 交易統計緩衝區 (Episode level)
        self.ep_infos: List[Dict[str, Any]] = []

        # 全域回合計數（跨 dump，不會清空）
        self.total_episodes: int = 0
        # 固定視窗：最近 100 回合的最大回撤（max drawdown）
        self.recent_episode_max_dd: Deque[float] = deque(maxlen=100)
        # 最近 100 回合：最大手續費/費用比等「極端值」追蹤
        self.recent_total_fees: Deque[float] = deque(maxlen=100)
        self.recent_fees_to_equity: Deque[float] = deque(maxlen=100)
        
    def _on_step(self) -> bool:
        # SB3 的 locals['infos'] 包含所有並行環境的 info
        infos = self.locals.get("infos", [])
        
        for info in infos:
            # 1. 收集 Cost (Step level)
            if "cost" in info:
                self.cost_buffer.append(float(info["cost"]))
            
            # 2. 收集 Episode 結束時的統計 (Episode level)
            # VecMonitor 會在 episode 結束時加入 info["episode"] = {"r": ep_return, "l": ep_len, ...}
            # 我們以此判斷「確實結束了一個回合」，並把同一筆 info 裡的回合摘要一起存起來。
            if "episode" in info:
                self.total_episodes += 1
                # Env 在 done 時提供 episode_max_dd；若舊模型/舊環境未提供，fallback 到 current_dd
                dd_val = float(info.get("episode_max_dd", info.get("current_dd", 0.0)))
                # 理論範圍 0~1（保守 clamp，避免偶發 NaN/Inf 影響統計）
                if not np.isfinite(dd_val):
                    dd_val = 0.0
                dd_val = float(np.clip(dd_val, 0.0, 1.0))
                self.recent_episode_max_dd.append(dd_val)

                fees_val = float(info.get("total_fees", 0.0))
                if not np.isfinite(fees_val) or fees_val < 0.0:
                    fees_val = 0.0
                self.recent_total_fees.append(float(fees_val))

                f2e = float(info.get("fees_to_equity_ratio", 0.0))
                if not np.isfinite(f2e) or f2e < 0.0:
                    f2e = 0.0
                self.recent_fees_to_equity.append(float(f2e))
                self.ep_infos.append(info)

        # 3. 定期更新 λ
        if self.n_calls % self.update_freq == 0:
            if len(self.cost_buffer) > 0:
                avg_cost = np.mean(self.cost_buffer)
                new_lambda = self.controller.update(avg_cost)
                
                # 寫入 TensorBoard
                self.logger.record("lagrangian/lambda", new_lambda)
                self.logger.record("lagrangian/avg_cost", avg_cost)
                self.logger.record("lagrangian/cost_violation", avg_cost - self.controller.cost_limit)

        # 4. 定期顯示交易統計 (Console + TB)
        if len(self.ep_infos) >= self.log_freq:
            self._dump_trade_stats()
            
        return True

    def _dump_trade_stats(self) -> None:
        """計算並顯示平均交易狀態"""
        n = len(self.ep_infos)
        if n == 0:
            return
            
        # ---- Episode return（用 VecMonitor 的 ep reward，這是訓練端真正看到的 reward）----
        ep_returns = [float(x.get("episode", {}).get("r", 0.0)) for x in self.ep_infos]
        ep_lengths = [int(x.get("episode", {}).get("l", 0)) for x in self.ep_infos]

        # ---- 交易統計（來自 TradingEnvironment done info）----
        total_fees = [float(x.get("total_fees", 0.0)) for x in self.ep_infos]
        total_fees_ratio = [float(x.get("total_fees_ratio", 0.0)) for x in self.ep_infos]
        long_entries = [int(x.get("long_entry_count", 0)) for x in self.ep_infos]
        short_entries = [int(x.get("short_entry_count", 0)) for x in self.ep_infos]
        final_pos = [float(x.get("final_position_size", 0.0)) for x in self.ep_infos]
        stop_counts = [int(x.get("episode_stop_loss_count", 0)) for x in self.ep_infos]
        liq_counts = [int(x.get("episode_liq_count", 0)) for x in self.ep_infos]
        ep_max_dds = [float(x.get("episode_max_dd", x.get("current_dd", 0.0))) for x in self.ep_infos]
        profits = [float(x.get("profit", 0.0)) for x in self.ep_infos]
        holding_steps = [int(x.get("episode_holding_steps", 0)) for x in self.ep_infos]
        turnover_notional = [float(x.get("episode_turnover_notional", 0.0)) for x in self.ep_infos]
        fees_to_equity = [float(x.get("fees_to_equity_ratio", 0.0)) for x in self.ep_infos]

        # 終止原因統計
        reasons = [str(x.get("termination_reason", "")) for x in self.ep_infos if "termination_reason" in x]
        reason_counter = Counter(reasons)
        
        # 計算統計
        avg_ep_return = float(np.mean(ep_returns)) if ep_returns else 0.0
        avg_ep_len = float(np.mean(ep_lengths)) if ep_lengths else 0.0
        avg_total_fees = float(np.mean(total_fees)) if total_fees else 0.0
        avg_total_fees_ratio = float(np.mean(total_fees_ratio)) if total_fees_ratio else 0.0
        avg_long_entries = float(np.mean(long_entries)) if long_entries else 0.0
        avg_short_entries = float(np.mean(short_entries)) if short_entries else 0.0
        avg_final_pos = float(np.mean(final_pos)) if final_pos else 0.0
        avg_stop_count = float(np.mean(stop_counts)) if stop_counts else 0.0
        avg_liq_count = float(np.mean(liq_counts)) if liq_counts else 0.0
        avg_ep_max_dd = float(np.mean(ep_max_dds)) if ep_max_dds else 0.0
        max_dd_last_100 = float(max(self.recent_episode_max_dd)) if len(self.recent_episode_max_dd) > 0 else 0.0

        # 最近 N 回合：Win Rate / Profit Factor（用 env profit；與訓練 reward 不同）
        win_rate = float(np.mean([1.0 if p > 0.0 else 0.0 for p in profits])) if profits else 0.0
        gross_profit = float(sum(p for p in profits if p > 0.0))
        gross_loss_abs = float(abs(sum(p for p in profits if p < 0.0)))
        if gross_loss_abs > 0.0:
            profit_factor = float(gross_profit / gross_loss_abs)
        else:
            profit_factor = float("inf") if gross_profit > 0.0 else 0.0

        # Overtrading hints: holding / turnover
        avg_holding_steps = float(np.mean(holding_steps)) if holding_steps else 0.0
        avg_turnover_notional = float(np.mean(turnover_notional)) if turnover_notional else 0.0

        # Fee pain: max fees last100 + fees-to-equity
        max_fees_last_100 = float(max(self.recent_total_fees)) if len(self.recent_total_fees) > 0 else 0.0
        avg_fees_to_equity = float(np.mean(fees_to_equity)) if fees_to_equity else 0.0
        max_fees_to_equity_last_100 = float(max(self.recent_fees_to_equity)) if len(self.recent_fees_to_equity) > 0 else 0.0
        
        # 寫入 Logger
        self.logger.record("trade/avg_episode_return", avg_ep_return)
        self.logger.record("trade/avg_episode_len", avg_ep_len)
        self.logger.record("trade/avg_total_fees", avg_total_fees)
        self.logger.record("trade/avg_total_fees_ratio", avg_total_fees_ratio)
        self.logger.record("trade/avg_long_entries", avg_long_entries)
        self.logger.record("trade/avg_short_entries", avg_short_entries)
        self.logger.record("trade/avg_final_position_size", avg_final_pos)
        self.logger.record("trade/avg_stop_loss_count", avg_stop_count)
        self.logger.record("trade/avg_liq_count", avg_liq_count)
        self.logger.record("trade/avg_episode_max_dd", avg_ep_max_dd)
        self.logger.record("trade/max_dd_last_100", max_dd_last_100)
        self.logger.record("trade/total_episodes", float(self.total_episodes))
        self.logger.record("trade/win_rate_last_n", win_rate)
        self.logger.record("trade/profit_factor_last_n", profit_factor)
        self.logger.record("trade/avg_holding_steps_last_n", avg_holding_steps)
        self.logger.record("trade/avg_turnover_notional_last_n", avg_turnover_notional)
        self.logger.record("trade/max_fees_last_100", max_fees_last_100)
        self.logger.record("trade/avg_fees_to_equity_last_n", avg_fees_to_equity)
        self.logger.record("trade/max_fees_to_equity_last_100", max_fees_to_equity_last_100)
        self.logger.record("trade/lambda", float(self.controller.current_lambda))
        
        # Console 輸出
        if self.verbose > 0:
            print(f"\n[Trade Stats @ {self.num_timesteps} steps] (last {n} episodes)")
            print(f"  Total Episodes: {self.total_episodes}")
            print(f"  Avg Episode Return: {avg_ep_return:.4f}")
            print(f"  Avg Episode Len: {avg_ep_len:.2f}")
            print(f"  Avg Fees: {avg_total_fees:.4f}  (ratio={avg_total_fees_ratio:.6f})")
            print(f"  Max Fees (last 100): {max_fees_last_100:.4f}")
            print(f"  Fees-to-Equity: avg(last N)={avg_fees_to_equity:.6f} | max(last 100)={max_fees_to_equity_last_100:.6f}")
            print(f"  Avg Long Entries: {avg_long_entries:.2f} | Avg Short Entries: {avg_short_entries:.2f}")
            print(f"  Avg Final Position Size: {avg_final_pos:.6f}")
            print(f"  Avg Stop Loss Count: {avg_stop_count:.2f} | Avg Liq Count: {avg_liq_count:.2f}")
            print(f"  Max Drawdown (last 100): {max_dd_last_100:.4f}")
            print(f"  Win Rate (last N): {win_rate:.2%} | Profit Factor (last N): {profit_factor:.4f}")
            print(f"  Avg Holding Steps (last N): {avg_holding_steps:.2f} | Avg Turnover Notional (last N): {avg_turnover_notional:.4f}")
            if reason_counter:
                print(f"  Termination Reasons: {dict(reason_counter)}")
            print(f"  Current Lambda: {self.controller.current_lambda:.4f}")
            print("-" * 30)
            
        # 清空 buffer
        self.ep_infos = []

