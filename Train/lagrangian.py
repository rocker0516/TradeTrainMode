from __future__ import annotations

import multiprocessing
import numpy as np
import gymnasium as gym
import torch
from collections import Counter, deque, defaultdict
from typing import Any, Dict, List, Tuple, Deque, Optional

from stable_baselines3.common.callbacks import BaseCallback


class SharedLagrangianController:
    """
    跨進程共享的 Lagrangian Multiplier (λ) 控制器。
    
    功能：
    - 維護全域 λ 值 (shared_lambda)。
    - 提供 update() 方法，根據 cost_violation (cost - limit) 調整 λ。
    - 實作 P-Control 與 Clamping 機制。
    """
    def __init__(
        self,
        cost_limit: float,
        kp: float = 0.1,  # P-gain
        lambda_init: float = 0.0,
        lambda_min: float = 0.0,
        lambda_max: float = 5.0,  # Clamp 上限
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
    並負責統計「原始主線獎勵」與「累積成本」供 Callback 顯示。
    
    R' = (Reward * scale) - (λ * Cost)
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
        
        # 累計當前回合數據 (for statistics)
        self.ep_ret_orig = 0.0
        self.ep_cost = 0.0
        self.ep_cost_breakdown = defaultdict(float)
        
    def reset(self, **kwargs):
        self.ep_ret_orig = 0.0
        self.ep_cost = 0.0
        self.ep_cost_breakdown.clear()
        return self.env.reset(**kwargs)
        
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. 取得原始數據
        # Env 回傳的 reward 即為「原始主線獎勵 (Log Return)」
        raw_reward = float(reward)
        cost = float(info.get("cost", 0.0))
        breakdown = info.get("cost_breakdown", {})
        
        # 2. 累計回合統計
        self.ep_ret_orig += raw_reward
        self.ep_cost += cost
        for k, v in breakdown.items():
            self.ep_cost_breakdown[k] += float(v)
            
        # 3. 計算 Lagrangian Reward (給 Agent 訓練用)
        lam = self.controller.current_lambda
        modified_reward = (raw_reward * self.reward_scale) - (lam * cost)
        
        # 4. 若回合結束，將累計統計注入 info 供 Callback 讀取
        if terminated or truncated:
            info["episode_metrics"] = {
                "return_orig": self.ep_ret_orig,
                "return_cost": self.ep_cost,
                "cost_breakdown": dict(self.ep_cost_breakdown)
            }
            # Reset 在 reset() 做，這裡不急著清空，避免 info 引用錯誤
        
        # Debug info
        info["lag_lambda"] = lam
        info["original_reward"] = raw_reward
        info["modified_reward"] = modified_reward
        
        return obs, modified_reward, terminated, truncated, info


class LagrangianCallback(BaseCallback):
    """
    訓練回調：
    1. 更新 λ (Update Lambda)
    2. 顯示詳細交易與訓練統計 (Display Stats)
       - 包含 Q-Value, 主線獎勵, Cost 細節
    """
    def __init__(
        self,
        controller: SharedLagrangianController,
        update_freq: int = 1000,   # 多少 global steps 更新一次 λ
        log_freq: int = 20,        # 多少 episodes 顯示一次統計 (Request: 20)
        window_size: int = 100,    # 統計視窗大小 (Request: 100)
        verbose: int = 1
    ) -> None:
        super().__init__(verbose)
        self.controller = controller
        self.update_freq = update_freq
        self.log_freq = log_freq
        self.window_size = window_size
        
        # Lambda Update Buffer
        self.cost_buffer: Deque[float] = deque(maxlen=int(update_freq))
        
        # Stats Buffer (存最近 N 回合的 info)
        self.ep_infos: Deque[Dict[str, Any]] = deque(maxlen=window_size)

        # Global counters
        self.total_episodes: int = 0
        self.last_log_episode: int = 0
        
    def _on_step(self) -> bool:
        # SB3 的 locals['infos'] 包含所有並行環境的 info
        infos = self.locals.get("infos", [])
        
        for info in infos:
            # 1. 收集 Cost (Step level, for Lambda update)
            if "cost" in info:
                self.cost_buffer.append(float(info["cost"]))
            
            # 2. 收集 Episode 結束時的統計
            # VecMonitor 在回合結束時會加入 "episode" key
            if "episode" in info:
                self.total_episodes += 1
                self.ep_infos.append(info)

        # 3. 定期更新 λ
        if self.n_calls % self.update_freq == 0:
            if len(self.cost_buffer) > 0:
                avg_cost = np.mean(self.cost_buffer)
                new_lambda = self.controller.update(avg_cost)
                
                # 寫入 TensorBoard
                self.logger.record("lagrangian/lambda", new_lambda)
                self.logger.record("lagrangian/avg_cost_step", avg_cost)
                self.logger.record("lagrangian/cost_violation", avg_cost - self.controller.cost_limit)

        # 4. 定期顯示統計 (每 log_freq 回合)
        # 檢查是否累積了足夠的新回合
        if (self.total_episodes - self.last_log_episode) >= self.log_freq:
            self._dump_stats()
            self.last_log_episode = self.total_episodes
            
        return True

    def _get_q_values_stats(self) -> Tuple[float, float, float]:
        """從 Replay Buffer 採樣計算 Q 值統計 (Mean, Min, Max)。"""
        if not hasattr(self.model, "replay_buffer") or self.model.replay_buffer is None:
            return 0.0, 0.0, 0.0
        
        if self.model.replay_buffer.size() < 1000:
            return 0.0, 0.0, 0.0

        # 採樣一個 batch
        batch = self.model.replay_buffer.sample(batch_size=256)
        
        # 使用 Critic 評估 (需轉為 Tensor，SB3 sample 出來已是 Tensor)
        # Critic 回傳 tuple of Q-values (q1, q2, ...)
        with torch.no_grad():
            q_values = self.model.critic(batch.observations, batch.actions)
            # 將 q1, q2 合併取 mean/min
            # q_values is tuple of tensors [batch, 1]
            q_concat = torch.cat(q_values, dim=1) # [batch, n_critics]
            
            q_mean = float(q_concat.mean().item())
            q_min = float(q_concat.min().item())
            q_max = float(q_concat.max().item())
            
        return q_mean, q_min, q_max

    def _dump_stats(self) -> None:
        """顯示詳細統計資訊"""
        if len(self.ep_infos) == 0:
            return

        # --- 1. 提取數據 ---
        # 透過 Wrapper 注入的 "episode_metrics" 獲取精確的主線與成本統計
        ep_ret_origs = []
        ep_costs = []
        cost_breakdowns = defaultdict(list)
        
        for info in self.ep_infos:
            metrics = info.get("episode_metrics", {})
            if metrics:
                ep_ret_origs.append(metrics.get("return_orig", 0.0))
                ep_costs.append(metrics.get("return_cost", 0.0))
                for k, v in metrics.get("cost_breakdown", {}).items():
                    cost_breakdowns[k].append(v)
        
        # 環境原生統計 (TradingEnv)
        profits = [float(x.get("profit", 0.0)) for x in self.ep_infos]
        total_fees = [float(x.get("total_fees", 0.0)) for x in self.ep_infos]
        liq_counts = [int(x.get("episode_liq_count", 0)) for x in self.ep_infos]
        max_dds = [float(x.get("episode_max_dd", 0.0)) for x in self.ep_infos]
        
        # --- 2. 計算平均 ---
        avg_ret_orig = np.mean(ep_ret_origs) if ep_ret_origs else 0.0
        avg_cost = np.mean(ep_costs) if ep_costs else 0.0
        
        avg_breakdown = {}
        for k, v_list in cost_breakdowns.items():
            avg_breakdown[k] = np.mean(v_list) if v_list else 0.0
            
        avg_profit = np.mean(profits) if profits else 0.0
        avg_fee = np.mean(total_fees) if total_fees else 0.0
        avg_liq = np.mean(liq_counts) if liq_counts else 0.0
        avg_dd = np.mean(max_dds) if max_dds else 0.0
        
        # Win Rate
        wins = sum(1 for p in profits if p > 0)
        win_rate = (wins / len(profits)) * 100 if profits else 0.0
        
        # Q-Values
        q_mean, q_min, q_max = self._get_q_values_stats()
        
        # --- 3. 顯示排版 ---
        # 使用 print 直接輸出到 console，方便查看
        print("\n" + "="*60)
        print(f"  STATS (Last {len(self.ep_infos)} Episodes) @ Global Step {self.num_timesteps}")
        print("="*60)
        
        # Section 1: Main Reward (Training Objective)
        print(f"[{'MAIN REWARD':^20}]")
        print(f"  Avg Original Return (LogRet): {avg_ret_orig:8.4f}")
        print(f"  Avg Profit (USDT)           : {avg_profit:8.2f}")
        print(f"  Win Rate                    : {win_rate:8.1f} %")
        print("-" * 60)
        
        # Section 2: Cost Line (Constraints)
        print(f"[{'COST LINE':^20}] Lambda: {self.controller.current_lambda:.4f}")
        print(f"  Avg Episode Cost            : {avg_cost:8.4f}")
        if avg_breakdown:
            print("  --- Cost Breakdown ---")
            for k, v in avg_breakdown.items():
                if v > 1e-6: # 只顯示非零項
                    print(f"    - {k:<20}: {v:8.4f}")
        else:
            print("    (No costs triggered)")
        print("-" * 60)
        
        # Section 3: Training Health (Q-Values & Losses)
        print(f"[{'TRAINING HEALTH':^20}]")
        print(f"  Q-Value (Mean)              : {q_mean:8.4f}")
        print(f"  Q-Value (Min/Max)           : {q_min:8.4f} / {q_max:8.4f}")
        # Entropy/Loss 等可從 Tensorboard 查看，這裡顯示最關鍵的 Q 值即可
        print("-" * 60)
        
        # Section 4: Trade Execution Stats
        print(f"[{'TRADE STATS':^20}]")
        print(f"  Avg Max Drawdown            : {avg_dd*100:8.2f} %")
        print(f"  Avg Liq Count               : {avg_liq:8.4f}")
        print(f"  Avg Fees                    : {avg_fee:8.2f}")
        print("="*60 + "\n")

        # --- 4. 寫入 TensorBoard (Optional) ---
        self.logger.record("custom/avg_ret_orig", avg_ret_orig)
        self.logger.record("custom/avg_profit", avg_profit)
        self.logger.record("custom/q_mean", q_mean)
        self.logger.record("custom/ep_cost", avg_cost)
