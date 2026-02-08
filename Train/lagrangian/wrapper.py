from __future__ import annotations

import gymnasium as gym
from collections import defaultdict
from typing import Any, Dict, Tuple

from Train.lagrangian.controllers import MultiSharedLagrangianController


class LagrangianRewardWrapper(gym.Wrapper):
    """
    環境包裝器：將原始 Reward 修正為 Lagrangian Reward。
    並負責統計「原始主線獎勵」與「累積成本」供 Callback 顯示。
    
    R' = (Reward * scale) - (λ * Cost)
    """
    def __init__(
        self, 
        env: gym.Env, 
        controller: Any,
        reward_scale: float = 1.0,
        cost_penalty_normalize: float = 2000.0,
    ) -> None:
        super().__init__(env)
        self.controller = controller
        self.reward_scale = float(reward_scale)
        self.cost_penalty_normalize = max(1.0, float(cost_penalty_normalize))

        # --- Cross-process lambda sync (for SubprocVecEnv / Windows spawn) ---
        # 在 SubprocVecEnv（spawn）情境下，controller 的共享值不一定會在各子進程保持同步。
        # 因此允許 Callback 透過 VecEnv.env_method() 將最新 λ 明確同步到每個子進程的 wrapper。
        self._synced_lambdas: Dict[str, float] | None = None
        self._synced_lambda: float | None = None
        
        # 累計當前回合數據 (for statistics)
        self.ep_ret_orig = 0.0
        self.ep_ret_orig_scaled = 0.0
        self.ep_ret_total = 0.0  # 實際給 agent 訓練的 reward（modified reward）累計
        self.ep_cost = 0.0
        self.ep_cost_breakdown = defaultdict(float)
        # cost 對 reward 的「懲罰貢獻」：每步 (λ * cost_component) 累計
        # 注意：實際 modified_reward 公式是減掉它，所以輸出時通常會以負號呈現。
        self.ep_cost_penalty_total = 0.0
        self.ep_cost_penalty_breakdown = defaultdict(float)
        
    def reset(self, **kwargs):
        self.ep_ret_orig = 0.0
        self.ep_ret_orig_scaled = 0.0
        self.ep_ret_total = 0.0
        self.ep_cost = 0.0
        self.ep_cost_breakdown.clear()
        self.ep_cost_penalty_total = 0.0
        self.ep_cost_penalty_breakdown.clear()
        return self.env.reset(**kwargs)

    def set_lagrangian_lambdas(self, lambdas: Dict[str, float]) -> None:
        """
        由主進程 Callback 呼叫，用於同步 multi-lambda 到子進程。

        Args:
            lambdas: e.g. {"risk": 0.01, "fric": 0.0}
        """
        self._synced_lambdas = {str(k): float(v) for k, v in dict(lambdas).items()}

    def set_lagrangian_lambda(self, lam: float) -> None:
        """
        由主進程 Callback 呼叫，用於同步 single-lambda 到子進程。
        """
        self._synced_lambda = float(lam)
        
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. 取得原始數據
        # Env 回傳的 reward 即為「原始主線獎勵 (Log Return)」
        raw_reward = float(reward)
        cost = float(info.get("cost", 0.0))
        breakdown = info.get("cost_breakdown", {})
        
        # 2. 累計回合統計
        self.ep_ret_orig += raw_reward
        self.ep_ret_orig_scaled += raw_reward * self.reward_scale
        self.ep_cost += cost
        for k, v in breakdown.items():
            self.ep_cost_breakdown[k] += float(v)
            
        # 3. 計算 Lagrangian Reward (給 Agent 訓練用)
        modified_reward: float

        # --- multi-lambda path ---
        if isinstance(self.controller, MultiSharedLagrangianController):
            # 優先使用 callback 同步過來的 λ（避免子進程讀到的 shared 值不同步）
            lams = self._synced_lambdas if self._synced_lambdas is not None else self.controller.current_lambdas
            channel_costs = {
                k: float(info.get(f"cost_{k}", 0.0))
                for k in self.controller.channel_configs.keys()
            }
            penalty_raw = float(sum(lams[k] * channel_costs[k] for k in channel_costs.keys()))
            penalty = penalty_raw / self.cost_penalty_normalize
            modified_reward = (raw_reward * self.reward_scale) - penalty
            self.ep_ret_total += float(modified_reward)
            self.ep_cost_penalty_total += float(penalty)

            # penalty breakdown：依元件對應通道（未知元件 -> 使用 λ 總和）
            component_to_channel = {
                "death_cost": "risk",
                "fric_cost": "fric",
            }
            lam_sum = float(sum(lams.values()))
            for k, v in breakdown.items():
                ch = component_to_channel.get(str(k))
                lam_k = float(lams.get(ch, lam_sum)) if ch is not None else lam_sum
                self.ep_cost_penalty_breakdown[k] += float(lam_k) * float(v)

            # Debug info（相容 + 詳細）
            info["lag_lambdas"] = dict(lams)
            for k, v in lams.items():
                info[f"lag_lambda_{k}"] = float(v)
            info["lag_lambda"] = float(lam_sum)
        else:
            # --- single-lambda path ---
            lam = float(self._synced_lambda) if self._synced_lambda is not None else float(getattr(self.controller, "current_lambda", 0.0))
            penalty_raw = lam * float(cost)
            penalty = penalty_raw / self.cost_penalty_normalize
            modified_reward = (raw_reward * self.reward_scale) - penalty
            self.ep_ret_total += float(modified_reward)
            self.ep_cost_penalty_total += float(penalty)
            for k, v in breakdown.items():
                self.ep_cost_penalty_breakdown[k] += float(lam) * float(v)
            info["lag_lambda"] = lam
        
        # 4. 若回合結束，將累計統計注入 info 供 Callback 讀取
        if terminated or truncated:
            info["episode_metrics"] = {
                "return_orig": self.ep_ret_orig,
                "return_orig_scaled": self.ep_ret_orig_scaled,
                "return_total": self.ep_ret_total,
                "return_cost": self.ep_cost,
                "cost_breakdown": dict(self.ep_cost_breakdown),
                # penalty 是「正數大小」（= λ * cost），總 reward 公式會減掉它
                "cost_penalty_total": self.ep_cost_penalty_total,
                "cost_penalty_breakdown": dict(self.ep_cost_penalty_breakdown),
            }
            # Reset 在 reset() 做，這裡不急著清空，避免 info 引用錯誤
        
        # Debug info
        info["original_reward"] = raw_reward
        info["modified_reward"] = modified_reward
        
        return obs, modified_reward, terminated, truncated, info

