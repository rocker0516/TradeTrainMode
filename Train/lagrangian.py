from __future__ import annotations

import multiprocessing
import numpy as np
import gymnasium as gym
import torch
from collections import Counter, deque, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Deque, Optional

from stable_baselines3.common.callbacks import BaseCallback


def compute_trade_stats(ep_infos: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    計算「TRADE STATS」視窗統計（純計算，方便測試）。

    Args:
        ep_infos: Callback 收集到的 episode 結束 info（VecMonitor 會在結束時注入 "episode" key）。

    Returns:
        dict，包含：
        - avg_fee, avg_liq, avg_dd, max_dd, max_dd_excl_liq, max_dd_excl_stop_loss
        - avg_long_entries, avg_short_entries, avg_long_closes, avg_short_closes
        - avg_stop_loss, avg_holding_steps, avg_trade_count, avg_active_exits
        - stop_loss_rate_pct, active_exit_rate_pct, exit_coverage_rate_pct, stop_loss_per_entry_pct
        - total_long_entries, total_short_entries, total_long_closes, total_short_closes（視窗內總計）
        - total_entries, total_closes
        - long_entry_share_pct, short_entry_share_pct: 多/空開倉佔比（%）
        - direction_bias_pct: (long-short)/total_entries*100（正=偏多，負=偏空，0=中性）
    """
    if not ep_infos:
        return {
            "avg_fee": 0.0,
            "avg_liq": 0.0,
            "avg_dd": 0.0,
            "max_dd": 0.0,
            "max_dd_excl_liq": 0.0,
            "max_dd_excl_stop_loss": 0.0,
            "avg_long_entries": 0.0,
            "avg_short_entries": 0.0,
            "avg_long_closes": 0.0,
            "avg_short_closes": 0.0,
            "avg_stop_loss": 0.0,
            "avg_holding_steps": 0.0,
            "avg_trade_count": 0.0,
            "avg_active_exits": 0.0,
            "stop_loss_rate_pct": 0.0,
            "active_exit_rate_pct": 0.0,
            "exit_coverage_rate_pct": 0.0,
            "stop_loss_per_entry_pct": 0.0,
            "total_long_entries": 0,
            "total_short_entries": 0,
            "total_long_closes": 0,
            "total_short_closes": 0,
            "total_entries": 0,
            "total_closes": 0,
            "long_entry_share_pct": 0.0,
            "short_entry_share_pct": 0.0,
            "direction_bias_pct": 0.0,
        }

    total_fees = [float(x.get("total_fees", 0.0)) for x in ep_infos]
    liq_counts = [int(x.get("episode_liq_count", 0)) for x in ep_infos]
    max_dds = [float(x.get("episode_max_dd", 0.0)) for x in ep_infos]
    long_entries = [int(x.get("long_entry_count", 0)) for x in ep_infos]
    short_entries = [int(x.get("short_entry_count", 0)) for x in ep_infos]
    long_closes = [int(x.get("long_close_count", 0)) for x in ep_infos]
    short_closes = [int(x.get("short_close_count", 0)) for x in ep_infos]
    stop_losses = [int(x.get("episode_stop_loss_count", 0)) for x in ep_infos]
    active_exits = [int(x.get("episode_active_exit_count", 0)) for x in ep_infos]
    holding_steps = [int(x.get("episode_holding_steps", 0)) for x in ep_infos]
    trade_counts = [int(x.get("episode_trade_count", 0)) for x in ep_infos]

    # --- max drawdown variants ---
    # 「排除」定義：整個 episode 只要曾發生過 해당事件，就把該 episode 從 max_dd 計算樣本排除。
    dd_pairs = list(zip(max_dds, liq_counts, stop_losses))
    def _safe_max(xs: List[float]) -> float:
        if not xs:
            return 0.0
        return float(np.max(xs))

    max_dd_all = _safe_max(list(max_dds))
    max_dd_excl_liq = _safe_max([dd for dd, liq, _sl in dd_pairs if int(liq) <= 0])
    # 你要求的口徑：排除止損「同時也排除強平」=> 僅保留 (stop_loss==0 且 liq==0) 的回合
    max_dd_excl_stop_loss = _safe_max([dd for dd, liq, sl in dd_pairs if int(sl) <= 0 and int(liq) <= 0])

    total_stop_losses = int(np.sum(stop_losses)) if stop_losses else 0
    total_active_exits = int(np.sum(active_exits)) if active_exits else 0
    total_exits = int(total_stop_losses + total_active_exits)
    if total_exits > 0:
        stop_loss_rate_pct = (total_stop_losses / total_exits) * 100.0
        active_exit_rate_pct = (total_active_exits / total_exits) * 100.0
    else:
        stop_loss_rate_pct = 0.0
        active_exit_rate_pct = 0.0

    total_closes = int(np.sum(long_closes) + np.sum(short_closes)) if (long_closes or short_closes) else 0
    if total_closes > 0:
        exit_coverage_rate_pct = (total_exits / total_closes) * 100.0
    else:
        exit_coverage_rate_pct = 0.0
    
    total_entries = int(np.sum(long_entries) + np.sum(short_entries)) if (long_entries or short_entries) else 0
    if total_entries > 0:
        stop_loss_per_entry_pct = (total_stop_losses / total_entries) * 100.0
    else:
        stop_loss_per_entry_pct = 0.0

    # 視窗內總計與方向偏好（便於解讀訓練時多/空偏好）
    total_long_entries = int(np.sum(long_entries)) if long_entries else 0
    total_short_entries = int(np.sum(short_entries)) if short_entries else 0
    total_long_closes = int(np.sum(long_closes)) if long_closes else 0
    total_short_closes = int(np.sum(short_closes)) if short_closes else 0
    total_closes = total_long_closes + total_short_closes
    if total_entries > 0:
        long_entry_share_pct = (total_long_entries / total_entries) * 100.0
        short_entry_share_pct = (total_short_entries / total_entries) * 100.0
        direction_bias_pct = ((total_long_entries - total_short_entries) / total_entries) * 100.0
    else:
        long_entry_share_pct = 0.0
        short_entry_share_pct = 0.0
        direction_bias_pct = 0.0

    return {
        "avg_fee": float(np.mean(total_fees)) if total_fees else 0.0,
        "avg_liq": float(np.mean(liq_counts)) if liq_counts else 0.0,
        "avg_dd": float(np.mean(max_dds)) if max_dds else 0.0,
        "max_dd": float(max_dd_all),
        "max_dd_excl_liq": float(max_dd_excl_liq) if np.isfinite(max_dd_excl_liq) else 0.0,
        "max_dd_excl_stop_loss": float(max_dd_excl_stop_loss) if np.isfinite(max_dd_excl_stop_loss) else 0.0,
        "avg_long_entries": float(np.mean(long_entries)) if long_entries else 0.0,
        "avg_short_entries": float(np.mean(short_entries)) if short_entries else 0.0,
        "avg_long_closes": float(np.mean(long_closes)) if long_closes else 0.0,
        "avg_short_closes": float(np.mean(short_closes)) if short_closes else 0.0,
        "avg_stop_loss": float(np.mean(stop_losses)) if stop_losses else 0.0,
        "avg_holding_steps": float(np.mean(holding_steps)) if holding_steps else 0.0,
        "avg_trade_count": float(np.mean(trade_counts)) if trade_counts else 0.0,
        "avg_active_exits": float(np.mean(active_exits)) if active_exits else 0.0,
        "stop_loss_rate_pct": float(stop_loss_rate_pct),
        "active_exit_rate_pct": float(active_exit_rate_pct),
        "exit_coverage_rate_pct": float(exit_coverage_rate_pct),
        "stop_loss_per_entry_pct": float(stop_loss_per_entry_pct),
        "total_long_entries": total_long_entries,
        "total_short_entries": total_short_entries,
        "total_long_closes": total_long_closes,
        "total_short_closes": total_short_closes,
        "total_entries": total_entries,
        "total_closes": total_closes,
        "long_entry_share_pct": float(long_entry_share_pct),
        "short_entry_share_pct": float(short_entry_share_pct),
        "direction_bias_pct": float(direction_bias_pct),
    }


def compute_end_result_stats(ep_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    計算「回合結束結果」視窗統計（純計算，方便測試）。

    統計內容：
    - terminated / truncated 次數與比例
    - termination_reason 各類型次數與比例
    - Avg Episode Length（VecMonitor: info["episode"]["l"]）
    - Avg Final Balance（TradingEnv: info["final_balance"]）

    Args:
        ep_infos: Callback 收集到的 episode 結束 info。

    Returns:
        dict（結構穩定，適合印出/寫 TB）。
    """
    n = int(len(ep_infos))
    if n == 0:
        return {
            "n": 0,
            "terminated_count": 0,
            "truncated_count": 0,
            "terminated_rate": 0.0,
            "truncated_rate": 0.0,
            "reason_counts": {},
            "reason_rates": {},
            "avg_episode_len": 0.0,
            "avg_final_balance": 0.0,
        }

    terminated_flags = [bool(x.get("terminated", False)) for x in ep_infos]
    truncated_flags = [bool(x.get("truncated", False)) for x in ep_infos]
    terminated_count = int(sum(terminated_flags))
    truncated_count = int(sum(truncated_flags))

    # termination_reason（缺失則歸類 unknown）
    reasons: List[str] = []
    for x in ep_infos:
        r = x.get("termination_reason", "unknown")
        r = str(r) if r is not None else "unknown"
        r = r.strip() or "unknown"
        reasons.append(r)

    reason_counts = dict(Counter(reasons))
    reason_rates = {k: (v / n) * 100.0 for k, v in reason_counts.items()}

    # VecMonitor episode length
    ep_lens: List[int] = []
    for x in ep_infos:
        ep = x.get("episode", {})
        if isinstance(ep, dict):
            try:
                ep_lens.append(int(ep.get("l", 0)))
            except (TypeError, ValueError):
                ep_lens.append(0)
        else:
            ep_lens.append(0)

    final_balances: List[float] = []
    for x in ep_infos:
        try:
            final_balances.append(float(x.get("final_balance", 0.0)))
        except (TypeError, ValueError):
            final_balances.append(0.0)

    return {
        "n": n,
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
        "terminated_rate": (terminated_count / n) * 100.0,
        "truncated_rate": (truncated_count / n) * 100.0,
        "reason_counts": reason_counts,
        "reason_rates": reason_rates,
        "avg_episode_len": float(np.mean(ep_lens)) if ep_lens else 0.0,
        "avg_final_balance": float(np.mean(final_balances)) if final_balances else 0.0,
    }


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


@dataclass(frozen=True)
class LagrangianChannelConfig:
    """多通道 Lagrangian 參數（每條成本線各自一組）。"""

    cost_limit: float
    kp: float = 0.1
    lambda_init: float = 0.0
    lambda_min: float = 0.0
    lambda_max: float = 5.0


class MultiSharedLagrangianController:
    """
    多通道的 Shared Lagrangian Controller。

    目的：
    - 將不同「成本線」分開更新 λ，避免單一 cost 混合後尺度不一致。
    - 每個通道有自己的 limit 與 λ（P-control + clamp）。
    """

    def __init__(self, channel_configs: Dict[str, LagrangianChannelConfig]) -> None:
        if not channel_configs:
            raise ValueError("channel_configs must not be empty.")
        self.channel_configs: Dict[str, LagrangianChannelConfig] = dict(channel_configs)
        self._lambda_vals: Dict[str, multiprocessing.Value] = {
            k: multiprocessing.Value("d", float(cfg.lambda_init))
            for k, cfg in self.channel_configs.items()
        }

    @property
    def current_lambdas(self) -> Dict[str, float]:
        """取得當前所有 λ（跨進程共享值）。"""
        out: Dict[str, float] = {}
        for k, v in self._lambda_vals.items():
            with v.get_lock():
                out[k] = float(v.value)
        return out

    @property
    def current_lambda(self) -> float:
        """相容舊版：回傳所有 λ 的和（僅供 debug/舊 key）。"""
        vals = self.current_lambdas
        return float(sum(vals.values()))

    def update(self, avg_costs: Dict[str, float]) -> Dict[str, float]:
        """
        根據各通道 avg_cost 更新對應 λ。
        公式：λ_new = clamp(λ_old + kp * (avg_cost - limit))
        """
        new_vals: Dict[str, float] = {}
        for k, cfg in self.channel_configs.items():
            avg_cost = float(avg_costs.get(k, 0.0))
            violation = avg_cost - float(cfg.cost_limit)
            v = self._lambda_vals[k]
            with v.get_lock():
                new_val = float(v.value) + float(cfg.kp) * float(violation)
                new_val = max(float(cfg.lambda_min), min(float(cfg.lambda_max), float(new_val)))
                v.value = float(new_val)
                new_vals[k] = float(new_val)
        return new_vals


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
        reward_scale: float = 1.0
    ) -> None:
        super().__init__(env)
        self.controller = controller
        self.reward_scale = float(reward_scale)

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
            lambdas: e.g. {"risk": 0.01, "fric": 0.0, "sl_buf": 0.02}
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
            penalty = float(sum(lams[k] * channel_costs[k] for k in channel_costs.keys()))
            modified_reward = (raw_reward * self.reward_scale) - penalty
            self.ep_ret_total += float(modified_reward)
            self.ep_cost_penalty_total += float(penalty)

            # penalty breakdown：依元件對應通道（未知元件 -> 使用 λ 總和）
            component_to_channel = {
                "death_cost": "risk",
                "stop_loss_event_cost": "sl_event",
                "fric_cost": "fric",
                "sl_buf_cost": "sl_buf",
                "stop_missing_cost": "sl_buf",
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
            modified_reward = (raw_reward * self.reward_scale) - (lam * cost)
            self.ep_ret_total += float(modified_reward)
            self.ep_cost_penalty_total += float(lam) * float(cost)
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
                "cost_breakdown": dict(self.ep_cost_breakdown)
                ,
                # penalty 是「正數大小」（= λ * cost），總 reward 公式會減掉它
                "cost_penalty_total": self.ep_cost_penalty_total,
                "cost_penalty_breakdown": dict(self.ep_cost_penalty_breakdown),
            }
            # Reset 在 reset() 做，這裡不急著清空，避免 info 引用錯誤
        
        # Debug info
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
        controller: Any,
        update_freq: int = 1000,   # 多少 global steps 更新一次 λ
        log_freq: int = 20,        # 多少 episodes 顯示一次統計 (Request: 20)
        window_size: int = 100,    # 統計視窗大小 (Request: 100)
        verbose: int = 1,
        reward_scale: float = 1.0,  # 用於顯示統計時的說明
        cost_window_steps: Optional[int] = None,  # cost 平均視窗步數；None 則用 update_freq*n_envs，設大則 λ 更平滑
    ) -> None:
        super().__init__(verbose)
        self.controller = controller
        self.update_freq = update_freq
        self.log_freq = log_freq
        self.window_size = window_size
        self.reward_scale = float(reward_scale)
        self.cost_window_steps = int(cost_window_steps) if cost_window_steps is not None and int(cost_window_steps) > 0 else None
        # SubprocVecEnv 下，每個 global step 會收到 n_envs 筆 info。
        # 若 cost buffer 的 maxlen 只用 update_freq，會等效變成只看 (update_freq / n_envs) 個 steps，
        # 導致 avg_cost 視窗過短而常常趨近 0，進而讓 λ 被拉回 0（看起來像「成本失效」）。
        self._cost_buffer_n_envs: int | None = None
        # 允許測試或外部覆寫 logger（SB3 BaseCallback.logger 預設為唯讀 property）
        self._logger_override = None
        
        # Lambda Update Buffer
        # 實際 maxlen 會在 _on_step() 依 n_envs 與 cost_window_steps 動態設定
        self.cost_buffer: Deque[float] = deque(maxlen=int(update_freq))
        self.cost_buffers: Dict[str, Deque[float]] = {}
        if isinstance(self.controller, MultiSharedLagrangianController):
            self.cost_buffers = {
                k: deque(maxlen=int(update_freq))
                for k in self.controller.channel_configs.keys()
            }
        
        # Stats Buffer (存最近 N 回合的 info)
        self.ep_infos: Deque[Dict[str, Any]] = deque(maxlen=window_size)

        # Global counters
        self.total_episodes: int = 0
        self.last_log_episode: int = 0

    @property
    def logger(self):  # type: ignore[override]
        """
        取得 logger。

        - 預設：沿用 SB3 BaseCallback 行為（從 self.model.logger 取得）。
        - 測試/注入：若外部指定 callback.logger = xxx，則優先回傳覆寫值。
        """
        if self._logger_override is not None:
            return self._logger_override
        return super().logger

    @logger.setter
    def logger(self, value) -> None:  # type: ignore[override]
        self._logger_override = value

    def on_step(self) -> bool:  # type: ignore[override]
        """
        兼容性 on_step：

        - 正常 SB3 訓練流程：model 會提供 num_timesteps，行為與 BaseCallback.on_step 等價。
        - 單元測試/Mock：允許 model 缺少 num_timesteps，避免測試因 SB3 內部細節崩潰。
        """
        self.n_calls += 1
        if hasattr(self.model, "num_timesteps"):
            # SB3 期望的欄位：用於 log / scheduler 等
            self.num_timesteps = int(getattr(self.model, "num_timesteps"))
        return bool(self._on_step())
        
    def _on_step(self) -> bool:
        # SB3 的 locals['infos'] 包含所有並行環境的 info
        infos = self.locals.get("infos", [])

        # --- Ensure cost buffer window matches global steps (handles n_envs) ---
        try:
            n_envs = max(1, int(len(infos)))
        except (TypeError, ValueError):
            n_envs = 1
        if self._cost_buffer_n_envs != n_envs:
            # cost_window_steps > 0 時拉長視窗，使 avg_cost 更平滑、λ 較少震盪（見 docs/lambda_tuning_optimization.md）
            window_steps = int(self.cost_window_steps) if self.cost_window_steps is not None and self.cost_window_steps > 0 else int(self.update_freq)
            target_maxlen = window_steps * int(n_envs)
            target_maxlen = max(1, target_maxlen)
            if isinstance(self.controller, MultiSharedLagrangianController):
                for k in self.controller.channel_configs.keys():
                    old = self.cost_buffers.get(k)
                    self.cost_buffers[k] = deque(old or [], maxlen=target_maxlen)
            else:
                self.cost_buffer = deque(self.cost_buffer, maxlen=target_maxlen)
            self._cost_buffer_n_envs = int(n_envs)
        
        for info in infos:
            # 1. 收集 Cost (Step level, for Lambda update)
            if isinstance(self.controller, MultiSharedLagrangianController):
                for k in self.controller.channel_configs.keys():
                    key = f"cost_{k}"
                    if key in info:
                        self.cost_buffers[k].append(float(info[key]))
            else:
                if "cost" in info:
                    self.cost_buffer.append(float(info["cost"]))
            
            # 2. 收集 Episode 結束時的統計
            # VecMonitor 在回合結束時會加入 "episode" key
            if "episode" in info:
                self.total_episodes += 1
                self.ep_infos.append(info)

        # 3. 定期更新 λ
        if self.n_calls % self.update_freq == 0:
            if isinstance(self.controller, MultiSharedLagrangianController):
                avg_costs: Dict[str, float] = {}
                for k, buf in self.cost_buffers.items():
                    avg_costs[k] = float(np.mean(buf)) if len(buf) > 0 else 0.0
                new_lams = self.controller.update(avg_costs)

                # TensorBoard：每條 λ + 相容總和
                self.logger.record("lagrangian/lambda", float(sum(new_lams.values())))
                for k, lam in new_lams.items():
                    self.logger.record(f"lagrangian/lambda_{k}", float(lam))
                    avg_c = float(avg_costs.get(k, 0.0))
                    self.logger.record(f"lagrangian/avg_cost_{k}", avg_c)
                    self.logger.record(
                        f"lagrangian/cost_violation_{k}",
                        avg_c - float(self.controller.channel_configs[k].cost_limit),
                    )

                # --- Sync lambdas to sub-process envs (Route B) ---
                # SubprocVecEnv（spawn）下，子進程可能讀不到主進程更新後的 shared λ；
                # 這裡用 VecEnv.env_method 明確同步到每個 LagrangianRewardWrapper。
                try:
                    if getattr(self, "training_env", None) is not None:
                        self.training_env.env_method("set_lagrangian_lambdas", dict(new_lams))
                except Exception:
                    # 同步失敗不應中斷訓練（例如 unit test 沒有 vec env）
                    pass
            else:
                if len(self.cost_buffer) > 0:
                    avg_cost = np.mean(self.cost_buffer)
                    new_lambda = self.controller.update(avg_cost)
                    
                    # 寫入 TensorBoard
                    self.logger.record("lagrangian/lambda", new_lambda)
                    # 相容：保留舊 key（部分測試/既有圖表可能用到）
                    self.logger.record("lagrangian/avg_cost_step", avg_cost)
                    self.logger.record("lagrangian/avg_cost", avg_cost)
                    self.logger.record("lagrangian/cost_violation", avg_cost - self.controller.cost_limit)

                    # --- Sync single lambda to sub-process envs (Route B) ---
                    try:
                        if getattr(self, "training_env", None) is not None:
                            self.training_env.env_method("set_lagrangian_lambda", float(new_lambda))
                    except Exception:
                        pass

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
        ep_ret_orig_scaleds = []
        ep_ret_totals = []
        ep_costs = []
        cost_breakdowns = defaultdict(list)
        cost_penalty_totals = []
        cost_penalty_breakdowns = defaultdict(list)
        
        for info in self.ep_infos:
            metrics = info.get("episode_metrics", {})
            if metrics:
                ep_ret_origs.append(metrics.get("return_orig", 0.0))
                ep_ret_orig_scaleds.append(metrics.get("return_orig_scaled", 0.0))
                ep_ret_totals.append(metrics.get("return_total", 0.0))
                ep_costs.append(metrics.get("return_cost", 0.0))
                for k, v in metrics.get("cost_breakdown", {}).items():
                    cost_breakdowns[k].append(v)
                if "cost_penalty_total" in metrics:
                    cost_penalty_totals.append(metrics.get("cost_penalty_total", 0.0))
                for k, v in metrics.get("cost_penalty_breakdown", {}).items():
                    cost_penalty_breakdowns[k].append(v)

        # VecMonitor 的 episode 統計（wrapped env 的 reward 回報；也就是訓練端「主線 reward」episode return）
        # SB3/VecMonitor: info["episode"] = {"r": ep_return, "l": ep_len, "t": elapsed_sec}
        ep_main_rewards = []
        for info in self.ep_infos:
            ep = info.get("episode", {})
            if isinstance(ep, dict):
                ep_main_rewards.append(float(ep.get("r", 0.0)))
        
        # 環境原生統計 (TradingEnv)
        profits = [float(x.get("profit", 0.0)) for x in self.ep_infos]
        trade_stats = compute_trade_stats(list(self.ep_infos))
        
        # --- 2. 計算平均 ---
        avg_ret_orig = np.mean(ep_ret_origs) if ep_ret_origs else 0.0
        avg_main_reward = np.mean(ep_main_rewards) if ep_main_rewards else 0.0
        avg_ret_orig_scaled = np.mean(ep_ret_orig_scaleds) if ep_ret_orig_scaleds else 0.0
        # 若 wrapper 有提供 return_total，優先用它（避免 VecMonitor 受其他 wrapper 影響）
        avg_total_reward = np.mean(ep_ret_totals) if ep_ret_totals else avg_main_reward
        avg_cost = np.mean(ep_costs) if ep_costs else 0.0
        avg_cost_penalty_total = np.mean(cost_penalty_totals) if cost_penalty_totals else 0.0
        
        avg_breakdown = {}
        for k, v_list in cost_breakdowns.items():
            avg_breakdown[k] = np.mean(v_list) if v_list else 0.0
            
        avg_profit = np.mean(profits) if profits else 0.0
        avg_fee = float(trade_stats["avg_fee"])
        avg_liq = float(trade_stats["avg_liq"])
        # Drawdown：三條口徑（你要求的拆分）
        max_dd = float(trade_stats.get("max_dd", trade_stats.get("avg_dd", 0.0)))
        max_dd_excl_liq = float(trade_stats.get("max_dd_excl_liq", 0.0))
        max_dd_excl_stop_loss = float(trade_stats.get("max_dd_excl_stop_loss", 0.0))
        avg_long_entries = float(trade_stats["avg_long_entries"])
        avg_short_entries = float(trade_stats["avg_short_entries"])
        avg_long_closes = float(trade_stats["avg_long_closes"])
        avg_short_closes = float(trade_stats["avg_short_closes"])
        avg_stop_loss = float(trade_stats["avg_stop_loss"])
        avg_active_exits = float(trade_stats["avg_active_exits"])
        stop_loss_rate_pct = float(trade_stats["stop_loss_rate_pct"])
        active_exit_rate_pct = float(trade_stats["active_exit_rate_pct"])
        exit_coverage_rate_pct = float(trade_stats["exit_coverage_rate_pct"])
        stop_loss_per_entry_pct = float(trade_stats.get("stop_loss_per_entry_pct", 0.0))
        avg_holding_steps = float(trade_stats.get("avg_holding_steps", 0.0))
        avg_trade_count = float(trade_stats.get("avg_trade_count", 0.0))
        total_entries = int(trade_stats.get("total_entries", 0))
        total_closes = int(trade_stats.get("total_closes", 0))
        long_entry_share_pct = float(trade_stats.get("long_entry_share_pct", 0.0))
        short_entry_share_pct = float(trade_stats.get("short_entry_share_pct", 0.0))
        direction_bias_pct = float(trade_stats.get("direction_bias_pct", 0.0))

        end_stats = compute_end_result_stats(list(self.ep_infos))
        terminated_count = int(end_stats["terminated_count"])
        truncated_count = int(end_stats["truncated_count"])
        terminated_rate = float(end_stats["terminated_rate"])
        truncated_rate = float(end_stats["truncated_rate"])
        avg_episode_len = float(end_stats["avg_episode_len"])
        avg_final_balance = float(end_stats["avg_final_balance"])
        
        # Win Rate
        wins = sum(1 for p in profits if p > 0)
        win_rate = (wins / len(profits)) * 100 if profits else 0.0
        
        # Q-Values
        q_mean, q_min, q_max = self._get_q_values_stats()
        
        # --- 3. 顯示排版 ---
        # 使用 print 直接輸出到 console，方便查看
        def _fmt_small(x: float, *, fixed: int = 6, sci: int = 2, sci_threshold: float = 1e-6) -> str:
            """
            讓 console 顯示更穩定：
            - 數值夠大：用固定小數（例如 0.012345）
            - 數值很小但非 0：用科學記號避免印成 0.000000（例如 1.23e-08）
            """
            try:
                v = float(x)
            except (TypeError, ValueError):
                return str(x)
            if v == 0.0:
                return f"{0.0:.{fixed}f}"
            if abs(v) < float(sci_threshold):
                return f"{v:.{sci}e}"
            return f"{v:.{fixed}f}"

        # λ 一律用科學記號（你要求：明確顯示 COST LINES 的 Lambdas 數值）。
        def _fmt_lambda(x: float) -> str:
            try:
                return f"{float(x):.3e}"
            except (TypeError, ValueError):
                return str(x)

        # COST LINES 裡的 avg/limit 也一律用科學記號（避免 0.000100 看起來不夠醒目）。
        def _fmt_cost_scalar(x: float) -> str:
            try:
                return f"{float(x):.3e}"
            except (TypeError, ValueError):
                return str(x)

        print("\n" + "="*60)
        print(
            f"  STATS (Last {len(self.ep_infos)} Episodes | Total Episodes {self.total_episodes}) "
            f"@ Global Step {self.num_timesteps}"
        )
        print("="*60)
        
        # Section 1: Main Reward (Training Objective)
        # 用更直觀的標籤區分「原始市場表現」與「RL 訓練訊號」
        print(f"[{'MAIN REWARD':^20}]")
        print("  --- RL Training Signal (What Agent Sees) ---")
        print(f"  Total Reward (R_total)      : {avg_total_reward:8.4f}  [= R_scaled - (λ * Cost)]")
        print(f"  Scaled Reward (R_scaled)    : {avg_ret_orig_scaled:8.4f}  [= LogRet * {self.reward_scale}]")
        print(f"  Cost Penalty (-λ * C)       : {-avg_cost_penalty_total:8.4f}")
        
        print("  --- Original Market Performance ---")
        print(f"  Log Return Sum (LogRet)     : {avg_ret_orig:8.4f}")
        # 將 Log Return 換算成簡單的 ROI% 估算 (exp(sum_log_ret) - 1)，供參考
        roi_est = (np.exp(avg_ret_orig) - 1.0) * 100.0
        print(f"  Est. ROI (from LogRet)      : {roi_est:8.2f} %")
        print(f"  Avg Profit (USDT)           : {avg_profit:8.2f}")
        print(f"  Win Rate                    : {win_rate:8.1f} %")
        print("-" * 60)
        
        # Section 2: Cost Line (Constraints)
        # 關鍵：顯示每步平均 Cost (與 Cost Limit 對齊)
        # avg_episode_len 已經在上面計算過
        avg_cost_per_step = avg_cost / max(1.0, avg_episode_len)
        if isinstance(self.controller, MultiSharedLagrangianController):
            lams = self.controller.current_lambdas
            print(f"[{'COST LINES':^20}] Lambdas (sum={_fmt_lambda(sum(lams.values()))})")
            for k in self.controller.channel_configs.keys():
                limit = float(self.controller.channel_configs[k].cost_limit)
                avg_c = 0.0
                if k in self.cost_buffers and len(self.cost_buffers[k]) > 0:
                    avg_c = float(np.mean(self.cost_buffers[k]))
                vio = avg_c - limit
                # 說明：這裡的 avg 是「最近 update_freq steps 的 per-step 平均」。
                # death_cost 通常只在回合終止那一步 =1，因此即使 death rate 很高，短視窗內也可能出現 avg=0。
                death_rate_ep = float(avg_breakdown.get("death_cost", 0.0)) * 100.0
                print(
                    f"  - {k:<8} λ={_fmt_lambda(lams.get(k, 0.0))}  limit={_fmt_cost_scalar(limit)}  avg={_fmt_cost_scalar(avg_c)}  "
                   # f"death_rate_ep={death_rate_ep:6.2f}%  "
                    f"[{'OK' if vio <= 0 else 'VIOLATION'}]"
                )
            # 仍顯示 aggregated cost（方便對照舊圖表）
            print(f"  Avg Cost (Per Step)         : {_fmt_cost_scalar(avg_cost_per_step)}  [aggregate]")
        else:
            cost_limit = float(self.controller.cost_limit)
            violation = avg_cost_per_step - cost_limit
            print(f"[{'COST LINE':^20}] Lambda: {_fmt_lambda(float(self.controller.current_lambda))}")
            print(f"  Cost Limit (Per Step)       : {_fmt_cost_scalar(cost_limit)}")
            print(f"  Avg Cost (Per Step)         : {_fmt_cost_scalar(avg_cost_per_step)}  [{'OK' if violation <= 0 else 'VIOLATION'}]")
        print(f"  Avg Cost (Episode Total)    : {avg_cost:8.4f}")
        
        if avg_breakdown:
            print("  --- Cost Breakdown ---")
            for k, v in avg_breakdown.items():
                if v > 1e-6: 
                    # 若是 death_cost (單次=1.0)，則平均值即為發生率
                    if k == "death_cost":
                        print(f"    - {k:<20}: {v:.4f} (Rate: {v*100:.2f}%)")
                    # freq 通道 (c_freq 0~1)：顯示每步平均
                    elif k == "fric_cost":
                        per_step = v / max(1.0, avg_episode_len)
                        print(f"    - {k:<20}: {v:.4f} (Avg/step: {per_step:.4f})")
                    else:
                        print(f"    - {k:<20}: {v:8.4f}")
            
            if cost_penalty_breakdowns:
                print("  --- Cost Penalty Breakdown (=-λ*C) ---")
                for k, v_list in cost_penalty_breakdowns.items():
                    avg_pen = float(np.mean(v_list)) if v_list else 0.0
                    if abs(avg_pen) > 1e-6:
                        print(f"    - {k:<20}: {-avg_pen:8.4f}")
        else:
            print("    (No costs triggered)")
        print("-" * 60)
        
        # Section 3: Training Health (Q-Values & Losses)
        print(f"[{'TRAINING HEALTH':^20}]")
        print(f"  Q-Value (Mean)              : {q_mean:8.4f}")
        print(f"  Q-Value (Min/Max)           : {q_min:8.4f} / {q_max:8.4f}")
        # Entropy/Loss 等可從 Tensorboard 查看，這裡顯示最關鍵的 Q 值即可
        print("-" * 60)
        
        # Section 4: Trade Execution Stats（對齊實際交易口徑：方向偏好、風險、成交量、出場方式）
        print(f"[{'TRADE STATS':^20}]")
        # --- 方向偏好：一目了然多/空偏好（依「開倉次數」口徑）---
        print("  --- Direction Preference (from entries) ---")
        print(f"  Long / Short Share         : Long {long_entry_share_pct:5.1f} %  |  Short {short_entry_share_pct:5.1f} %")
        print(f"  Direction Bias             : {direction_bias_pct:+6.2f} %  (0=neutral, +=long, -=short)")
        print(f"  Total in Window            : Entries L+S = {total_entries},  Closes L+S = {total_closes}")
        print("  --- Risk & Fees ---")
        print(f"  Max Drawdown (Window)       : {max_dd*100:8.2f} %  [all episodes]")
        print(f"  Max DD Excl. Liquidation    : {max_dd_excl_liq*100:8.2f} %")
        print(f"  Max DD Excl. Stop Loss      : {max_dd_excl_stop_loss*100:8.2f} %")
        print(f"  Avg Liq Count               : {avg_liq:8.4f}")
        print(f"  Avg Fees                    : {avg_fee:8.2f}")
        print("  --- Volume (per-episode averages) ---")
        print(f"  Avg Long Entries            : {avg_long_entries:8.2f}  (open-from-flat events)")
        print(f"  Avg Short Entries           : {avg_short_entries:8.2f}")
        print(f"  Avg Long Close Count        : {avg_long_closes:8.2f}  (full or partial close events)")
        print(f"  Avg Short Close Count       : {avg_short_closes:8.2f}")
        print(f"  Avg Holding Steps           : {avg_holding_steps:8.2f}  (steps with position)")
        print(f"  Avg Steps With Trade        : {avg_trade_count:8.2f}  (steps where position changed)")
        print("  --- Exit Mix ---")
        print(f"  Avg Active Exit Count       : {avg_active_exits:8.2f} ({active_exit_rate_pct:5.1f}%)")
        print(f"  Avg Stop Loss Count         : {avg_stop_loss:8.2f} ({stop_loss_rate_pct:5.1f}%)")
        print(f"  Stop Loss / Entry Rate      : {stop_loss_per_entry_pct:8.2f} %  (tighter SL => higher)")
        print(f"  Exit Coverage (AE+SL)/Close : {exit_coverage_rate_pct:8.2f} %  (of all close events)")
        print("-" * 60)

        # Section 5: End Results (Episode termination summary)
        print(f"[{'END RESULTS':^20}]")
        print(
            f"  Terminated/Truncated        : {terminated_count:4d} / {truncated_count:4d} "
            f"({terminated_rate:5.1f}% / {truncated_rate:5.1f}%)"
        )
        print(f"  Avg Episode Length          : {avg_episode_len:8.2f}")
        print(f"  Avg Final Balance           : {avg_final_balance:8.2f}")
        print("  --- Termination Reasons ---")
        reason_counts = dict(end_stats.get("reason_counts", {}))
        reason_rates = dict(end_stats.get("reason_rates", {}))
        # 常見 reason 先列出，其他再補
        preferred_order = ["liq_triggered", "balance_insufficient", "max_steps_reached", "data_exhausted", "unknown"]
        printed = set()
        for k in preferred_order:
            if k in reason_counts:
                printed.add(k)
                print(f"    - {k:<20}: {int(reason_counts[k]):4d} ({float(reason_rates.get(k, 0.0)):5.1f}%)")
        for k in sorted(reason_counts.keys()):
            if k in printed:
                continue
            print(f"    - {k:<20}: {int(reason_counts[k]):4d} ({float(reason_rates.get(k, 0.0)):5.1f}%)")

        print("="*60 + "\n")

        # --- 4. 寫入 TensorBoard (Optional) ---
        self.logger.record("custom/avg_main_reward", avg_main_reward)
        self.logger.record("custom/avg_total_reward", avg_total_reward)
        self.logger.record("custom/avg_ret_orig", avg_ret_orig)
        self.logger.record("custom/avg_ret_orig_scaled", avg_ret_orig_scaled)
        self.logger.record("custom/avg_cost_penalty_total", avg_cost_penalty_total)
        self.logger.record("custom/avg_profit", avg_profit)
        self.logger.record("custom/q_mean", q_mean)
        self.logger.record("custom/ep_cost", avg_cost)
        self.logger.record("custom/avg_fee", avg_fee)
        # 相容保留：avg_dd 仍記錄「平均 episode_max_dd」
        self.logger.record("custom/avg_dd", float(trade_stats.get("avg_dd", 0.0)))
        # 新增：視窗內最大 DD（你要求的口徑）
        self.logger.record("custom/max_dd", max_dd)
        self.logger.record("custom/max_dd_excl_liq", max_dd_excl_liq)
        self.logger.record("custom/max_dd_excl_stop_loss", max_dd_excl_stop_loss)
        self.logger.record("custom/avg_liq", avg_liq)
        self.logger.record("custom/avg_long_entries", avg_long_entries)
        self.logger.record("custom/avg_short_entries", avg_short_entries)
        self.logger.record("custom/avg_stop_loss", avg_stop_loss)
        self.logger.record("custom/avg_active_exits", avg_active_exits)
        self.logger.record("custom/stop_loss_rate_pct", stop_loss_rate_pct)
        self.logger.record("custom/active_exit_rate_pct", active_exit_rate_pct)
        self.logger.record("custom/stop_loss_per_entry_pct", stop_loss_per_entry_pct)
        self.logger.record("custom/avg_holding_steps", avg_holding_steps)
        self.logger.record("custom/avg_trade_count", avg_trade_count)
        self.logger.record("custom/avg_long_closes", avg_long_closes)
        self.logger.record("custom/avg_short_closes", avg_short_closes)
        self.logger.record("custom/exit_coverage_rate_pct", exit_coverage_rate_pct)
        self.logger.record("custom/direction_bias_pct", direction_bias_pct)
        self.logger.record("custom/long_entry_share_pct", long_entry_share_pct)
        self.logger.record("custom/short_entry_share_pct", short_entry_share_pct)
        self.logger.record("custom/total_entries", total_entries)
        self.logger.record("custom/total_closes", total_closes)
        self.logger.record("custom/total_episodes", int(self.total_episodes))
        self.logger.record("custom/terminated_rate", terminated_rate)
        self.logger.record("custom/truncated_rate", truncated_rate)
        self.logger.record("custom/avg_episode_len", avg_episode_len)
        self.logger.record("custom/avg_final_balance", avg_final_balance)
        for k, v in dict(end_stats.get("reason_counts", {})).items():
            # 只記錄有限長度 key，避免 logger key 太亂
            safe_k = str(k).replace(" ", "_")[:64]
            self.logger.record(f"custom/end_reason_count/{safe_k}", int(v))
        for k, v in dict(end_stats.get("reason_rates", {})).items():
            safe_k = str(k).replace(" ", "_")[:64]
            self.logger.record(f"custom/end_reason_rate/{safe_k}", float(v))

        # cost penalty breakdown to TensorBoard（以負號呈現）
        for k, v_list in cost_penalty_breakdowns.items():
            avg_pen = float(np.mean(v_list)) if v_list else 0.0
            if abs(avg_pen) <= 1e-12:
                continue
            safe_k = str(k).replace(" ", "_")[:64]
            self.logger.record(f"custom/cost_penalty/{safe_k}", -avg_pen)
