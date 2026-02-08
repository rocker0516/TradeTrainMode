from __future__ import annotations

import numpy as np
import torch
from collections import deque, defaultdict
from typing import Any, Dict, Deque, Tuple

from stable_baselines3.common.callbacks import BaseCallback

from Train.lagrangian.controllers import MultiSharedLagrangianController
from Train.lagrangian.stats import compute_trade_stats, compute_end_result_stats


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
        reward_scale: float = 1.0  # 新增：用於顯示統計時的說明
    ) -> None:
        super().__init__(verbose)
        self.controller = controller
        self.update_freq = update_freq
        self.log_freq = log_freq
        self.window_size = window_size
        self.reward_scale = float(reward_scale)
        # SubprocVecEnv 下，每個 global step 會收到 n_envs 筆 info。
        # 若 cost buffer 的 maxlen 只用 update_freq，會等效變成只看 (update_freq / n_envs) 個 steps，
        # 導致 avg_cost 視窗過短而常常趨近 0，進而讓 λ 被拉回 0（看起來像「成本失效」）。
        self._cost_buffer_n_envs: int | None = None
        # 允許測試或外部覆寫 logger（SB3 BaseCallback.logger 預設為唯讀 property）
        self._logger_override = None
        
        # Lambda Update Buffer
        # 先用 update_freq 當 base；實際 maxlen 會在 _on_step() 依 n_envs 動態擴充
        # 若 channel 有 window_steps（如 trade_freq 的「最近 N 步」），則該 channel 固定用該長度
        self.cost_buffer: Deque[float] = deque(maxlen=int(update_freq))
        self.cost_buffers: Dict[str, Deque[float]] = {}
        if isinstance(self.controller, MultiSharedLagrangianController):
            self.cost_buffers = {}
            for k, cfg in self.controller.channel_configs.items():
                w = getattr(cfg, "window_steps", None)
                maxlen = int(w) if w is not None else int(update_freq)
                self.cost_buffers[k] = deque(maxlen=max(1, maxlen))
        
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
            if isinstance(self.controller, MultiSharedLagrangianController):
                for k, cfg in self.controller.channel_configs.items():
                    w = getattr(cfg, "window_steps", None)
                    target_maxlen = int(w) if w is not None else (int(self.update_freq) * int(n_envs))
                    target_maxlen = max(1, target_maxlen)
                    old = self.cost_buffers.get(k)
                    self.cost_buffers[k] = deque(old or [], maxlen=target_maxlen)
            else:
                target_maxlen = max(1, int(self.update_freq) * int(n_envs))
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
                    # 若是 fric_cost，也顯示 Per Step 值
                    elif k == "fric_cost":
                        per_step = v / max(1.0, avg_episode_len)
                        print(f"    - {k:<20}: {v:.4f} (Step: {per_step:.6f})")
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
        
        # Section 4: Trade Execution Stats
        print(f"[{'TRADE STATS':^20}]")
        print(f"  Max Drawdown (Window)       : {max_dd*100:8.2f} %  [all episodes]")
        print(f"  Max DD Excl. Liquidation    : {max_dd_excl_liq*100:8.2f} %")
        print(f"  Max DD Excl. Stop Loss      : {max_dd_excl_stop_loss*100:8.2f} %")
        print(f"  Avg Liq Count               : {avg_liq:8.4f}")
        print(f"  Avg Fees                    : {avg_fee:8.2f}")
        print(f"  Avg Long Entries            : {avg_long_entries:8.4f}")
        print(f"  Avg Short Entries           : {avg_short_entries:8.4f}")
        print(f"  Avg Long Close Count        : {avg_long_closes:8.4f}")
        print(f"  Avg Short Close Count       : {avg_short_closes:8.4f}")
        print(f"  Avg Active Exit Count       : {avg_active_exits:8.4f} ({active_exit_rate_pct:5.1f}%)")
        print(f"  Avg Stop Loss Count         : {avg_stop_loss:8.4f} ({stop_loss_rate_pct:5.1f}%)")
        # 讓你判斷「止損線是否過緊」的輔助指標（越高通常越緊）
        print(f"  Stop Loss / Entry Rate      : {stop_loss_per_entry_pct:8.2f} %")
        print(f"  Avg Holding Steps           : {avg_holding_steps:8.2f}")
        print(f"  Avg Trade Steps (traded)    : {avg_trade_count:8.2f}")
        print(f"  Exit Coverage (AE+SL)/Close  : {exit_coverage_rate_pct:8.2f} %")
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

