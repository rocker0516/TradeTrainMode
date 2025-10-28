"""
SAC 交易模型訓練腳本（使用 Stable-Baselines3）

使用 SB3 內建的 SAC 算法進行訓練。
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 添加父目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
from gymnasium.wrappers import TimeLimit

from Env.trading_env import TradingEnvironment
from Env.reward import (
    RewardCalculator,
    create_default_calculator,
)
from Train.models.sac_lstm_policy import RecurrentSACPolicy
from Train.utils.wrappers import RiskPenaltyWrapper, InfoLoggerWrapper
from Train.utils.lagrangian import LagrangianController


class TradingEnvFactory:
    """
    可序列化的環境工廠，用於多進程向量環境。

    每個子進程都會持有各自獨立的 RewardCalculator 與環境狀態，
    以確保「每回合獨立獎勵計算」。
    """

    def __init__(
        self,
        *,
        data_path: str,
        initial_balance: float,
        transaction_fee: float,
        window_size: int,
        leverage: float,
        min_balance: float,
        min_trade_qty: float,
        margin_mode: str,
        random_start: bool,
        use_rudder: bool = True,
        use_lagrangian_wrapper: bool = True,
        stop_loss_cooldown_window: int = 6,
        stop_loss_cooldown_length: int = 12,
        stop_loss_cooldown_limit: int = 2,
        cooldown_hold_ratio: float = 0.0,
    ) -> None:
        self.data_path = data_path
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        self.leverage = leverage
        self.min_balance = min_balance
        self.min_trade_qty = min_trade_qty
        self.margin_mode = margin_mode
        self.random_start = random_start
        self.use_rudder = use_rudder
        self.use_lagrangian_wrapper = use_lagrangian_wrapper
        self.stop_loss_cooldown_window = int(stop_loss_cooldown_window)
        self.stop_loss_cooldown_length = int(stop_loss_cooldown_length)
        self.stop_loss_cooldown_limit = int(stop_loss_cooldown_limit)
        self.cooldown_hold_ratio = float(cooldown_hold_ratio)

    def __call__(self):
        df = pd.read_csv(self.data_path)
        
        # 創建環境（支援 RUDDER）
        if self.use_rudder:
            env = TradingEnvironment(
                df=df,
                initial_balance=self.initial_balance,
                transaction_fee=self.transaction_fee,
                window_size=self.window_size,
                leverage=self.leverage,
                min_balance=self.min_balance,
                min_trade_qty=self.min_trade_qty,
                margin_mode=self.margin_mode,
                use_rudder=True,  # 啟用 RUDDER
                random_start=self.random_start,
                stop_loss_cooldown_window=self.stop_loss_cooldown_window,
                stop_loss_cooldown_length=self.stop_loss_cooldown_length,
                stop_loss_cooldown_limit=self.stop_loss_cooldown_limit,
                cooldown_hold_ratio=self.cooldown_hold_ratio,
            )
            
            # 添加 Wrapper（Lagrangian + InfoLogger）
            if self.use_lagrangian_wrapper:
                env = RiskPenaltyWrapper(env, lambda_prob=1.0, lambda_cvar=1.0)
            env = InfoLoggerWrapper(env)
        else:
            # 舊版（向後兼容）
            rc = create_default_calculator()
            env = TradingEnvironment(
                df=df,
                initial_balance=self.initial_balance,
                transaction_fee=self.transaction_fee,
                window_size=self.window_size,
                leverage=self.leverage,
                min_balance=self.min_balance,
                min_trade_qty=self.min_trade_qty,
                margin_mode=self.margin_mode,
                reward_calculator=rc,
                use_rudder=False,
                random_start=self.random_start,
                stop_loss_cooldown_window=self.stop_loss_cooldown_window,
                stop_loss_cooldown_length=self.stop_loss_cooldown_length,
                stop_loss_cooldown_limit=self.stop_loss_cooldown_limit,
                cooldown_hold_ratio=self.cooldown_hold_ratio,
            )
        
        return Monitor(env)


class InfoTensorboardCallback(BaseCallback):
    """Log selected info keys and positive/negative reward traces to TensorBoard."""

    def __init__(
        self,
        *,
        log_keys: list[str],
        prefix: str = 'env',
        smooth: float | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.log_keys = log_keys
        self.prefix = prefix
        self.smooth = smooth if smooth is not None and 0.0 < smooth < 1.0 else None
        self._reward_pos = 0.0
        self._reward_neg = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get('infos')
        if infos is None:
            infos = []

        rewards_local = self.locals.get('rewards')
        rewards_array = np.asarray(rewards_local, dtype=float) if rewards_local is not None else np.asarray([], dtype=float)

        if infos:
            aggregated: dict[str, list[float]] = {}
            for info in infos:
                for key in self.log_keys:
                    if key in info:
                        aggregated.setdefault(key, []).append(float(info[key]))
            for key, values in aggregated.items():
                if values:
                    self.logger.record(f"{self.prefix}/{key}", float(np.mean(values)))

        if rewards_array.size > 0:
            reward_value = float(np.mean(rewards_array))
            reward_pos = max(reward_value, 0.0)
            reward_neg = min(reward_value, 0.0)

            if self.smooth is not None:
                alpha = self.smooth
                self._reward_pos = alpha * self._reward_pos + (1.0 - alpha) * reward_pos
                self._reward_neg = alpha * self._reward_neg + (1.0 - alpha) * reward_neg
                self.logger.record(f"{self.prefix}/reward_pos", self._reward_pos)
                self.logger.record(f"{self.prefix}/reward_neg", self._reward_neg)
            else:
                self.logger.record(f"{self.prefix}/reward_pos", reward_pos)
                self.logger.record(f"{self.prefix}/reward_neg", reward_neg)

        return True


class TradingCallback(BaseCallback):
    """
    自定義訓練回調（支援 RUDDER + Lagrangian）
    
    記錄訓練過程中的詳細信息，並定期更新 Lagrangian 乘子。
    """
    
    def __init__(
        self,
        log_interval: int = 1,
        verbose: int = 1,
        print_step_episode: bool = False,
        lagrangian_controller: LagrangianController | None = None,
        lagrangian_update_freq: int = 100,  # 每 N 個 episode 更新一次 λ
    ):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.print_step_episode = bool(print_step_episode)
        self.lagrangian_controller = lagrangian_controller
        self.lagrangian_update_freq = lagrangian_update_freq
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_profit_rates = []
        self.episode_failures = []  # True 表示失敗（非 data_exhausted）
        self.episode_stop_losses = []  # 止損觸發次數
        self.episode_liquidations = []  # 清算觸發次數
        
        # RUDDER + Lagrangian 統計
        self.episode_outcomes = []  # outcome 總和
        self.episode_cost_probs = []  # cost_prob 總和（機率違規）
        self.episode_forced_closes = []  # 強制平倉次數
        self.episode_risk_costs = []  # 平均 risk cost
        self.episode_struct_costs = []  # 平均 struct cost
        self.episode_step_costs = []  # 平均 step cost
        self.episode_trade_costs = []  # 平均交易次數成本
        self.episode_entry_costs = []  # 平均進場成本
        self.episode_drawdown_costs = []  # 平均 drawdown 成本
        self.episode_stop_streak_costs = []  # 平均連續止損成本
        
        # 向量環境的逐環境狀態
        self.current_episode_reward = None  # 將在 training start 時初始化為 (n_envs,) 向量
        self.current_episode_length = None
        self.env_episode_indices = None  # 每個環境的回合序號
    
    def _on_training_start(self) -> None:
        try:
            n_envs = getattr(self.training_env, 'num_envs', 1)
        except Exception:
            n_envs = 1
        self.current_episode_reward = np.zeros(n_envs, dtype=float)
        self.current_episode_length = np.zeros(n_envs, dtype=int)
        self.env_episode_indices = np.ones(n_envs, dtype=int)

    def _on_step(self) -> bool:
        """每步後調用（支援向量環境）"""
        rewards = self.locals.get('rewards')
        dones = self.locals.get('dones')
        infos = self.locals.get('infos')
        if rewards is None or dones is None:
            return True

        n_envs = len(rewards)
        # 累積獎勵與步數
        self.current_episode_reward[:n_envs] += rewards
        self.current_episode_length[:n_envs] += 1

        # 可選：每步輸出任一環境的回合數
        if self.print_step_episode:
            try:
                print(f"[train] step={self.num_timesteps}", end='\r')
            except Exception:
                pass

        # 處理結束的環境們
        for i in range(n_envs):
            if not dones[i]:
                continue

            # 取得該環境的底層 env
            try:
                wrapped_env = self.training_env.envs[i]
                env = wrapped_env.unwrapped if hasattr(wrapped_env, 'unwrapped') else wrapped_env
            except Exception:
                env = None

            final_balance = None
            profit_rate = None
            term_result = 'unknown'
            failure_flag = False
            stop_loss_flag = False
            liq_flag = False
            episode_stop_loss_count = 0
            episode_liq_count = 0
            long_close_count = None
            short_close_count = None
            total_fees = None
            episode_steps = None
            long_entry_count = None
            short_entry_count = None
            episode_max_steps = None

            # 從 infos[i] 讀取結算資訊
            total_outcome = 0.0
            cost_prob_sum = 0.0
            forced_close_count = 0
            risk_cost_avg = 0.0
            struct_cost_avg = 0.0
            step_cost_avg = 0.0
            trade_cost_rate = 0.0
            drawdown_cost_avg = 0.0
            stop_streak_cost_avg = 0.0
            try:
                info_i = infos[i] if isinstance(infos, (list, tuple)) and i < len(infos) else {}
                if isinstance(info_i, dict):
                    reason = info_i.get('termination_reason')
                    failure_flag = (reason is not None and reason != 'data_exhausted')
                    # 止損不終止 episode，從 info 中的標記讀取（非 termination_reason）
                    stop_loss_flag = bool(info_i.get('stop_loss_triggered', False))
                    liq_flag = (reason == 'liq_triggered')
                    if 'final_balance' in info_i:
                        final_balance = float(info_i['final_balance'])
                        profit_rate = float(info_i.get('profit_rate', 0.0))
                    if 'long_close_count' in info_i:
                        long_close_count = int(info_i['long_close_count'])
                    if 'short_close_count' in info_i:
                        short_close_count = int(info_i['short_close_count'])
                    if 'total_fees' in info_i:
                        total_fees = float(info_i['total_fees'])
                    if 'episode_steps' in info_i:
                        episode_steps = int(info_i['episode_steps'])
                    if 'long_entry_count' in info_i:
                        long_entry_count = int(info_i['long_entry_count'])
                    if 'short_entry_count' in info_i:
                        short_entry_count = int(info_i['short_entry_count'])
                    if 'episode_max_steps' in info_i:
                        episode_max_steps = int(info_i['episode_max_steps'])
                    # 本 episode 止損與清算次數
                    if 'episode_stop_loss_count' in info_i:
                        episode_stop_loss_count = int(info_i['episode_stop_loss_count'])
                    if 'episode_liq_count' in info_i:
                        episode_liq_count = int(info_i['episode_liq_count'])
                    if 'episode_stats' in info_i:
                        stats = info_i['episode_stats']
                        total_outcome = float(stats.get('total_outcome', 0.0))
                        cost_prob_sum = float(stats.get('cost_prob_sum', 0.0))
                        forced_close_count = int(stats.get('forced_close_count', 0))
                        risk_cost_avg = float(stats.get('risk_cost_avg', 0.0))
                        struct_cost_avg = float(stats.get('struct_cost_avg', 0.0))
                        step_cost_avg = float(stats.get('avg_step_cost', 0.0))
                        trade_cost_rate = float(stats.get('trade_cost_rate', 0.0))
                        entry_cost_rate = float(stats.get('entry_cost_rate', 0.0))
                        drawdown_cost_avg = float(stats.get('drawdown_cost_avg', 0.0))
                        stop_streak_cost_avg = float(stats.get('stop_streak_cost_avg', 0.0))
                    else:
                        risk_cost_avg = float(info_i.get('constraint_risk_cost', 0.0))
                        struct_cost_avg = float(info_i.get('constraint_struct_cost', 0.0))
                        step_cost_avg = float(info_i.get('step_cost', 0.0))
                        trade_cost_rate = float(info_i.get('trade_count_cost', 0.0))
                        entry_cost_rate = float(info_i.get('entry_cost', 0.0))
                        drawdown_cost_avg = float(info_i.get('drawdown_cost', 0.0))
                        stop_streak_cost_avg = float(info_i.get('stop_loss_streak_cost', 0.0))
                    if reason == 'data_exhausted':
                        term_result = 'success(data_exhausted)'
                    elif reason is not None:
                        term_result = f"fail({reason})"
            except Exception:
                pass

            # 若 info 未提供，則從 env 讀取
            if (final_balance is None or profit_rate is None) and env is not None:
                try:
                    final_balance = float(env.total_value)
                    profit = final_balance - env.initial_balance
                    profit_rate = (profit / env.initial_balance) * 100
                except Exception:
                    final_balance = 0.0
                    profit_rate = 0.0

            # 記錄聚合回合資料
            self.episode_rewards.append(self.current_episode_reward[i])
            self.episode_lengths.append(int(self.current_episode_length[i]))
            self.episode_profit_rates.append(float(profit_rate))
            self.episode_failures.append(bool(failure_flag))
            self.episode_stop_losses.append(bool(stop_loss_flag))
            self.episode_liquidations.append(bool(liq_flag))
            self.episode_outcomes.append(float(total_outcome))
            self.episode_cost_probs.append(float(cost_prob_sum))
            self.episode_forced_closes.append(int(forced_close_count))
            self.episode_risk_costs.append(float(risk_cost_avg))
            self.episode_struct_costs.append(float(struct_cost_avg))
            self.episode_step_costs.append(float(step_cost_avg))
            self.episode_trade_costs.append(float(trade_cost_rate))
            self.episode_entry_costs.append(float(entry_cost_rate))
            self.episode_drawdown_costs.append(float(drawdown_cost_avg))
            self.episode_stop_streak_costs.append(float(stop_streak_cost_avg))

            episode_num = len(self.episode_rewards)
            if episode_num % self.log_interval == 0:
                print(
                    f"Episode {episode_num:4d} | "
                    f"Env#{i} | "
                    f"Steps: {episode_steps if episode_steps is not None else int(self.current_episode_length[i]):4d}/{episode_max_steps if episode_max_steps is not None else '?'} | "
                    f"Reward: {self.current_episode_reward[i]:8.2f} | "
                    f"Balance: {final_balance if final_balance is not None else 0.0:10.2f} | "
                    f"ProfitRate: {profit_rate if profit_rate is not None else 0.0:+.2f}% | "
                    f"Fees: {total_fees if total_fees is not None else 0.0:.4f} | "
                    f"Entries L/S: {long_entry_count if long_entry_count is not None else 0}/{short_entry_count if short_entry_count is not None else 0} | "
                    f"Stops: {episode_stop_loss_count} | Liqs: {episode_liq_count} | "
                    f"Result: {term_result}"
                )

            # 每 10 回合統計
            if episode_num % 10 == 0:
                last10_rates = self.episode_profit_rates[-10:]
                last10_fails = self.episode_failures[-10:]
                last10_stops = self.episode_stop_losses[-10:]
                last10_liqs = self.episode_liquidations[-10:]
                avg_rate_10 = float(np.mean(last10_rates)) if len(last10_rates) > 0 else 0.0
                total_fail_10 = int(np.sum(last10_fails)) if len(last10_fails) > 0 else 0
                total_stops_10 = int(np.sum(last10_stops)) if len(last10_stops) > 0 else 0
                total_liqs_10 = int(np.sum(last10_liqs)) if len(last10_liqs) > 0 else 0
                
                # RUDDER 統計
                rudder_stats = ""
                if len(self.episode_outcomes) > 0:
                    last10_outcomes = self.episode_outcomes[-10:]
                    avg_outcome_10 = float(np.mean(last10_outcomes)) if len(last10_outcomes) > 0 else 0.0
                    rudder_stats += f" | avg_outcome={avg_outcome_10:+.2f}"
                
                print(
                    f"[10-episode stats] avg_profit_rate={avg_rate_10:+.2f}% | "
                    f"failures={total_fail_10}/10 (stop_loss={total_stops_10}, liq={total_liqs_10})"
                    f"{rudder_stats}"
                )
            
            # Lagrangian 控制器更新（每 N 個 episode）
            if self.lagrangian_controller is not None and episode_num % self.lagrangian_update_freq == 0:
                self._update_lagrangian(episode_num)

            # 重置該環境的回合累積器（獨立回合）
            self.current_episode_reward[i] = 0.0
            self.current_episode_length[i] = 0
            self.env_episode_indices[i] += 1

        return True
    
    def _update_lagrangian(self, episode_num: int) -> None:
        """更新 Lagrangian 乘子（基於最近 N 個 episode 的統計）"""
        if self.lagrangian_controller is None:
            return
        
        # 計算最近 N 個 episode 的平均成本
        N = min(self.lagrangian_update_freq, len(self.episode_stop_losses))
        if N == 0:
            return
        
        # cost_prob: 止損或清算的比例
        recent_stops = self.episode_stop_losses[-N:]
        recent_liqs = self.episode_liquidations[-N:]
        cost_prob_samples = np.array([
            1.0 if (stop or liq) else 0.0
            for stop, liq in zip(recent_stops, recent_liqs)
        ]).reshape(-1, 1)
        
        # loss_cvar_sample: 從 outcome 計算（負 outcome 視為損失）
        if len(self.episode_outcomes) >= N:
            recent_outcomes = self.episode_outcomes[-N:]
            loss_cvar_samples = np.array([
                max(0.0, -outcome) for outcome in recent_outcomes
            ]).reshape(-1, 1)
        else:
            loss_cvar_samples = np.zeros((N, 1))

        if len(self.episode_risk_costs) >= N:
            recent_risk_costs = self.episode_risk_costs[-N:]
            risk_cost_samples = np.array(recent_risk_costs).reshape(-1, 1)
        else:
            risk_cost_samples = np.zeros((N, 1))

        if len(self.episode_struct_costs) >= N:
            recent_struct_costs = self.episode_struct_costs[-N:]
            struct_cost_samples = np.array(recent_struct_costs).reshape(-1, 1)
        else:
            struct_cost_samples = np.zeros((N, 1))

        if len(self.episode_step_costs) >= N:
            recent_step_costs = self.episode_step_costs[-N:]
            step_cost_samples = np.array(recent_step_costs).reshape(-1, 1)
        else:
            step_cost_samples = np.zeros((N, 1))

        if len(self.episode_trade_costs) >= N:
            recent_trade_costs = self.episode_trade_costs[-N:]
            trade_cost_samples = np.array(recent_trade_costs).reshape(-1, 1)
        else:
            trade_cost_samples = np.zeros((N, 1))

        if len(self.episode_entry_costs) >= N:
            recent_entry_costs = self.episode_entry_costs[-N:]
            entry_cost_samples = np.array(recent_entry_costs).reshape(-1, 1)
        else:
            entry_cost_samples = np.zeros((N, 1))

        if len(self.episode_drawdown_costs) >= N:
            recent_drawdown_costs = self.episode_drawdown_costs[-N:]
            drawdown_cost_samples = np.array(recent_drawdown_costs).reshape(-1, 1)
        else:
            drawdown_cost_samples = np.zeros((N, 1))

        if len(self.episode_stop_streak_costs) >= N:
            recent_stop_costs = self.episode_stop_streak_costs[-N:]
            stop_streak_cost_samples = np.array(recent_stop_costs).reshape(-1, 1)
        else:
            stop_streak_cost_samples = np.zeros((N, 1))
        
        # 更新 Lagrangian 乘子
        try:
            stats = self.lagrangian_controller.update(
                cost_prob_batch=cost_prob_samples,
                loss_cvar_batch=loss_cvar_samples,
                risk_cost_batch=risk_cost_samples,
                struct_cost_batch=struct_cost_samples,
                step_cost_batch=step_cost_samples,
                trade_cost_batch=trade_cost_samples,
                entry_cost_batch=entry_cost_samples,
                drawdown_cost_batch=drawdown_cost_samples,
                stop_streak_cost_batch=stop_streak_cost_samples,
            )
            
            if self.logger is not None:
                self.logger.record("lagrangian/lambda_prob", float(stats.get("lambda_prob", 0.0)))
                self.logger.record("lagrangian/lambda_cvar", float(stats.get("lambda_cvar", 0.0)))
                self.logger.record("lagrangian/lambda_risk", float(stats.get("lambda_risk", 0.0)))
                self.logger.record("lagrangian/lambda_struct", float(stats.get("lambda_struct", 0.0)))
                self.logger.record("lagrangian/lambda_step_cost", float(stats.get("lambda_step_cost", 0.0)))
                self.logger.record("lagrangian/lambda_trade", float(stats.get("lambda_trade_count", 0.0)))
                self.logger.record("lagrangian/lambda_entry", float(stats.get("lambda_entry", 0.0)))
                self.logger.record("lagrangian/lambda_drawdown", float(stats.get("lambda_drawdown", 0.0)))
                self.logger.record("lagrangian/lambda_stop_streak", float(stats.get("lambda_stop_streak", 0.0)))
                self.logger.record("lagrangian/violation_prob", float(stats.get("violation_prob", 0.0)))
                self.logger.record("lagrangian/violation_cvar", float(stats.get("violation_cvar", 0.0)))
                self.logger.record("lagrangian/violation_risk", float(stats.get("violation_risk", 0.0)))
                self.logger.record("lagrangian/violation_struct", float(stats.get("violation_struct", 0.0)))
                self.logger.record("lagrangian/violation_step_cost", float(stats.get("violation_step_cost", 0.0)))
                self.logger.record("lagrangian/violation_trade", float(stats.get("violation_trade_count", 0.0)))
                self.logger.record("lagrangian/violation_entry", float(stats.get("violation_entry_cost", 0.0)))
                self.logger.record("lagrangian/violation_drawdown", float(stats.get("violation_drawdown_cost", 0.0)))
                self.logger.record("lagrangian/violation_stop_streak", float(stats.get("violation_stop_streak_cost", 0.0)))
                self.logger.record("lagrangian/mean_risk_cost", float(stats.get("mean_risk_cost", 0.0)))
                self.logger.record("lagrangian/mean_struct_cost", float(stats.get("mean_struct_cost", 0.0)))
                self.logger.record("lagrangian/mean_step_cost", float(stats.get("mean_step_cost", 0.0)))
                self.logger.record("lagrangian/mean_trade_cost", float(stats.get("mean_trade_cost", 0.0)))
                self.logger.record("lagrangian/mean_entry_cost", float(stats.get("mean_entry_cost", 0.0)))
                self.logger.record("lagrangian/mean_drawdown_cost", float(stats.get("mean_drawdown_cost", 0.0)))
                self.logger.record("lagrangian/mean_stop_streak_cost", float(stats.get("mean_stop_streak_cost", 0.0)))
                self.logger.dump(step=self.num_timesteps)

            # 更新所有環境的 wrapper λ
            try:
                if hasattr(self.training_env, 'envs'):
                    # 向量環境
                    for env_wrapper in self.training_env.envs:
                        self._update_env_lambdas(env_wrapper, stats)
                else:
                    # 單環境
                    self._update_env_lambdas(self.training_env, stats)
            except Exception as e:
                print(f"[警告] 更新環境 λ 失敗: {e}")
            
            print(
                f"\n[Lagrangian Update @ Episode {episode_num}] "
                f"λ_prob={stats['lambda_prob']:.4f}, "
                f"λ_cvar={stats['lambda_cvar']:.4f}, "
                f"λ_risk={stats.get('lambda_risk', 0.0):.4f}, "
                f"λ_struct={stats.get('lambda_struct', 0.0):.4f}, "
                f"λ_step={stats.get('lambda_step_cost', 0.0):.4f}, "
                f"λ_trade={stats.get('lambda_trade_count', 0.0):.4f}, "
                f"λ_entry={stats.get('lambda_entry', 0.0):.4f}, "
                f"λ_drawdown={stats.get('lambda_drawdown', 0.0):.4f}, "
                f"λ_stop_streak={stats.get('lambda_stop_streak', 0.0):.4f} | "
                f"violation_prob={stats['violation_prob']:.4f}, "
                f"violation_cvar={stats['violation_cvar']:.4f}, "
                f"violation_risk={stats.get('violation_risk', 0.0):.4f}, "
                f"violation_struct={stats.get('violation_struct', 0.0):.4f}, "
                f"violation_step={stats.get('violation_step_cost', 0.0):.4f}, "
                f"violation_trade={stats.get('violation_trade_count', 0.0):.4f}, "
                f"violation_entry={stats.get('violation_entry_cost', 0.0):.4f}, "
                f"violation_drawdown={stats.get('violation_drawdown_cost', 0.0):.4f}, "
                f"violation_stop_streak={stats.get('violation_stop_streak_cost', 0.0):.4f}"
            )
        except Exception as e:
            print(f"[錯誤] Lagrangian 更新失敗: {e}")
    
    def _update_env_lambdas(self, env_wrapper, stats: dict) -> None:
        """遞迴查找並更新 RiskPenaltyWrapper 的 λ"""
        if hasattr(env_wrapper, 'update_lambdas'):
            env_wrapper.update_lambdas(
                lambda_prob=stats['lambda_prob'],
                lambda_cvar=stats['lambda_cvar'],
                lambda_risk=stats.get('lambda_risk'),
                lambda_struct=stats.get('lambda_struct'),
                lambda_step_cost=stats.get('lambda_step_cost'),
                lambda_trade_count=stats.get('lambda_trade_count'),
                lambda_entry=stats.get('lambda_entry'),
                lambda_drawdown=stats.get('lambda_drawdown'),
                lambda_stop_streak=stats.get('lambda_stop_streak'),
            )
        elif hasattr(env_wrapper, 'env'):
            self._update_env_lambdas(env_wrapper.env, stats)
        elif hasattr(env_wrapper, 'unwrapped'):
            # 最後嘗試 unwrapped
            pass
    
    def _on_training_end(self) -> None:
        """訓練結束時調用"""
        if len(self.episode_rewards) > 0:
            print(f"\n{'='*60}")
            print("訓練總結:")
            print(f"  總回合數: {len(self.episode_rewards)}")
            print(f"  平均獎勵: {np.mean(self.episode_rewards):.2f}")
            print(f"  最佳獎勵: {np.max(self.episode_rewards):.2f}")
            print(f"  最差獎勵: {np.min(self.episode_rewards):.2f}")
            if len(self.episode_profit_rates) > 0:
                print(f"  平均收益率: {np.mean(self.episode_profit_rates):+.2f}%")
            if len(self.episode_failures) > 0:
                print(f"  總失敗次數: {int(np.sum(self.episode_failures))}")
            print(f"{'='*60}\n")


class VerboseEvalCallback(EvalCallback):
    """
    評估時打印提示訊息的回調。
    每當達到 eval_freq 觸發評估時，會打印當前全域步數與設定。
    """
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.n_calls % self.eval_freq) == 0:
            try:
                print(f"\n----- 評估開始 (global_steps={self.num_timesteps}, eval_freq={self.eval_freq}, n_eval_episodes={self.n_eval_episodes}) -----")
            except Exception:
                print("\n----- 評估開始 -----")
        return super()._on_step()


class RestartingEvalCallback(BaseCallback):
    """
    自訂評估回調：固定頻率觸發評估，遇到非「數據用完」的終止會自動從頭重啟，並統計成功/失敗次數。
    成功: info['termination_reason'] == 'data_exhausted'
    失敗: 其他 terminated 或被 TimeLimit 截斷（truncated=True）
    """
    def __init__(self, eval_env, eval_freq: int = 10000, n_eval_episodes: int = 5, deterministic: bool = True, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = bool(deterministic)
        self.success_episodes = 0
        self.failure_episodes = 0
        self.truncated_episodes = 0

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.n_calls % self.eval_freq) == 0:
            print(f"\n----- 自訂評估開始 (global_steps={self.num_timesteps}, eval_freq={self.eval_freq}, n_eval_episodes={self.n_eval_episodes}) -----")
            self._run_evaluation()
        return True

    def _run_evaluation(self) -> None:
        success = 0
        failure = 0
        truncated_cnt = 0
        episodes = 0
        try:
            while episodes < self.n_eval_episodes:
                obs, _ = self.eval_env.reset()
                while True:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    if terminated or truncated:
                        if truncated:
                            truncated_cnt += 1
                            failure += 1
                        else:
                            reason = info.get('termination_reason') if isinstance(info, dict) else None
                            if reason == 'data_exhausted':
                                success += 1
                            else:
                                failure += 1
                        episodes += 1
                        break
        except Exception:
            print("\n[錯誤位置] 評估 (evaluation) 發生例外，即將拋出")
            raise
        self.success_episodes += success
        self.failure_episodes += failure
        self.truncated_episodes += truncated_cnt
        print(f"自訂評估結果: 本輪 success={success}, failure={failure}, truncated={truncated_cnt} | 累計 success={self.success_episodes}, failure={self.failure_episodes}")

class TerminationAwareEvalCallback(EvalCallback):
    """
    自訂評估：
    - 若 done 且 info['termination_reason'] != 'data_exhausted'，視為失敗，重啟一個新 episode，累計失敗計數
    - 若 done 且為 'data_exhausted'，視為成功，累計成功計數
    - 可與 TimeLimit 搭配使用，步數達上限時 SB3 會設 truncated；此處以 terminated/done 為主
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.success_episodes = 0
        self.failure_episodes = 0

    def _on_step(self) -> bool:
        return super()._on_step()

    def _on_event(self) -> None:
        # 在單輪 evaluate_policy 完成後被調用，累計成功/失敗次數
        try:
            # 最近一次評估的 episode info 存在 self.last_eval_ep_info (SB3 內部不提供)
            # 這裡退一步：根據評估環境的最後 done 時印出的訊息做統計
            pass
        except Exception:
            pass


def create_environment(
    data_path: str,
    initial_balance: float = 10000.0,
    min_balance: float = 100.0,
    leverage: float = 10.0,
    transaction_fee: float = 0.001,
    window_size: int = 288,
    margin_mode: str = 'isolated',
    start_date: str | None = None,
    end_date: str | None = None,
    random_start: bool = False,
    use_rudder: bool = True,
    use_lagrangian_wrapper: bool = True,
    stop_loss_cooldown_window: int = 6,
    stop_loss_cooldown_length: int = 12,
    stop_loss_cooldown_limit: int = 2,
    cooldown_hold_ratio: float = 0.0,
) -> TradingEnvironment:
    """
    創建交易環境
    
    Args:
        data_path: 數據路徑
        initial_balance: 初始資金
        leverage: 槓桿倍數
        transaction_fee: 交易手續費率
        window_size: 觀察窗口大小
        reward_mode: 獎勵模式
        margin_mode: 保證金模式
        stop_loss_cooldown_window: 停損冷靜期觸發時計算連續止損的滑動窗口（步數）
        stop_loss_cooldown_length: 停損冷靜期啟動後暫停進場的期間（步數）
        stop_loss_cooldown_limit: 滑動窗口內的止損觸發上限
        cooldown_hold_ratio: 冷靜期內維持的持倉比例
        
    Returns:
        交易環境實例
    """
    # 加載數據
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"數據文件不存在: {data_path}")
    
    print(f"加載數據: {data_path}")
    df = pd.read_csv(data_path)

    # 可選：依日期範圍過濾
    if start_date is not None or end_date is not None:
        dt_col = None
        for candidate in ('datetime', 'date', 'time', 'timestamp'):
            if candidate in df.columns:
                dt_col = candidate
                break
        if dt_col is None:
            print("警告: 指定了日期範圍，但資料無日期欄位，跳過日期過濾")
        else:
            df[dt_col] = pd.to_datetime(df[dt_col])
            if start_date is not None:
                df = df[df[dt_col] >= pd.to_datetime(start_date)]
            if end_date is not None:
                df = df[df[dt_col] <= pd.to_datetime(end_date)]
            df = df.reset_index(drop=True)
            print(f"已按照日期範圍過濾: start={start_date}, end={end_date}, 形狀: {df.shape}")

    print(f"數據形狀: {df.shape}")
    
    # 創建環境（支援 RUDDER + Lagrangian）
    if use_rudder:
        print(f"獎勵模式: SAC-Lagrangian + RUDDER")
        print(f"  PBRS 勢能: survival=0.5, struct=0.2, extreme_entry=0.3")
        print(f"  Shaping 權重: risk=20.0, struct=6.0, survival=0.5, entry=0.0 (移轉至成本線)")
        print(f"  Outcome 權重: w_outcome=1.0, stop_loss_penalty=50.0, liq_penalty=90.0")
        print(f"  Lagrangian: target_prob=0.03 (3%), target_cvar=0.01 (1%)")
        
        env = TradingEnvironment(
            df=df,
            initial_balance=initial_balance,
            transaction_fee=transaction_fee,
            window_size=window_size,
            leverage=leverage,
            min_balance=min_balance,
            min_trade_qty=0.001,
            margin_mode=margin_mode,
            use_rudder=True,
            random_start=random_start,
            max_daily_trades=6,
            stop_loss_cooldown_window=stop_loss_cooldown_window,
            stop_loss_cooldown_length=stop_loss_cooldown_length,
            stop_loss_cooldown_limit=stop_loss_cooldown_limit,
            cooldown_hold_ratio=cooldown_hold_ratio,
        )
        
        # 添加 Wrapper
        if use_lagrangian_wrapper:
            env = RiskPenaltyWrapper(env, lambda_prob=1.0, lambda_cvar=1.0, lambda_entry=0.0)
            env = RiskPenaltyWrapper(env, lambda_prob=1.0, lambda_cvar=1.0, lambda_entry=0.0)
        env = InfoLoggerWrapper(env)
    else:
        # 舊版（向後兼容）
        reward_calculator = create_default_calculator()
        info = reward_calculator.get_info()
        print(f"獎勵配置: {info['type']} (舊版)")
        print(f"  權重 - stop_loss:{info['weights']['stop_loss']}, terminal:{info['weights']['terminal']}, return:{info['weights']['return']}, risk:{info['weights']['risk']}, struct:{info['weights']['struct']}")
        print(f"  止損規則: 2.5 ATR | 尺度 - return_clip:{info['scales']['return_clip']}, risk_threshold:{info['scales']['risk_threshold']}")
        
        env = TradingEnvironment(
            df=df,
            initial_balance=initial_balance,
            transaction_fee=transaction_fee,
            window_size=window_size,
            leverage=leverage,
            min_balance=min_balance,
            min_trade_qty=0.001,
            margin_mode=margin_mode,
            reward_calculator=reward_calculator,
            use_rudder=False,
            random_start=random_start,
            max_daily_trades=12,
            stop_loss_cooldown_window=stop_loss_cooldown_window,
            stop_loss_cooldown_length=stop_loss_cooldown_length,
            stop_loss_cooldown_limit=stop_loss_cooldown_limit,
            cooldown_hold_ratio=cooldown_hold_ratio,
        )
    
    print(f"環境創建成功:")
    print(f"  觀察空間: {env.observation_space.shape}")
    print(f"  最小資金: {min_balance}")
    print(f"  動作空間: {env.action_space.shape}")
    print(f"  初始資金: {initial_balance}")
    print(f"  槓桿倍數: {leverage}")
    print(f"  保證金模式: {margin_mode}")
    
    return env


def train_sac(
    env: TradingEnvironment,
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    buffer_size: int = 100000,
    learning_starts: int = 1000,
    batch_size: int = 256,
    tau: float = 0.005,
    gamma: float = 0.995,
    model_dir: str = './models',
    log_dir: str = './logs',
    save_freq: int = 10000,
    eval_freq: int = 10000,
    eval_episodes: int = 100,
    device: str = 'auto',
    load_model: str | None = None,
    n_envs: int = 1,
    train_data_path: str | None = None,
    eval_data_path: str | None = None,
    eval_start_date: str | None = None,
    eval_end_date: str | None = None,
    eval_max_steps: int | None = None,
    use_rudder: bool = True,
    use_lagrangian: bool = True,
    lagrangian_update_freq: int = 100,
) -> SAC:
    """
    訓練 SAC 模型
    
    Args:
        env: 訓練環境
        total_timesteps: 總訓練步數
        learning_rate: 學習率
        buffer_size: 經驗回放緩衝區大小
        learning_starts: 開始學習前的隨機步數
        batch_size: 批次大小
        tau: 軟更新係數
        gamma: 折扣因子
        model_dir: 模型保存目錄
        log_dir: 日誌目錄
        save_freq: 保存頻率
        eval_freq: 評估頻率
        eval_episodes: 評估回合數
        device: 訓練設備
        load_model: 加載已有模型路徑
        
    Returns:
        訓練好的 SAC 模型
    """
    # 創建目錄
    model_dir = Path(model_dir)
    log_dir = Path(log_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 檢查設備
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用設備: {device}")
    
    # 建立向量化或單環境
    monitor_dir_path = log_dir / 'monitor'
    monitor_dir_path.mkdir(parents=True, exist_ok=True)

    if n_envs is not None and int(n_envs) > 1:
        print(f"建立向量化環境: n_envs={int(n_envs)} (SubprocVecEnv)")
        
        # 從 wrapped env 中提取參數
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        if hasattr(base_env, 'unwrapped'):
            base_env = base_env.unwrapped
        
        factory = TradingEnvFactory(
            data_path=(train_data_path or eval_data_path or './Data/BTCUSDT_futures_volume_5years_5min.csv'),
            initial_balance=base_env.initial_balance,
            transaction_fee=base_env.transaction_fee,
            window_size=base_env.window_size,
            leverage=base_env.leverage,
            min_balance=base_env.min_balance,
            min_trade_qty=base_env.min_trade_qty,
            margin_mode=base_env.margin_mode,
            random_start=True,
            use_rudder=use_rudder,
            use_lagrangian_wrapper=(use_rudder and use_lagrangian),
            stop_loss_cooldown_window=base_env.stop_loss_cooldown_window,
            stop_loss_cooldown_length=base_env.stop_loss_cooldown_length,
            stop_loss_cooldown_limit=base_env.stop_loss_cooldown_limit,
            cooldown_hold_ratio=base_env.cooldown_hold_ratio,
        )
        vec_env = make_vec_env(factory, n_envs=int(n_envs), vec_env_cls=SubprocVecEnv, monitor_dir=str(monitor_dir_path))
        env_for_training = vec_env
    else:
        env_for_training = Monitor(env)
    
    # 創建或加載模型
    if load_model and Path(load_model).exists():
        print(f"加載模型: {load_model}")
        model = SAC.load(load_model, env=env_for_training, device=device)
        print("模型加載成功")
    else:
        print("創建新模型（SAC + LSTM）...")
        model = SAC(
            policy=RecurrentSACPolicy,
            env=env_for_training,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=1,
            gradient_steps=1,  # 降低梯度步數，提升穩定性
            ent_coef='auto',
            target_entropy=-0.5,  # action_dim=1 的適當 target entropy
            verbose=1,
            device=device,
            tensorboard_log=str(log_dir),
            policy_kwargs={
                'lstm_hidden_size': 64,   # 降低以節省記憶體
                'lstm_layers': 1,         # 單層 LSTM
                'net_arch': [128, 128],   # 較小的 MLP
            },
        )
        print("模型創建成功（SAC + LSTM, hidden=64, layers=1）")
    
    # 創建回調
    callbacks = []
    
    info_log_keys = [
        'reward_shaping_total',
        'reward_shaping_pbrs',
        'reward_shaping_return',
        'reward_shaping_survival',
        'constraint_risk_cost',
        'constraint_struct_cost',
        'step_cost',
        'cost_prob',
    ]
    callbacks.append(
        InfoTensorboardCallback(
            log_keys=info_log_keys,
            prefix='env',
            smooth=0.9,
        )
    )

    # 創建 Lagrangian 控制器（若啟用）
    lagrangian_controller = None
    if use_rudder and use_lagrangian:
        trade_target_per_step = 30.0 / 288.0
        entry_target_per_step = trade_target_per_step
        entry_penalty = 0.6
        base_env_for_target = env
        try:
            while hasattr(base_env_for_target, 'env'):
                base_env_for_target = base_env_for_target.env
            if hasattr(base_env_for_target, 'unwrapped'):
                base_env_for_target = base_env_for_target.unwrapped
            cost_calc_attr = getattr(base_env_for_target, 'cost_calc', None)
            if cost_calc_attr is not None:
                if hasattr(cost_calc_attr, 'trade_target_per_step'):
                    trade_target_per_step = float(cost_calc_attr.trade_target_per_step)
                if hasattr(cost_calc_attr, 'entry_target_per_step'):
                    entry_target_per_step = float(cost_calc_attr.entry_target_per_step)
                if hasattr(cost_calc_attr, 'entry_penalty'):
                    entry_penalty = float(cost_calc_attr.entry_penalty)
        except Exception:
            pass

        lagrangian_controller = LagrangianController(
            lambda_prob_init=1.0,              # 初始 λ_prob：止損/強平違規的懲罰強度起點
            lambda_cvar_init=1.0,              # 初始 λ_cvar：尾損（CVaR）懲罰起點
            lambda_risk_init=0.0,              # 初始 λ_risk：margin buffer 違規懲罰，先從 0 起避免早期干預
            lambda_struct_init=0.0,            # 初始 λ_struct：結構違規（MAE/極值距離）懲罰起點
            lambda_step_cost_init=0.0,         # 初始 λ_step_cost：交易成本（手續費/滑點）懲罰起點
            lambda_trade_count_init=0.0,       # 初始 λ_trade_count：交易次數懲罰（目前主要依硬限制）
            lambda_entry_init=0.0,             # 初始 λ_entry：進場次數懲罰，0 表示先不施壓
            lambda_drawdown_init=0.0,          # 初始 λ_drawdown：最大回撤懲罰
            lambda_stop_streak_init=0.0,       # 初始 λ_stop_streak：連續止損懲罰
            lr_prob=0.01,                      # λ_prob 更新速率：違規超標時提升懲罰的梯度步幅
            lr_cvar=0.01,                      # λ_cvar 更新速率：控制尾損懲罰調整速度
            lr_risk=0.005,                     # λ_risk 更新速率：較低以免 margin buffer 波動時過度放大懲罰
            lr_struct=0.005,                   # λ_struct 更新速率：限制結構指標的調整幅度
            lr_step_cost=0.01,                 # λ_step_cost 更新速率：費用超標時提高懲罰的速度
            lr_trade_count=0.01,               # λ_trade_count 更新速率：若重啟交易次數成本可快速調整
            lr_entry=0.01,                     # λ_entry 更新速率：進場頻率超標時懲罰增長速度
            lr_drawdown=0.01,                  # λ_drawdown 更新速率
            lr_stop_streak=0.01,               # λ_stop_streak 更新速率
            target_prob=0.20,                  # 允許的止損/強平平均比例上限（暫時放寬）
            target_cvar=0.50,                  # 允許的 CVaR 尾損平均上限（暫時放寬）
            target_risk=0.10,                  # 允許 margin buffer 違規指標平均值（>0 代表容許一定程度風險）
            target_struct=0.10,                # 允許結構違規（MAE/極值距離）平均值
            target_step_cost=trade_target_per_step,        # 交易成本目標：對應每日 30 筆成本換算到單步
            target_trade_count=trade_target_per_step,      # 交易次數目標：每日 30 筆 → 單步期望
            target_entry_cost=entry_target_per_step * entry_penalty,  # 進場成本目標：每日 30 筆進場 * 單筆成本
            target_drawdown_cost=0.02,          # 最大回撤成本目標（允許少量超標）
            target_stop_streak_cost=0.0,        # 連續止損成本目標
            lambda_min=0.001,                  # λ 下界，避免完全為 0 時沒有懲罰信號
            lambda_max=100.0,                  # λ 上界，防止乘子無限制爆炸
        )
        print(f"\n已啟用 Lagrangian 控制器:")
        print(f"  target_prob={lagrangian_controller.target_prob:.2%} (機率違規目標)")
        print(f"  target_cvar={lagrangian_controller.target_cvar:.2%} (CVaR 尾損目標)")
        print(f"  target_trade_per_step={trade_target_per_step:.4f} (~{trade_target_per_step * 288:.1f}/天)")
        print(f"  target_entry_per_step={entry_target_per_step:.4f} (~{entry_target_per_step * 288:.1f}/天) (權重 {entry_penalty})")
        print(f"  target_drawdown_cost={lagrangian_controller.target_drawdown_cost:.4f}")
        print(f"  target_stop_streak_cost={lagrangian_controller.target_stop_streak_cost:.4f}")
        print(f"  update_freq={lagrangian_update_freq} episodes")
    
    # 訓練日誌回調
    training_callback = TradingCallback(
        log_interval=1,
        print_step_episode=True,
        lagrangian_controller=lagrangian_controller,
        lagrangian_update_freq=lagrangian_update_freq,
    )
    callbacks.append(training_callback)
    
    # 檢查點回調
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(model_dir),
        name_prefix='sac_trading'
    )
    callbacks.append(checkpoint_callback)
    
    # 評估回調：若提供 eval_start/end_date 則使用指定切片，否則複製訓練切片
    if eval_freq and eval_freq > 0:
        if eval_start_date is not None or eval_end_date is not None or eval_data_path is not None:
            # 從 wrapped env 中提取參數
            base_env = env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            if hasattr(base_env, 'unwrapped'):
                base_env = base_env.unwrapped
            
            eval_env = create_environment(
                data_path=eval_data_path or './Data/BTCUSDT_futures_volume_5years_5min.csv',
                initial_balance=base_env.initial_balance,
                min_balance=base_env.min_balance,
                leverage=base_env.leverage,
                transaction_fee=base_env.transaction_fee,
                window_size=base_env.window_size,
                margin_mode=base_env.margin_mode,
                start_date=eval_start_date,
                end_date=eval_end_date,
                random_start=False,
                use_rudder=use_rudder,
                use_lagrangian_wrapper=False,  # 評估時不使用 Lagrangian wrapper
                stop_loss_cooldown_window=base_env.stop_loss_cooldown_window,
                stop_loss_cooldown_length=base_env.stop_loss_cooldown_length,
                stop_loss_cooldown_limit=base_env.stop_loss_cooldown_limit,
                cooldown_hold_ratio=base_env.cooldown_hold_ratio,
            )
            if eval_max_steps is not None and eval_max_steps > 0:
                eval_env = TimeLimit(eval_env, max_episode_steps=int(eval_max_steps))
            eval_env = Monitor(eval_env)
        else:
            if hasattr(env_for_training, 'get_attr'):
                initial_balance = env_for_training.get_attr('initial_balance')[0]
                transaction_fee = env_for_training.get_attr('transaction_fee')[0]
                window_size = env_for_training.get_attr('window_size')[0]
                leverage = env_for_training.get_attr('leverage')[0]
                min_balance = env_for_training.get_attr('min_balance')[0]
                min_trade_qty = env_for_training.get_attr('min_trade_qty')[0]
                margin_mode = env_for_training.get_attr('margin_mode')[0]
                stop_loss_cooldown_window = env_for_training.get_attr('stop_loss_cooldown_window')[0]
                stop_loss_cooldown_length = env_for_training.get_attr('stop_loss_cooldown_length')[0]
                stop_loss_cooldown_limit = env_for_training.get_attr('stop_loss_cooldown_limit')[0]
                cooldown_hold_ratio = env_for_training.get_attr('cooldown_hold_ratio')[0]
                eval_env = create_environment(
                    data_path=eval_data_path or './Data/BTCUSDT_futures_volume_5years_5min.csv',
                    initial_balance=initial_balance,
                    min_balance=min_balance,
                    leverage=leverage,
                    transaction_fee=transaction_fee,
                    window_size=window_size,
                    margin_mode=margin_mode,
                    random_start=False,
                    stop_loss_cooldown_window=stop_loss_cooldown_window,
                    stop_loss_cooldown_length=stop_loss_cooldown_length,
                    stop_loss_cooldown_limit=stop_loss_cooldown_limit,
                    cooldown_hold_ratio=cooldown_hold_ratio,
                )
            else:
                base_env = env.unwrapped
                eval_env = TradingEnvironment(
                    df=base_env.df.copy(),
                    initial_balance=base_env.initial_balance,
                    transaction_fee=base_env.transaction_fee,
                    window_size=base_env.window_size,
                    leverage=base_env.leverage,
                    min_balance=base_env.min_balance,
                    min_trade_qty=base_env.min_trade_qty,
                    margin_mode=base_env.margin_mode,
                    reward_calculator=base_env.reward_calculator,  # 直接使用相同的獎勵計算器
                    random_start=False,
                    stop_loss_cooldown_window=base_env.stop_loss_cooldown_window,
                    stop_loss_cooldown_length=base_env.stop_loss_cooldown_length,
                    stop_loss_cooldown_limit=base_env.stop_loss_cooldown_limit,
                    cooldown_hold_ratio=base_env.cooldown_hold_ratio,
                )
            if eval_max_steps is not None and eval_max_steps > 0:
                eval_env = TimeLimit(eval_env, max_episode_steps=int(eval_max_steps))
            eval_env = Monitor(eval_env)

        # 安全性檢查：評估資料長度需大於 window_size+1，否則跳過評估
        try:
            eval_steps_available = len(eval_env.unwrapped.df) - eval_env.unwrapped.window_size - 1
        except Exception:
            eval_steps_available = 0

        if eval_steps_available <= 0:
            print("[評估跳過] 評估資料不足（長度 <= window_size+1），已略過本次評估以避免錯誤")
        else:
            # 使用自訂可重啟評估回調
            eval_callback = RestartingEvalCallback(
                eval_env=eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=min(eval_episodes, 3) if eval_steps_available < 100 else eval_episodes,
                deterministic=True,
                verbose=1
            )
            callbacks.append(eval_callback)
    
    # 開始訓練
    print(f"\n{'='*60}")
    print(f"開始訓練 - 總步數: {total_timesteps}")
    print(f"{'='*60}\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=None,
            progress_bar=True
        )
    except Exception as e:
        print("\n[錯誤位置] 訓練 (learn) 發生例外，即將拋出")
        raise
    
    # 保存最終模型
    final_model_path = model_dir / 'final_model.zip'
    model.save(str(final_model_path))
    print(f"\n最終模型已保存至: {final_model_path}")
    
    return model


def main() -> None:
    """主函數"""
    parser = argparse.ArgumentParser(description='SAC 交易模型訓練（SB3）')
    
    # 訓練參數（支持回合或步數）
    parser.add_argument('--timesteps', type=int, default=10_000_000,
                       help='總訓練步數（若未提供，將使用回合模式）')
    parser.add_argument('--random_start', action='store_true', default=False,
                       help='啟用回合隨機起點（每回合從隨機時間開始）')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'quick_test'],
                       help='運行模式')
    parser.add_argument('--n_envs', type=int, default=40,
                       help='並行環境數量（>1 啟用多進程）')
    
    # 數據和路徑
    parser.add_argument('--data', type=str, 
                       default='./Data/BTCUSDT_futures_volume_5years_5min.csv',
                       help='訓練數據路徑')
    parser.add_argument('--start_date', type=str, default='2020-01-01',
                       help='訓練資料開始日期（YYYY-MM-DD 或可解析字串）')
    parser.add_argument('--end_date', type=str, default='2025-09-30',
                       help='訓練資料結束日期（YYYY-MM-DD 或可解析字串）')
    parser.add_argument('--model_dir', type=str, default='./models',
                       help='模型保存目錄')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日誌目錄')
    
    # 模型參數
    parser.add_argument('--load_model', type=str, default=None,
                       help='加載已有模型路徑（繼續訓練）')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='訓練設備')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='學習率')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='批次大小')
    parser.add_argument('--buffer_size', type=int, default=3_000_000 ,
                       help='經驗回放緩衝區大小')
    
    # 環境參數
    parser.add_argument('--margin_mode', type=str, default='isolated',
                       choices=['isolated', 'cross'],# isolated: 隔離保證金(逐倉), cross: 交叉保證金(全倉)
                       help='保證金模式')
    parser.add_argument('--leverage', type=float, default=10.0,
                       help='槓桿倍數')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                       help='初始資金')
    parser.add_argument('--min_balance', type=float, default=100.0,
                       help='最小資金')
    parser.add_argument('--fee_rate', type=float, default=0.01,
                       help='交易手續費率')
    parser.add_argument('--window_size', type=int, default= 24 * 60 //5 // 12,#5min bars * 24 hours * 7 days = 288 bars * 7 days = 2016 bars
                       help='觀察窗口大小')
    parser.add_argument('--stop_loss_cooldown_window', type=int, default=6,#6步 = 30分鐘
                       help='停損冷靜期觸發的滑動窗口步數')
    parser.add_argument('--stop_loss_cooldown_length', type=int, default=36,#12步 = 1小時
                       help='停損冷靜期啟動後暫停進場的步數長度')
    parser.add_argument('--stop_loss_cooldown_limit', type=int, default=2,
                       help='滑動窗口內允許的最大止損次數（達到即啟動冷靜期）')
    parser.add_argument('--cooldown_hold_ratio', type=float, default=0.0,
                       help='停損冷靜期內維持的持倉比例（0 代表保持空倉）')
    
    # RUDDER + Lagrangian 參數
    parser.add_argument('--use_rudder', action='store_true', default=True,
                       help='啟用 RUDDER reward 系統')
    parser.add_argument('--no_rudder', action='store_false', dest='use_rudder',
                       help='禁用 RUDDER（使用舊版 reward）')
    parser.add_argument('--use_lagrangian', action='store_true', default=True,
                       help='啟用 Lagrangian 約束控制')
    parser.add_argument('--no_lagrangian', action='store_false', dest='use_lagrangian',
                       help='禁用 Lagrangian')
    parser.add_argument('--lagrangian_update_freq', type=int, default=25,
                       help='Lagrangian 更新頻率（每 N 個 episode）')
    
    # 評估資料設定
    parser.add_argument('--eval_use_last_month', action='store_true',
                       help='使用最新1個月作為評估資料集（自動計算日期範圍）')
    parser.add_argument('--eval_start_date', type=str, default='2025-10-01',
                       help='評估資料開始日期（YYYY-MM-DD）')
    parser.add_argument('--eval_end_date', type=str, default='2025-10-31',
                       help='評估資料結束日期（YYYY-MM-DD）')
    parser.add_argument('--eval_max_steps', type=int, default=2880,
                       help='每個評估回合的最大片長（步數），超過即截斷')
    parser.add_argument('--eval_freq', type=int, default=1_000_000,
                       help='評估頻率（步數）')
    args = parser.parse_args()
    
    # 快速測試模式
    if args.mode == 'quick_test':
        print("\n=== 快速測試模式 ===")
        args.timesteps = 10000
        args.buffer_size = 10000
        args.window_size = 100
        print("已調整參數為快速測試模式\n")
    
    print("\n" + "="*60)
    print("SAC 交易模型訓練系統（Stable-Baselines3）")
    print("="*60 + "\n")
    
    try:
        # 創建環境
        # 1) timesteps 模式：一次性以總步數訓練

        if args.timesteps is not None:
            env = create_environment(
                data_path=args.data,
                initial_balance=args.initial_balance,
                min_balance=args.min_balance,
                leverage=args.leverage,
                transaction_fee=args.fee_rate,
                window_size=args.window_size,
                start_date=args.start_date,
                end_date=args.end_date,
                random_start=args.random_start,
                use_rudder=args.use_rudder,
                use_lagrangian_wrapper=(args.use_rudder and args.use_lagrangian),
                stop_loss_cooldown_window=args.stop_loss_cooldown_window,
                stop_loss_cooldown_length=args.stop_loss_cooldown_length,
                stop_loss_cooldown_limit=args.stop_loss_cooldown_limit,
                cooldown_hold_ratio=args.cooldown_hold_ratio,
            )

            if args.eval_use_last_month:
                # 從來源 CSV 推算最新日期
                df_tmp = pd.read_csv(args.data)
                dt_col = None
                for candidate in ('datetime', 'date', 'time', 'timestamp'):
                    if candidate in df_tmp.columns:
                        dt_col = candidate
                        break
                if dt_col is not None:
                    df_tmp[dt_col] = pd.to_datetime(df_tmp[dt_col])
                    max_dt = df_tmp[dt_col].max()
                    eval_end_date = max_dt.strftime('%Y-%m-%d')
                    eval_start_date = (max_dt - timedelta(days=30)).strftime('%Y-%m-%d')
                    print(f"使用最新1個月作評估: {eval_start_date} ~ {eval_end_date}")
                else:
                    print("警告: 資料無日期欄位，無法自動計算最新1個月評估區間")
            else:
                eval_start_date = args.eval_start_date
                eval_end_date = args.eval_end_date

            model = train_sac(
                env=env,
                total_timesteps=args.timesteps,#總訓練步數
                learning_rate=args.lr,#學習率
                learning_starts=args.batch_size * 1000,#開始學習前的隨機步數(batch_size * 1000 = 256000)
                buffer_size=args.buffer_size,#經驗回放緩衝區大小
                batch_size=args.batch_size,#批次大小
                model_dir=args.model_dir,#模型保存目錄
                log_dir=args.log_dir,#日誌目錄
                device=args.device,#訓練設備
                load_model=args.load_model,#加載已有模型路徑
                n_envs=args.n_envs,
                train_data_path=args.data,
                eval_data_path=args.data,#評估數據路徑
                eval_start_date=eval_start_date,#評估資料開始日期
                eval_end_date=eval_end_date,#評估資料結束日期
                eval_max_steps=args.eval_max_steps,#每個評估回合的最大片長（步數），超過即截斷
                eval_freq=args.eval_freq,#評估頻率（步數）
                use_rudder=args.use_rudder,#啟用 RUDDER
                use_lagrangian=args.use_lagrangian,#啟用 Lagrangian
                lagrangian_update_freq=args.lagrangian_update_freq,#Lagrangian 更新頻率
            )
       
        print(f"\n{'='*60}")
        print("訓練完成！")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print("\n\n訓練被用戶中斷")
        
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

