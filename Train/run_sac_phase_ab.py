"""
Phase A / Phase B SAC 訓練入口。

Phase A：關閉 regime 硬投影與過多 hard override，讓 policy 直接控制倉位。
Phase B：在 Phase A 基礎上只加 cost_risk 懲罰（不把 cost_trade_freq / cost_flat 放進 reward）。

使用方式：
    python -m Train.run_sac_phase_ab --phase A --timesteps 300000
    python -m Train.run_sac_phase_ab --phase B --timesteps 300000 --lambda-risk 0.05 --reward-scale 10

TensorBoard 儀表板（僅顯示實際 PnL 兩條，方便觀察）：
    1) 訓練時寫入 log（指定目錄）：
       python -m Train.run_sac_phase_ab --phase B --tb-log logs/phase_b
    2) 啟動 TensorBoard：tensorboard --logdir=logs/phase_b
    3) 瀏覽器 http://localhost:6006 只會看到 episode_stats/log_return_sum_mean、episode_stats/final_balance_mean。
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor

from Env.trading_env import TradingEnvironment
from Eval.eval_triggers import (
    CompositeTrigger,
    EvalTriggerContext,
    ExpressionTrigger,
    MetricRule,
    MetricTrigger,
    StepTrigger,
)
from Eval.phase_ab_evaluator import PhaseABEvaluator, build_single_env_builder


# ---------------------------------------------------------------------------
# Phase A: 關閉 regime action 硬投影
# ---------------------------------------------------------------------------


class TradingEnvPhaseA(TradingEnvironment):
    """
    Phase A 環境：不對 action 做 regime 投影，讓 policy 輸出直接進入 ActionProcessor。
    """

    def _apply_regime_action_projection(self, action: np.ndarray) -> np.ndarray:
        """不做投影，直接回傳 action。"""
        return action


# ---------------------------------------------------------------------------
# Phase B: 只吃 cost_risk 的懲罰 Wrapper
# ---------------------------------------------------------------------------


class RiskOnlyPenaltyWrapper(gym.Wrapper):
    """
    Phase B：reward_mod = reward_scale * reward - lambda_risk * cost_risk - lambda_buffer * cost_risk_dense - lambda_fee * cost_fric。
    不把 cost_trade_freq / cost_flat 放進懲罰，避免主線被約束吞掉。
    cost_risk 為事件型（死亡）；cost_risk_dense 為每步 dense 緩衝懲罰（方案 B 獨立通道）。
    cost_fric 為摩擦成本（手續費相關）；lambda_fee_max 是「手續費懲罰」的權重，不是環境的每筆交易手續費率
    （每筆手續費率由環境的 transaction_fee / Config.TRANSACTION_FEE 決定）。

    註：episode_trade_count 約等於 execution_rate × episode_steps。同一資料集下 episode 長度由
    max_episode_steps（288*31）與 random_start 決定，分佈固定，故若策略學到的「交易頻率」相近，
    平均交易次數會落在相近區間。若要顯著改變交易次數：可提高 lambda_fee_max、在 reward 中加
    trade_count 懲罰、或啟用 trade_freq_window_steps / trade_freq_cost_limit 做硬性上限。
    """

    def __init__(
        self,
        env: gym.Env,
        lambda_risk: float = 0.05,
        lambda_buffer: float = 0.1,
        reward_scale: float = 10.0,
        lambda_fee_max: float = 0.0,
        fee_anneal_steps: int = 0,
    ) -> None:
        super().__init__(env)
        self.lambda_risk = float(lambda_risk)
        self.lambda_buffer = float(lambda_buffer)
        self.reward_scale = float(reward_scale)
        # 手續費「懲罰」權重退火（非 transaction_fee）：lambda_fee(t) 從 0 線性升到 lambda_fee_max
        self.lambda_fee_max = float(lambda_fee_max)
        self.fee_anneal_steps = max(0, int(fee_anneal_steps))

    def _current_lambda_fee(self) -> float:
        """
        依當前 training_timestep 決定手續費懲罰權重。

        規則：
        - fee_anneal_steps <= 0：不退火，恆為 lambda_fee_max
        - 0 < step < fee_anneal_steps：線性從 0 升到 lambda_fee_max
        - step >= fee_anneal_steps：固定為 lambda_fee_max
        """
        if self.lambda_fee_max <= 0.0:
            return 0.0
        if self.fee_anneal_steps <= 0:
            return self.lambda_fee_max
        step = int(getattr(self, "training_timestep", 0) or 0)
        if step <= 0:
            return 0.0
        factor = max(0.0, min(1.0, float(step) / float(self.fee_anneal_steps)))
        return self.lambda_fee_max * factor

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        cost_risk = float(info.get("cost_risk", 0.0))
        cost_risk_dense = float(info.get("cost_risk_dense", 0.0))
        # 手續費相關成本（由 CostCalculator.compute 輸出），若不存在則視為 0
        cost_fric = float(info.get("cost_fric", 0.0))
        lambda_fee = self._current_lambda_fee()
        reward_mod = (
            self.reward_scale * float(reward)
            - self.lambda_risk * cost_risk
            - self.lambda_buffer * cost_risk_dense
            - lambda_fee * cost_fric
        )
        info["reward_raw"] = float(reward)
        info["reward_mod"] = float(reward_mod)
        info["cost_risk_used"] = cost_risk
        info["cost_risk_dense_used"] = cost_risk_dense
        info["cost_fric_used"] = cost_fric
        info["lambda_fee"] = float(lambda_fee)
        return obs, reward_mod, terminated, truncated, info


# ---------------------------------------------------------------------------
# 輔助 reward 退火：主線 log_return 不變，regime/conviction 隨訓練步數線性衰減
# ---------------------------------------------------------------------------


class RewardAnnealWrapper(gym.Wrapper):
    """
    依 info 的 reward_log_return / reward_regime_bonus / reward_conviction_bonus 重組 reward：
    reward = reward_log_return + anneal_factor * (reward_regime_bonus + reward_conviction_bonus)。
    anneal_factor 從 1 線性降到 0（anneal_steps 步內）；anneal_steps<=0 表示不退火（恆為 1）。
    training_timestep 由 PhaseABStatsCallback 每步寫入，供計算 factor。
    """

    def __init__(self, env: gym.Env, anneal_steps: int = 0) -> None:
        super().__init__(env)
        self.anneal_steps = max(0, int(anneal_steps))

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        r_log = float(info.get("reward_log_return", reward))
        r_reg = float(info.get("reward_regime_bonus", 0.0))
        r_conv = float(info.get("reward_conviction_bonus", 0.0))
        step = int(getattr(self, "training_timestep", 0))
        if self.anneal_steps > 0 and step >= 0:
            factor = max(0.0, 1.0 - float(step) / float(self.anneal_steps))
        else:
            factor = 1.0
        reward_new = r_log + factor * (r_reg + r_conv)
        info["reward_anneal_factor"] = float(factor)
        return obs, float(reward_new), terminated, truncated, info


# ---------------------------------------------------------------------------
# 把 action 統計寫入 info，供 callback 彙總
# ---------------------------------------------------------------------------


class ActionStatsInfoWrapper(gym.Wrapper):
    """
    每步將 base env 的 _last_action_effects 寫入 info，
    供 PhaseABStatsCallback 計算 override_rate、tracking_error、execution_rate。
    """

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        base = self._get_base_env()
        if hasattr(base, "_last_action_effects") and base._last_action_effects:
            eff = base._last_action_effects
            info["action_overridden_flag"] = float(eff.get("action_overridden_flag", 0.0))
            info["last_action_raw"] = float(eff.get("last_action_raw", 0.0))
            info["last_final_pos_pct"] = float(eff.get("last_final_pos_pct", 0.0))
            info["trade_executed_flag"] = float(eff.get("trade_executed_flag", 0.0))
        return obs, reward, terminated, truncated, info

    def _get_base_env(self) -> gym.Env:
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return env


# ---------------------------------------------------------------------------
# Callback：每 N 步彙總 override_rate、tracking_error、execution_rate
# ---------------------------------------------------------------------------


class PhaseABStatsCallback(BaseCallback):
    """
    從每個 env 的 info 彙總：
    1) action 統計（每 log_freq 步）：override_rate、tracking_error、execution_rate（僅印出，不寫 TensorBoard）
    2) episode 結束時累積 log_return_sum、final_balance，每 log_freq 步寫入「僅含此兩條」的 TensorBoard（tb_log_dir）
    """

    def __init__(
        self,
        log_freq: int = 1000,
        verbose: int = 1,
        tb_log_dir: Optional[str] = None,
        lambda_buffer: Optional[float] = None,
        reward_scale: Optional[float] = None,
        lambda_risk: Optional[float] = None,
    ) -> None:
        super().__init__(verbose=verbose)
        self.log_freq = max(1, int(log_freq))
        self._log_return_buf: list[float] = []
        self._final_balance_buf: list[float] = []
        self._regime_bonus_buf: list[float] = []
        self._conviction_bonus_buf: list[float] = []
        self._cost_risk_buf: list[float] = []
        self._cost_risk_dense_buf: list[float] = []
        self._trade_count_buf: list[float] = []
        self._total_fees_buf: list[float] = []
        self._profit_buf: list[float] = []
        # 每步 reward 明細（來自 RiskOnlyPenaltyWrapper 的 info，供 TensorBoard reward_decomp）
        self._reward_raw_step: list[float] = []
        self._reward_mod_step: list[float] = []
        self._cost_risk_used_step: list[float] = []
        self._cost_risk_dense_used_step: list[float] = []
        self._cost_fric_used_step: list[float] = []
        self._lambda_fee_step: list[float] = []
        self._penalty_fee_step: list[float] = []
        self._tb_log_dir = (tb_log_dir or "").strip()
        self._tb_writer = None
        self._reward_scale = float(reward_scale) if reward_scale is not None else None
        self._lambda_risk = float(lambda_risk) if lambda_risk is not None else None
        self._lambda_buffer = float(lambda_buffer) if lambda_buffer is not None else None
        if self._tb_log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(self._tb_log_dir, exist_ok=True)
                self._tb_writer = SummaryWriter(log_dir=self._tb_log_dir)
            except Exception:
                self._tb_writer = None

    def _on_training_end(self) -> None:
        if getattr(self, "_tb_writer", None) is not None:
            try:
                self._tb_writer.close()
            except Exception:
                pass
            self._tb_writer = None

    def _on_step(self) -> bool:
        # 供 RewardAnnealWrapper 讀取當前訓練步數（退火用）
        try:
            for env in getattr(self.training_env, "envs", []):
                setattr(env, "training_timestep", self.num_timesteps)
        except Exception:
            pass

        infos = self.locals.get("infos")
        if not infos:
            return True

        # --- 每步：若有 episode 結束，累積實際 PnL、終局 equity、reward 分解 ---
        for i in range(len(infos)):
            info = infos[i] if isinstance(infos, (list, tuple)) else infos
            if not isinstance(info, dict):
                continue
            if "episode_log_return_sum" in info:
                try:
                    self._log_return_buf.append(float(info["episode_log_return_sum"]))
                except (TypeError, ValueError):
                    pass
            if "final_balance" in info:
                try:
                    self._final_balance_buf.append(float(info["final_balance"]))
                except (TypeError, ValueError):
                    pass
            if "episode_regime_alignment_bonus_sum" in info:
                try:
                    self._regime_bonus_buf.append(float(info["episode_regime_alignment_bonus_sum"]))
                except (TypeError, ValueError):
                    pass
            if "episode_conviction_bonus_sum" in info:
                try:
                    self._conviction_bonus_buf.append(float(info["episode_conviction_bonus_sum"]))
                except (TypeError, ValueError):
                    pass
            if "episode_cost_risk_sum" in info:
                try:
                    self._cost_risk_buf.append(float(info["episode_cost_risk_sum"]))
                except (TypeError, ValueError):
                    pass
            if "episode_cost_risk_dense_sum" in info:
                try:
                    self._cost_risk_dense_buf.append(float(info["episode_cost_risk_dense_sum"]))
                except (TypeError, ValueError):
                    pass
            if "episode_trade_count" in info:
                try:
                    self._trade_count_buf.append(float(info["episode_trade_count"]))
                except (TypeError, ValueError):
                    pass
            if "total_fees" in info:
                try:
                    self._total_fees_buf.append(float(info["total_fees"]))
                except (TypeError, ValueError):
                    pass
            if "profit" in info:
                try:
                    self._profit_buf.append(float(info["profit"]))
                except (TypeError, ValueError):
                    pass
            # 每步 reward 明細（Phase B 時 RiskOnlyPenaltyWrapper 會寫入）
            if "reward_raw" in info:
                try:
                    self._reward_raw_step.append(float(info["reward_raw"]))
                except (TypeError, ValueError):
                    pass
            if "reward_mod" in info:
                try:
                    self._reward_mod_step.append(float(info["reward_mod"]))
                except (TypeError, ValueError):
                    pass
            if "cost_risk_used" in info:
                try:
                    self._cost_risk_used_step.append(float(info["cost_risk_used"]))
                except (TypeError, ValueError):
                    pass
            if "cost_risk_dense_used" in info:
                try:
                    self._cost_risk_dense_used_step.append(float(info["cost_risk_dense_used"]))
                except (TypeError, ValueError):
                    pass
            if "cost_fric_used" in info:
                try:
                    self._cost_fric_used_step.append(float(info["cost_fric_used"]))
                except (TypeError, ValueError):
                    pass
            if "lambda_fee" in info:
                try:
                    self._lambda_fee_step.append(float(info["lambda_fee"]))
                except (TypeError, ValueError):
                    pass
            if "cost_fric_used" in info and "lambda_fee" in info:
                try:
                    cost_fric_val = float(info["cost_fric_used"])
                    lambda_fee_val = float(info["lambda_fee"])
                    self._penalty_fee_step.append(lambda_fee_val * cost_fric_val)
                except (TypeError, ValueError):
                    pass

        # --- 每 log_freq 步：寫入 action 統計（logger 僅 stdout）+ reward 分解與輔助/主線比寫入 TensorBoard ---
        if self.n_calls % self.log_freq != 0:
            return True
        step = getattr(self, "num_timesteps", self.n_calls)
        if self._tb_writer is not None:
            try:
                # Reward 分解：主線 log_return、輔助 regime/conviction、cost_risk
                if self._log_return_buf:
                    val = float(np.mean(self._log_return_buf))
                    self.logger.record("episode_stats/reward_log_return_sum_mean", val)
                    self._tb_writer.add_scalar("episode_stats/reward_log_return_sum_mean", val, step)
                    self._tb_writer.add_scalar("episode_stats/log_return_sum_mean", val, step)
                if self._regime_bonus_buf:
                    val = float(np.mean(self._regime_bonus_buf))
                    self.logger.record("episode_stats/reward_regime_bonus_sum_mean", val)
                    self._tb_writer.add_scalar("episode_stats/reward_regime_bonus_sum_mean", val, step)
                if self._conviction_bonus_buf:
                    val = float(np.mean(self._conviction_bonus_buf))
                    self.logger.record("episode_stats/reward_conviction_bonus_sum_mean", val)
                    self._tb_writer.add_scalar("episode_stats/reward_conviction_bonus_sum_mean", val, step)
                if self._cost_risk_buf:
                    val = float(np.mean(self._cost_risk_buf))
                    self.logger.record("episode_stats/cost_risk_sum_mean", val)
                    self._tb_writer.add_scalar("episode_stats/cost_risk_sum_mean", val, step)
                if self._cost_risk_dense_buf:
                    val = float(np.mean(self._cost_risk_dense_buf))
                    self.logger.record("episode_stats/cost_risk_dense_sum_mean", val)
                    self._tb_writer.add_scalar("episode_stats/cost_risk_dense_sum_mean", val, step)
                if self._trade_count_buf:
                    val = float(np.mean(self._trade_count_buf))
                    self.logger.record("episode_stats/trade_count_mean", val)
                    self._tb_writer.add_scalar("episode_stats/trade_count_mean", val, step)
                if self._total_fees_buf:
                    val = float(np.mean(self._total_fees_buf))
                    self.logger.record("episode_stats/total_fees_mean", val)
                    self._tb_writer.add_scalar("episode_stats/total_fees_mean", val, step)
                # 輔助/主線比：(|regime|+|conviction|) / max(|log_return|, 1e-8)，>1 表示輔助項量級壓過主線
                if self._log_return_buf and (self._regime_bonus_buf or self._conviction_bonus_buf):
                    mean_log = float(np.mean(self._log_return_buf))
                    mean_reg = float(np.mean(self._regime_bonus_buf)) if self._regime_bonus_buf else 0.0
                    mean_conv = float(np.mean(self._conviction_bonus_buf)) if self._conviction_bonus_buf else 0.0
                    denom = max(abs(mean_log), 1e-8)
                    ratio = (abs(mean_reg) + abs(mean_conv)) / denom
                    self.logger.record("episode_stats/auxiliary_main_ratio", ratio)
                    self._tb_writer.add_scalar("episode_stats/auxiliary_main_ratio", ratio, step)
                # 手續費 / 利潤 比例：total_fees_mean / max(|profit_mean|, 1e-8)
                if self._total_fees_buf and self._profit_buf:
                    mean_fees = float(np.mean(self._total_fees_buf))
                    mean_profit = float(np.mean(self._profit_buf))
                    denom_profit = max(abs(mean_profit), 1e-8)
                    ratio_fp = mean_fees / denom_profit
                    self.logger.record("episode_stats/fees_profit_ratio_mean", ratio_fp)
                    self._tb_writer.add_scalar("episode_stats/fees_profit_ratio_mean", ratio_fp, step)
                # Reward 計算明細（reward_decomp）：每 log_freq 內步的平均，方便對照公式
                if self._reward_raw_step:
                    mean_raw = float(np.mean(self._reward_raw_step))
                    self._tb_writer.add_scalar("reward_decomp/reward_raw_mean", mean_raw, step)
                    if self._reward_scale is not None:
                        self._tb_writer.add_scalar("reward_decomp/reward_scaled_mean", self._reward_scale * mean_raw, step)
                if self._reward_mod_step:
                    self._tb_writer.add_scalar("reward_decomp/reward_mod_mean", float(np.mean(self._reward_mod_step)), step)
                if self._cost_risk_used_step:
                    mean_cr = float(np.mean(self._cost_risk_used_step))
                    self._tb_writer.add_scalar("reward_decomp/cost_risk_used_mean", mean_cr, step)
                    if self._lambda_risk is not None:
                        self._tb_writer.add_scalar("reward_decomp/penalty_risk_mean", self._lambda_risk * mean_cr, step)
                if self._cost_risk_dense_used_step:
                    mean_cd = float(np.mean(self._cost_risk_dense_used_step))
                    self._tb_writer.add_scalar("reward_decomp/cost_risk_dense_used_mean", mean_cd, step)
                    if self._lambda_buffer is not None:
                        self._tb_writer.add_scalar("reward_decomp/penalty_dense_mean", self._lambda_buffer * mean_cd, step)
                if self._cost_fric_used_step:
                    self._tb_writer.add_scalar(
                        "reward_decomp/cost_fric_used_mean",
                        float(np.mean(self._cost_fric_used_step)),
                        step,
                    )
                if self._lambda_fee_step:
                    self._tb_writer.add_scalar(
                        "reward_decomp/lambda_fee_mean",
                        float(np.mean(self._lambda_fee_step)),
                        step,
                    )
                if self._penalty_fee_step:
                    self._tb_writer.add_scalar(
                        "reward_decomp/penalty_fee_mean",
                        float(np.mean(self._penalty_fee_step)),
                        step,
                    )
            except Exception:
                pass
        # 保留原有 logger key 以相容既有腳本，並清空 buffer
        if self._log_return_buf:
            self.logger.record("episode_stats/log_return_sum_mean", float(np.mean(self._log_return_buf)))
            self._log_return_buf.clear()
        if self._final_balance_buf:
            val = float(np.mean(self._final_balance_buf))
            self.logger.record("episode_stats/final_balance_mean", val)
            if self._tb_writer is not None:
                try:
                    self._tb_writer.add_scalar("episode_stats/final_balance_mean", val, step)
                except Exception:
                    pass
            self._final_balance_buf.clear()
        if self._regime_bonus_buf:
            self._regime_bonus_buf.clear()
        if self._conviction_bonus_buf:
            self._conviction_bonus_buf.clear()
        if self._cost_risk_buf:
            self._cost_risk_buf.clear()
        if self._cost_risk_dense_buf:
            self._cost_risk_dense_buf.clear()
        if self._trade_count_buf:
            self._trade_count_buf.clear()
        if self._total_fees_buf:
            self._total_fees_buf.clear()
        if self._profit_buf:
            self._profit_buf.clear()
        if self._reward_raw_step:
            self._reward_raw_step.clear()
        if self._reward_mod_step:
            self._reward_mod_step.clear()
        if self._cost_risk_used_step:
            self._cost_risk_used_step.clear()
        if self._cost_risk_dense_used_step:
            self._cost_risk_dense_used_step.clear()
        if self._cost_fric_used_step:
            self._cost_fric_used_step.clear()
        if self._lambda_fee_step:
            self._lambda_fee_step.clear()
        if self._penalty_fee_step:
            self._penalty_fee_step.clear()

        overrides: list[float] = []
        tracking_errors: list[float] = []
        executions: list[float] = []
        for i in range(len(infos)):
            info = infos[i] if isinstance(infos, (list, tuple)) else infos
            if not isinstance(info, dict):
                continue
            overrides.append(float(info.get("action_overridden_flag", 0.0)))
            raw = float(info.get("last_action_raw", 0.0))
            final = float(info.get("last_final_pos_pct", 0.0))
            tracking_errors.append(abs(raw - final))
            executions.append(float(info.get("trade_executed_flag", 0.0)))
        if overrides:
            self.logger.record("phase_ab/override_rate", np.mean(overrides))
            self.logger.record("phase_ab/tracking_error", np.mean(tracking_errors))
            self.logger.record("phase_ab/execution_rate", np.mean(executions))
            if self.verbose >= 1:
                print(
                    f"[PhaseAB] override_rate={np.mean(overrides):.3f} "
                    f"tracking_error={np.mean(tracking_errors):.3f} "
                    f"execution_rate={np.mean(executions):.3f}"
                )
        return True


# ---------------------------------------------------------------------------
# 環境工廠
# ---------------------------------------------------------------------------


def _default_feature_symbols() -> tuple[str, ...]:
    try:
        from Eval.train_config import TrainConfig
        return TrainConfig.FEATURE_SYMBOLS
    except Exception:
        return ("BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT")


def make_env(
    phase: str,
    lambda_risk: float = 0.05,
    lambda_buffer: float = 0.1,
    reward_scale: float = 10.0,
    lambda_fee_max: float = 0.0,
    fee_anneal_steps: int = 0,
    action_repeat: int = 1,
    anneal_steps: int = 0,
    seed: Optional[int] = None,
    **env_kwargs: Any,
) -> Callable[[], gym.Env]:
    """
    回傳一個 thunk：呼叫後建立一個 Phase A 或 Phase B 的環境。
    主線 log_return 不變；輔助 regime/conviction 依 anneal_steps 線性退火（0=不退火）。
    Phase B 時套用 cost_risk（事件型）與 cost_risk_dense（dense 緩衝懲罰）雙通道。
    """

    def thunk() -> gym.Env:
        base_cls = TradingEnvPhaseA if phase in ("A", "B") else TradingEnvironment
        env = base_cls(**env_kwargs)
        if seed is not None:
            env.reset(seed=seed)
        env = ActionStatsInfoWrapper(env)
        if anneal_steps > 0:
            env = RewardAnnealWrapper(env, anneal_steps=anneal_steps)
        if phase == "B":
            env = RiskOnlyPenaltyWrapper(
                env,
                lambda_risk=lambda_risk,
                lambda_buffer=lambda_buffer,
                reward_scale=reward_scale,
                lambda_fee_max=lambda_fee_max,
                fee_anneal_steps=fee_anneal_steps,
            )
        if action_repeat and action_repeat > 1:
            from Env.wrappers import ActionRepeatWrapper
            env = ActionRepeatWrapper(env, repeat=action_repeat)
        return env

    return thunk


def get_phase_ab_env_kwargs(
    max_episode_steps: Optional[int] = None,
    regime_bonus_weight: float = 0.0,
    conviction_bonus_weight: float = 0.0,
    conviction_min_abs_pos: float = 0.2,
    conviction_trend_min_strength: float = 0.25,
    data_split_enabled: bool = True,
    holdout_months: int = 3,
    data_mode: str = "train",
    cost_fric_scale: float = 1.0,
) -> dict[str, Any]:
    """
    Phase A/B 共用的 env 參數：減少 hard override、主線 log-return。
    可選：小權重 regime/conviction 輔助 reward，讓「做對方向」有額外正訊號。
    訓練/評估時間切分：data_split_enabled=True 時，data_mode="train" 用非最近 N 月，
    data_mode="eval" 用最近 holdout_months 月，避免評估用訓練見過的資料。
    """
    try:
        from Eval.train_config import TrainConfig
        symbol = TrainConfig.SYMBOL
        feature_symbols = list(TrainConfig.FEATURE_SYMBOLS)
    except Exception:
        symbol = "BTCUSDT"
        feature_symbols = list(_default_feature_symbols())

    if max_episode_steps is None:
        max_episode_steps = 288 * 31

    mode = str(data_mode).strip().lower() or "train"
    env_kwargs = dict(
        env_id=0,
        random_start=True,
        window_size=14,
        window_size_1d=12,
        max_episode_steps=max_episode_steps,
        target_symbol=symbol,
        feature_symbols=feature_symbols,
        # 訓練/評估時間切分（預設：訓練用過去、評估用最近 holdout_months 月）
        data_split_enabled=bool(data_split_enabled),
        holdout_months=max(1, int(holdout_months)),
        data_mode=mode,
        # 降低 hard override
        no_trade_entry_threshold=0.3,
        no_trade_exit_threshold=0.05,
        max_step_pos_change_pct=0.5,
        min_position_change=0.05,
        trade_freq_window_steps=None,
        trade_freq_cost_limit=None,
        cost_fric_scale=float(cost_fric_scale),
        # 順向／regime 輔助 reward（小權重）：做對方向加分，主線仍是 log-return
        regime_alignment_bonus_weight=float(regime_bonus_weight),
        conviction_trend_bonus_weight=float(conviction_bonus_weight),
        conviction_min_abs_pos=float(conviction_min_abs_pos),
        conviction_trend_min_strength=float(conviction_trend_min_strength),
    )
    # 評估端若沿用預設 min_episode_steps，常會把可選起點範圍壓縮到單一點，
    # 導致每次 reset 都是同一起點。eval 模式改為放寬，確保 random_start 可生效。
    if mode == "eval":
        env_kwargs["min_episode_steps"] = 1
    return env_kwargs


def _parse_metric_rule(rule_text: str) -> Optional[MetricRule]:
    """解析 `metric>=value` 類型的規則字串。"""
    text = (rule_text or "").strip()
    if not text:
        return None
    for op in (">=", "<=", "==", "!=", ">", "<"):
        if op in text:
            left, right = text.split(op, 1)
            metric_name = left.strip()
            threshold_text = right.strip()
            if not metric_name:
                return None
            try:
                threshold = float(threshold_text)
            except ValueError:
                return None
            return MetricRule(metric_name=metric_name, op=op, threshold=threshold)
    return None


def _build_eval_trigger(args: argparse.Namespace) -> Optional[CompositeTrigger]:
    """依 CLI 參數建構評估觸發器。"""
    triggers: list[Any] = []

    step_trigger = StepTrigger(
        at_steps=tuple(int(s) for s in (args.eval_trigger_steps or []) if int(s) > 0),
        every_n_steps=int(args.eval_trigger_every),
        min_interval_steps=int(args.eval_trigger_min_interval),
        include_training_end=False,
    )
    if step_trigger.at_steps or step_trigger.every_n_steps > 0:
        triggers.append(step_trigger)

    metric_rules: list[MetricRule] = []
    for rule_text in args.eval_metric_rule or []:
        rule = _parse_metric_rule(rule_text)
        if rule is not None:
            metric_rules.append(rule)
    if metric_rules:
        triggers.append(MetricTrigger(rules=tuple(metric_rules), mode=args.eval_trigger_mode, include_training_end=False))

    if (args.eval_expression or "").strip():
        triggers.append(ExpressionTrigger(expression=args.eval_expression, include_training_end=False))

    if not triggers:
        return None
    return CompositeTrigger(triggers=tuple(triggers), mode=args.eval_trigger_mode)


class PhaseABEvaluationTriggerCallback(BaseCallback):
    """
    訓練期間依條件觸發評估，並可在訓練結束後再做一次評估。
    """

    def __init__(
        self,
        evaluator: PhaseABEvaluator,
        trigger: Optional[CompositeTrigger],
        log_freq: int = 1000,
        eval_on_train_end: bool = True,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.evaluator = evaluator
        self.trigger = trigger
        self.log_freq = max(1, int(log_freq))
        self.eval_on_train_end = bool(eval_on_train_end)
        self.eval_count = 0
        self.last_eval_step: Optional[int] = None
        self._latest_metrics: dict[str, float] = {}
        self._log_return_buf: list[float] = []
        self._final_balance_buf: list[float] = []
        self._cost_risk_buf: list[float] = []
        self._cost_risk_dense_buf: list[float] = []
        self._override_buf: list[float] = []
        self._tracking_error_buf: list[float] = []
        self._execution_buf: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos:
            for i in range(len(infos)):
                info = infos[i] if isinstance(infos, (list, tuple)) else infos
                if not isinstance(info, dict):
                    continue
                if "episode_log_return_sum" in info:
                    try:
                        self._log_return_buf.append(float(info["episode_log_return_sum"]))
                    except (TypeError, ValueError):
                        pass
                if "final_balance" in info:
                    try:
                        self._final_balance_buf.append(float(info["final_balance"]))
                    except (TypeError, ValueError):
                        pass
                if "episode_cost_risk_sum" in info:
                    try:
                        self._cost_risk_buf.append(float(info["episode_cost_risk_sum"]))
                    except (TypeError, ValueError):
                        pass
                if "episode_cost_risk_dense_sum" in info:
                    try:
                        self._cost_risk_dense_buf.append(float(info["episode_cost_risk_dense_sum"]))
                    except (TypeError, ValueError):
                        pass

                try:
                    self._override_buf.append(float(info.get("action_overridden_flag", 0.0)))
                    raw = float(info.get("last_action_raw", 0.0))
                    final = float(info.get("last_final_pos_pct", 0.0))
                    self._tracking_error_buf.append(abs(raw - final))
                    self._execution_buf.append(float(info.get("trade_executed_flag", 0.0)))
                except (TypeError, ValueError):
                    pass

        if self.n_calls % self.log_freq != 0:
            return True
        self._latest_metrics = self._collect_metrics()
        if self.trigger is None:
            return True

        context = EvalTriggerContext(
            step=int(getattr(self, "num_timesteps", self.n_calls)),
            metrics=self._latest_metrics,
            eval_count=self.eval_count,
            training_end=False,
            last_eval_step=self.last_eval_step,
        )
        if self.trigger.should_evaluate(context):
            current_step = int(getattr(self, "num_timesteps", self.n_calls))
            self.evaluator.evaluate(
                model=self.model,
                reason="train_trigger",
                step=current_step,
                print_result=True,
            )
            self.eval_count += 1
            self.last_eval_step = current_step
        return True

    def _on_training_end(self) -> None:
        if not self.eval_on_train_end:
            return
        current_step = int(getattr(self, "num_timesteps", self.n_calls))
        if not self._latest_metrics:
            self._latest_metrics = self._collect_metrics()
        context = EvalTriggerContext(
            step=current_step,
            metrics=self._latest_metrics,
            eval_count=self.eval_count,
            training_end=True,
            last_eval_step=self.last_eval_step,
        )
        # 訓練結束評估預設強制執行；若你要受 trigger 控制，可自行改為 trigger 判斷
        _ = context
        self.evaluator.evaluate(
            model=self.model,
            reason="training_end",
            step=current_step,
            print_result=True,
        )
        self.eval_count += 1
        self.last_eval_step = current_step

    def _collect_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if self._log_return_buf:
            metrics["log_return_sum_mean"] = float(np.mean(self._log_return_buf))
            self._log_return_buf.clear()
        if self._final_balance_buf:
            metrics["final_balance_mean"] = float(np.mean(self._final_balance_buf))
            self._final_balance_buf.clear()
        if self._cost_risk_buf:
            metrics["cost_risk_sum_mean"] = float(np.mean(self._cost_risk_buf))
            self._cost_risk_buf.clear()
        if self._cost_risk_dense_buf:
            metrics["cost_risk_dense_sum_mean"] = float(np.mean(self._cost_risk_dense_buf))
            self._cost_risk_dense_buf.clear()
        if self._override_buf:
            metrics["override_rate"] = float(np.mean(self._override_buf))
            self._override_buf.clear()
        if self._tracking_error_buf:
            metrics["tracking_error"] = float(np.mean(self._tracking_error_buf))
            self._tracking_error_buf.clear()
        if self._execution_buf:
            metrics["execution_rate"] = float(np.mean(self._execution_buf))
            self._execution_buf.clear()
        return metrics


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase A/B SAC 訓練")
    parser.add_argument("--phase", choices=["A", "B"], default="B", help="Phase A=只放寬控制, B=再加 cost_risk 懲罰")
    parser.add_argument("--timesteps", type=int, default=9_000_000) # 288 * 21 * 48 * 20 = 261,360,000
    parser.add_argument("--n-envs", type=int, default=48)
    parser.add_argument("--lambda-risk", type=float, default=1.0, help="Phase B 時 cost_risk（事件型）的權重")
    parser.add_argument("--lambda-buffer", type=float, default=0.01, help="Phase B 時 cost_risk_dense（dense 緩衝懲罰）的權重")
    parser.add_argument("--reward-scale", type=float, default=10.0, help="Phase B 時主線 reward 放大倍數")
    parser.add_argument(
        "--lambda-fee-max",
        type=float,
        default=0.05,
        help="Phase B 時 cost_fric（手續費摩擦成本）的「懲罰權重」；非每筆交易手續費率（手續費率由環境 transaction_fee 決定）。0=不啟用手續費懲罰",
    )
    parser.add_argument(
        "--fee-anneal-steps",
        type=int,
        default=3_000_000,
        help="手續費懲罰權重從 0 線性升到 lambda_fee_max 所需的步數（<=0 表示不退火，恆為 lambda_fee_max）",
    )
    parser.add_argument(
        "--cost-fric-scale",
        type=float,
        default=1000.0,
        help="Phase B 時 cost_fric 放大係數；原始 cost_fric=step_fee/equity 約 1e-4~1e-3，乘此係數後與 reward 同數量級，lambda_fee 才有效（預設 100）",
    )
    parser.add_argument("--action-repeat", type=int, default=1, help="Frame skip，1=每步決策")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-freq", type=int, default=1000, help="PhaseAB 統計與 log 間隔（步數）")
    # 訓練/評估時間切分（預設：訓練用過去、評估用最近 N 月，避免評估用訓練見過的資料）
    parser.add_argument("--holdout-months", type=int, default=1, help="評估用最近 N 個月資料；訓練用其餘過去資料（與 --no-data-split 互斥）")
    parser.add_argument("--no-data-split", action="store_true", help="停用訓練/評估時間切分，訓練與評估皆用完整資料")
    # 順向／regime 輔助 reward（小權重，做對方向加分）
    parser.add_argument("--regime-bonus-weight", type=float, default=0.00003, help="Regime 對齊 bonus 權重（A 多/C 空加分，依 dir_strength 加權）")
    parser.add_argument("--conviction-bonus-weight", type=float, default=0.03, help="Conviction 順向 bonus 權重（強訊號+大倉+同向加分）")
    parser.add_argument("--conviction-min-abs-pos", type=float, default=0.4, help="Conviction 生效最小持倉比例")
    parser.add_argument("--conviction-trend-min-strength", type=float, default=0.5, help="Conviction 生效最小趨勢強度")
    parser.add_argument("--anneal-steps", type=int, default=500_000, help="輔助 reward 退火步數（0=不退火，regime/conviction 全程滿權重）")
    # 評估參數（可訓練中觸發、訓練後觸發，或 eval-only）
    parser.add_argument("--eval-only", action="store_true", help="只做評估，不進行訓練")
    parser.add_argument("--eval-model-path", type=str, default="", help="評估模型路徑（空則沿用 --save-path）")
    parser.add_argument("--eval-episodes", type=int, default=10, help="每次評估回合數")
    parser.add_argument("--eval-seed", type=int, default=42, help="評估用 seed")
    parser.add_argument("--eval-report-path", type=str, default="", help="評估結果 JSON 輸出路徑，空則依 phase/lr/lb/rs/rb/cb 自動產生")
    parser.add_argument("--tb-log", type=str, default="", help="TensorBoard log 目錄，空則依 phase/lr/lb/rs/rb/cb 自動產生或不寫")
    parser.add_argument("--save-path", type=str, default="", help="模型儲存路徑，空則依 phase/lr/lb/rs/rb/cb 自動產生")
    parser.add_argument("--eval-deterministic", action=argparse.BooleanOptionalAction, default=True, help="評估是否使用 deterministic 動作(False=使用隨機動作)")
    parser.add_argument("--eval-on-train-end", action=argparse.BooleanOptionalAction, default=True, help="訓練結束後是否執行一次評估")
    parser.add_argument("--eval-trigger-steps", type=int, nargs="*", default=[], help="訓練中在指定步數觸發評估，可多個")
    parser.add_argument("--eval-trigger-every", type=int, default=3_000_000, help="訓練中每 N steps 觸發評估（0=停用）")
    parser.add_argument("--eval-trigger-min-interval", type=int, default=0, help="兩次訓練中評估最小間隔步數")
    parser.add_argument("--eval-trigger-mode", choices=["any", "all"], default="any", help="多條件組合模式")
    parser.add_argument("--eval-metric-rule", action="append", default=[], help="內建 metric 規則，例如 log_return_sum_mean>=0.2")
    parser.add_argument("--eval-expression", type=str, default="", help="自訂評估條件式，例如 step>=5_000_000 and log_return_sum_mean>0")
    args = parser.parse_args()

    # 路徑預設：依 phase/lr/lb/rs/rb/cb/lf/fas 產生（不可在 add_argument 時用 args，故在此補上）
    def _path_prefix() -> str:
        return (
            f"phase_{args.phase}_lr{str(args.lambda_risk).replace('.', '')}_lb{str(args.lambda_buffer).replace('.', '')}"
            f"_rs{str(args.reward_scale).replace('.', '')}_rb{str(args.regime_bonus_weight).replace('.', '')}_cb{str(args.conviction_bonus_weight).replace('.', '')}"
            f"_lf{str(args.lambda_fee_max).replace('.', '')}_fas{str(args.fee_anneal_steps).replace('.', '')}_cfs{str(args.cost_fric_scale).replace('.', '')}"
        )
    if not (getattr(args, "eval_report_path", "") or "").strip():
        args.eval_report_path = f"logs/{_path_prefix()}_eval.json"
    if not (getattr(args, "tb_log", "") or "").strip():
        args.tb_log = f"logs/{_path_prefix()}"
    if not (getattr(args, "save_path", "") or "").strip():
        args.save_path = f"models/{_path_prefix()}"

    data_split = not getattr(args, "no_data_split", False)
    holdout = max(1, int(getattr(args, "holdout_months", 3)))
    if data_split:
        print(f"[Data split] 訓練用「非最近 {holdout} 月」、評估用「最近 {holdout} 月」")
    else:
        print("[Data split] 已停用，訓練與評估皆使用完整資料")

    env_kwargs_train = get_phase_ab_env_kwargs(
        regime_bonus_weight=args.regime_bonus_weight,
        conviction_bonus_weight=args.conviction_bonus_weight,
        conviction_min_abs_pos=args.conviction_min_abs_pos,
        conviction_trend_min_strength=args.conviction_trend_min_strength,
        data_split_enabled=data_split,
        holdout_months=holdout,
        data_mode="train",
        cost_fric_scale=getattr(args, "cost_fric_scale", 100.0),
    )
    env_kwargs_eval = get_phase_ab_env_kwargs(
        regime_bonus_weight=args.regime_bonus_weight,
        conviction_bonus_weight=args.conviction_bonus_weight,
        conviction_min_abs_pos=args.conviction_min_abs_pos,
        conviction_trend_min_strength=args.conviction_trend_min_strength,
        data_split_enabled=data_split,
        holdout_months=holdout,
        data_mode="eval",
        cost_fric_scale=getattr(args, "cost_fric_scale", 100.0),
    )

    eval_env_thunk = make_env(
        phase=args.phase,
        lambda_risk=args.lambda_risk,
        lambda_buffer=args.lambda_buffer,
        reward_scale=args.reward_scale,
        lambda_fee_max=args.lambda_fee_max,
        fee_anneal_steps=args.fee_anneal_steps,
        action_repeat=args.action_repeat,
        anneal_steps=args.anneal_steps,
        seed=args.eval_seed,
        **env_kwargs_eval,
    )
    evaluator = PhaseABEvaluator(
        env_builder=build_single_env_builder(eval_env_thunk),
        n_episodes=args.eval_episodes,
        deterministic=args.eval_deterministic,
        seed=args.eval_seed,
        report_path=args.eval_report_path,
    )
    if args.eval_only:
        model_path = args.eval_model_path.strip() or args.save_path
        model = SAC.load(model_path, device=args.device)
        evaluator.evaluate(model=model, reason="eval_only", step=None, print_result=True)
        return

    vec_env: VecEnv = DummyVecEnv(
        [
            make_env(
                phase=args.phase,
                lambda_risk=args.lambda_risk,
                lambda_buffer=args.lambda_buffer,
                reward_scale=args.reward_scale,
                lambda_fee_max=args.lambda_fee_max,
                fee_anneal_steps=args.fee_anneal_steps,
                action_repeat=args.action_repeat,
                anneal_steps=args.anneal_steps,
                **env_kwargs_train,
            )
            for _ in range(args.n_envs)
        ]
    )
    vec_env = VecMonitor(vec_env)

    callbacks: list[BaseCallback] = [
        PhaseABStatsCallback(
            log_freq=args.log_freq,
            verbose=1,
            tb_log_dir=args.tb_log if args.tb_log else None,
            lambda_buffer=args.lambda_buffer if args.phase == "B" else None,
            reward_scale=args.reward_scale if args.phase == "B" else None,
            lambda_risk=args.lambda_risk if args.phase == "B" else None,
        ),
    ]
    eval_trigger = _build_eval_trigger(args)
    if eval_trigger is not None or args.eval_on_train_end:
        callbacks.append(
            PhaseABEvaluationTriggerCallback(
                evaluator=evaluator,
                trigger=eval_trigger,
                log_freq=args.log_freq,
                eval_on_train_end=args.eval_on_train_end,
                verbose=1,
            )
        )
    model = SAC(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=300_000,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        device=args.device,
    )
    # TensorBoard 僅由 callback 寫入「兩條 PnL」到 tb_log，主 logger 只 stdout 避免畫面雜亂
    if args.tb_log:
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(None, ["stdout"]))

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    model.save(args.save_path)
    vec_env.close()
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
