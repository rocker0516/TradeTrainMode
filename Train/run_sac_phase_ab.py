"""
Phase A / Phase B SAC 訓練入口。

Phase A：關閉 regime 硬投影與過多 hard override，讓 policy 直接控制倉位。
Phase B：在 Phase A 基礎上只加 cost_risk 懲罰（不把 cost_trade_freq / cost_flat 放進 reward）。

使用方式：
    python -m Train.run_sac_phase_ab --phase A --timesteps 300000
    python -m Train.run_sac_phase_ab --phase B --timesteps 300000 --lambda-risk 0.05 --reward-scale 10
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
    Phase B：reward_mod = reward_scale * reward - lambda_risk * cost_risk。
    不把 cost_trade_freq / cost_flat 放進懲罰，避免主線被約束吞掉。
    """

    def __init__(
        self,
        env: gym.Env,
        lambda_risk: float = 0.05,
        reward_scale: float = 10.0,
    ) -> None:
        super().__init__(env)
        self.lambda_risk = float(lambda_risk)
        self.reward_scale = float(reward_scale)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        cost_risk = float(info.get("cost_risk", 0.0))
        reward_mod = self.reward_scale * float(reward) - self.lambda_risk * cost_risk
        info["reward_raw"] = float(reward)
        info["reward_mod"] = float(reward_mod)
        info["cost_risk_used"] = cost_risk
        return obs, reward_mod, terminated, truncated, info


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
    從每個 env 的 info 彙總 action 統計並寫入 logger。
    需要 env 外層包 ActionStatsInfoWrapper（或 env 本身在 info 裡提供上述鍵）。
    """

    def __init__(
        self,
        log_freq: int = 1000,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.log_freq = max(1, int(log_freq))

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True
        infos = self.locals.get("infos")
        if not infos:
            return True
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
    reward_scale: float = 10.0,
    action_repeat: int = 1,
    seed: Optional[int] = None,
    **env_kwargs: Any,
) -> Callable[[], gym.Env]:
    """
    回傳一個 thunk：呼叫後建立一個 Phase A 或 Phase B 的環境。
    """

    def thunk() -> gym.Env:
        base_cls = TradingEnvPhaseA if phase in ("A", "B") else TradingEnvironment
        env = base_cls(**env_kwargs)
        if seed is not None:
            env.reset(seed=seed)
        env = ActionStatsInfoWrapper(env)
        if phase == "B":
            env = RiskOnlyPenaltyWrapper(env, lambda_risk=lambda_risk, reward_scale=reward_scale)
        if action_repeat and action_repeat > 1:
            from Env.wrappers import ActionRepeatWrapper
            env = ActionRepeatWrapper(env, repeat=action_repeat)
        return env

    return thunk


def get_phase_ab_env_kwargs(
    max_episode_steps: Optional[int] = None,
) -> dict[str, Any]:
    """Phase A/B 共用的 env 參數：減少 hard override、主線純 log-return。"""
    try:
        from Eval.train_config import TrainConfig
        symbol = TrainConfig.SYMBOL
        feature_symbols = list(TrainConfig.FEATURE_SYMBOLS)
    except Exception:
        symbol = "BTCUSDT"
        feature_symbols = list(_default_feature_symbols())

    if max_episode_steps is None:
        max_episode_steps = 288 * 31

    return dict(
        env_id=0,
        random_start=True,
        window_size=288,
        window_size_1d=30,
        max_episode_steps=max_episode_steps,
        target_symbol=symbol,
        feature_symbols=feature_symbols,
        # 降低 hard override
        no_trade_entry_threshold=0.0,
        no_trade_exit_threshold=0.0,
        max_step_pos_change_pct=1.0,
        min_position_change=0.0,
        trade_freq_window_steps=None,
        trade_freq_cost_limit=None,
        # 主線先純 log-return，不加 bonus
        conviction_trend_bonus_weight=0.0,
        regime_alignment_bonus_weight=0.0,
    )


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase A/B SAC 訓練")
    parser.add_argument("--phase", choices=["A", "B"], default="A", help="Phase A=只放寬控制, B=再加 cost_risk 懲罰")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--lambda-risk", type=float, default=0.05, help="Phase B 時 cost_risk 的權重")
    parser.add_argument("--reward-scale", type=float, default=10.0, help="Phase B 時主線 reward 放大倍數")
    parser.add_argument("--action-repeat", type=int, default=1, help="Frame skip，1=每步決策")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-path", type=str, default="models/sac_phase_ab")
    parser.add_argument("--log-freq", type=int, default=1000, help="PhaseAB 統計與 log 間隔（步數）")
    parser.add_argument("--tb-log", type=str, default="", help="TensorBoard log 目錄，空則不寫")
    args = parser.parse_args()

    env_kwargs = get_phase_ab_env_kwargs()

    vec_env: VecEnv = DummyVecEnv([
        make_env(
            phase=args.phase,
            lambda_risk=args.lambda_risk,
            reward_scale=args.reward_scale,
            action_repeat=args.action_repeat,
            **env_kwargs,
        )
        for _ in range(args.n_envs)
    ])
    vec_env = VecMonitor(vec_env)

    callbacks: list[BaseCallback] = [
        PhaseABStatsCallback(log_freq=args.log_freq, verbose=1),
    ]
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
    if args.tb_log:
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(args.tb_log, ["stdout", "tensorboard"]))

    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    model.save(args.save_path)
    vec_env.close()
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
