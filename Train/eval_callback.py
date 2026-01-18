from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


@dataclass(frozen=True)
class EvalConstraints:
    """評估約束條件（constraints_then_balance）。"""

    max_dd_limit: float
    mean_cost_limit: float


@dataclass(frozen=True)
class EvalConfig:
    """
    Periodic evaluation 設定。

    Notes:
    - eval_freq 以「model.num_timesteps」判斷，避免 VecEnv 下每次 callback call = n_envs steps 的語意偏差。
    - 成本 cost 以每步 info["cost"] 累加，最後以 mean_cost = sum(cost)/episode_steps 表示。
    - 平均持倉口徑：holding_ratio = episode_holding_steps / episode_steps。
    - 交易頻率口徑：trades_per_step = episode_trade_count / episode_steps。
    """

    enabled: bool
    eval_every_timesteps: int
    n_eval_episodes: int
    deterministic: bool
    reject_if_death_event: bool
    constraints: EvalConstraints
    save_best_model: bool
    best_model_path: str
    print_each_episode: bool = False
    print_prefix: str = "[EVAL]"


class ConstraintEvalCallback(BaseCallback):
    """
    週期性評估 callback：
    - 每隔 N timesteps 跑 n_eval_episodes
    - 若任一 episode 出現死亡事件（強平/資金耗盡） => 整次 eval 不合格
    - 否則套用約束（max_dd, mean_cost）後以 mean_final_balance 挑 best，並可保存 best model
    - 將指標寫入 TensorBoard（logger.record）
    """

    def __init__(self, eval_env: VecEnv, eval_config: EvalConfig, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_config = eval_config

        self._last_eval_timestep: int = 0
        self._best_mean_final_balance: float = float("-inf")

    def _on_step(self) -> bool:
        if not bool(self.eval_config.enabled):
            return True

        current_ts = int(getattr(self.model, "num_timesteps", 0))
        if (current_ts - self._last_eval_timestep) < int(self.eval_config.eval_every_timesteps):
            return True

        self._last_eval_timestep = current_ts

        results = self._run_eval(
            n_episodes=int(self.eval_config.n_eval_episodes),
            deterministic=bool(self.eval_config.deterministic),
            current_ts=current_ts,
        )
        self._log_eval_results(results, current_ts=current_ts)

        # 死亡事件淘汰：整次評估不合格
        if bool(self.eval_config.reject_if_death_event) and results.any_death_event:
            return True

        # 約束判斷
        if (results.mean_episode_max_dd is None) or (results.mean_cost is None) or (results.mean_final_balance is None):
            return True

        if results.mean_episode_max_dd > float(self.eval_config.constraints.max_dd_limit):
            return True

        if results.mean_cost > float(self.eval_config.constraints.mean_cost_limit):
            return True

        # constraints_then_balance：在通過約束後，以 mean_final_balance 選 best
        if results.mean_final_balance > self._best_mean_final_balance:
            self._best_mean_final_balance = float(results.mean_final_balance)
            if bool(self.eval_config.save_best_model):
                self._save_best_model()

        return True

    def _save_best_model(self) -> None:
        os.makedirs(os.path.dirname(self.eval_config.best_model_path), exist_ok=True)
        self.model.save(self.eval_config.best_model_path)

    def _run_eval(self, n_episodes: int, deterministic: bool, current_ts: int) -> "EvalResults":
        # VecEnv 介面：n_envs 必須是 1（我們在訓練入口會用 DummyVecEnv 建立單環境 eval）
        n_envs = int(getattr(self.eval_env, "num_envs", 1))
        if n_envs != 1:
            raise ValueError(f"ConstraintEvalCallback expects eval_env.num_envs == 1, got {n_envs}")

        try:
            initial_balance = float(self.eval_env.get_attr("initial_balance")[0])
        except Exception:
            initial_balance = 0.0

        per_episode: List[EpisodeEval] = []
        any_death = False

        for ep_idx in range(int(n_episodes)):
            obs = self.eval_env.reset()

            done = False
            ep_steps = 0
            cost_sum = 0.0
            last_info: Dict[str, Any] = {}

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, _reward, dones, infos = self.eval_env.step(action)
                ep_steps += 1

                info = infos[0] if isinstance(infos, (list, tuple)) and infos else {}
                last_info = info

                step_cost = info.get("cost", None)
                if step_cost is not None:
                    try:
                        cost_sum += float(step_cost)
                    except (TypeError, ValueError):
                        pass

                done = bool(dones[0])

            ep = EpisodeEval.from_episode_end_info(
                info=last_info,
                episode_steps=int(ep_steps),
                cost_sum=float(cost_sum),
                initial_balance=float(initial_balance),
            )
            per_episode.append(ep)

            if ep.is_death_event:
                any_death = True

            if bool(getattr(self.eval_config, "print_each_episode", False)):
                prefix = str(getattr(self.eval_config, "print_prefix", "[EVAL]"))
                print(
                    _format_episode_line(prefix=prefix, current_ts=int(current_ts), ep_idx=int(ep_idx), ep=ep),
                    flush=True,
                )

        return EvalResults.aggregate(per_episode=per_episode, any_death_event=any_death)

    def _log_eval_results(self, results: "EvalResults", current_ts: int) -> None:
        # 基礎資訊
        self.logger.record("eval/timesteps", float(current_ts))
        self.logger.record("eval/any_death_event", float(1.0 if results.any_death_event else 0.0))
        self.logger.record("eval/death_event_count", float(results.death_event_count))

        # 主要指標（可能為 None）
        if results.mean_final_balance is not None:
            self.logger.record("eval/mean_final_balance", float(results.mean_final_balance))
        if results.mean_return is not None:
            self.logger.record("eval/mean_return", float(results.mean_return))
        if results.mean_episode_max_dd is not None:
            self.logger.record("eval/mean_episode_max_dd", float(results.mean_episode_max_dd))
        if results.mean_cost is not None:
            self.logger.record("eval/mean_cost", float(results.mean_cost))

        # 你關心的統計
        if results.mean_stop_loss_count is not None:
            self.logger.record("eval/mean_stop_loss_count", float(results.mean_stop_loss_count))
        if results.mean_liq_count is not None:
            self.logger.record("eval/mean_liq_count", float(results.mean_liq_count))
        if results.mean_trades_per_step is not None:
            self.logger.record("eval/mean_trades_per_step", float(results.mean_trades_per_step))
        if results.mean_long_entry_count is not None:
            self.logger.record("eval/mean_long_entry_count", float(results.mean_long_entry_count))
        if results.mean_short_entry_count is not None:
            self.logger.record("eval/mean_short_entry_count", float(results.mean_short_entry_count))
        if results.mean_holding_ratio is not None:
            self.logger.record("eval/mean_holding_ratio", float(results.mean_holding_ratio))

        # best 追蹤
        self.logger.record("eval/best_mean_final_balance", float(self._best_mean_final_balance))

        # 重要：SB3 可能只在特定時機（log_interval/rollout end）dump logger。
        # 為了讓 eval 指標能「立刻」出現在 TensorBoard，這裡強制寫出本次 eval 的紀錄。
        self.logger.dump(step=int(current_ts))


@dataclass(frozen=True)
class EpisodeEval:
    """單一 episode 的評估結果（由 episode 結束時 info + 本地累加值構成）。"""

    final_balance: Optional[float]
    episode_max_dd: Optional[float]
    stop_loss_count: Optional[int]
    liq_count: Optional[int]
    trade_count: Optional[int]
    long_entry_count: Optional[int]
    short_entry_count: Optional[int]
    holding_steps: Optional[int]
    holding_ratio: Optional[float]
    trades_per_step: Optional[float]
    mean_cost: Optional[float]
    ret: Optional[float]
    termination_reason: Optional[str]
    is_death_event: bool

    @staticmethod
    def from_episode_end_info(
        info: Dict[str, Any],
        episode_steps: int,
        cost_sum: float,
        initial_balance: float,
    ) -> "EpisodeEval":
        final_balance = _to_float_or_none(info.get("final_balance", None))
        episode_max_dd = _to_float_or_none(info.get("episode_max_dd", None))
        termination_reason = info.get("termination_reason", None)
        termination_reason = str(termination_reason) if termination_reason is not None else None

        stop_loss_count = _to_int_or_none(info.get("episode_stop_loss_count", None))
        liq_count = _to_int_or_none(info.get("episode_liq_count", None))
        trade_count = _to_int_or_none(info.get("episode_trade_count", None))
        long_entry_count = _to_int_or_none(info.get("long_entry_count", None))
        short_entry_count = _to_int_or_none(info.get("short_entry_count", None))
        holding_steps = _to_int_or_none(info.get("episode_holding_steps", None))

        steps = max(1, int(episode_steps))
        holding_ratio = None
        if holding_steps is not None:
            holding_ratio = float(max(0, holding_steps)) / float(steps)

        trades_per_step = None
        if trade_count is not None:
            trades_per_step = float(max(0, trade_count)) / float(steps)

        mean_cost = float(cost_sum) / float(steps) if steps > 0 else None

        ret = None
        if (final_balance is not None) and (initial_balance > 0):
            ret = float(final_balance) / float(initial_balance) - 1.0

        # 死亡事件判定（雙保險）：termination_reason or liq_count>0
        is_death_event = False
        if termination_reason in {"liq_triggered", "balance_insufficient"}:
            is_death_event = True
        if (liq_count is not None) and (liq_count > 0):
            is_death_event = True

        return EpisodeEval(
            final_balance=final_balance,
            episode_max_dd=episode_max_dd,
            stop_loss_count=stop_loss_count,
            liq_count=liq_count,
            trade_count=trade_count,
            long_entry_count=long_entry_count,
            short_entry_count=short_entry_count,
            holding_steps=holding_steps,
            holding_ratio=holding_ratio,
            trades_per_step=trades_per_step,
            mean_cost=mean_cost,
            ret=ret,
            termination_reason=termination_reason,
            is_death_event=bool(is_death_event),
        )


@dataclass(frozen=True)
class EvalResults:
    """多個 episodes 聚合後的評估結果。"""

    any_death_event: bool
    death_event_count: int
    mean_final_balance: Optional[float]
    mean_return: Optional[float]
    mean_episode_max_dd: Optional[float]
    mean_cost: Optional[float]
    mean_stop_loss_count: Optional[float]
    mean_liq_count: Optional[float]
    mean_trades_per_step: Optional[float]
    mean_long_entry_count: Optional[float]
    mean_short_entry_count: Optional[float]
    mean_holding_ratio: Optional[float]

    @staticmethod
    def aggregate(per_episode: List[EpisodeEval], any_death_event: bool) -> "EvalResults":
        death_event_count = int(sum(1 for ep in per_episode if ep.is_death_event))

        return EvalResults(
            any_death_event=bool(any_death_event),
            death_event_count=int(death_event_count),
            mean_final_balance=_mean_or_none([ep.final_balance for ep in per_episode]),
            mean_return=_mean_or_none([ep.ret for ep in per_episode]),
            mean_episode_max_dd=_mean_or_none([ep.episode_max_dd for ep in per_episode]),
            mean_cost=_mean_or_none([ep.mean_cost for ep in per_episode]),
            mean_stop_loss_count=_mean_or_none([ep.stop_loss_count for ep in per_episode]),
            mean_liq_count=_mean_or_none([ep.liq_count for ep in per_episode]),
            mean_trades_per_step=_mean_or_none([ep.trades_per_step for ep in per_episode]),
            mean_long_entry_count=_mean_or_none([ep.long_entry_count for ep in per_episode]),
            mean_short_entry_count=_mean_or_none([ep.short_entry_count for ep in per_episode]),
            mean_holding_ratio=_mean_or_none([ep.holding_ratio for ep in per_episode]),
        )


def _to_float_or_none(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _to_int_or_none(v: Any) -> Optional[int]:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def _fmt_float(v: Optional[float], digits: int = 6) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def _fmt_int(v: Optional[int]) -> str:
    if v is None:
        return "NA"
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return "NA"


def _format_episode_line(prefix: str, current_ts: int, ep_idx: int, ep: EpisodeEval) -> str:
    return (
        f"{prefix} ts={current_ts} ep={ep_idx} "
        f"death={int(bool(ep.is_death_event))} term={ep.termination_reason or 'NA'} "
        f"final={_fmt_float(ep.final_balance, digits=2)} ret={_fmt_float(ep.ret, digits=6)} "
        f"dd={_fmt_float(ep.episode_max_dd, digits=6)} cost={_fmt_float(ep.mean_cost, digits=8)} "
        f"sl={_fmt_int(ep.stop_loss_count)} liq={_fmt_int(ep.liq_count)} "
        f"trades_per_step={_fmt_float(ep.trades_per_step, digits=6)} "
        f"long_entry={_fmt_int(ep.long_entry_count)} short_entry={_fmt_int(ep.short_entry_count)} "
        f"holding_ratio={_fmt_float(ep.holding_ratio, digits=6)}"
    )

