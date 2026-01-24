"""
Eval callback with detailed summary for production readiness check.
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


@dataclass(frozen=True)
class EvalConstraints:
    """評估約束條件（constraints_then_balance）。"""

    max_dd_limit: float
    mean_cost_limit: float


@dataclass(frozen=True)
class TrainEvalStartGateConfig:
    """
    訓練端「開始評估」的 gate 設定。

    目的：
    - 避免 agent 還在「大量死亡/提前結束」階段就開始 eval，讓你看到的 eval 指標更穩定且可解讀。

    規則（以最近 window_size 個 training episodes 為準）：
    - 若其中 max_steps_reached_count >= min_max_steps_reached_count，才允許開始跑 periodic eval。

    注意：
    - 這裡使用 training env 的 info["termination_reason"] 判斷 episode 是否為 max_steps_reached。
    - 當 gate 尚未達標時，eval callback 會直接跳過本次 eval（不更新 best、不寫入 eval 指標）。
    """

    enabled: bool = True
    window_size: int = 100
    min_max_steps_reached_count: int = 70


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
    # 只有訓練端達到「足夠多回合能撐到 max_steps」才開始 eval
    train_start_gate: TrainEvalStartGateConfig = field(default_factory=TrainEvalStartGateConfig)


class _TrainEpisodeWindowGate:
    """維護最近 N 個訓練回合，判斷是否可開始 eval。"""

    def __init__(self, *, config: TrainEvalStartGateConfig) -> None:
        self._cfg = config
        self._is_max_steps_hist: Deque[bool] = deque(maxlen=int(max(1, config.window_size)))
        self._max_steps_count: int = 0

    def update_from_infos(self, infos: Any) -> None:
        """
        從 SB3 callback locals 的 infos 更新 episode window。

        Args:
            infos: 一般是 List[dict]（VecEnv 每個 env 一個 info）
        """
        if not bool(getattr(self._cfg, "enabled", True)):
            return
        if not isinstance(infos, (list, tuple)):
            return

        for info in infos:
            if not isinstance(info, dict):
                continue

            # VecMonitor 在 episode 結束時會附加 "episode" dict；同時我們的 env 會給 termination_reason
            termination_reason = info.get("termination_reason", None)
            is_episode_end = ("episode" in info) or bool(info.get("terminated", False)) or bool(info.get("truncated", False))
            if (not is_episode_end) or (termination_reason is None):
                continue

            is_max_steps = bool(str(termination_reason) == "max_steps_reached")

            # 維護 max_steps_count（用 deque 的 pop 補償）
            if len(self._is_max_steps_hist) == self._is_max_steps_hist.maxlen:
                old = bool(self._is_max_steps_hist[0])
                if old:
                    self._max_steps_count = int(max(0, self._max_steps_count - 1))

            self._is_max_steps_hist.append(is_max_steps)
            if is_max_steps:
                self._max_steps_count = int(self._max_steps_count + 1)

    def is_ready(self) -> bool:
        """是否已達到可開始 eval 的條件。"""
        if not bool(getattr(self._cfg, "enabled", True)):
            return True
        if len(self._is_max_steps_hist) < int(self._cfg.window_size):
            return False
        return int(self._max_steps_count) >= int(self._cfg.min_max_steps_reached_count)

    def snapshot(self) -> Dict[str, Any]:
        """供 debug 使用的狀態快照。"""
        return {
            "enabled": bool(self._cfg.enabled),
            "window_size": int(self._cfg.window_size),
            "min_max_steps_reached_count": int(self._cfg.min_max_steps_reached_count),
            "seen_episodes": int(len(self._is_max_steps_hist)),
            "max_steps_reached_count": int(self._max_steps_count),
        }


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
        self._train_start_gate = _TrainEpisodeWindowGate(config=self.eval_config.train_start_gate)

    def _on_step(self) -> bool:
        if not bool(self.eval_config.enabled):
            return True

        # 更新「訓練端 episode window」：尚未達標前不跑 eval
        try:
            self._train_start_gate.update_from_infos(self.locals.get("infos", None))
        except Exception:
            # gate 是輔助功能，不應阻斷訓練
            pass
        if not bool(self._train_start_gate.is_ready()):
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
        # 你要求要看的 regime 指標（FeatureTransformer 5m columns）
        regime_features: Tuple[str, ...] = (
            # 波動面
            "atr_ratio_z",
            "rv_ratio_z",
            "bb_width_48_z",
            "hl_range_20_z",
            # 趨勢/盤整面
            "trend_strength_atr",
            "chop_48",
            "trend_flip_rate_48",
            "dir_persist_20",
            # 廣度面
            "alts_trend_up_ratio",
            "alts_ret_15m_std_z",
            "alts_trend_spread_std_z",
        )

        for ep_idx in range(int(n_episodes)):
            obs = self.eval_env.reset()

            done = False
            ep_steps = 0
            cost_sum = 0.0
            
            # 累積成本分解
            cost_risk_sum = 0.0
            cost_fric_sum = 0.0
            cost_sl_buf_sum = 0.0
            cost_sl_event_sum = 0.0

            last_info: Dict[str, Any] = {}
            # episode 起始資訊：由 env 提供（避免 callback 自己猜）
            start_ts: Optional[str] = None
            start_step: Optional[int] = None
            try:
                start_ts = self.eval_env.env_method("get_episode_start_timestamp")[0]
            except Exception:
                start_ts = None
            try:
                start_step = int(self.eval_env.env_method("get_episode_start_step")[0])
            except Exception:
                start_step = None

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, _reward, dones, infos = self.eval_env.step(action)
                ep_steps += 1

                info = infos[0] if isinstance(infos, (list, tuple)) and infos else {}
                last_info = info

                # 累加總 Cost
                step_cost = info.get("cost", None)
                if step_cost is not None:
                    try:
                        cost_sum += float(step_cost)
                    except (TypeError, ValueError):
                        pass
                
                # 累加成本分解
                def _add_cost(key: str, current_sum: float) -> float:
                    val = info.get(key, 0.0)
                    try:
                        return current_sum + float(val)
                    except (TypeError, ValueError):
                        return current_sum

                cost_risk_sum = _add_cost("cost_risk", cost_risk_sum)
                cost_fric_sum = _add_cost("cost_fric", cost_fric_sum)
                cost_sl_buf_sum = _add_cost("cost_sl_buf", cost_sl_buf_sum)
                cost_sl_event_sum = _add_cost("cost_sl_event", cost_sl_event_sum)

                done = bool(dones[0])

            # 以 info["episode_steps"] 為主（最貼近 env 內部結束時記錄），
            # 失敗再回退到 env_method 或 loop 計數。
            env_episode_steps: int = int(ep_steps)
            try:
                if "episode_steps" in last_info:
                    env_episode_steps = int(last_info.get("episode_steps", ep_steps))
            except Exception:
                pass
            if env_episode_steps <= 0:
                try:
                    env_episode_steps = int(self.eval_env.env_method("get_episode_steps")[0])
                except Exception:
                    env_episode_steps = int(ep_steps)

            # 取 regime 指標的 episode slice 統計（mean/std/quantiles）
            episode_regime_stats: Optional[Dict[str, Dict[str, float]]] = None
            try:
                episode_regime_stats = self.eval_env.env_method(
                    "get_episode_feature_stats",
                    list(regime_features),
                )[0]
            except Exception:
                episode_regime_stats = None

            ep = EpisodeEval.from_episode_end_info(
                info=last_info,
                episode_steps=int(env_episode_steps),
                cost_sum=float(cost_sum),
                cost_breakdown_sums={
                    "risk": cost_risk_sum,
                    "fric": cost_fric_sum,
                    "sl_buf": cost_sl_buf_sum,
                    "sl_event": cost_sl_event_sum,
                },
                initial_balance=float(initial_balance),
                constraints=self.eval_config.constraints,
                episode_start_timestamp=start_ts,
                episode_start_step=start_step,
                regime_stats=episode_regime_stats,
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

        results = EvalResults.aggregate(per_episode=per_episode, any_death_event=any_death)

        # 每次 eval 結束後：顯示整合的 Summary
        if bool(getattr(self.eval_config, "print_each_episode", False)):
             self._print_eval_summary(prefix=str(getattr(self.eval_config, "print_prefix", "[EVAL]")), results=results)
             # 原有的 Regime Summary 保留
             self._print_regime_summary(prefix=str(getattr(self.eval_config, "print_prefix", "[EVAL]")), features=regime_features, per_episode=per_episode)

        return results

    def _print_eval_summary(self, *, prefix: str, results: "EvalResults") -> None:
        """顯示易於觀察的 Gate 判斷與總結資訊。"""
        print(f"{prefix} ================== Eval Summary ==================", flush=True)
        
        # 1. Gate / Pass Rate
        pass_status = "PASS" if (results.pass_rate >= 1.0) else ("PARTIAL" if results.pass_rate > 0 else "FAIL")
        print(f"{prefix} [Gate] Status: {pass_status} | Pass Rate: {results.pass_rate:.1%} ({results.pass_count}/{results.total_count})", flush=True)
        if results.any_death_event:
            print(f"{prefix} [Gate] CRITICAL: Death Events Detected! Count: {results.death_event_count}", flush=True)

        # 2. Performance Distribution
        ret_p10 = _fmt_pct(results.ret_p10, 2)
        ret_med = _fmt_pct(results.ret_median, 2)
        ret_p90 = _fmt_pct(results.ret_p90, 2)
        print(
            f"{prefix} [Perf] Return: "
            f"P10={ret_p10}  Median={ret_med}  P90={ret_p90} | Mean={_fmt_pct(results.mean_return, 2)}",
            flush=True,
        )
        bal_mean = _fmt_float(results.mean_final_balance, 2)
        print(f"{prefix} [Balance] Mean Final Balance={bal_mean}", flush=True)
        
        # 3. Risk Tail
        dd_max = _fmt_float(results.dd_max, 4)
        dd_p90 = _fmt_float(results.dd_p90, 4)
        print(f"{prefix} [Risk] MaxDD : Max={dd_max}  P90={dd_p90} | Mean={_fmt_float(results.mean_episode_max_dd, 4)}", flush=True)

        # 4. Cost Breakdown (Mean per step)
        c_risk = _fmt_float(results.mean_cost_risk, 6)
        c_fric = _fmt_float(results.mean_cost_fric, 6)
        c_slb = _fmt_float(results.mean_cost_sl_buf, 6)
        c_sle = _fmt_float(results.mean_cost_sl_event, 6)
        print(f"{prefix} [Cost] Mean Breakdown: Risk={c_risk} Fric={c_fric} SL_Buf={c_slb} SL_Event={c_sle}", flush=True)

        # 5. Executability
        fees = _fmt_float(results.mean_total_fees_ratio, 4)
        to_notional = _fmt_float(results.mean_turnover_notional, 0)
        print(f"{prefix} [Exec] Fees/Equity={fees}  Turnover(Notional)={to_notional}", flush=True)

        # 6. Worst Episodes (Debug)
        if results.worst_episodes:
            print(f"{prefix} --- Worst 3 Episodes (by Return) ---", flush=True)
            for i, ep in enumerate(results.worst_episodes):
                # 簡化顯示：只列出關鍵 ID 與 原因
                print(
                    f"{prefix} #{i+1}: start_ts={ep.episode_start_timestamp} "
                    f"ret={_fmt_pct(ep.ret, 2)} dd={_fmt_float(ep.episode_max_dd, 4)} "
                    f"final_balance={_fmt_float(ep.final_balance, 4)} term={ep.termination_reason}",
                    flush=True,
                )

        print(f"{prefix} ==================================================", flush=True)

    def _print_regime_summary(self, *, prefix: str, features: Iterable[str], per_episode: List["EpisodeEval"]) -> None:
        """
        額外印出：
        - eval 資料整段的 regime 特徵分布（mean/std/q10/q50/q90）
        - death vs alive episodes 的 regime mean（用 episode slice 的 mean）
        """
        # 1) 整段資料的分布
        dist: Dict[str, Dict[str, float]] = {}
        try:
            dist = self.eval_env.env_method("get_feature_distribution_stats", list(features))[0]
        except Exception:
            dist = {}

        if dist:
            print(f"{prefix} --- Regime Feature Distribution (full eval window) ---", flush=True)
            for k in list(features):
                if k not in dist:
                    continue
                d = dist[k]
                mean = d.get("mean", 0.0)
                std = d.get("std", 0.0)
                q10 = d.get("q10", float("nan"))
                q50 = d.get("q50", float("nan"))
                q90 = d.get("q90", float("nan"))
                print(f"{prefix} {k:<22} mean={mean: .4f} std={std: .4f} q10={q10: .4f} q50={q50: .4f} q90={q90: .4f}", flush=True)

        # 2) death vs alive：用 episode slice 的 mean 做比較（最能對症）
        def _collect_episode_means(episodes: List["EpisodeEval"]) -> Dict[str, List[float]]:
            out: Dict[str, List[float]] = {str(f): [] for f in features}
            for ep in episodes:
                if not ep.regime_stats:
                    continue
                for f in features:
                    dd = ep.regime_stats.get(str(f), {})
                    if "mean" in dd:
                        try:
                            out[str(f)].append(float(dd["mean"]))
                        except (TypeError, ValueError):
                            pass
            return out

        deaths = [ep for ep in per_episode if bool(ep.is_death_event)]
        alives = [ep for ep in per_episode if not bool(ep.is_death_event)]
        if deaths and alives:
            d_means = _collect_episode_means(deaths)
            a_means = _collect_episode_means(alives)
            print(f"{prefix} --- Regime Mean (episode slice): death vs alive ---", flush=True)
            for f in list(features):
                dv = d_means.get(str(f), [])
                av = a_means.get(str(f), [])
                if not dv or not av:
                    continue
                print(
                    f"{prefix} {str(f):<22} alive_mean={float(np.mean(av)): .4f}  death_mean={float(np.mean(dv)): .4f}  (Δ={float(np.mean(dv) - np.mean(av)): .4f})",
                    flush=True,
                )

    def _log_eval_results(self, results: "EvalResults", current_ts: int) -> None:
        # 基礎資訊
        self.logger.record("eval/timesteps", float(current_ts))
        self.logger.record("eval/any_death_event", float(1.0 if results.any_death_event else 0.0))
        self.logger.record("eval/death_event_count", float(results.death_event_count))
        self.logger.record("eval/pass_rate", float(results.pass_rate))

        # 績效（分位數）
        if results.mean_final_balance is not None:
            self.logger.record("eval/mean_final_balance", float(results.mean_final_balance))
        if results.mean_return is not None:
            self.logger.record("eval/mean_return", float(results.mean_return))
        if results.ret_p10 is not None:
             self.logger.record("eval/ret_p10", float(results.ret_p10))
        if results.ret_median is not None:
             self.logger.record("eval/ret_median", float(results.ret_median))
        if results.ret_p90 is not None:
             self.logger.record("eval/ret_p90", float(results.ret_p90))

        # 風險（尾部）
        if results.mean_episode_max_dd is not None:
            self.logger.record("eval/mean_episode_max_dd", float(results.mean_episode_max_dd))
        if results.dd_p90 is not None:
            self.logger.record("eval/dd_p90", float(results.dd_p90))
        if results.dd_max is not None:
            self.logger.record("eval/dd_max", float(results.dd_max))

        # 成本總體
        if results.mean_cost is not None:
            self.logger.record("eval/mean_cost", float(results.mean_cost))
        
        # 成本分解
        if results.mean_cost_risk is not None: self.logger.record("eval/mean_cost_risk", float(results.mean_cost_risk))
        if results.mean_cost_fric is not None: self.logger.record("eval/mean_cost_fric", float(results.mean_cost_fric))
        if results.mean_cost_sl_buf is not None: self.logger.record("eval/mean_cost_sl_buf", float(results.mean_cost_sl_buf))
        if results.mean_cost_sl_event is not None: self.logger.record("eval/mean_cost_sl_event", float(results.mean_cost_sl_event))

        # 交易統計
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
        
        # 可執行性
        if results.mean_total_fees_ratio is not None:
            self.logger.record("eval/mean_total_fees_ratio", float(results.mean_total_fees_ratio))
        if results.mean_turnover_notional is not None:
            self.logger.record("eval/mean_turnover_notional", float(results.mean_turnover_notional))

        # best 追蹤
        self.logger.record("eval/best_mean_final_balance", float(self._best_mean_final_balance))

        # 強制寫入
        self.logger.dump(step=int(current_ts))


@dataclass(frozen=True)
class EpisodeEval:
    """單一 episode 的評估結果（由 episode 結束時 info + 本地累加值構成）。"""

    final_balance: Optional[float]
    episode_max_dd: Optional[float]
    episode_steps: Optional[int]
    episode_start_step: Optional[int]
    episode_start_timestamp: Optional[str]
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
    
    # 新增：可執行性指標
    total_fees_ratio: Optional[float] = None
    turnover_notional: Optional[float] = None
    final_position_size: Optional[float] = None
    episode_active_exit_count: Optional[int] = None
    
    # 新增：成本分解 (Mean per step)
    mean_cost_risk: Optional[float] = None
    mean_cost_fric: Optional[float] = None
    mean_cost_sl_buf: Optional[float] = None
    mean_cost_sl_event: Optional[float] = None
    
    # 新增：是否通過上線門檻
    passed: bool = False

    # 額外：episode slice 的 regime 統計（每個 feature: mean/std/q10/q50/q90）
    regime_stats: Optional[Dict[str, Dict[str, float]]] = None

    @staticmethod
    def from_episode_end_info(
        info: Dict[str, Any],
        episode_steps: int,
        cost_sum: float,
        initial_balance: float,
        constraints: EvalConstraints,
        *,
        cost_breakdown_sums: Dict[str, float] = None,
        episode_start_timestamp: Optional[str] = None,
        episode_start_step: Optional[int] = None,
        regime_stats: Optional[Dict[str, Dict[str, float]]] = None,
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
        
        # 新增欄位讀取
        total_fees_ratio = _to_float_or_none(info.get("total_fees_ratio", None))
        turnover_notional = _to_float_or_none(info.get("episode_turnover_notional", None))
        final_position_size = _to_float_or_none(info.get("final_position_size", None))
        episode_active_exit_count = _to_int_or_none(info.get("episode_active_exit_count", None))

        steps = max(1, int(episode_steps))
        holding_ratio = None
        if holding_steps is not None:
            holding_ratio = float(max(0, holding_steps)) / float(steps)

        trades_per_step = None
        if trade_count is not None:
            trades_per_step = float(max(0, trade_count)) / float(steps)

        mean_cost = float(cost_sum) / float(steps) if steps > 0 else None
        
        # 成本分解均值
        c_sums = cost_breakdown_sums or {}
        mean_cost_risk = float(c_sums.get("risk", 0.0)) / float(steps) if steps > 0 else 0.0
        mean_cost_fric = float(c_sums.get("fric", 0.0)) / float(steps) if steps > 0 else 0.0
        mean_cost_sl_buf = float(c_sums.get("sl_buf", 0.0)) / float(steps) if steps > 0 else 0.0
        mean_cost_sl_event = float(c_sums.get("sl_event", 0.0)) / float(steps) if steps > 0 else 0.0

        ret = None
        if (final_balance is not None) and (initial_balance > 0):
            ret = float(final_balance) / float(initial_balance) - 1.0

        # 死亡事件判定（雙保險）：termination_reason or liq_count>0
        is_death_event = False
        if termination_reason in {"liq_triggered", "balance_insufficient"}:
            is_death_event = True
        if (liq_count is not None) and (liq_count > 0):
            is_death_event = True
            
        # Pass 判定
        # 1. 不死
        # 2. MaxDD <= limit
        # 3. MeanCost <= limit
        # 4. Ret >= 0
        passed = True
        if is_death_event:
            passed = False
        if (episode_max_dd is not None) and (episode_max_dd > constraints.max_dd_limit):
            passed = False
        if (mean_cost is not None) and (mean_cost > constraints.mean_cost_limit):
            passed = False
        if (ret is not None) and (ret < 0):
            passed = False

        return EpisodeEval(
            final_balance=final_balance,
            episode_max_dd=episode_max_dd,
            episode_steps=int(episode_steps),
            episode_start_step=_to_int_or_none(info.get("episode_start_step", episode_start_step)),
            episode_start_timestamp=str(info.get("episode_start_timestamp", episode_start_timestamp)) if (info.get("episode_start_timestamp", episode_start_timestamp) is not None) else None,
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
            total_fees_ratio=total_fees_ratio,
            turnover_notional=turnover_notional,
            final_position_size=final_position_size,
            episode_active_exit_count=episode_active_exit_count,
            mean_cost_risk=mean_cost_risk,
            mean_cost_fric=mean_cost_fric,
            mean_cost_sl_buf=mean_cost_sl_buf,
            mean_cost_sl_event=mean_cost_sl_event,
            passed=passed,
            regime_stats=regime_stats,
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
    
    # 新增：分位數與尾部風險
    ret_p10: Optional[float]
    ret_median: Optional[float]
    ret_p90: Optional[float]
    dd_p90: Optional[float]
    dd_max: Optional[float]
    
    # 新增：Gate 統計
    pass_rate: float
    pass_count: int
    total_count: int
    
    # 新增：執行性
    mean_total_fees_ratio: Optional[float]
    mean_turnover_notional: Optional[float]
    
    # 新增：成本分解
    mean_cost_risk: Optional[float]
    mean_cost_fric: Optional[float]
    mean_cost_sl_buf: Optional[float]
    mean_cost_sl_event: Optional[float]
    
    # 用於顯示
    worst_episodes: List[EpisodeEval]

    @staticmethod
    def aggregate(per_episode: List[EpisodeEval], any_death_event: bool) -> "EvalResults":
        death_event_count = int(sum(1 for ep in per_episode if ep.is_death_event))
        pass_count = int(sum(1 for ep in per_episode if ep.passed))
        total_count = len(per_episode)
        pass_rate = float(pass_count) / float(total_count) if total_count > 0 else 0.0

        # 計算分位數
        rets = [ep.ret for ep in per_episode if ep.ret is not None]
        dds = [ep.episode_max_dd for ep in per_episode if ep.episode_max_dd is not None]
        
        def _quantile(data, q):
            if not data: return None
            return float(np.quantile(data, q))
            
        ret_p10 = _quantile(rets, 0.1)
        ret_median = _quantile(rets, 0.5)
        ret_p90 = _quantile(rets, 0.9)
        dd_p90 = _quantile(dds, 0.9)
        dd_max = float(np.max(dds)) if dds else None
        
        # 找出最差 3 個 episode (依 ret 排序)
        valid_eps = [ep for ep in per_episode if ep.ret is not None]
        worst_episodes = sorted(valid_eps, key=lambda x: x.ret)[:3]

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
            ret_p10=ret_p10,
            ret_median=ret_median,
            ret_p90=ret_p90,
            dd_p90=dd_p90,
            dd_max=dd_max,
            pass_rate=pass_rate,
            pass_count=pass_count,
            total_count=total_count,
            mean_total_fees_ratio=_mean_or_none([ep.total_fees_ratio for ep in per_episode]),
            mean_turnover_notional=_mean_or_none([ep.turnover_notional for ep in per_episode]),
            mean_cost_risk=_mean_or_none([ep.mean_cost_risk for ep in per_episode]),
            mean_cost_fric=_mean_or_none([ep.mean_cost_fric for ep in per_episode]),
            mean_cost_sl_buf=_mean_or_none([ep.mean_cost_sl_buf for ep in per_episode]),
            mean_cost_sl_event=_mean_or_none([ep.mean_cost_sl_event for ep in per_episode]),
            worst_episodes=worst_episodes,
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


def _fmt_pct(v: Optional[float], digits: int = 2) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v) * 100:.{digits}f}%"
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
    # 新增顯示：start_step, fee%, pos (final pos)
    # 格式：[EVAL] ts=... ep=... PASS/FAIL ...
    pass_str = "PASS" if ep.passed else "FAIL"
    term_str = ep.termination_reason or 'NA'
    
    return (
        f"{prefix} ts={current_ts} ep={ep_idx} [{pass_str}] "
        f"term={term_str} start={ep.episode_start_timestamp or 'NA'}(idx={_fmt_int(ep.episode_start_step)}) "
        f"steps={_fmt_int(ep.episode_steps)} "
        f"ret={_fmt_pct(ep.ret, 2)} dd={_fmt_float(ep.episode_max_dd, 4)} "
        f"fee%={_fmt_float(ep.total_fees_ratio, 4)} "
        f"cost={_fmt_float(ep.mean_cost, 6)} "
        f"sl={_fmt_int(ep.stop_loss_count)} "
        f"tr/s={_fmt_float(ep.trades_per_step, 4)} "
        f"hold={_fmt_float(ep.holding_ratio, 2)} "
        f"bal={_fmt_float(ep.final_balance, 2)} "
        f"pos={_fmt_float(ep.final_position_size, 4)}"
    )
