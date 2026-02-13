"""
週期性評估 callback，用於訓練過程中的模型評估和最佳模型保存。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from Train.eval.config import EvalConfig
from Train.eval.gate import TrainEpisodeWindowGate
from Train.eval.models import EpisodeEval, EvalResults
from Train.eval.utils import (
    fmt_float,
    fmt_pct,
    format_episode_line,
)


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
        self._train_start_gate = TrainEpisodeWindowGate(config=self.eval_config.train_start_gate)

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

        min_ret = float(self.eval_config.constraints.min_mean_return)
        if results.mean_return is None or results.mean_return < min_ret:
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

    def _run_eval(self, n_episodes: int, deterministic: bool, current_ts: int) -> EvalResults:
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

                done = bool(dones[0])

            # episode end: optional render (VecEnv-safe)
            #
            # 重要：env 端須啟用 render_on_done（由 EVAL_RENDER_EACH_EPISODE 控制），
            # 於終止那一步把 render_path 塞回 info；callback 僅讀取並可選列印。
            if bool(getattr(self.eval_config, "render_each_episode", False)):
                out_path = None
                try:
                    out_path = last_info.get("render_path", None) if isinstance(last_info, dict) else None
                except Exception:
                    out_path = None
                if out_path:
                    prefix = str(getattr(self.eval_config, "print_prefix", "[EVAL]"))
                    print(f"{prefix} render_saved: {out_path}", flush=True)

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
                    format_episode_line(prefix=prefix, current_ts=int(current_ts), ep_idx=int(ep_idx), ep=ep),
                    flush=True,
                )

        results = EvalResults.aggregate(per_episode=per_episode, any_death_event=any_death)

        # 每次 eval 結束後：顯示整合的 Summary
        if bool(getattr(self.eval_config, "print_each_episode", False)):
             self._print_eval_summary(prefix=str(getattr(self.eval_config, "print_prefix", "[EVAL]")), results=results)
             # 原有的 Regime Summary 保留
             self._print_regime_summary(prefix=str(getattr(self.eval_config, "print_prefix", "[EVAL]")), features=regime_features, per_episode=per_episode)

        return results

    def _print_eval_summary(self, *, prefix: str, results: EvalResults) -> None:
        """顯示易於觀察的 Gate 判斷與總結資訊。"""
        print(f"{prefix} ================== Eval Summary ==================", flush=True)
        
        # 1. Gate / Pass Rate
        pass_status = "PASS" if (results.pass_rate >= 1.0) else ("PARTIAL" if results.pass_rate > 0 else "FAIL")
        print(f"{prefix} [Gate] Status: {pass_status} | Pass Rate: {results.pass_rate:.1%} ({results.pass_count}/{results.total_count})", flush=True)
        if results.any_death_event:
            print(f"{prefix} [Gate] CRITICAL: Death Events Detected! Count: {results.death_event_count}", flush=True)

        # 2. Performance Distribution
        ret_p10 = fmt_pct(results.ret_p10, 2)
        ret_med = fmt_pct(results.ret_median, 2)
        ret_p90 = fmt_pct(results.ret_p90, 2)
        print(
            f"{prefix} [Perf] Return: "
            f"P10={ret_p10}  Median={ret_med}  P90={ret_p90} | Mean={fmt_pct(results.mean_return, 2)}",
            flush=True,
        )
        bal_mean = fmt_float(results.mean_final_balance, 2)
        print(f"{prefix} [Balance] Mean Final Balance={bal_mean}", flush=True)
        
        # 3. Risk Tail
        dd_max = fmt_float(results.dd_max, 4)
        dd_p90 = fmt_float(results.dd_p90, 4)
        print(f"{prefix} [Risk] MaxDD : Max={dd_max}  P90={dd_p90} | Mean={fmt_float(results.mean_episode_max_dd, 4)}", flush=True)

        # 4. Cost Breakdown (Mean per step)
        c_risk = fmt_float(results.mean_cost_risk, 6)
        c_fric = fmt_float(results.mean_cost_fric, 6)
        print(f"{prefix} [Cost] Mean Breakdown: Risk={c_risk} Fric={c_fric}", flush=True)

        # 5. Executability
        fees = fmt_float(results.mean_total_fees_ratio, 4)
        to_notional = fmt_float(results.mean_turnover_notional, 0)
        print(f"{prefix} [Exec] Fees/Equity={fees}  Turnover(Notional)={to_notional}", flush=True)

        # 6. Worst Episodes (Debug)
        if results.worst_episodes:
            print(f"{prefix} --- Worst 3 Episodes (by Return) ---", flush=True)
            for i, ep in enumerate(results.worst_episodes):
                # 簡化顯示：只列出關鍵 ID 與 原因
                print(
                    f"{prefix} #{i+1}: start_ts={ep.episode_start_timestamp} "
                    f"ret={fmt_pct(ep.ret, 2)} dd={fmt_float(ep.episode_max_dd, 4)} "
                    f"final_balance={fmt_float(ep.final_balance, 4)} term={ep.termination_reason}",
                    flush=True,
                )

        print(f"{prefix} ==================================================", flush=True)

    def _print_regime_summary(self, *, prefix: str, features: Iterable[str], per_episode: List[EpisodeEval]) -> None:
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
        def _collect_episode_means(episodes: List[EpisodeEval]) -> Dict[str, List[float]]:
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

    def _log_eval_results(self, results: EvalResults, current_ts: int) -> None:
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

