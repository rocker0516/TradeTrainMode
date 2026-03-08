from __future__ import annotations

"""
Phase A/B 模型評估器。

用途：
- 用訓練同設定的 env 進行多回合評估
- 回傳結構化結果，並可選擇輸出 JSON 報告
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor


@dataclass
class EvalSummaryStats:
    """單一指標的統計摘要。"""

    mean: float
    std: float
    min: float
    max: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": float(self.mean),
            "std": float(self.std),
            "min": float(self.min),
            "max": float(self.max),
        }


def _summary(values: list[float]) -> EvalSummaryStats:
    """計算 mean/std/min/max；空值時回傳 0。"""
    if not values:
        return EvalSummaryStats(mean=0.0, std=0.0, min=0.0, max=0.0)
    arr = np.asarray(values, dtype=np.float64)
    return EvalSummaryStats(
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
    )


class PhaseABEvaluator:
    """
    Phase A/B 評估器。

    Args:
        env_builder: 建立單一 VecEnv 的工廠函式（每次評估會建立新 env）
        n_episodes: 評估回合數
        deterministic: 是否 deterministic action
        seed: 評估 reset seed（可選）
        report_path: 可選 JSON 報告輸出路徑
    """

    def __init__(
        self,
        env_builder: Callable[[], VecEnv],
        n_episodes: int = 20,
        deterministic: bool = True,
        seed: Optional[int] = None,
        report_path: str = "",
    ) -> None:
        self.env_builder = env_builder
        self.n_episodes = max(1, int(n_episodes))
        self.deterministic = bool(deterministic)
        self.seed = seed
        self.report_path = (report_path or "").strip()

    def evaluate(
        self,
        model: SAC,
        *,
        reason: str = "manual",
        step: Optional[int] = None,
        print_result: bool = True,
    ) -> dict[str, Any]:
        """執行多回合評估並回傳結果字典。"""
        env = self.env_builder()
        try:
            episode_results = self._run_episodes(model=model, env=env)
            payload = self._build_payload(episode_results=episode_results, reason=reason, step=step)
            if print_result:
                self._print_payload(payload)
            if self.report_path:
                self._write_report(payload)
            return payload
        finally:
            try:
                env.close()
            except Exception:
                pass

    def _run_episodes(self, model: SAC, env: VecEnv) -> list[dict[str, float]]:
        episodes: list[dict[str, float]] = []
        for idx in range(self.n_episodes):
            obs = env.reset()
            if self.seed is not None:
                try:
                    obs = env.reset(seed=int(self.seed) + idx)
                except TypeError:
                    obs = env.reset()

            done = False
            action_overrides: list[float] = []
            action_tracking_errors: list[float] = []
            action_execs: list[float] = []

            episode_item: dict[str, float] = {
                "episode_log_return_sum": 0.0,
                "final_balance": 0.0,
                "episode_cost_risk_sum": 0.0,
                "override_rate": 0.0,
                "tracking_error": 0.0,
                "execution_rate": 0.0,
            }

            while not done:
                action, _ = model.predict(obs, deterministic=self.deterministic)
                obs, _, dones, infos = env.step(action)
                info = infos[0] if isinstance(infos, (list, tuple)) and infos else {}

                if isinstance(info, dict):
                    action_overrides.append(float(info.get("action_overridden_flag", 0.0)))
                    raw = float(info.get("last_action_raw", 0.0))
                    final = float(info.get("last_final_pos_pct", 0.0))
                    action_tracking_errors.append(abs(raw - final))
                    action_execs.append(float(info.get("trade_executed_flag", 0.0)))

                    if "episode_log_return_sum" in info:
                        episode_item["episode_log_return_sum"] = float(info["episode_log_return_sum"])
                    if "final_balance" in info:
                        episode_item["final_balance"] = float(info["final_balance"])
                    if "episode_cost_risk_sum" in info:
                        episode_item["episode_cost_risk_sum"] = float(info["episode_cost_risk_sum"])

                done = bool(dones[0]) if isinstance(dones, np.ndarray) else bool(dones)

            if action_overrides:
                episode_item["override_rate"] = float(np.mean(action_overrides))
                episode_item["tracking_error"] = float(np.mean(action_tracking_errors))
                episode_item["execution_rate"] = float(np.mean(action_execs))
            episodes.append(episode_item)
        return episodes

    def _build_payload(
        self,
        *,
        episode_results: list[dict[str, float]],
        reason: str,
        step: Optional[int],
    ) -> dict[str, Any]:
        keys = (
            "episode_log_return_sum",
            "final_balance",
            "episode_cost_risk_sum",
            "override_rate",
            "tracking_error",
            "execution_rate",
        )
        summary: dict[str, dict[str, float]] = {}
        for key in keys:
            vals = [float(item.get(key, 0.0)) for item in episode_results]
            summary[key] = _summary(vals).to_dict()
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "reason": reason,
            "step": int(step) if step is not None else None,
            "episodes": int(self.n_episodes),
            "deterministic": bool(self.deterministic),
            "results": episode_results,
            "summary": summary,
        }

    def _print_payload(self, payload: dict[str, Any]) -> None:
        summary = payload.get("summary", {})
        print(
            "[Eval] "
            f"reason={payload.get('reason')} "
            f"step={payload.get('step')} "
            f"episodes={payload.get('episodes')} "
            f"deterministic={payload.get('deterministic')}"
        )
        for key in (
            "episode_log_return_sum",
            "final_balance",
            "episode_cost_risk_sum",
            "override_rate",
            "tracking_error",
            "execution_rate",
        ):
            item = summary.get(key, {})
            print(
                f"[Eval] {key}: "
                f"mean={float(item.get('mean', 0.0)):.6f} "
                f"std={float(item.get('std', 0.0)):.6f} "
                f"min={float(item.get('min', 0.0)):.6f} "
                f"max={float(item.get('max', 0.0)):.6f}"
            )

    def _write_report(self, payload: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.report_path) or ".", exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[Eval] report written: {self.report_path}")


def build_single_env_builder(env_thunk: Callable[[], Any]) -> Callable[[], VecEnv]:
    """
    將單環境 thunk 包成可重建的 VecEnv builder。

    評估每次建立新 env，避免狀態殘留。
    """

    def _builder() -> VecEnv:
        env = DummyVecEnv([env_thunk])
        env = VecMonitor(env)
        return env

    return _builder
