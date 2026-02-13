"""
評估配置相關的 dataclass。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalConstraints:
    """評估約束條件（constraints_then_balance）。"""

    max_dd_limit: float
    mean_cost_limit: float
    min_mean_return: float = 0.20
    """最低平均收益率門檻（例如 0.20 = +20%），未達標視為 eval 失敗"""


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

    # 預設關閉：避免在一般使用/單元測試時「第一次 eval 永遠不會跑」。
    # 若你在訓練端需要 gate，請在訓練入口顯式啟用並設定 window/min_count。
    enabled: bool = False
    window_size: int = 100
    min_max_steps_reached_count: int = 90


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
    # Render
    # - True：每個 eval episode 結束時呼叫 env.render()（用於出圖/存檔/人工觀察）
    # - 注意：是否真的顯示視窗取決於 env 端 render_show + matplotlib backend（策略 A）
    render_each_episode: bool = False
    # 只有訓練端達到「足夠多回合能撐到 max_steps」才開始 eval
    train_start_gate: TrainEvalStartGateConfig = field(default_factory=TrainEvalStartGateConfig)

