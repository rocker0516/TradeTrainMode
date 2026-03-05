"""
Log Return Reward Calculator (純主線版本)

設計目標：
- 主線只保留資產對數報酬與終局懲罰。
- 行為/風險懲罰移至成本線 (Lagrangian) 處理。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class RewardCalculator:
    """
    主線獎勵計算器：僅使用「淨」對數報酬，可調整權重。
    - Equity 已內含手續費、滑點、利息。
    - 終局懲罰改移至成本線（death cost）處理，不再在主線扣分。
    - base_log_ret_weight：控制 log-return 影響力（方案 B）
    """
    c_liq: float = 0.0
    fee_limit_penalty: float = 0.0
    base_log_ret_weight: float = 1.0

    # --- Debug / diagnostics (set on every compute call) ---
    last_conviction_bonus: float = 0.0
    last_conviction_active: bool = False
    last_regime_alignment_bonus: float = 0.0

    # Regime 對齊 bonus（A 多加分、C 空加分，依 dir_strength 加權）；0 表示不啟用
    regime_alignment_bonus_weight: float = 0.0

    def compute(
        self,
        *,
        last_equity: float,
        new_equity: float,
        done: bool = False,
        termination_reason: str | None = None,
        episode_steps: int | None = None,
        episode_max_steps: int | None = None,
        **kwargs
    ) -> float:
        """
        計算主線獎勵：log return 乘以權重；可選 regime 對齊 bonus（需 env 傳入 gate_flags、regime_score）。
        """
        self.last_conviction_bonus = 0.0
        self.last_conviction_active = False
        self.last_regime_alignment_bonus = 0.0

        safe_last = max(last_equity, 1e-8)
        safe_new = max(new_equity, 1e-8)
        log_ret = np.log(safe_new / safe_last)
        reward = float(self.base_log_ret_weight * log_ret)

        # Regime 對齊 bonus（reward sign alignment）：A 狀態多頭加分、C 狀態空頭加分，依 dir_strength 加權
        w_reg = float(getattr(self, "regime_alignment_bonus_weight", 0.0))
        if w_reg > 0.0 and "gate_flags" in kwargs and "regime_score" in kwargs:
            gate_flags = kwargs["gate_flags"]
            regime_score = kwargs["regime_score"]
            if hasattr(gate_flags, "__len__") and len(gate_flags) >= 3 and hasattr(regime_score, "__len__") and len(regime_score) >= 3:
                gate_A = float(gate_flags[0])
                gate_C = float(gate_flags[2])
                dir_strength = float(regime_score[2])
                pos_pct = float(kwargs.get("position_pct", 0.0))
                regime_dir = 1.0 if gate_A >= 0.5 else (-1.0 if gate_C <= -0.5 else 0.0)
                align = pos_pct * regime_dir
                regime_bonus = w_reg * dir_strength * np.clip(align, -1.0, 1.0)
                self.last_regime_alignment_bonus = float(
                    np.nan_to_num(regime_bonus, nan=0.0, posinf=0.0, neginf=0.0)
                )
                reward += float(regime_bonus)
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        return reward
    
    def get_info(self) -> dict:
        return {
            'type': 'log_return_only',
            'c_liq': self.c_liq,
            'base_log_ret_weight': self.base_log_ret_weight,
        }

# 工廠函數
def create_default_calculator(
    c_liq: float = 10.0,
    fee_limit_penalty: float = 2.0,
    base_log_ret_weight: float = 1.0,
    conviction_trend_bonus_weight: float = 0.0,
    conviction_trend_min_strength: float = 0.8,
    conviction_min_abs_pos: float = 0.15,
    conviction_trend_score_scale: float = 10.0,
    regime_alignment_bonus_weight: float = 0.0,
) -> RewardCalculator:
    """
    工廠函數：建立 reward calculator。

    Notes:
    - 預設仍是「純 log-return」(conviction_trend_bonus_weight=0) => 不改變現有行為。
    - 若 conviction_trend_bonus_weight > 0，則啟用「強訊號 + 大倉 + 同向」的 conviction bonus。
    - regime_alignment_bonus_weight > 0：啟用 regime 對齊 bonus（A 多加分、C 空加分，依 dir_strength 加權）。
    """
    w_reg = float(regime_alignment_bonus_weight)
    if float(conviction_trend_bonus_weight) <= 0.0:
        calc = RewardCalculator(
            c_liq=0.0,
            fee_limit_penalty=0.0,
            base_log_ret_weight=base_log_ret_weight,
            regime_alignment_bonus_weight=w_reg,
        )
        return calc
    calc = ConvictionTrendRewardCalculator(
        c_liq=0.0,
        fee_limit_penalty=0.0,
        base_log_ret_weight=base_log_ret_weight,
        conviction_trend_bonus_weight=float(conviction_trend_bonus_weight),
        conviction_trend_min_strength=float(conviction_trend_min_strength),
        conviction_min_abs_pos=float(conviction_min_abs_pos),
        conviction_trend_score_scale=float(conviction_trend_score_scale),
        regime_alignment_bonus_weight=w_reg,
    )
    return calc


@dataclass
class ConvictionTrendRewardCalculator(RewardCalculator):
    """
    Conviction + Trend Alignment Reward (可選 shaping)；
    若 env 傳入 gate_flags / regime_score，且 regime_alignment_bonus_weight > 0，會再加上 regime 對齊 bonus。

    目標：讓 agent 在「訊號明確」時，願意用「較大倉位」去承擔風險並賺取主線收益，
    而不是收斂到接近 0 的曝險。

    設計原則：
    - 順向 +、反向 -：align = pos_pct * trend_dir，正則加分、負則扣分（同權重 w）。
    - 只在訊號強度 >= 門檻、且 abs_position_pct >= 門檻時才啟用（避免小倉刷分/刷罰）。

    需要的 kwargs（由 env 提供）：
    - position_pct: [-1, 1] 以 equity*leverage 正規化的 signed exposure
    - abs_position_pct: [0, 1] position_pct 絕對值
    - trend_score: (ma50-ma200)/ma200 小數比，會乘上 conviction_trend_score_scale 後再 tanh 算 strength
    """
    conviction_trend_bonus_weight: float = 0.0
    conviction_trend_min_strength: float = 0.8
    conviction_min_abs_pos: float = 0.15
    conviction_trend_score_scale: float = 10.0

    def compute(
        self,
        *,
        last_equity: float,
        new_equity: float,
        done: bool = False,
        termination_reason: str | None = None,
        episode_steps: int | None = None,
        episode_max_steps: int | None = None,
        **kwargs
    ) -> float:
        base = super().compute(
            last_equity=last_equity,
            new_equity=new_equity,
            done=done,
            termination_reason=termination_reason,
            episode_steps=episode_steps,
            episode_max_steps=episode_max_steps,
            **kwargs
        )

        w = float(self.conviction_trend_bonus_weight)
        if w <= 0.0:
            return float(base)

        pos_pct = float(kwargs.get("position_pct", 0.0))
        abs_pos_pct = float(kwargs.get("abs_position_pct", abs(pos_pct)))
        trend_score = float(kwargs.get("trend_score", 0.0))
        scale = float(self.conviction_trend_score_scale)

        # 將小數比 trend_score 縮放後再 tanh，使 strength 能達門檻（例：scale=10 時 0.05→strength≈0.46）
        scaled = scale * np.clip(trend_score, -5.0, 5.0)
        trend_dir = float(np.tanh(scaled))
        strength = float(abs(trend_dir))

        # Gate 1: require strong enough trend signal
        s0 = float(self.conviction_trend_min_strength)
        if strength < s0:
            return float(base)

        # Gate 2: require sufficiently large exposure（避免極小倉位刷分/刷罰）
        p0 = float(self.conviction_min_abs_pos)
        if abs_pos_pct < p0:
            return float(base)

        # 順向 + / 反向 -：align = pos_pct * trend_dir，正=順向加分、負=反向扣分
        align = float(pos_pct * trend_dir)
        gate_s = float(np.clip((strength - s0) / max(1e-8, (1.0 - s0)), 0.0, 1.0))
        gate_p = float(np.clip((abs_pos_pct - p0) / max(1e-8, (1.0 - p0)), 0.0, 1.0))
        bonus = w * gate_s * gate_p * align  # align>0 加分，align<0 扣分

        self.last_conviction_bonus = float(bonus)
        self.last_conviction_active = True

        out = float(base + bonus)
        # Regime 對齊 bonus（與 base RewardCalculator 相同邏輯，依 gate_flags + regime_score）
        w_reg = float(getattr(self, "regime_alignment_bonus_weight", 0.0))
        if w_reg > 0.0 and "gate_flags" in kwargs and "regime_score" in kwargs:
            gate_flags = kwargs["gate_flags"]
            regime_score = kwargs["regime_score"]
            if hasattr(gate_flags, "__len__") and len(gate_flags) >= 3 and hasattr(regime_score, "__len__") and len(regime_score) >= 3:
                gate_A = float(gate_flags[0])
                gate_C = float(gate_flags[2])
                dir_strength = float(regime_score[2])
                pos_pct = float(kwargs.get("position_pct", 0.0))
                regime_dir = 1.0 if gate_A >= 0.5 else (-1.0 if gate_C <= -0.5 else 0.0)
                align = pos_pct * regime_dir
                regime_bonus = w_reg * dir_strength * np.clip(align, -1.0, 1.0)
                self.last_regime_alignment_bonus = float(
                    np.nan_to_num(regime_bonus, nan=0.0, posinf=0.0, neginf=0.0)
                )
                out += float(regime_bonus)
        return float(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0))
