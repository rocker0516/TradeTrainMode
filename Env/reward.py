"""
SAC-Lagrangian + RUDDER 獎勵系統

核心設計：
1. 每步 reward = PBRS shaping（小額引導）
2. 大額 outcome（收益/止損/強平）透過 RUDDER 回填到進場步
3. 成本（手續費/滑點）記錄在當步，供 Lagrangian 約束頭使用

PBRS 勢能函數 Φ：
- Φ_survival：margin_buffer 越高越好
- Φ_struct：MAE/ATR 越小、距極值越遠越好
- Φ_extreme_entry：持倉且接近極值時較高

每步 shaping reward：
- r' = γ Φ(s') - Φ(s)（PBRS 勢能差）
- 風險懲罰（凸性）
- 結構性懲罰（MAE + 極值距離）
- 進場行為抑制
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import math
import numpy as np


def _compute_risk_cost(
    *,
    margin_buffer: Optional[float],
    has_position: bool,
    risk_threshold: float,
    risk_danger: float,
) -> float:
    """Compute normalized risk cost based on margin buffer violation."""

    if not has_position or margin_buffer is None or risk_threshold <= 0:
        return 0.0

    if margin_buffer >= risk_threshold:
        return 0.0

    if margin_buffer >= risk_danger:
        deficit = (risk_threshold - margin_buffer) / max(risk_threshold - risk_danger, 1e-6)
        return float(np.clip(deficit, 0.0, 1.0))

    deficit2 = (risk_danger - margin_buffer) / max(risk_danger, 1e-6)
    return float(np.clip(deficit2, 0.0, 1.0) ** 2)


def _compute_struct_cost(
    *,
    mae_atr: Optional[float],
    dist_to_extreme_atr: Optional[float],
    has_position: bool,
    mae_atr_scale: float,
    extreme_safe_atr: float,
) -> float:
    """Compute normalized structural cost based on MAE and extreme-distance."""

    if not has_position:
        return 0.0

    cost = 0.0
    if mae_atr is not None and mae_atr_scale > 0:
        mae_norm = float(np.clip(mae_atr / mae_atr_scale, 0.0, 1.0))
        cost += mae_norm * 0.5
    if dist_to_extreme_atr is not None and extreme_safe_atr > 0:
        proximity = 1.0 - float(np.clip(dist_to_extreme_atr / extreme_safe_atr, 0.0, 1.0))
        cost += float(np.clip(proximity, 0.0, 1.0)) * 0.5
    return float(np.clip(cost, 0.0, 1.0))


@dataclass
class PBRSPotentialCalculator:
    """PBRS 勢能函數計算器（用於計算 γΦ(s') - Φ(s)）"""
    
    # 勢能權重
    w_phi_survival: float = 0.5        # 存活勢能（margin_buffer）
    w_phi_struct: float = 0.2          # 結構性勢能（MAE + 極值距離）
    w_phi_extreme_entry: float = 0.3   # 極值進場勢能
    
    # 尺度參數
    risk_threshold: float = 0.85       # 安全區上界
    risk_danger: float = 0.25          # 危險區下界
    mae_atr_scale: float = 3.0         # MAE 達 3 ATR → 勢能降為 0
    extreme_safe_atr: float = 3.0      # 距極值 3 ATR 視為安全
    gamma: float = 0.99                # 折扣因子
    
    def compute_potential(
        self,
        *,
        margin_buffer: Optional[float] = None,
        mae_atr: Optional[float] = None,
        dist_to_extreme_atr: Optional[float] = None,
        has_position: bool = False,
    ) -> float:
        """計算當前狀態的勢能 Φ(s)
        
        Args:
            margin_buffer: 保證金緩衝（0-1，越高越安全）
            mae_atr: 最大不利移動（ATR 單位）
            dist_to_extreme_atr: 距極值距離（ATR 單位）
            has_position: 是否持有倉位
            
        Returns:
            勢能值（無界，越高越好）
        """
        phi_total = 0.0
        
        # 1) Φ_survival：margin_buffer 越高越好（線性正相關）
        if margin_buffer is not None and self.w_phi_survival > 0:
            # 正規化到 [0, 1]
            phi_survival = float(np.clip(margin_buffer, 0.0, 1.0))
            phi_total += phi_survival * self.w_phi_survival
        
        # 2) Φ_struct：MAE 越小、距極值越遠越好
        if self.w_phi_struct > 0:
            phi_struct = 0.0
            # MAE/ATR：越小越好，正規化到 [0, 1]
            if mae_atr is not None and self.mae_atr_scale > 0:
                mae_norm = float(np.clip(mae_atr / self.mae_atr_scale, 0.0, 1.0))
                phi_struct += (1.0 - mae_norm) * 0.5  # 佔結構性勢能 50%
            else:
                phi_struct += 0.5  # 無 MAE 時給予最佳勢能
            
            # 距極值距離：越遠越好，正規化到 [0, 1]
            if dist_to_extreme_atr is not None and self.extreme_safe_atr > 0:
                dist_norm = float(np.clip(dist_to_extreme_atr / self.extreme_safe_atr, 0.0, 1.0))
                phi_struct += dist_norm * 0.5  # 佔結構性勢能 50%
            else:
                phi_struct += 0.5  # 無極值距離時給予最佳勢能
            
            phi_total += phi_struct * self.w_phi_struct
        
        # 3) Φ_extreme_entry：持倉且接近極值時較高（鼓勵在極值進場）
        if self.w_phi_extreme_entry > 0 and has_position:
            if dist_to_extreme_atr is not None and self.extreme_safe_atr > 0:
                # 距離越近勢能越高：1 - normalized_distance
                proximity = 1.0 - float(np.clip(dist_to_extreme_atr / self.extreme_safe_atr, 0.0, 1.0))
                phi_extreme = float(np.clip(proximity, 0.0, 1.0))
                phi_total += phi_extreme * self.w_phi_extreme_entry
        
        return float(phi_total)


@dataclass(frozen=True)
class ShapingRewardBreakdown:
    """Shaping reward 拆解與對應約束成本。"""

    total: float
    pbrs: float
    entry: float
    return_component: float
    survival_component: float
    risk_penalty: float
    struct_penalty: float
    risk_cost: float
    struct_cost: float


@dataclass
class ShapingRewardCalculator:
    """每步 Shaping Reward 計算器（小額引導，不含大額 outcome）"""
    
    # PBRS 勢能計算器
    potential_calc: PBRSPotentialCalculator = None
    
    # Shaping 權重（小額）
    w_risk: float = 20.0               # 風險懲罰（凸性）
    w_struct: float = 6.0              # 結構性懲罰（MAE + 極值距離）
    w_survival: float = 0.5            # 存活獎勵權重
    w_entry: float = 0.0               # 進場懲罰移至成本線
    w_entry_streak: float = 0.0
    
    # 尺度參數（與勢能計算器共用）
    risk_threshold: float = 0.80
    risk_danger: float = 0.20
    mae_atr_scale: float = 4.0
    extreme_safe_atr: float = 4.5
    
    def __post_init__(self):
        if self.potential_calc is None:
            self.potential_calc = PBRSPotentialCalculator(
                risk_threshold=self.risk_threshold,
                risk_danger=self.risk_danger,
                mae_atr_scale=self.mae_atr_scale,
                extreme_safe_atr=self.extreme_safe_atr,
            )

    def compute_with_breakdown(
        self,
        *,
        margin_buffer: Optional[float] = None,
        mae_atr: Optional[float] = None,
        dist_to_extreme_atr: Optional[float] = None,
        has_position: bool = False,
        prev_margin_buffer: Optional[float] = None,
        prev_mae_atr: Optional[float] = None,
        prev_dist_to_extreme_atr: Optional[float] = None,
        prev_has_position: bool = False,
        entry_happened: bool = False,
        entry_streak_count: Optional[int] = None,
        last_equity: Optional[float] = None,
        new_equity: Optional[float] = None,
        initial_equity: Optional[float] = None,
        **_: Any,
    ) -> ShapingRewardBreakdown:
        """計算當步 shaping reward 並提供拆解資訊。"""

        phi_current = self.potential_calc.compute_potential(
            margin_buffer=margin_buffer,
            mae_atr=mae_atr,
            dist_to_extreme_atr=dist_to_extreme_atr,
            has_position=has_position,
        )
        phi_prev = self.potential_calc.compute_potential(
            margin_buffer=prev_margin_buffer,
            mae_atr=prev_mae_atr,
            dist_to_extreme_atr=prev_dist_to_extreme_atr,
            has_position=prev_has_position,
        )
        pbrs_component = self.potential_calc.gamma * phi_current - phi_prev

        risk_cost = _compute_risk_cost(
            margin_buffer=margin_buffer,
            has_position=has_position,
            risk_threshold=self.risk_threshold,
            risk_danger=self.risk_danger,
        )
        risk_penalty = -risk_cost * self.w_risk

        struct_cost = _compute_struct_cost(
            mae_atr=mae_atr,
            dist_to_extreme_atr=dist_to_extreme_atr,
            has_position=has_position,
            mae_atr_scale=self.mae_atr_scale,
            extreme_safe_atr=self.extreme_safe_atr,
        )
        struct_penalty = -struct_cost * self.w_struct

        return_component = 0.0
        if (
            initial_equity is not None
            and initial_equity > 1e-8
            and last_equity is not None
            and new_equity is not None
            and last_equity > 1e-8
        ):
            last_ratio = float(last_equity / initial_equity)
            new_ratio = float(new_equity / initial_equity)
            return_component = float(np.clip(new_ratio - last_ratio, -1.0, 1.0))

        survival_component = 0.0
        buffer_value = float(np.clip(margin_buffer if margin_buffer is not None else 1.0, 0.0, 1.0))
        if self.w_survival > 0.0:
            survival_component = buffer_value * self.w_survival
            if (
                initial_equity is not None
                and initial_equity > 1e-8
                and new_equity is not None
            ):
                equity_ratio = float(max(0.0, new_equity / initial_equity))
                if equity_ratio > 1.0:
                    bonus_steps = int(math.floor(equity_ratio - 1.0))
                    if bonus_steps > 0:
                        survival_component *= 1.0 + float(bonus_steps)

        total = float(pbrs_component + return_component + survival_component)
        breakdown = ShapingRewardBreakdown(
            total=total,
            pbrs=float(pbrs_component),
            entry=0.0,
            return_component=float(return_component),
            survival_component=float(survival_component),
            risk_penalty=float(risk_penalty),
            struct_penalty=float(struct_penalty),
            risk_cost=float(risk_cost),
            struct_cost=float(struct_cost),
        )
        self.last_breakdown = breakdown
        return breakdown
    
    def compute(
        self,
        *,
        # 當前步狀態（用於計算 Φ(s')）
        margin_buffer: Optional[float] = None,
        mae_atr: Optional[float] = None,
        dist_to_extreme_atr: Optional[float] = None,
        has_position: bool = False,
        # 上一步狀態（用於計算 Φ(s)）
        prev_margin_buffer: Optional[float] = None,
        prev_mae_atr: Optional[float] = None,
        prev_dist_to_extreme_atr: Optional[float] = None,
        prev_has_position: bool = False,
        # 進場事件
        entry_happened: bool = False,
        entry_streak_count: Optional[int] = None,
        **kwargs  # 忽略其他參數
    ) -> float:
        """計算當步 shaping reward（PBRS + 小額懲罰）
        
        Returns:
            小額 reward（用於引導學習，不含大額 outcome）
        """
        breakdown = self.compute_with_breakdown(
            margin_buffer=margin_buffer,
            mae_atr=mae_atr,
            dist_to_extreme_atr=dist_to_extreme_atr,
            has_position=has_position,
            prev_margin_buffer=prev_margin_buffer,
            prev_mae_atr=prev_mae_atr,
            prev_dist_to_extreme_atr=prev_dist_to_extreme_atr,
            prev_has_position=prev_has_position,
            entry_happened=entry_happened,
            entry_streak_count=entry_streak_count,
            last_equity=kwargs.get('last_equity'),
            new_equity=kwargs.get('new_equity'),
            initial_equity=kwargs.get('initial_equity'),
        )
        return breakdown.total


@dataclass
class OutcomeCalculator:
    """Outcome 計算器（大額獎勵/懲罰，透過 RUDDER 回填到進場步）"""
    
    w_outcome: float = 1.0             # outcome 基礎權重（可調整放大正向報酬）
    
    # 止損/強平額外懲罰（合併進 outcome）
    stop_loss_penalty: float = 50.0    # 止損額外懲罰（會依 ATR 比例調整）
    liq_penalty: float = 90.0          # 強平額外懲罰（會依 ATR 比例調整）
    atr_floor: float = 1.0             # ATR 比例下限，避免除以 0
    atr_cap: float = 6.0               # ATR 比例上限，避免懲罰爆炸
    penalty_scale: float = 1.0         # 其他自訂縮放（保留彈性）
    
    def compute_outcome_delta(
        self,
        *,
        realized_pnl: float,
        notional: float,
        exit_reason: str,  # 'close' | 'reduce' | 'stop_loss' | 'liq' | 'forced_close_on_done'
        atr_multiple: Optional[float] = None,
    ) -> float:
        """計算單次出場事件的 outcome delta（回填到進場步）
        
        Args:
            realized_pnl: 已實現損益（美元）
            notional: 倉位名目價值（美元，用於正規化）
            exit_reason: 出場原因
            atr_multiple: 以 ATR 表示的移動或波動程度，用於調整懲罰量級
            
        Returns:
            outcome_delta（無界，會被回填到進場步）
        """
        if notional <= 1e-8:
            return 0.0
        
        # 方案 B：clip(realized_pnl / notional, -1, 1)
        outcome_base = float(np.clip(realized_pnl / notional, -1.0, 1.0))
        outcome_delta = outcome_base * self.w_outcome
        
        # 依 ATR 動態調整懲罰比重，避免固定常數在不同波動下過大或過小
        atr_ratio = float(atr_multiple) if atr_multiple is not None else 1.0
        atr_ratio = float(np.clip(atr_ratio, self.atr_floor, self.atr_cap))
        atr_ratio *= self.penalty_scale

        # 止損/強平額外懲罰（一起回填）
        if exit_reason == 'stop_loss':
            outcome_delta -= self.stop_loss_penalty * atr_ratio
        elif exit_reason == 'liq':
            outcome_delta -= self.liq_penalty * atr_ratio
        
        return float(outcome_delta)


@dataclass
class CostCalculator:
    """成本計算器（記錄在當步，供 Lagrangian 使用）"""
    
    # 機率違規參數
    cost_prob_target: float = 0.03     # 目標止損/強平機率（δ，例如 3%）
    
    # CVaR 尾損參數
    cvar_alpha: float = 0.1            # CVaR 置信水平（α，例如 10%）
    cvar_target: float = 0.01          # CVaR 目標上界（b_cvar，例如 1%）
    
    # 風險/結構參數（與 PBRS 共用）
    risk_threshold: float = 0.85
    risk_danger: float = 0.25
    mae_atr_scale: float = 3.0
    extreme_safe_atr: float = 3.0

    # 交易頻率約束參數（實際值由工廠函式配置）
    trade_target_per_day: float = 30.0
    entry_target_per_day: float = 30.0
    steps_per_day: int = 288
    trade_penalty: float = 1.0
    entry_penalty: float = 0.6
    entry_streak_penalty: float = 0.3
    drawdown_threshold: float = 0.15
    drawdown_weight: float = 1.0
    stop_loss_streak_limit: int = 2
    stop_loss_streak_penalty: float = 0.5
    
    def compute_step_cost(
        self,
        *,
        fee: float = 0.0,
        slippage: float = 0.0,
        funding: float = 0.0,
    ) -> float:
        """計算當步總成本（手續費 + 滑點 + 資金費用）"""
        return float(fee + slippage + funding)
    
    def compute_cost_prob(
        self,
        *,
        stop_loss_triggered: bool = False,
        liq_triggered: bool = False,
    ) -> float:
        """計算機率違規成本（1 若本步止損/強平，否則 0）"""
        return 1.0 if (stop_loss_triggered or liq_triggered) else 0.0
    
    def compute_cvar_loss_sample(
        self,
        *,
        outcome_delta: float,
    ) -> float:
        """計算 CVaR 尾損樣本（用於估計 CVaR）
        
        Args:
            outcome_delta: 當步 outcome（若有出場事件）
            
        Returns:
            loss_sample = max(0, -outcome_delta)
        """
        return float(max(0.0, -outcome_delta))

    def compute_risk_cost(
        self,
        *,
        margin_buffer: Optional[float],
        has_position: bool,
    ) -> float:
        """計算 margin buffer 違規成本。"""

        return _compute_risk_cost(
            margin_buffer=margin_buffer,
            has_position=has_position,
            risk_threshold=self.risk_threshold,
            risk_danger=self.risk_danger,
        )

    def compute_struct_cost(
        self,
        *,
        mae_atr: Optional[float],
        dist_to_extreme_atr: Optional[float],
        has_position: bool,
    ) -> float:
        """計算結構性違規成本。"""

        return _compute_struct_cost(
            mae_atr=mae_atr,
            dist_to_extreme_atr=dist_to_extreme_atr,
            has_position=has_position,
            mae_atr_scale=self.mae_atr_scale,
            extreme_safe_atr=self.extreme_safe_atr,
        )

    @property
    def trade_target_per_step(self) -> float:
        return float(self.trade_target_per_day / max(1, self.steps_per_day))

    def compute_trade_count_cost(
        self,
        *,
        traded: bool,
    ) -> float:
        """計算交易次數成本（每次交易扣除固定成本）。"""

        if not traded:
            return 0.0
        return float(self.trade_penalty)

    @property
    def entry_target_per_step(self) -> float:
        return float(self.entry_target_per_day / max(1, self.steps_per_day))

    def compute_entry_cost(
        self,
        *,
        entry_happened: bool,
        entry_streak_count: Optional[int],
    ) -> float:
        if not entry_happened:
            return 0.0
        cost = self.entry_penalty
        if entry_streak_count is not None and entry_streak_count > 1:
            cost += self.entry_streak_penalty * (entry_streak_count - 1)
        return float(cost)

    def compute_drawdown_cost(
        self,
        *,
        peak_equity: float,
        current_equity: float,
    ) -> float:
        if peak_equity <= 1e-8:
            return 0.0
        drawdown = 1.0 - (current_equity / peak_equity)
        if drawdown <= self.drawdown_threshold:
            return 0.0
        excess = drawdown - self.drawdown_threshold
        scale = max(1e-6, 1.0 - self.drawdown_threshold)
        return float(self.drawdown_weight * (excess / scale))

    def compute_stop_loss_streak_cost(
        self,
        *,
        consecutive_stop_losses: int,
    ) -> float:
        if consecutive_stop_losses <= self.stop_loss_streak_limit:
            return 0.0
        excess = consecutive_stop_losses - self.stop_loss_streak_limit
        return float(excess * self.stop_loss_streak_penalty)


# 工廠函數
def create_default_reward_system():
    """創建預設的獎勵系統（PBRS + Shaping + Outcome + Cost）"""
    potential_calc = PBRSPotentialCalculator(
        w_phi_survival=0.5,
        w_phi_struct=0.2,
        w_phi_extreme_entry=0.3,
        gamma=0.99,
    )
    
    shaping_calc = ShapingRewardCalculator(
        potential_calc=potential_calc,
        w_risk=10.0,
        w_struct=3.0,
        w_entry=0.0,
        w_entry_streak=0.0,
    )
    
    outcome_calc = OutcomeCalculator(
        w_outcome=1.0,
        stop_loss_penalty=20.0,
        liq_penalty=90.0,
    )
    
    cost_calc = CostCalculator(
        cost_prob_target=0.03,
        cvar_alpha=0.1,
        cvar_target=0.01,
        risk_threshold=0.80,
        risk_danger=0.20,
        mae_atr_scale=4.0,
        extreme_safe_atr=4.5,
        trade_target_per_day=8.0,
        entry_target_per_day=8.0,
        trade_penalty=1.0,
        entry_penalty=1.0,
        entry_streak_penalty=0.6,
        drawdown_threshold=0.15,
        drawdown_weight=1.0,
        stop_loss_streak_limit=2,
        stop_loss_streak_penalty=0.5,
    )
    
    return {
        'shaping': shaping_calc,
        'outcome': outcome_calc,
        'cost': cost_calc,
        'potential': potential_calc,
    }


# 向後兼容：保留舊的 RewardCalculator 接口（用於不支援 RUDDER 的舊訓練腳本）
@dataclass
class RewardCalculator:
    """向後兼容的獎勵計算器（整合所有獎勵，不分離 outcome）
    
    注意：這是舊版接口，新訓練應使用 create_default_reward_system()
    """
    
    # 權重（以「避免止損 + 存活」為最優先目標）
    w_stop_loss: float = 50.0
    w_terminal: float = 90.0
    w_risk: float = 3.0
    w_struct: float = 5.0
    w_return: float = 10.0
    w_entry: float = 0.7
    w_entry_streak: float = 0.3
    
    # 尺度參數
    return_clip: float = 0.02
    risk_threshold: float = 0.85
    risk_danger: float = 0.25
    mae_atr_scale: float = 3.0
    extreme_safe_atr: float = 3.0
    terminal_steps_scale: float = 3000.0
    
    def compute(
        self,
        *,
        last_equity: float,
        new_equity: float,
        done: bool = False,
        termination_reason: Optional[str] = None,
        margin_buffer: Optional[float] = None,
        turnover_ratio: Optional[float] = None,
        dist_to_extreme_atr: Optional[float] = None,
        mae_atr: Optional[float] = None,
        leverage_ratio: Optional[float] = None,
        realized_pnl_step: Optional[float] = None,
        has_position: bool = False,
        unrealized_pnl: Optional[float] = None,
        traded: bool = False,
        entry_happened: bool = False,
        entry_streak_count: Optional[int] = None,
        entry_streak_side: Optional[int] = None,
        episode_steps: Optional[int] = None,
        stop_loss_triggered: bool = False,
        **kwargs
    ) -> float:
        """計算加權獎勵（舊版整合版本，向後兼容）"""
        total = 0.0
        
        # 0) 止損懲罰
        if stop_loss_triggered:
            total += -1.0 * self.w_stop_loss
        
        # 1) 終局懲罰
        if done and termination_reason not in (None, 'data_exhausted'):
            survival = min(max((episode_steps or 0) / self.terminal_steps_scale, 0.0), 1.0)
            terminal_penalty = -1.0 * self.w_terminal * (1.0 - survival)
            total += terminal_penalty
        
        # 2) 收益
        if last_equity > 1e-8 and new_equity > 1e-8 and self.return_clip > 0:
            log_ret = float(np.log(new_equity / last_equity))
            ret_norm = float(np.clip(log_ret / self.return_clip, -1.0, 1.0))
            total += ret_norm * self.w_return
        
        # 3) 風險
        if margin_buffer is not None and self.w_risk > 0:
            risk_pen = 0.0
            if margin_buffer < self.risk_threshold:
                if margin_buffer >= self.risk_danger:
                    deficit = (self.risk_threshold - margin_buffer) / max(
                        self.risk_threshold - self.risk_danger, 1e-6
                    )
                    risk_pen = -float(np.clip(deficit, 0.0, 1.0))
                else:
                    deficit2 = (self.risk_danger - margin_buffer) / max(self.risk_danger, 1e-6)
                    risk_pen = -float(np.clip(deficit2, 0.0, 1.0) ** 2)
            total += risk_pen * self.w_risk
        
        # 4) 結構性
        struct_pen = 0.0
        if mae_atr is not None and self.mae_atr_scale > 0:
            mae_norm = float(np.clip(mae_atr / self.mae_atr_scale, 0.0, 1.0))
            struct_pen -= mae_norm
        if dist_to_extreme_atr is not None and self.extreme_safe_atr > 0:
            proximity = 1.0 - float(np.clip(dist_to_extreme_atr / self.extreme_safe_atr, 0.0, 1.0))
            struct_pen -= float(np.clip(proximity, 0.0, 1.0))
        struct_pen = float(np.clip(struct_pen, -1.0, 0.0))
        total += struct_pen * self.w_struct
        
        # 5) 進場懲罰
        if entry_happened:
            entry_pen = -1.0 * self.w_entry
            if entry_streak_count is not None and entry_streak_count > 1:
                entry_pen -= float(self.w_entry_streak * (entry_streak_count - 1))
            total += entry_pen
        
        return float(total)
    
    def get_info(self) -> dict:
        return {
            'type': 'legacy_integrated',
            'weights': {
                'stop_loss': self.w_stop_loss,
                'terminal': self.w_terminal,
                'return': self.w_return,
                'risk': self.w_risk,
                'struct': self.w_struct,
                'entry': self.w_entry,
                'entry_streak': self.w_entry_streak,
            },
            'scales': {
                'return_clip': self.return_clip,
                'risk_threshold': self.risk_threshold,
                'risk_danger': self.risk_danger,
                'mae_atr_scale': self.mae_atr_scale,
                'extreme_safe_atr': self.extreme_safe_atr,
                'terminal_steps_scale': self.terminal_steps_scale,
            }
        }


def create_default_calculator() -> RewardCalculator:
    """向後兼容的工廠函數"""
    return RewardCalculator()
