"""
統一正規化獎勵（依優先序）：
1) 終局（terminal）  2) 收益（return）  3) 風險（risk）  4) 結構性風險（struct）

原則：
- 各子項先正規化到固定尺度（[-1, 1] 或 [-1, 0]），再乘權重線性組合。
- 終局以 -1 表示失敗（非 data_exhausted），權重最大。
- 收益使用對數報酬縮放。
- 風險用 margin_buffer 的凸性懲罰（越接近強平懲罰越大）。
- 結構性風險含 MAE/ATR 與極值鄰近（dist_to_extreme/ATR），權重最小。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class RewardCalculator:
    """
    分層懲罰：止損 > 清算/資金不足
    目標：避免觸發止損，次要才是盈利
    """
    # 權重（以「避免止損 + 存活」為最優先目標）
    w_stop_loss: float = 50.0     # 觸發止損懲罰（固定，清晰訊號）
    w_terminal: float = 120.0     # 清算/資金不足（步數歸一化，極端失敗）
    w_risk: float = 15.0          # 每步風險（降低，因有止損保護）
    w_struct: float = 5.0         # 結構性風險（降低，止損已處理 MAE）
    w_return: float = 12.0        # 收益（維持）

    # 尺度參數
    return_clip: float = 0.02         # ±2% log return → ±1
    risk_threshold: float = 0.85      # 安全區上界
    risk_danger: float = 0.25         # 危險區下界
    mae_atr_scale: float = 3.0        # MAE 達 3 ATR → -1
    extreme_safe_atr: float = 3.0     # 距極值 3 ATR 視為安全
    terminal_steps_scale: float = 1000.0  # 終局步數歸一化基準
    
    def compute(
        self,
        *,
        last_equity: float,
        new_equity: float,
        done: bool = False,
        termination_reason: str | None = None,
        margin_buffer: float | None = None,
        turnover_ratio: float | None = None,
        dist_to_extreme_atr: float | None = None,
        mae_atr: float | None = None,
        leverage_ratio: float | None = None,
        realized_pnl_step: float | None = None,
        has_position: bool = False,
        unrealized_pnl: float | None = None,
        traded: bool = False,
        episode_steps: int | None = None,
        stop_loss_triggered: bool = False,
        **kwargs  # 忽略其他舊參數以保持兼容
    ) -> float:
        """計算加權獎勵（分層：止損 > 終局 > 風險 > 結構性 > 收益）。"""
        total = 0.0

        # 0) 止損懲罰（固定 -50，清晰訊號，優先於終局）
        if stop_loss_triggered:
            total += -1.0 * self.w_stop_loss

        # 1) 終局懲罰（步數歸一化，僅用於清算/資金不足）
        if done and termination_reason not in (None, 'data_exhausted'):
            # 步數歸一化：活越久懲罰「絕對值」越大，但「相對每步收益」仍可見
            # 例：100 步清算 → -120 × 0.1 = -12
            #     5000 步清算 → -120 × 5.0 = -600
            steps_normalized = max(episode_steps or 100, 1) / self.terminal_steps_scale
            terminal_penalty = -1.0 * self.w_terminal * steps_normalized
            total += float(np.clip(terminal_penalty, -self.w_terminal * 10, 0.0))  # 上限 10x

        # 2) 收益：log return 正規化到 [-1, 1]
        if last_equity > 1e-8 and new_equity > 1e-8 and self.return_clip > 0:
            log_ret = float(np.log(new_equity / last_equity))
            ret_norm = float(np.clip(log_ret / self.return_clip, -1.0, 1.0))
            total += ret_norm * self.w_return

        # 3) 風險：margin_buffer 的凸性懲罰 [-1, 0]
        if margin_buffer is not None and self.w_risk > 0:
            risk_pen = 0.0
            if margin_buffer < self.risk_threshold:
                if margin_buffer >= self.risk_danger:
                    # 線性區
                    deficit = (self.risk_threshold - margin_buffer) / max(self.risk_threshold - self.risk_danger, 1e-6)
                    risk_pen = -float(np.clip(deficit, 0.0, 1.0))
                else:
                    # 危險區：二次懲罰
                    deficit2 = (self.risk_danger - margin_buffer) / max(self.risk_danger, 1e-6)
                    risk_pen = -float(np.clip(deficit2, 0.0, 1.0) ** 2)
            total += risk_pen * self.w_risk

        # 4) 結構性：MAE/ATR 與 極值鄰近，合成後 clip 到 [-1, 0]
        struct_pen = 0.0
        if mae_atr is not None and self.mae_atr_scale > 0:
            mae_norm = float(np.clip(mae_atr / self.mae_atr_scale, 0.0, 1.0))
            struct_pen -= mae_norm
        if dist_to_extreme_atr is not None and self.extreme_safe_atr > 0:
            proximity = 1.0 - float(np.clip(dist_to_extreme_atr / self.extreme_safe_atr, 0.0, 1.0))
            struct_pen -= float(np.clip(proximity, 0.0, 1.0))
        struct_pen = float(np.clip(struct_pen, -1.0, 0.0))
        total += struct_pen * self.w_struct

        return float(total)
    
    def get_info(self) -> dict:
        return {
            'type': 'stop_loss_priority',
            'weights': {
                'stop_loss': self.w_stop_loss,
                'terminal': self.w_terminal,
                'return': self.w_return,
                'risk': self.w_risk,
                'struct': self.w_struct,
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

# 工廠函數保留以便外部一致使用名稱
def create_default_calculator() -> RewardCalculator:
    return RewardCalculator()
