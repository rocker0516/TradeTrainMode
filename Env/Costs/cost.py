from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

# Freq 通道（交易頻率/調倉幅度）常數
FREQ_BUFFER_SIZE = 500
FREQ_BUFFER_MIN_SAMPLES = 30
FREQ_TAU_QUANTILE = 0.5   # q50(turnover)；可改為 0.6
FREQ_SCALE_QUANTILE = 0.9
FREQ_TAU_DEFAULT = 0.05
FREQ_SCALE_DEFAULT = 0.2
FREQ_EPS = 1e-12


@dataclass(frozen=True)
class CostWeights:
    """
    成本權重/設定 (新架構)
    雖然公式已標準化，但保留此類別以備未來擴充 (例如是否啟用某條線的開關)。
    目前主要用於佔位，參數皆預設為 1.0 或由 Config 控制。
    """
    pass


class CostCalculator:
    """
    正規化成本計算器 (Normalized Cost Calculator)

    設計原則：
    所有成本皆正規化為「佔當前權益的比例」或 0~1 無量綱，
    確保約束條件在不同資金規模下具有尺度不變性 (Scale Invariance)。

    公式：
    1. Death Cost: 1.0 (若發生爆倉/破產，視為損失 100% 權益)
    2. Fric/Freq Cost: 控制「交易頻率/調倉幅度」，與手續費脫鉤，避免重複懲罰。
       turnover_t = |pos_t - pos_{t-1}|，c_freq = clip((turnover_t - tau)/scale, 0, 1)；
       tau = q50(turnover)，scale = max(q90(turnover) - tau, eps)。
    """

    def __init__(self, weights: CostWeights | None = None) -> None:
        self.weights = weights or CostWeights()
        self._turnover_buffer: deque[float] = deque(maxlen=FREQ_BUFFER_SIZE)

    def reset(self) -> None:
        """Episode 重置時清空 turnover 緩衝，tau/scale 下個 episode 重新累積。"""
        self._turnover_buffer.clear()

    def compute(
        self,
        *,
        liq_triggered: bool,
        equity: float,
        min_balance: float,
        step_fee: float,
        **kwargs
    ) -> Dict[str, float]:
        """
        計算正規化成本。

        Args:
            liq_triggered: 是否觸發爆倉
            equity: 當前權益 (E_t)
            min_balance: 最低資金門檻
            step_fee: 本步產生的手續費 (絕對金額；供總 cost 或其它用途，freq 通道不用)
            pos_t: （kwargs）本步實際執行後的持倉比例 (-1~1)，用於 freq 通道
            pos_prev: （kwargs）上一步實際執行後的持倉比例 (-1~1)

        Returns:
            Dict:
            - cost: 總正規化成本 (供單一 Lambda 使用)
            - cost_risk: 死亡成本 (1.0 or 0.0)
            - cost_fric: 頻率/調倉成本 c_freq (0~1)，與手續費脫鉤
            - cost_sl_buf / cost_sl_event / cost_breakdown: 其餘分項
        """
        # 防除以零保護：使用 min_balance 或極小值做為分母下限
        # 若 equity 已經低於 0，則保護值為 1e-4，避免負值或除零炸裂
        safe_equity = max(equity, 1e-4)

        # 1. 死亡/風險成本 (c_risk)
        # 定義：發生死亡事件 = 100% 權益損失風險實現 -> Cost = 1.0
        is_dead = liq_triggered or (equity <= min_balance)
        c_death = 1.0 if is_dead else 0.0

        # 1b. 止損事件成本（事件型；獨立成本線，不屬於 risk/sl_buf）
        # 定義：若本 step 觸發止損，給一個固定成本（0~1）。
        # 注意：若本 step 同時是死亡事件，death_cost 已主導；此事件成本在該步視為 0（避免重複懲罰）。
        stop_loss_triggered = bool(kwargs.get("stop_loss_triggered", False))
        stop_loss_event_cost = kwargs.get("stop_loss_event_cost", 0.0)
        try:
            stop_loss_event_cost = float(stop_loss_event_cost)
        except (TypeError, ValueError):
            stop_loss_event_cost = 0.0
        stop_loss_event_cost = float(min(1.0, max(0.0, stop_loss_event_cost)))
        c_stop_event = stop_loss_event_cost if (stop_loss_triggered and not is_dead) else 0.0

        # 風險通道：只代表死亡事件（你要求「止損獨立出來不能涵蓋在 risk」）。
        c_risk = float(c_death)

        # 2. Freq 通道 (c_freq)：控制「交易頻率/調倉幅度」，與手續費脫鉤
        # turnover_t = |pos_t - pos_{t-1}|；c_freq = clip((turnover_t - tau)/scale, 0, 1)
        # tau = q50(turnover)，scale = max(q90(turnover) - tau, eps)；pos 為實際執行後持倉比例
        pos_t = kwargs.get("pos_t")
        pos_prev = kwargs.get("pos_prev")
        if pos_t is not None and pos_prev is not None:
            pos_t_f = float(pos_t)
            pos_prev_f = float(pos_prev)
            turnover_t = float(np.clip(abs(pos_t_f - pos_prev_f), 0.0, 2.0))
            self._turnover_buffer.append(turnover_t)
            buf = np.array(self._turnover_buffer, dtype=float)
            if len(buf) >= FREQ_BUFFER_MIN_SAMPLES:
                tau = float(np.quantile(buf, FREQ_TAU_QUANTILE))
                q90 = float(np.quantile(buf, FREQ_SCALE_QUANTILE))
                scale = max(q90 - tau, FREQ_EPS)
            else:
                tau = FREQ_TAU_DEFAULT
                scale = FREQ_SCALE_DEFAULT
            raw = (turnover_t - tau) / scale
            c_freq = float(np.clip(raw, 0.0, 1.0))
        else:
            c_freq = 0.0

        # 3. Stop-Buffer Cost（止損成本線）
        # 定義距離（以 ATR 正規化）：d_t = |P_t - SL_t| / ATR_t
        # 成本：c_sl_buf = clip( max(0, d_min - d_t) / d_scale, 0, 1 )
        #
        # 語義：不是罰虧損，而是罰「你把倉位放在快撞止損的地方還不撤」。
        # 若持倉但缺少 SL 或 ATR 無法估計，視為不安全 -> stop_missing_cost = 1.0。
        has_position = bool(kwargs.get("has_position", False))
        current_price = float(kwargs.get("current_price", 0.0) or 0.0)
        stop_loss_price = float(kwargs.get("stop_loss_price", 0.0) or 0.0)
        atr = float(kwargs.get("atr", 0.0) or 0.0)
        d_min = float(kwargs.get("stop_buffer_d_min", 0.3))
        d_scale = float(kwargs.get("stop_buffer_d_scale", max(d_min, 1e-12)))
        d_scale_safe = max(d_scale, 1e-12)

        sl_buf_cost = 0.0
        stop_missing_cost = 0.0
        if has_position:
            # stop_loss_price==0 表示未設定；atr<=0 表示無法估計（資料不足或極端情況）
            if stop_loss_price <= 0.0 or atr <= 1e-12:
                stop_missing_cost = 1.0
            else:
                d_t = abs(current_price - stop_loss_price) / atr
                raw = max(0.0, d_min - d_t) / d_scale_safe
                sl_buf_cost = min(1.0, max(0.0, raw))

        # channel 值：若缺 SL，直接視為最大不安全；否則使用 buffer 公式
        c_sl_buf = max(sl_buf_cost, stop_missing_cost)

        # 總成本：death + freq + sl_buf + stop_event（freq 為 0~1 無量綱）
        total_cost = c_death + c_freq + c_sl_buf + c_stop_event

        return {
            "cost": float(total_cost),
            "cost_risk": float(c_risk),
            "cost_fric": float(c_freq),  # 對外仍用 cost_fric 鍵名，實為 freq 通道 (0~1)
            "cost_sl_buf": float(c_sl_buf),
            "cost_sl_event": float(c_stop_event),
            "cost_breakdown": {
                "death_cost": float(c_death),
                "stop_loss_event_cost": float(c_stop_event),
                "fric_cost": float(c_freq),
                "sl_buf_cost": float(sl_buf_cost),
                "stop_missing_cost": float(stop_missing_cost),
            },
        }
