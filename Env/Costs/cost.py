from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

# =============================================================================
# Freq 通道常數（控制 cost_fric：約束「調倉幅度」的用法，與手續費脫鉤）
# turnover_t = |pos_t - pos_prev|（每步持倉變動 0~2）；tau/scale 由近期 turnover 分布估計
# =============================================================================

# 緩衝區大小：最近多少步的 turnover 用來估計 tau / scale（tau = 分位數(turnover)）
# 調大：tau/scale 對近期行情反應較慢、較平滑；調小：反應快但易受少數步影響
FREQ_BUFFER_SIZE = 288 * 3

# 最少樣本數：buffer 內至少幾筆 turnover 才用「分位數估計」tau/scale；不足則用 _DEFAULT
# 調大：episode 前段更久用固定 tau/scale；調小：更快切換到依資料估計
FREQ_BUFFER_MIN_SAMPLES = 288

# tau 的分位數：tau = 近期 turnover 的第 FREQ_TAU_QUANTILE 分位（例 0.5 = 中位數）
# 作用：門檻——小於/大於 tau 決定是否被罰（依 FREQ_PENALIZE_SMALL_TRADES）
# 調大（例 0.6）：tau 變大 → 懲罰小改倉時「小」的範圍變寬；懲罰大改倉時「大」的門檻變高，整體 cost 易降
# 調小（例 0.4）：tau 變小 → 更多步被算進懲罰區，整體 cost 易升
FREQ_TAU_QUANTILE = 0.3

# scale 用分位數：scale = q90(turnover) - tau（僅在 FREQ_PENALIZE_SMALL_TRADES=False 時用）
# 作用：turnover > tau 時 cost = (turnover_t - tau)/scale，scale 愈大 cost 上升愈慢
# 調大（例 0.95）：scale 變大 → 同一 turnover 的 cost 變小，大改倉被罰得較輕
# 調小：scale 變小 → 大改倉更容易頂到 cost 上限
FREQ_SCALE_QUANTILE = 0.9

# buffer 不足時 tau 的固定值（在 FREQ_BUFFER_MIN_SAMPLES 未達時使用）
# 調大：前段「小改倉罰」門檻變高、「大改倉罰」門檻也變高，前段 cost 整體偏小
FREQ_TAU_DEFAULT = 0.05

# buffer 不足時 scale 的固定值（僅在 FREQ_PENALIZE_SMALL_TRADES=False 時用）
# 調大：前段大改倉的 cost 上升較慢；調小：前段大改倉更容易被重罰
FREQ_SCALE_DEFAULT = 0.2

# 數值保護：scale 分母下限，避免除零
FREQ_EPS = 1e-12

# 輸出縮放：cost_fric = (0~1 的 c_freq_raw) × FREQ_COST_SCALE，故 cost_fric ∈ [0, FREQ_COST_SCALE]
# 調大：整條 fric 尺度變大，若 FRIC_COST_LIMIT 不變則易 violation、λ 易升
# 調小：整條 fric 尺度變小，須同步把 Train/train_config 的 FRIC_COST_LIMIT 改為「原意每步上限 × FREQ_COST_SCALE」
FREQ_COST_SCALE: float = 0.001

# 懲罰方向（二選一）：
# True   → 懲罰「小於 tau」的 turnover：0 < turnover < tau 才罰，不改倉(0)不罰、≥ tau 不罰
#          效果：抑制高頻極小步，鼓勵「要嘛不改、要嘛一次改夠大」
# False  → 懲罰「大於 tau」的 turnover：turnover > tau 才罰，cost 隨 (turnover - tau)/scale 上升
#          效果：抑制單步大改倉，鼓勵少改倉或小步改倉
FREQ_PENALIZE_SMALL_TRADES: bool = True

# =============================================================================
# 交易頻率通道常數（cost_trade_freq：約束「有交易的步數比例」，與 fric 型態約束區分）
# cost_trade_freq = 近期 W 步內「有發生調倉」的步數 / W，即 rolling ratio ∈ [0, 1]
# Fric 約束「調倉幅度型態」；本線約束「有交易的步數比例」。同一筆交易會同時影響兩者，語意不同。
# =============================================================================
TRADE_FREQ_WINDOW_SIZE: int = 288
TRADE_FREQ_MIN_SAMPLES: int = 288 / 4 # 288 / 4 = 72 筆資料


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
    2. Fric/Freq Cost: 控制「交易頻率/調倉幅度」，與手續費脫鉤。
       turnover_t = |pos_t - pos_{t-1}|，tau = q50(turnover)。
       FREQ_PENALIZE_SMALL_TRADES=True 時懲罰 0<turnover<tau（小改倉罰）；
       False 時懲罰 turnover>tau（大改倉罰，原邏輯）。
    """

    def __init__(self, weights: CostWeights | None = None) -> None:
        self.weights = weights or CostWeights()
        self._turnover_buffer: deque[float] = deque(maxlen=FREQ_BUFFER_SIZE)
        self._traded_buffer: deque[float] = deque(maxlen=TRADE_FREQ_WINDOW_SIZE)

    def reset(self) -> None:
        """Episode 重置時清空 turnover / traded 緩衝，下個 episode 重新累積。"""
        self._turnover_buffer.clear()
        self._traded_buffer.clear()

    def get_trade_freq_ratio(self) -> float:
        """
        回傳視窗內「有交易」步數比例（不修改 buffer）。
        在每步執行前呼叫時，buffer 只含之前步數，故為「到上一步為止」的滾動比例。
        """
        if len(self._traded_buffer) >= TRADE_FREQ_MIN_SAMPLES:
            return float(np.mean(self._traded_buffer))
        return 0.0

    def get_trade_freq_window_trade_count(self) -> int:
        """視窗內有交易的步數（整數）。"""
        return int(sum(self._traded_buffer))

    def get_trade_freq_window_len(self) -> int:
        """視窗當前長度（未滿時為實際累積步數）。"""
        return len(self._traded_buffer)

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
            episode_step: （kwargs）本步前已執行的步數（0-indexed），供 risk 剩餘步數加權
            max_episode_steps: （kwargs）本 episode 最大步數，供 risk 剩餘步數加權

        Returns:
            Dict:
            - cost: 總正規化成本 (供單一 Lambda 使用)
            - cost_risk: 死亡成本，依剩餘步數加權 (0~1)：剩餘越多懲罰越大
            - cost_fric: 頻率/調倉成本 c_freq (0~1)，與手續費脫鉤
            - cost_sl_buf / cost_sl_event / cost_breakdown: 其餘分項
        """
        # 防除以零保護：使用 min_balance 或極小值做為分母下限
        # 若 equity 已經低於 0，則保護值為 1e-4，避免負值或除零炸裂
        safe_equity = max(equity, 1e-4)

        # 1. 死亡/風險成本 (c_risk)
        # 定義：發生死亡事件 = 100% 權益損失風險實現 -> Cost = 1.0
        # 僅 risk_cost：剩餘步數越多懲罰越大（前期死亡比後期死亡更重）
        is_dead = liq_triggered or (equity <= min_balance)
        c_death = 1.0 if is_dead else 0.0
        episode_step = kwargs.get("episode_step")
        max_episode_steps = kwargs.get("max_episode_steps")
        if episode_step is not None and max_episode_steps is not None:
            try:
                step_i = int(episode_step)
                max_s = int(max_episode_steps)
            except (TypeError, ValueError):
                step_i, max_s = 0, 1
            if max_s <= 0:
                risk_weight = 1.0
            else:
                remaining_steps = max(0, max_s - step_i - 1)  # 本步之後剩餘步數
                risk_weight = max(0.0, min(1.0, float(remaining_steps) / float(max_s)))
            c_risk = float(c_death * risk_weight)
        else:
            c_risk = float(c_death)

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

        # 2. Freq 通道 (c_freq)：控制「交易頻率/調倉幅度」，與手續費脫鉤
        # turnover_t = |pos_t - pos_{t-1}|；tau = q50(turnover)
        # FREQ_PENALIZE_SMALL_TRADES=True：懲罰 0 < turnover_t < tau（小改倉罰）→ 抑制高頻極小步
        # FREQ_PENALIZE_SMALL_TRADES=False：懲罰 turnover_t > tau（大改倉罰，原邏輯）
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
            tau_safe = max(tau, FREQ_EPS)
            if FREQ_PENALIZE_SMALL_TRADES:
                # 懲罰小於 tau：0 < turnover_t < tau → cost = (tau - turnover_t)/tau；其餘 0
                # 不改倉(turnover=0)不罰；小改倉罰；改倉 >= tau 不罰
                if turnover_t <= 0.0:
                    c_freq_raw = 0.0
                elif turnover_t < tau:
                    c_freq_raw = float((tau_safe - turnover_t) / tau_safe)
                else:
                    c_freq_raw = 0.0
            else:
                # 原邏輯：懲罰大於 tau
                raw = (turnover_t - tau) / scale
                c_freq_raw = float(np.clip(raw, 0.0, 1.0))
            c_freq = c_freq_raw * FREQ_COST_SCALE
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

        # 4. 交易頻率通道 (c_trade_freq)：近期步數中「有發生調倉」的步數比率 ∈ [0, 1]
        traded = bool(kwargs.get("traded", False))
        self._traded_buffer.append(1.0 if traded else 0.0)
        if len(self._traded_buffer) >= TRADE_FREQ_MIN_SAMPLES:
            c_trade_freq = float(np.mean(self._traded_buffer))
        else:
            c_trade_freq = 0.0

        # 總成本：risk + freq + sl_buf + stop_event + trade_freq
        total_cost = c_risk + c_freq + c_sl_buf + c_stop_event + c_trade_freq

        return {
            "cost": float(total_cost),
            "cost_risk": float(c_risk),
            "cost_fric": float(c_freq),  # freq 通道，輸出已乘 FREQ_COST_SCALE，尺度 [0, FREQ_COST_SCALE]
            "cost_sl_buf": float(c_sl_buf),
            "cost_sl_event": float(c_stop_event),
            "cost_trade_freq": float(c_trade_freq),
            "cost_breakdown": {
                "death_cost": float(c_risk),  # 與 cost_risk 一致（含剩餘步數加權）
                "stop_loss_event_cost": float(c_stop_event),
                "fric_cost": float(c_freq),
                "sl_buf_cost": float(sl_buf_cost),
                "stop_missing_cost": float(stop_missing_cost),
                "trade_freq_cost": float(c_trade_freq),
            },
        }
