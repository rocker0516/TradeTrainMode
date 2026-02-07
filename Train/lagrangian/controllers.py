from __future__ import annotations

import multiprocessing
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class LagrangianChannelConfig:
    """多通道 Lagrangian 參數（每條成本線各自一組）。"""

    cost_limit: float
    kp: float = 0.1
    lambda_init: float = 0.0
    lambda_min: float = 0.0
    lambda_max: float = 5.0


class SharedLagrangianController:
    """
    跨進程共享的 Lagrangian Multiplier (λ) 控制器。
    
    功能：
    - 維護全域 λ 值 (shared_lambda)。
    - 提供 update() 方法，根據 cost_violation (cost - limit) 調整 λ。
    - 實作 P-Control 與 Clamping 機制。
    """
    def __init__(
        self,
        cost_limit: float,
        kp: float = 0.1,  # P-gain
        lambda_init: float = 0.0,
        lambda_min: float = 0.0,
        lambda_max: float = 5.0,  # Clamp 上限
    ) -> None:
        self.cost_limit = float(cost_limit)
        self.kp = float(kp)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        
        # 使用 multiprocessing.Value 讓並行環境能讀取同一份 λ
        self._lambda_val = multiprocessing.Value('d', float(lambda_init))
    
    @property
    def current_lambda(self) -> float:
        with self._lambda_val.get_lock():
            return self._lambda_val.value
            
    def update(self, avg_cost: float) -> float:
        """
        根據平均 Cost 更新 λ。
        公式：λ_new = clamp(λ_old + kp * (avg_cost - limit))
        """
        violation = avg_cost - self.cost_limit
        with self._lambda_val.get_lock():
            new_val = self._lambda_val.value + self.kp * violation
            new_val = max(self.lambda_min, min(self.lambda_max, new_val))
            self._lambda_val.value = new_val
            return new_val


class MultiSharedLagrangianController:
    """
    多通道的 Shared Lagrangian Controller。

    目的：
    - 將不同「成本線」分開更新 λ，避免單一 cost 混合後尺度不一致。
    - 每個通道有自己的 limit 與 λ（P-control + clamp）。
    """

    def __init__(self, channel_configs: Dict[str, LagrangianChannelConfig]) -> None:
        if not channel_configs:
            raise ValueError("channel_configs must not be empty.")
        self.channel_configs: Dict[str, LagrangianChannelConfig] = dict(channel_configs)
        self._lambda_vals: Dict[str, multiprocessing.Value] = {
            k: multiprocessing.Value("d", float(cfg.lambda_init))
            for k, cfg in self.channel_configs.items()
        }

    @property
    def current_lambdas(self) -> Dict[str, float]:
        """取得當前所有 λ（跨進程共享值）。"""
        out: Dict[str, float] = {}
        for k, v in self._lambda_vals.items():
            with v.get_lock():
                out[k] = float(v.value)
        return out

    @property
    def current_lambda(self) -> float:
        """相容舊版：回傳所有 λ 的和（僅供 debug/舊 key）。"""
        vals = self.current_lambdas
        return float(sum(vals.values()))

    def update(self, avg_costs: Dict[str, float]) -> Dict[str, float]:
        """
        根據各通道 avg_cost 更新對應 λ。
        公式：λ_new = clamp(λ_old + kp * (avg_cost - limit))
        """
        new_vals: Dict[str, float] = {}
        for k, cfg in self.channel_configs.items():
            avg_cost = float(avg_costs.get(k, 0.0))
            violation = avg_cost - float(cfg.cost_limit)
            v = self._lambda_vals[k]
            with v.get_lock():
                new_val = float(v.value) + float(cfg.kp) * float(violation)
                new_val = max(float(cfg.lambda_min), min(float(cfg.lambda_max), float(new_val)))
                v.value = float(new_val)
                new_vals[k] = float(new_val)
        return new_vals

