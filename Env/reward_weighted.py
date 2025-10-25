"""
加權/正則化版本的獎勵計算器

核心改進：
1. 每個獎勵功能有獨立的權重係數
2. 各功能的獎勵貢獻被正則化到相似尺度
3. 更容易調整和平衡不同目標的重要性
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np


RewardMode = Literal['delta_equity', 'pct', 'log']


@dataclass
class WeightedRewardCalculator:
    """
    加權獎勵計算器 - 解決尺度不平衡問題
    
    設計理念：
    - 每個功能有獨立的權重係數（weight_xxx）
    - 各功能內部正則化到相似範圍（通常 -10 到 +10）
    - 最終獎勵 = Σ(功能獎勵 * 權重)
    
    優點：
    - 各獎勵項在可比較的尺度上
    - 容易調整功能重要性
    - 訓練更穩定
    
    參數說明：
        mode: 基礎獎勵模式（log 推薦）
        
        權重係數（調整各功能的重要性）：
        weight_equity: 權益增長的重要性（1.0 = 標準）
        weight_trade_cost: 交易成本的重要性（越大越抑制交易）
        weight_holding: 持倉的重要性（越大越鼓勵持倉）
        weight_risk: 風險控制的重要性（越大越規避風險）
        weight_failure: 失敗懲罰的重要性（越大越避免失敗）
    """
    mode: RewardMode = 'log'
    
    # 權重係數（可調節各功能的重要性）
    weight_equity: float = 10.0      # 權益增長權重
    weight_trade_cost: float = 5.0   # 交易成本權重（抑制頻繁交易）
    weight_holding: float = 2.0      # 持倉獎勵權重（鼓勵持倉）
    weight_risk: float = 3.0         # 風險控制權重
    weight_failure: float = 20.0     # 失敗懲罰權重
    
    # 內部正則化參數（通常不需要調整）
    _equity_scale: float = 100.0     # 將 log returns 轉換為 -10~+10 範圍
    _trade_cost_scale: float = 2.0   # 交易成本縮放
    _holding_scale: float = 1.0      # 持倉獎勵縮放
    _risk_scale: float = 10.0        # 風險懲罰縮放
    _failure_scale: float = 100.0    # 失敗懲罰縮放
    
    def compute(
        self, *,
        last_equity: float,
        new_equity: float,
        done: bool = False,
        termination_reason: str | None = None,
        margin_buffer: float | None = None,
        position_change: float | None = None,
        has_position: bool = False,
        unrealized_pnl: float | None = None,
        traded: bool = False,
    ) -> float:
        """
        計算加權正則化獎勵
        
        Returns:
            float: 總獎勵 = Σ(功能獎勵 * 權重)
        """
        total_reward = 0.0
        
        # ========================================
        # 功能 1: 權益增長（正則化到 -10 ~ +10）
        # ========================================
        if self.weight_equity > 0:
            if self.mode == 'pct':
                if last_equity > 0:
                    equity_reward = (new_equity / last_equity - 1.0) * self._equity_scale
                else:
                    equity_reward = 0.0
            elif self.mode == 'log':
                if last_equity > 0 and new_equity > 0:
                    # log(1.01) ≈ 0.01, * 100 = 1.0
                    # log(1.10) ≈ 0.095, * 100 = 9.5
                    equity_reward = float(np.log(new_equity / last_equity)) * self._equity_scale
                else:
                    equity_reward = 0.0
            else:  # delta_equity
                # 正則化：以初始資金為基準
                if last_equity > 0:
                    equity_reward = (new_equity - last_equity) / last_equity * self._equity_scale
                else:
                    equity_reward = 0.0
            
            # Clip 到合理範圍避免極端值
            equity_reward = float(np.clip(equity_reward, -50.0, 50.0))
            total_reward += equity_reward * self.weight_equity
        
        # ========================================
        # 功能 2: 交易成本（交易懲罰 + 倉位變動懲罰）
        # ========================================
        if self.weight_trade_cost > 0:
            trade_cost = 0.0
            
            # 2.1 固定交易懲罰（每次 -2.0，正則化到 -10 範圍）
            if traded:
                trade_cost -= 2.0
            
            # 2.2 倉位變動比例懲罰（正則化）
            if position_change is not None and position_change > 1e-8:
                # 假設典型倉位變動為 0.1 BTC @ 50000 = 5000 USD
                # 相對於 10000 equity = 0.5 的比例
                # 我們希望這個產生約 -5 的懲罰
                if last_equity > 0:
                    # 估算倉位變動的市值（假設平均價格 50000）
                    position_value_change = position_change * 50000.0
                    turnover_ratio = position_value_change / max(last_equity, 1.0)
                    # 正則化：典型 turnover_ratio 0.5 → 懲罰 -5
                    trade_cost -= turnover_ratio * 10.0
            
            # Clip 交易成本到合理範圍
            trade_cost = float(np.clip(trade_cost, -20.0, 0.0))
            total_reward += trade_cost * self.weight_trade_cost
        
        # ========================================
        # 功能 3: 持倉獎勵（正則化到 0 ~ +10）
        # ========================================
        if self.weight_holding > 0:
            holding_reward = 0.0
            
            if has_position:
                # 基礎持倉時間獎勵（正則化：每步 +1.0）
                holding_reward += 1.0
                
                # 獲利持倉額外獎勵（正則化到 0 ~ +5）
                if unrealized_pnl is not None and unrealized_pnl > 0 and last_equity > 0:
                    pnl_ratio = unrealized_pnl / last_equity
                    # pnl_ratio 0.01 (1%) → +5, 0.02 (2%) → +10
                    holding_reward += min(pnl_ratio * 500.0, 5.0)
                
                # 虧損持倉小幅懲罰（鼓勵止損）
                elif unrealized_pnl is not None and unrealized_pnl < 0 and last_equity > 0:
                    pnl_ratio = abs(unrealized_pnl) / last_equity
                    # 虧損 1% → -2.5
                    holding_reward -= min(pnl_ratio * 250.0, 2.5)
            
            # Clip 到合理範圍
            holding_reward = float(np.clip(holding_reward, -5.0, 10.0))
            total_reward += holding_reward * self.weight_holding
        
        # ========================================
        # 功能 4: 風險控制（保證金緩衝懲罰）
        # ========================================
        if self.weight_risk > 0:
            risk_penalty = 0.0
            
            if margin_buffer is not None:
                # 正則化：低緩衝時產生 -10 的懲罰
                safe_threshold = 0.6
                if margin_buffer < safe_threshold:
                    if margin_buffer >= 0.3:
                        # 警戒區：線性懲罰
                        # deficit 0.3 → -10
                        deficit = safe_threshold - margin_buffer
                        risk_penalty = -(deficit / 0.3) * 10.0
                    else:
                        # 危險區：指數懲罰
                        # deficit 0.2 → -20
                        deficit = 0.3 - margin_buffer
                        risk_penalty = -((deficit / 0.3) ** 2) * 20.0
            
            # Clip 到合理範圍
            risk_penalty = float(np.clip(risk_penalty, -30.0, 0.0))
            total_reward += risk_penalty * self.weight_risk
        
        # ========================================
        # 功能 5: 失敗懲罰（極大懲罰避免爆倉）
        # ========================================
        if self.weight_failure > 0:
            if done and termination_reason is not None and termination_reason != 'data_exhausted':
                # 正則化：失敗給予 -100 的懲罰
                failure_penalty = -100.0
                total_reward += failure_penalty * self.weight_failure
        
        return total_reward
    
    def get_info(self) -> dict:
        """返回當前配置資訊（用於記錄）"""
        return {
            'mode': self.mode,
            'weight_equity': self.weight_equity,
            'weight_trade_cost': self.weight_trade_cost,
            'weight_holding': self.weight_holding,
            'weight_risk': self.weight_risk,
            'weight_failure': self.weight_failure,
        }


# 預設配置（平衡版）
def create_balanced_calculator() -> WeightedRewardCalculator:
    """創建平衡的獎勵計算器"""
    return WeightedRewardCalculator(
        weight_equity=10.0,
        weight_trade_cost=5.0,
        weight_holding=2.0,
        weight_risk=3.0,
        weight_failure=20.0,
    )


# 保守配置（強調風險控制）
def create_conservative_calculator() -> WeightedRewardCalculator:
    """創建保守的獎勵計算器（強調風險控制和減少交易）"""
    return WeightedRewardCalculator(
        weight_equity=8.0,
        weight_trade_cost=10.0,  # 更高的交易成本權重
        weight_holding=3.0,
        weight_risk=5.0,         # 更高的風險權重
        weight_failure=30.0,     # 更高的失敗懲罰
    )


# 激進配置（強調收益）
def create_aggressive_calculator() -> WeightedRewardCalculator:
    """創建激進的獎勵計算器（強調收益增長）"""
    return WeightedRewardCalculator(
        weight_equity=15.0,      # 更高的權益權重
        weight_trade_cost=3.0,   # 較低的交易成本權重
        weight_holding=1.0,
        weight_risk=2.0,
        weight_failure=15.0,
    )


# 趨勢追蹤配置（鼓勵持倉）
def create_trend_following_calculator() -> WeightedRewardCalculator:
    """創建趨勢追蹤的獎勵計算器（強調持倉和趨勢）"""
    return WeightedRewardCalculator(
        weight_equity=10.0,
        weight_trade_cost=8.0,   # 較高的交易成本，抑制頻繁交易
        weight_holding=5.0,      # 較高的持倉獎勵
        weight_risk=3.0,
        weight_failure=20.0,
    )

