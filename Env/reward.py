"""
獎勵計算器模組

提供多種獎勵計算策略：
1. RewardCalculatorBalanced（方案 B）：平衡版獎勵（推薦）
   - 存活獎勵 + 收益獎勵 + 風險懲罰 + 止損/強平懲罰 + 終局獎勵
   - 適合生產環境，平衡風險與收益

2. RewardCalculator（原方案）：複雜版獎勵
   - 保留原有的多層次獎勵結構
   - 適合進階優化

使用方式：
```python
# 方案 B（推薦）
calculator = RewardCalculatorBalanced()

# 原方案
calculator = RewardCalculator()
```
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class RewardCalculatorBalanced:
    """
    平衡版獎勵計算器（方案 B）
    
    設計原則：
    - 鼓勵長期存活（每步小額正獎勵）
    - 獎勵收益，使用對數報酬標準化
    - 懲罰接近危險區域（margin buffer）
    - 懲罰止損/強平（但不終止）
    - 終局獎勵與整體表現掛鉤
    """
    # 權重配置
    w_survival: float = 0.1           # 存活獎勵（每步基礎分）
    w_return: float = 10.0            # 收益獎勵
    w_risk: float = 10.0              # 風險懲罰
    w_stop_loss: float = 20.0         # 止損懲罰
    w_liquidation: float = 30.0       # 強平懲罰
    w_terminal_success: float = 50.0  # 終局成功基礎分
    w_terminal_fail: float = 100.0    # 終局失敗懲罰
    w_terminal_return_bonus: float = 200.0  # 終局報酬率加成
    
    # 尺度參數
    return_clip: float = 0.02         # ±2% log return → ±1
    risk_threshold: float = 0.5       # 風險區域閾值（margin buffer < 0.5 觸發懲罰）
    
    def compute(
        self,
        *,
        last_equity: float,
        new_equity: float,
        done: bool = False,
        termination_reason: str | None = None,
        margin_buffer: float | None = None,
        realized_pnl_step: float | None = None,
        episode_steps: int | None = None,
        stop_loss_triggered: bool = False,
        initial_balance: float | None = None,
        **kwargs  # 忽略其他舊參數以保持兼容
    ) -> float:
        """
        計算平衡版獎勵
        
        Args:
            last_equity: 上一步權益
            new_equity: 當前步權益
            done: 是否終止
            termination_reason: 終止原因
            margin_buffer: 保證金緩衝比例 [0, 1]
            realized_pnl_step: 當前步已實現損益
            episode_steps: 回合步數
            stop_loss_triggered: 是否觸發止損
            initial_balance: 初始資金（用於計算終局報酬率）
            **kwargs: 其他參數
            
        Returns:
            float: 獎勵值
        """
        total = 0.0
        
        # 1. 存活獎勵（每步基礎分，鼓勵長期存活）
        if not done:
            total += self.w_survival
        
        # 2. 收益獎勵（log return 標準化）
        if last_equity > 1e-8 and new_equity > 1e-8 and self.return_clip > 0:
            log_ret = float(np.log(new_equity / last_equity))
            ret_norm = float(np.clip(log_ret / self.return_clip, -1.0, 1.0))
            total += ret_norm * self.w_return
        
        # 3. 風險懲罰（margin buffer 低於閾值時二次懲罰）
        if margin_buffer is not None and self.w_risk > 0:
            if margin_buffer < self.risk_threshold:
                # 二次懲罰：越接近 0 懲罰越大
                risk_penalty = ((self.risk_threshold - margin_buffer) / self.risk_threshold) ** 2
                total -= risk_penalty * self.w_risk
        
        # 4. 止損懲罰（觸發時扣分，但不終止）
        if stop_loss_triggered:
            total -= self.w_stop_loss
        
        # 5. 強平懲罰（觸發時扣分，但不終止）
        # 注意：liq_triggered 需要從 kwargs 傳入
        if kwargs.get('liq_triggered', False):
            total -= self.w_liquidation
        
        # 6. 終局獎勵
        if done:
            if termination_reason == 'data_exhausted':
                # 成功：基礎分 + 報酬率加成
                total += self.w_terminal_success
                
                # 報酬率加成（盈利越多獎勵越高）
                if initial_balance is not None and initial_balance > 1e-8:
                    profit_rate = (new_equity / initial_balance) - 1.0
                    total += profit_rate * self.w_terminal_return_bonus
            
            elif termination_reason == 'balance_insufficient':
                # 失敗：資金不足
                total -= self.w_terminal_fail
        
        return float(total)
    
    def get_info(self) -> dict:
        """返回獎勵配置資訊"""
        return {
            'type': 'balanced_reward',
            'weights': {
                'survival': self.w_survival,
                'return': self.w_return,
                'risk': self.w_risk,
                'stop_loss': self.w_stop_loss,
                'liquidation': self.w_liquidation,
                'terminal_success': self.w_terminal_success,
                'terminal_fail': self.w_terminal_fail,
                'terminal_return_bonus': self.w_terminal_return_bonus,
            },
            'scales': {
                'return_clip': self.return_clip,
                'risk_threshold': self.risk_threshold,
            }
        }


# ========== 原方案（保留向後兼容） ==========


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

# 工廠函數：默認使用平衡版（方案 B）
def create_default_calculator() -> RewardCalculatorBalanced:
    """創建默認獎勵計算器（方案 B：平衡版）"""
    return RewardCalculatorBalanced()


def create_advanced_calculator() -> RewardCalculator:
    """創建進階獎勵計算器（原方案：複雜版）"""
    return RewardCalculator()
