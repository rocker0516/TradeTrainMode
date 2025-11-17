"""
資訊收集器（可擴展設計）

提供抽象基類 InfoCollector，允許用戶自定義每步/回合結束時的資訊收集邏輯。
包含默認實現和擴展實現範例。

符合 SOLID 原則：
- 單一職責：只負責資訊收集與統計
- 開放封閉：透過繼承擴展，不修改基類
- 介面隔離：分離 step 和 episode 資訊
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class InfoCollector(ABC):
    """
    資訊收集器抽象基類
    
    子類需實現：
    1. collect_step_info() - 收集每步資訊
    2. collect_episode_info() - 收集回合結束資訊
    """
    
    @abstractmethod
    def collect_step_info(
        self,
        *,
        current_step: int,
        current_price: float,
        position_size: float,
        equity: float,
        stop_loss_triggered: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """
        收集每步資訊
        
        Args:
            current_step: 當前步數
            current_price: 當前價格
            position_size: 持倉數量
            equity: 當前權益
            stop_loss_triggered: 是否觸發止損
            **kwargs: 其他可選參數
            
        Returns:
            Dict[str, Any]: 資訊字典
        """
        pass
    
    @abstractmethod
    def collect_episode_info(
        self,
        *,
        termination_reason: str,
        final_equity: float,
        initial_balance: float,
        episode_steps: int,
        executor: Any,
        data_len: int,
        window_size: int,
        episode_max_steps: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        收集回合結束資訊
        
        Args:
            termination_reason: 終止原因
            final_equity: 最終權益
            initial_balance: 初始資金
            episode_steps: 回合步數
            executor: 交易執行器（包含止損/強平統計）
            data_len: 數據長度
            window_size: 窗口大小
            episode_max_steps: 回合最大步數
            **kwargs: 其他可選參數
            
        Returns:
            Dict[str, Any]: 資訊字典
        """
        pass


class DefaultInfoCollector(InfoCollector):
    """
    默認資訊收集器
    
    收集基礎資訊：
    - 每步：價格、持倉、權益
    - 回合：成功/失敗、報酬率、交易統計
    """
    
    def collect_step_info(
        self,
        *,
        current_step: int,
        current_price: float,
        position_size: float,
        equity: float,
        stop_loss_triggered: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """收集每步基礎資訊"""
        return {
            'step': int(current_step),
            'price': float(current_price),
            'position_size': float(position_size),
            'equity': float(equity),
            'stop_loss_triggered': bool(stop_loss_triggered),
        }
    
    def collect_episode_info(
        self,
        *,
        termination_reason: str,
        final_equity: float,
        initial_balance: float,
        episode_steps: int,
        executor: Any,
        data_len: int,
        window_size: int,
        episode_max_steps: int,
        **kwargs
    ) -> Dict[str, Any]:
        """收集回合結束基礎資訊（統計從 executor 獲取）"""
        # 計算報酬率
        profit_rate = ((final_equity - initial_balance) / initial_balance) * 100.0
        
        # 判斷成功/失敗
        is_success = (termination_reason == 'data_exhausted')
        
        # 計算完成度
        completion_rate = (episode_steps / max(episode_max_steps, 1)) * 100.0
        
        return {
            'termination_reason': str(termination_reason),
            'is_success': bool(is_success),
            'final_equity': float(final_equity),
            'profit_rate': float(profit_rate),
            'episode_steps': int(episode_steps),
            'completion_rate': float(completion_rate),
            'total_fees': float(executor.total_fees),
            'long_trades': int(executor.long_close_count),
            'short_trades': int(executor.short_close_count),
            'total_trades': int(executor.long_close_count + executor.short_close_count),
            'stop_loss_count': int(executor.total_stop_loss_count),
            'liq_count': int(executor.total_liq_count),
        }


class ExtendedInfoCollector(InfoCollector):
    """
    擴展資訊收集器（範例）
    
    額外收集：
    - 每步：未實現損益、保證金使用率
    - 回合：最大回撤、夏普比率、勝率
    """
    
    def __init__(self):
        """初始化擴展統計"""
        self.equity_history = []
        self.pnl_history = []
    
    def reset(self) -> None:
        """重置歷史記錄"""
        self.equity_history = []
        self.pnl_history = []
    
    def collect_step_info(
        self,
        *,
        current_step: int,
        current_price: float,
        position_size: float,
        equity: float,
        stop_loss_triggered: bool,
        unrealized_pnl: Optional[float] = None,
        margin_usage: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """收集每步擴展資訊"""
        # 記錄權益歷史
        self.equity_history.append(equity)
        
        info = {
            'step': int(current_step),
            'price': float(current_price),
            'position_size': float(position_size),
            'equity': float(equity),
            'stop_loss_triggered': bool(stop_loss_triggered),
        }
        
        # 擴展資訊
        if unrealized_pnl is not None:
            info['unrealized_pnl'] = float(unrealized_pnl)
        
        if margin_usage is not None:
            info['margin_usage'] = float(margin_usage)
        
        return info
    
    def collect_episode_info(
        self,
        *,
        termination_reason: str,
        final_equity: float,
        initial_balance: float,
        episode_steps: int,
        executor: Any,
        data_len: int,
        window_size: int,
        episode_max_steps: int,
        **kwargs
    ) -> Dict[str, Any]:
        """收集回合結束擴展資訊（統計從 executor 獲取）"""
        # 基礎資訊
        profit_rate = ((final_equity - initial_balance) / initial_balance) * 100.0
        is_success = (termination_reason == 'data_exhausted')
        completion_rate = (episode_steps / max(episode_max_steps, 1)) * 100.0
        
        # 計算最大回撤
        max_drawdown = self._compute_max_drawdown()
        
        # 計算夏普比率（簡化版）
        sharpe_ratio = self._compute_sharpe_ratio(initial_balance)
        
        # 計算勝率
        win_rate = self._compute_win_rate(executor)
        
        # 計算平均交易損益
        avg_trade_pnl = self._compute_avg_trade_pnl(executor)
        
        info = {
            'termination_reason': str(termination_reason),
            'is_success': bool(is_success),
            'final_equity': float(final_equity),
            'profit_rate': float(profit_rate),
            'episode_steps': int(episode_steps),
            'completion_rate': float(completion_rate),
            'total_fees': float(executor.total_fees),
            'long_trades': int(executor.long_close_count),
            'short_trades': int(executor.short_close_count),
            'total_trades': int(executor.long_close_count + executor.short_close_count),
            'stop_loss_count': int(executor.total_stop_loss_count),
            'liq_count': int(executor.total_liq_count),
            # 擴展統計
            'max_drawdown': float(max_drawdown),
            'sharpe_ratio': float(sharpe_ratio),
            'win_rate': float(win_rate),
            'avg_trade_pnl': float(avg_trade_pnl),
        }
        
        return info
    
    def _compute_max_drawdown(self) -> float:
        """計算最大回撤（百分比）"""
        if len(self.equity_history) < 2:
            return 0.0
        
        import numpy as np
        equity_arr = np.array(self.equity_history)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / np.clip(peak, 1e-8, None)
        max_dd = float(np.max(drawdown)) * 100.0
        return max_dd
    
    def _compute_sharpe_ratio(self, initial_balance: float) -> float:
        """計算夏普比率（簡化版：無風險利率=0）"""
        if len(self.equity_history) < 2:
            return 0.0
        
        import numpy as np
        equity_arr = np.array(self.equity_history)
        returns = np.diff(equity_arr) / np.clip(equity_arr[:-1], 1e-8, None)
        
        if len(returns) == 0 or np.std(returns) < 1e-8:
            return 0.0
        
        sharpe = float(np.mean(returns) / np.std(returns))
        return sharpe
    
    def _compute_win_rate(self, executor: Any) -> float:
        """計算勝率（盈利交易比例）"""
        if len(executor.closed_trades) == 0:
            return 0.0
        
        win_count = sum(1 for trade in executor.closed_trades if trade['realized_pnl'] > 0)
        win_rate = (win_count / len(executor.closed_trades)) * 100.0
        return float(win_rate)
    
    def _compute_avg_trade_pnl(self, executor: Any) -> float:
        """計算平均交易損益"""
        if len(executor.closed_trades) == 0:
            return 0.0
        
        total_pnl = sum(trade['realized_pnl'] for trade in executor.closed_trades)
        avg_pnl = total_pnl / len(executor.closed_trades)
        return float(avg_pnl)
