"""
帳戶特徵構建器（可擴展設計）

提供抽象基類 AccountFeatureBuilder，允許用戶自定義帳戶狀態特徵。
包含默認實現和擴展實現範例。

符合 SOLID 原則：
- 單一職責：只負責帳戶特徵計算
- 開放封閉：透過繼承擴展，不修改基類
- 介面隔離：專注的抽象方法
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np


class AccountFeatureBuilder(ABC):
    """
    帳戶特徵構建器抽象基類
    
    子類需實現：
    1. feature_names() - 返回特徵名稱列表
    2. feature_count() - 返回特徵數量
    3. compute_features() - 計算特徵值
    """
    
    @abstractmethod
    def feature_names(self) -> List[str]:
        """返回特徵名稱列表"""
        pass
    
    @abstractmethod
    def feature_count(self) -> int:
        """返回特徵數量"""
        pass
    
    @abstractmethod
    def compute_features(
        self,
        *,
        position_size: float,
        position_value: float,
        equity: float,
        wallet_balance: float,
        current_price: float,
        initial_balance: float,
        **kwargs
    ) -> np.ndarray:
        """
        計算帳戶特徵
        
        Args:
            position_size: 持倉數量（正數=多，負數=空）
            position_value: 持倉價值（絕對值）
            equity: 當前權益
            wallet_balance: 錢包餘額
            current_price: 當前價格
            initial_balance: 初始資金
            **kwargs: 其他可選參數
            
        Returns:
            np.ndarray: 特徵值陣列 [feature_count]
        """
        pass


class DefaultAccountFeatureBuilder(AccountFeatureBuilder):
    """
    默認帳戶特徵構建器
    
    特徵：
    1. position_ratio: 持倉方向與強度 [-1, 1]
    2. equity_ratio: 權益比例 (equity / initial_balance)
    3. wallet_ratio: 錢包比例 (wallet_balance / initial_balance)
    4. unrealized_ratio: 未實現損益比例 ((equity - wallet) / initial_balance)
    """
    
    def feature_names(self) -> List[str]:
        return [
            'position_ratio',
            'equity_ratio',
            'wallet_ratio',
            'unrealized_ratio'
        ]
    
    def feature_count(self) -> int:
        return len(self.feature_names())
    
    def compute_features(
        self,
        *,
        position_size: float,
        position_value: float,
        equity: float,
        wallet_balance: float,
        current_price: float,
        initial_balance: float,
        **kwargs
    ) -> np.ndarray:
        """計算基礎帳戶特徵"""
        # 1. 持倉方向與強度（歸一化到 [-1, 1]）
        # position_ratio = position_value / equity (帶符號)
        if equity > 1e-8:
            position_ratio = (position_size * current_price) / equity
            position_ratio = float(np.clip(position_ratio, -1.0, 1.0))
        else:
            position_ratio = 0.0
        
        # 2. 權益比例
        if initial_balance > 1e-8:
            equity_ratio = equity / initial_balance
        else:
            equity_ratio = 1.0
        
        # 3. 錢包比例
        if initial_balance > 1e-8:
            wallet_ratio = wallet_balance / initial_balance
        else:
            wallet_ratio = 1.0
        
        # 4. 未實現損益比例
        unrealized_pnl = equity - wallet_balance
        if initial_balance > 1e-8:
            unrealized_ratio = unrealized_pnl / initial_balance
        else:
            unrealized_ratio = 0.0
        
        return np.array([
            position_ratio,
            equity_ratio,
            wallet_ratio,
            unrealized_ratio
        ], dtype=np.float32)


class ExtendedAccountFeatureBuilder(AccountFeatureBuilder):
    """
    擴展帳戶特徵構建器（範例）
    
    額外特徵：
    5. leverage_usage: 槓桿使用率 (position_value / equity)
    6. profit_loss_ratio: 盈虧比 ((equity - initial) / initial)
    7. position_side: 持倉方向 (-1=空, 0=平, 1=多)
    8. equity_drawdown: 從初始資金的回撤 (max(0, (initial - equity) / initial))
    """
    
    def feature_names(self) -> List[str]:
        return [
            'position_ratio',
            'equity_ratio',
            'wallet_ratio',
            'unrealized_ratio',
            'leverage_usage',
            'profit_loss_ratio',
            'position_side',
            'equity_drawdown'
        ]
    
    def feature_count(self) -> int:
        return 8
    
    def compute_features(
        self,
        *,
        position_size: float,
        position_value: float,
        equity: float,
        wallet_balance: float,
        current_price: float,
        initial_balance: float,
        leverage: float = 10.0,
        **kwargs
    ) -> np.ndarray:
        """計算擴展帳戶特徵"""
        # 基礎特徵（1-4）
        if equity > 1e-8:
            position_ratio = (position_size * current_price) / equity
            position_ratio = float(np.clip(position_ratio, -1.0, 1.0))
        else:
            position_ratio = 0.0
        
        if initial_balance > 1e-8:
            equity_ratio = equity / initial_balance
            wallet_ratio = wallet_balance / initial_balance
            unrealized_ratio = (equity - wallet_balance) / initial_balance
        else:
            equity_ratio = 1.0
            wallet_ratio = 1.0
            unrealized_ratio = 0.0
        
        # 擴展特徵（5-8）
        # 5. 槓桿使用率
        if equity > 1e-8:
            leverage_usage = position_value / equity
            leverage_usage = float(np.clip(leverage_usage / leverage, 0.0, 1.0))
        else:
            leverage_usage = 0.0
        
        # 6. 盈虧比
        if initial_balance > 1e-8:
            profit_loss_ratio = (equity - initial_balance) / initial_balance
        else:
            profit_loss_ratio = 0.0
        
        # 7. 持倉方向
        if abs(position_size) < 1e-8:
            position_side = 0.0
        elif position_size > 0:
            position_side = 1.0
        else:
            position_side = -1.0
        
        # 8. 權益回撤
        if initial_balance > 1e-8 and equity < initial_balance:
            equity_drawdown = (initial_balance - equity) / initial_balance
        else:
            equity_drawdown = 0.0
        
        return np.array([
            position_ratio,
            equity_ratio,
            wallet_ratio,
            unrealized_ratio,
            leverage_usage,
            profit_loss_ratio,
            position_side,
            equity_drawdown
        ], dtype=np.float32)
