from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import pandas as pd


@dataclass
class TradeContext:
    """交易執行所需的上下文（由環境在當步提供）。

    欄位：
        df: K 線資料，需包含 'close','high','low'。
        current_step: 當前步索引。
        balance: 可用資金（USDT）。
        total_value: 總資產（可用資金 + 未實現損益）。
        btc_held: 當前持倉數量（>0 多倉，<0 空倉）。
        avg_entry_price: 當前持倉平均成本價。
        take_profit_price: 當前持倉止盈價（0 表示未設）。
        stop_loss_price: 當前持倉止損價（0 表示未設）。
        leverage: 槓桿倍數。
        fee_rate: 手續費率（按名目金額計）。
        min_trade_amount: 最低名目下單金額（USDT）。
        min_balance: 最低可接受總資產（用於終止/風險控制）。
        last_position: 上一步持倉數量。
        stop_triggered_this_step: 若本步觸發止盈/止損，值為 'take_profit' 或 'stop_loss'，否則 None。
    """

    df: pd.DataFrame
    current_step: int
    balance: float
    total_value: float
    btc_held: float
    avg_entry_price: float
    take_profit_price: float
    stop_loss_price: float
    leverage: float
    fee_rate: float
    min_trade_amount: float
    min_balance: float
    last_position: float
    stop_triggered_this_step: Optional[str] = None
    position_holding_time: int = 0
    steps_in_episode: int = 0
    slippage_bps: float = 0.0  # 以基點(bps)表示的滑點，1 bps = 0.01%
    warmup_steps: int = 0


@dataclass
class TradeAction:
    """連續動作：倉位百分比、止盈%、止損%。"""
    position_percent: float
    take_profit_percent: float
    stop_loss_percent: float


@dataclass
class TradeResult:
    """交易執行結果。"""
    balance: float
    total_value: float
    btc_held: float
    avg_entry_price: float
    take_profit_price: float
    stop_loss_price: float
    stop_triggered_this_step: Optional[str]
    closed_trade: Optional[Dict]
    forced_liquidation: bool


class TradingExecutor:
    """合約槓桿交易執行器：負責下單、止盈止損與強平邏輯。

    設計重點：
    - 手續費按名目金額（notional = size * price）計算，不因槓桿而縮小。
    - 保證金鎖定：required_margin = notional / leverage。
    - 平倉返還鎖定保證金並結算盈虧與手續費。
    - 強平條件：未實現虧損絕對值超過已用保證金的 80%（可調）。
    """

    def __init__(self, liquidation_margin_ratio: float = 0.8) -> None:
        self.liquidation_margin_ratio = liquidation_margin_ratio

    # ---- 公開入口 ----
    def execute(self, ctx: TradeContext, action: np.ndarray) -> TradeResult:
        current_price = float(ctx.df.iloc[ctx.current_step]['close'])
        current_high = float(ctx.df.iloc[ctx.current_step]['high'])
        current_low = float(ctx.df.iloc[ctx.current_step]['low'])

        stop_triggered = None
        closed_trade: Optional[Dict] = None
        forced_liq = False

        # 1) 先判斷止盈/止損
        if ctx.btc_held != 0:
            if ctx.btc_held > 0:
                if current_low <= ctx.stop_loss_price and ctx.stop_loss_price > 0:
                    stop_triggered = 'stop_loss'
                    ctx = self._close_position(ctx, exit_price=ctx.stop_loss_price)
                    closed_trade = ctx.closed_trade  # type: ignore[attr-defined]
                elif current_high >= ctx.take_profit_price and ctx.take_profit_price > 0:
                    stop_triggered = 'take_profit'
                    ctx = self._close_position(ctx, exit_price=ctx.take_profit_price)
                    closed_trade = ctx.closed_trade  # type: ignore[attr-defined]
            else:
                if current_high >= ctx.stop_loss_price and ctx.stop_loss_price > 0:
                    stop_triggered = 'stop_loss'
                    ctx = self._close_position(ctx, exit_price=ctx.stop_loss_price)
                    closed_trade = ctx.closed_trade  # type: ignore[attr-defined]
                elif current_low <= ctx.take_profit_price and ctx.take_profit_price > 0:
                    stop_triggered = 'take_profit'
                    ctx = self._close_position(ctx, exit_price=ctx.take_profit_price)
                    closed_trade = ctx.closed_trade  # type: ignore[attr-defined]

        # 2) 目標倉位（使用帳戶價值與槓桿推導）
        # 動作解碼：position ∈ [-1,1]；TP/SL ∈ [0,1] → 0–100% / 0–20%
        raw_pos, raw_tp, raw_sl = float(action[0]), float(action[1]), float(action[2])
        tp_pct = float(np.clip(raw_tp, 0.0, 1.0)) * 1.0   # 0%~100%
        sl_pct = float(np.clip(raw_sl, 0.0, 1.0)) * 0.20  # 0%~20%
        act = TradeAction(raw_pos, tp_pct * 100.0, sl_pct * 100.0)
        target_position = ctx.total_value * act.position_percent * ctx.leverage / max(current_price, 1e-12)
        diff_position = target_position - ctx.btc_held

        # 若方向相反，先平倉再依目標開倉
        if act.position_percent == 0.0:
            if ctx.btc_held != 0:
                ctx = self._close_position(ctx, exit_price=current_price)
                closed_trade = ctx.closed_trade  # type: ignore[attr-defined]
        else:
            if ctx.btc_held * target_position < 0:
                # 方向相反：先平倉
                if ctx.btc_held != 0:
                    ctx = self._close_position(ctx, exit_price=current_price)
                    closed_trade = ctx.closed_trade  # type: ignore[attr-defined]
                # 再按目標全量開倉
                ctx = self._open_or_adjust(ctx, target_position - ctx.btc_held, current_price)
            else:
                # 同向調整
                if abs(diff_position) > 0:
                    ctx = self._open_or_adjust(ctx, diff_position, current_price)

        # 3) 更新止盈止損（根據最新持倉方向）
        if ctx.btc_held > 0:
            tp = current_price * (1.0 + act.take_profit_percent * 0.01)
            sl = current_price * (1.0 - act.stop_loss_percent * 0.01)
        elif ctx.btc_held < 0:
            tp = current_price * (1.0 - act.take_profit_percent * 0.01)
            sl = current_price * (1.0 + act.stop_loss_percent * 0.01)
        else:
            tp = 0.0
            sl = 0.0
        ctx.take_profit_price = tp
        ctx.stop_loss_price = sl

        # 4) 更新總資產與強平檢查
        if ctx.btc_held != 0:
            position_size = abs(ctx.btc_held)
            margin_used = position_size * ctx.avg_entry_price / max(ctx.leverage, 1e-12) if ctx.avg_entry_price > 0 else 0.0
            unrealized = self._unrealized_pnl(ctx, current_price)
            # Warm-up 期間放寬強平
            margin_ratio = self.liquidation_margin_ratio
            if ctx.steps_in_episode < ctx.warmup_steps:
                margin_ratio = margin_ratio * 0.5
            if margin_used > 0 and unrealized < -margin_used * margin_ratio:
                # 強平
                forced_liq = True
                ctx = self._close_position(ctx, exit_price=self._apply_slippage(current_price, ctx, is_exit=True))
                closed_trade = ctx.closed_trade  # type: ignore[attr-defined]
                ctx.total_value = ctx.balance
            else:
                ctx.total_value = float(np.clip(ctx.balance + unrealized, 0.0, 1e12))
        else:
            ctx.total_value = float(np.clip(ctx.balance, 0.0, 1e12))

        return TradeResult(
            balance=ctx.balance,
            total_value=ctx.total_value,
            btc_held=ctx.btc_held,
            avg_entry_price=ctx.avg_entry_price,
            take_profit_price=ctx.take_profit_price,
            stop_loss_price=ctx.stop_loss_price,
            stop_triggered_this_step=stop_triggered,
            closed_trade=closed_trade,
            forced_liquidation=forced_liq,
        )

    # ---- 私有輔助 ----
    def _unrealized_pnl(self, ctx: TradeContext, current_price: float) -> float:
        if ctx.avg_entry_price <= 0 or ctx.btc_held == 0:
            return 0.0
        size = abs(ctx.btc_held)
        if ctx.btc_held > 0:
            return (current_price - ctx.avg_entry_price) * size
        return (ctx.avg_entry_price - current_price) * size

    def _open_or_adjust(self, ctx: TradeContext, delta_size: float, price: float) -> TradeContext:
        exec_price = self._apply_slippage(price, ctx, is_exit=False)
        notional = abs(delta_size) * exec_price
        if notional < ctx.min_trade_amount:
            return ctx
        required_margin = notional / max(ctx.leverage, 1e-12)
        if required_margin > ctx.balance:
            return ctx
        # 入場手續費（名目 * 費率）
        entry_fee = notional * ctx.fee_rate
        if entry_fee > ctx.balance - required_margin:
            return ctx

        old_pos = ctx.btc_held
        old_size = abs(old_pos)
        ctx.btc_held = ctx.btc_held + delta_size
        ctx.balance = ctx.balance - required_margin - entry_fee

        # 更新平均成本（加倉/減倉）
        new_size = abs(ctx.btc_held)
        if new_size == 0:
            ctx.avg_entry_price = 0.0
        else:
            if old_size == 0:
                ctx.avg_entry_price = exec_price
            elif np.sign(old_pos) == np.sign(ctx.btc_held):
                total_cost = old_size * ctx.avg_entry_price + abs(delta_size) * exec_price
                ctx.avg_entry_price = total_cost / new_size
            else:
                # 同步處理減倉時不改變 avg_entry_price（直到完全平倉）
                ctx.avg_entry_price = ctx.avg_entry_price if old_size > new_size else price

        return ctx

    def _close_position(self, ctx: TradeContext, exit_price: float) -> TradeContext:
        if ctx.btc_held == 0:
            return ctx

        size = abs(ctx.btc_held)
        # 實現盈虧
        if ctx.btc_held > 0:
            pnl = (exit_price - ctx.avg_entry_price) * size
        else:
            pnl = (ctx.avg_entry_price - exit_price) * size

        # 出場手續費（名目 * 費率）
        exit_fee = size * exit_price * ctx.fee_rate
        # 返還保證金
        margin_locked = size * ctx.avg_entry_price / max(ctx.leverage, 1e-12)

        net_pnl = pnl - exit_fee
        ctx.balance = float(np.clip(ctx.balance + margin_locked + net_pnl, 0.0, 1e12))

        # 建立關閉交易紀錄（供上層保存）
        closed_trade = {
            'pnl': net_pnl,
            'entry_price': ctx.avg_entry_price,
            'exit_price': exit_price,
            'position_size': ctx.btc_held,
            'return_rate': (net_pnl / (size * ctx.avg_entry_price)) if ctx.avg_entry_price > 0 else 0.0,
            'hold_time': ctx.position_holding_time,
        }
        # 暫存在 ctx 以回傳（不在本模組維護清單）
        setattr(ctx, 'closed_trade', closed_trade)

        ctx.btc_held = 0.0
        ctx.avg_entry_price = 0.0
        return ctx

    def _apply_slippage(self, price: float, ctx: TradeContext, is_exit: bool) -> float:
        if ctx.slippage_bps <= 0:
            return price
        slip = price * (ctx.slippage_bps / 10000.0)
        # 簡化：入場/出場同向增加成本
        return price + slip


