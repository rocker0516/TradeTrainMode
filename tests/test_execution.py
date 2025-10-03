import numpy as np
import pandas as pd

from Env.execution import TradingExecutor, TradeContext


def make_df(close: float, high: float, low: float):
    return pd.DataFrame([
        {'close': close, 'high': high, 'low': low}
    ])


def test_open_long_with_margin_and_fee():
    df = make_df(100.0, 101.0, 99.0)
    ctx = TradeContext(
        df=df,
        current_step=0,
        balance=1000.0,
        total_value=1000.0,
        btc_held=0.0,
        avg_entry_price=0.0,
        take_profit_price=0.0,
        stop_loss_price=0.0,
        leverage=10.0,
        fee_rate=0.001,
        min_trade_amount=10.0,
        min_balance=0.0,
        last_position=0.0,
    )

    executor = TradingExecutor()
    # 50% 倉位，理論名目 = 1000 * 0.5 * 10 = 5000 USDT
    # 目標張數 = 5000 / 100 = 50
    action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    result = executor.execute(ctx, action)

    # 檢查保證金與費用有被扣除（入場費 5000 * 0.001 = 5）
    assert result.btc_held > 0
    assert result.avg_entry_price == 100.0
    # 保證金 = 5000/10 = 500；餘額應 <= 1000 - 500 - 5
    assert result.balance <= 495.0 + 1e-6


def test_close_long_realizes_pnl_and_fee():
    df = make_df(110.0, 112.0, 109.0)
    ctx = TradeContext(
        df=df,
        current_step=0,
        balance=500.0,
        total_value=1000.0,
        btc_held=50.0,
        avg_entry_price=100.0,
        take_profit_price=0.0,
        stop_loss_price=0.0,
        leverage=10.0,
        fee_rate=0.001,
        min_trade_amount=10.0,
        min_balance=0.0,
        last_position=50.0,
    )

    executor = TradingExecutor()
    # 平倉：position_percent = 0
    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    result = executor.execute(ctx, action)

    # 實現盈虧： (110-100)*50 = 500；出場費：50*110*0.001=5.5
    # 加上返還保證金：50*100/10=500
    # 新餘額 = 500 + 500 + 500 - 5.5 = 1494.5
    assert abs(result.balance - 1494.5) < 1e-3
    assert result.btc_held == 0
    assert result.closed_trade is not None


def test_liquidation_when_unrealized_loss_exceeds_margin_threshold():
    df = make_df(80.0, 81.0, 79.0)
    ctx = TradeContext(
        df=df,
        current_step=0,
        balance=500.0,
        total_value=1000.0,
        btc_held=50.0,
        avg_entry_price=100.0,
        take_profit_price=0.0,
        stop_loss_price=0.0,
        leverage=10.0,
        fee_rate=0.001,
        min_trade_amount=10.0,
        min_balance=0.0,
        last_position=50.0,
    )

    executor = TradingExecutor(liquidation_margin_ratio=0.8)
    # 持倉未實現損益 = (80-100)*50 = -1000；已用保證金 = 50*100/10 = 500
    # |-1000| > 0.8*500 = 400 → 觸發強平
    action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    result = executor.execute(ctx, action)
    assert result.forced_liquidation is True
    assert result.btc_held == 0


def test_stops_trigger_correctly_for_long():
    df = make_df(100.0, 105.0, 95.0)
    ctx = TradeContext(
        df=df,
        current_step=0,
        balance=500.0,
        total_value=1000.0,
        btc_held=10.0,
        avg_entry_price=100.0,
        take_profit_price=104.0,
        stop_loss_price=96.0,
        leverage=10.0,
        fee_rate=0.001,
        min_trade_amount=10.0,
        min_balance=0.0,
        last_position=10.0,
    )
    executor = TradingExecutor()
    action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    result = executor.execute(ctx, action)

    assert result.stop_triggered_this_step in ('take_profit', 'stop_loss')


def test_stops_trigger_take_profit_for_short():
    df = make_df(100.0, 101.0, 95.0)
    ctx = TradeContext(
        df=df,
        current_step=0,
        balance=500.0,
        total_value=1000.0,
        btc_held=-10.0,  # 空倉
        avg_entry_price=100.0,
        take_profit_price=96.0,  # 低於現價，low=95 會觸發
        stop_loss_price=105.0,
        leverage=10.0,
        fee_rate=0.001,
        min_trade_amount=10.0,
        min_balance=0.0,
        last_position=-10.0,
    )
    executor = TradingExecutor()
    action = np.array([-0.5, 0.0, 0.0], dtype=np.float32)
    result = executor.execute(ctx, action)
    assert result.stop_triggered_this_step == 'take_profit'
    assert result.btc_held == 0


def test_stops_trigger_stop_loss_for_short():
    df = make_df(100.0, 106.0, 99.0)
    ctx = TradeContext(
        df=df,
        current_step=0,
        balance=500.0,
        total_value=1000.0,
        btc_held=-10.0,  # 空倉
        avg_entry_price=100.0,
        take_profit_price=96.0,
        stop_loss_price=105.0,  # high=106 會觸發
        leverage=10.0,
        fee_rate=0.001,
        min_trade_amount=10.0,
        min_balance=0.0,
        last_position=-10.0,
    )
    executor = TradingExecutor()
    action = np.array([-0.5, 0.0, 0.0], dtype=np.float32)
    result = executor.execute(ctx, action)
    assert result.stop_triggered_this_step == 'stop_loss'
    assert result.btc_held == 0


