import math

import pytest

from Env.trade_executor import TradeExecutor


def approx(a: float, b: float, tol: float = 1e-8) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=tol)


def new_executor(
    initial_balance: float = 1000.0,
    fee_rate: float = 0.1,
    leverage: float = 10.0,
    min_trade_qty: float = 0.001,
    maintenance_margin_rate: float = 0.005,
    margin_mode: str = "cross",
    stop_loss_atr: float = 1.0,
):
    return TradeExecutor(
        initial_balance=initial_balance,
        fee_rate=fee_rate,
        leverage=leverage,
        min_trade_qty=min_trade_qty,
        maintenance_margin_rate=maintenance_margin_rate,
        margin_mode=margin_mode,
        stop_loss_atr=stop_loss_atr,
    )


def test_open_long_position_and_fee_and_margin():
    ex = new_executor()
    price = 10000.0
    equity_before = ex.equity(price)
    assert approx(equity_before, 1000.0)

    ex.execute(
        position_percent=0.5,
        current_price=price,
        high=price,
        low=price,
        equity=equity_before,
        atr=0.0,
    )

    # size = equity * p * lev / price = 1000 * 0.5 * 10 / 10000 = 0.5
    assert approx(ex.position.size, 0.5)
    # fee_rate 0.1 = 0.1% => fee = notional * 0.001 = (0.5*10000) * 0.001 = 5
    assert approx(ex.wallet_balance, 1000.0 - 5.0)
    # margin = notional/leverage = 5000/10 = 500
    assert approx(ex.used_margin, 500.0)
    # equity = wallet + UPNL (0) = 995
    assert approx(ex.equity(price), 995.0)


def test_unrealized_pnl_updates_equity_for_long():
    ex = new_executor()
    p0 = 10000.0
    ex.execute(
        position_percent=0.5,
        current_price=p0,
        high=p0,
        low=p0,
        equity=ex.equity(p0),
        atr=0.0,
    )
    # price up 100
    p1 = 10100.0
    # equity should be 995 + (10100-10000)*0.5 = 1045
    assert approx(ex.equity(p1), 1045.0)


def test_reduce_position_realizes_pnl_and_releases_margin():
    ex = new_executor()
    p0 = 10000.0
    ex.execute(
        position_percent=0.5,
        current_price=p0,
        high=p0,
        low=p0,
        equity=ex.equity(p0),
        atr=0.0,
    )
    # move price up and reduce to 20% target
    p1 = 10100.0
    eq_before = ex.equity(p1)
    wallet_before = ex.wallet_balance
    ex.execute(
        position_percent=0.2,
        current_price=p1,
        high=p1,
        low=p1,
        equity=eq_before,
        atr=0.0,
    )

    # Execute 在 risk_base=None 時會用保守基準（以「下單當下」為準）：
    # base_amount = min(wallet_before, equity_before)
    base_amount = min(wallet_before, eq_before)
    target_size = (base_amount * 0.2 * 10.0) / p1
    assert approx(ex.position.size, target_size, tol=1e-6)

    # Realized pnl on closed ~ (10100-10000)*(0.5 - target)
    closed = 0.5 - target_size
    realized = (p1 - p0) * closed
    fee = closed * p1 * 0.001
    # wallet after reduce = 995 + realized - fee
    expected_wallet = 995.0 + realized - fee
    assert approx(ex.wallet_balance, expected_wallet, tol=1e-6)
    # used margin should be margin for remaining size at entry price
    expected_margin = abs(target_size) * p0 / 10.0
    assert approx(ex.used_margin, expected_margin, tol=1e-6)


def test_reverse_from_long_to_short_closes_then_opens_new():
    ex = new_executor()
    p0 = 10000.0
    ex.execute(
        position_percent=0.5,
        current_price=p0,
        high=p0,
        low=p0,
        equity=ex.equity(p0),
        atr=0.0,
    )
    # price down to 9900 and reverse to -0.5
    p1 = 9900.0
    eq = ex.equity(p1)
    ex.execute(
        position_percent=-0.5,
        current_price=p1,
        high=p1,
        low=p1,
        equity=eq,
        atr=0.0,
    )
    # Position should be short now.
    # 注意：execute 反轉時，會先用「反轉前」的 base_amount 計算 target_size，
    # close 後再用同一個 base_amount 重算 target（不會用 close 後 wallet 重新計算）。
    realized_close = (p1 - p0) * 0.5
    fee_close = 0.5 * p1 * 0.001
    wallet_after_close = 1000.0 - 5.0 + realized_close - fee_close
    # base_amount = min(wallet_before_close, equity_before_close)
    # wallet_before_close = 995
    # equity_before_close = 995 + (9900-10000)*0.5 = 945
    base_amount = min(995.0, 945.0)
    target_short = (base_amount * 0.5 * 10.0) / p1
    # Opening fee
    fee_open = target_short * p1 * 0.001
    # Wallet after open
    expected_wallet = wallet_after_close - fee_open
    assert ex.position.size < 0
    assert approx(abs(ex.position.size), target_short, tol=1e-6)
    assert approx(ex.wallet_balance, expected_wallet, tol=1e-5)


def test_stop_loss_triggers_close_on_next_execute():
    ex = new_executor(stop_loss_atr=1.0)
    p0 = 100.0
    ex.execute(
        position_percent=0.5,
        current_price=p0,
        high=p0,
        low=p0,
        equity=ex.equity(p0),
        atr=1.0,
    )
    # Stops should be set; next candle hits stop loss low
    # For long: stop_loss = entry - atr*stop_loss_atr = 100 - 1 = 99.0
    assert ex.position.stop_loss_price > 0
    # Next execute with low below stop triggers close before any target sizing
    p1 = 100.0
    ex.execute(
        position_percent=0.5,
        current_price=p1,
        high=100.0,
        low=98.0,
        equity=ex.equity(p1),
        atr=1.0,
    )
    assert ex.position.size == 0.0
    assert ex.stop_loss_triggered is True


def test_min_trade_qty_gates_small_trades():
    ex = new_executor(min_trade_qty=0.05)
    p = 100.0
    eq = ex.equity(p)
    # position_percent so small that size < 0.05
    small_p = 0.01  # size = 1000*0.01*10/100 = 1.0 -> exceeds 0.05; make smaller
    small_p = 0.0004  # size = 0.04 < 0.05 -> should not trade
    ex.execute(
        position_percent=small_p,
        current_price=p,
        high=p,
        low=p,
        equity=eq,
        atr=0.0,
    )
    assert ex.position.size == 0.0


def test_insufficient_balance_caps_position():
    ex = new_executor(initial_balance=50.0, fee_rate=0.1, leverage=10.0, min_trade_qty=0.001)
    p = 1000.0
    eq = ex.equity(p)
    # Target with 100% would be size = 50*1*10/1000 = 0.5, notional 500, margin 50, fee 0.5 (0.1%)
    # Available balance for margin+fee is 50; since fee>0, effective size will be slightly less than 0.5
    ex.execute(
        position_percent=1.0,
        current_price=p,
        high=p,
        low=p,
        equity=eq,
        atr=0.0,
    )
    assert ex.position.size > 0.0
    assert ex.wallet_balance >= 0.0


def test_short_position_pnl_signs():
    ex = new_executor()
    p0 = 100.0
    ex.execute(
        position_percent=-0.5,
        current_price=p0,
        high=p0,
        low=p0,
        equity=ex.equity(p0),
        atr=0.0,
    )
    # price down -> profit
    p1 = 90.0
    assert ex.equity(p1) > ex.wallet_balance
    # price up -> loss
    p2 = 110.0
    assert ex.equity(p2) < ex.wallet_balance


# ------- Liquidation tests -------
def test_long_liquidates_at_derived_price_intrabar():
    ex = new_executor(stop_loss_atr=0.0)
    p0 = 100.0
    # open 100% long to maximize clarity
    ex.execute(
        position_percent=1.0,
        current_price=p0,
        high=p0,
        low=p0,
        equity=ex.equity(p0),
        atr=0.0,
    )
    # After open: actual size limited by available balance after fee
    # Target: 1000*1*10/100 = 100.0, but fee and margin constraint reduce it
    # Available after initial: 1000; need margin+fee, so actual ~99.0
    # fee = 99*100*0.001 = 9.9; margin = 9900/10 = 990; wallet = 1000-9.9 = 990.1
    assert approx(ex.position.size, 99.0)
    assert approx(ex.wallet_balance, 990.1, tol=0.2)

    # maintenance_margin_rate default is 0.005
    mmr = 0.005
    # liquidation price for long: p = (s*e - w) / (s*(1-m))
    s = ex.position.size
    e = ex.position.entry_price
    w = ex.wallet_balance
    liq_price = (s * e - w) / (s * (1.0 - mmr))

    # intrabar low breaches liquidation price -> position closed
    ex.execute(
        position_percent=1.0,
        current_price=liq_price,  # close price equal to liq
        high=liq_price,
        low=liq_price - 1.0,  # breach below to ensure trigger
        equity=ex.equity(liq_price),
        atr=0.0,
    )
    assert ex.position.size == 0.0
