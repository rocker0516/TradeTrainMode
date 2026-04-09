import math

from Env.Executors.trade_executor import TradeExecutor


def make_executor(**overrides) -> TradeExecutor:
    params = dict(
        initial_balance=1000.0,
        fee_rate=0.0,            # 測試滑點時關閉手續費干擾
        leverage=1.0,
        min_trade_qty=0.0,
        maintenance_margin_rate=0.005,
        margin_mode="cross",
        stop_loss_atr=0.0,
        stop_loss_liq_buffer_pct=0.0,
        min_position_change=0.0,
        execution_cost_mode=0,
        spread_half_bps=0.0,
        min_notional=0.0,
        slip_base_bps=0.0,
        slip_vol_coeff=0.0,
        slip_size_coeff=0.0,
    )
    params.update(overrides)
    return TradeExecutor(**params)


def test_min_notional_blocks_all_trades():
    ex = make_executor(execution_cost_mode=2, min_notional=50.0)
    # 嘗試以 price=100、期望變動名目約 10（小於 50 門檻）
    ex.execute(
        position_percent=0.01,   # 1000*0.01/100 = 0.1 單位，名目=10
        current_price=100.0,
        high=100.0,
        low=100.0,
        equity=ex.initial_balance,
        atr=0.0,
        risk_base=ex.initial_balance,
        bar_volume=0.0,
        adv_notional=1e6,
    )
    assert ex.position.size == 0.0, "min_notional 應阻擋小額成交"
    assert math.isclose(ex.wallet_balance, ex.initial_balance, rel_tol=1e-9)


def test_slippage_size_ratio_cost_on_entry_and_exit():
    # 僅啟用 slippage（bit 4）
    ex = make_executor(
        execution_cost_mode=4,
        slip_base_bps=0.0,
        slip_vol_coeff=0.0,
        slip_size_coeff=15.0,   # 係數
    )
    price = 100.0
    adv_notional = 1000.0      # ADV 名目

    # 進場：期望 Δnotional = 100 → size_ratio = 0.1 → slip_bps = 1.5 → cost = 100*1.5e-4 = 0.015
    ex.execute(
        position_percent=0.1,   # 1000*0.1/100 = 1 單位
        current_price=price,
        high=price,
        low=price,
        equity=ex.initial_balance,
        atr=0.0,
        risk_base=ex.initial_balance,
        bar_volume=0.0,
        adv_notional=adv_notional,
    )
    expected_entry_slip = 100.0 * 1.5e-4
    assert ex.position.size != 0.0
    assert ex.total_slippage_cost > 0.0
    assert math.isclose(ex.total_slippage_cost, expected_entry_slip, rel_tol=1e-9)

    # 出場：名目同為 100 → 再扣一次相同滑點
    ex.execute(
        position_percent=0.0,   # 全平
        current_price=price,
        high=price,
        low=price,
        equity=ex.equity(price),
        atr=0.0,
        risk_base=ex.initial_balance,
        bar_volume=0.0,
        adv_notional=adv_notional,
    )
    expected_total = expected_entry_slip * 2.0
    assert math.isclose(ex.total_slippage_cost, expected_total, rel_tol=1e-9)
    assert ex.position.size == 0.0


def test_forced_stop_exit_bypasses_min_notional_and_uses_passed_adv_notional():
    ex = make_executor(
        execution_cost_mode=6,  # min_notional + slippage
        min_notional=50.0,
        slip_base_bps=0.0,
        slip_vol_coeff=0.0,
        slip_size_coeff=15.0,
    )
    ex.position.size = 0.1
    ex.position.entry_price = 100.0
    ex.position.stop_loss_price = 95.0

    ex.execute(
        position_percent=0.1,
        current_price=100.0,
        high=100.0,
        low=94.0,               # 觸發 stop loss
        equity=ex.equity(100.0),
        atr=0.0,
        risk_base=ex.initial_balance,
        bar_volume=0.0,
        adv_notional=1000.0,
    )

    expected_slip = (0.1 * 95.0) * ((15.0 * ((0.1 * 95.0) / 1000.0)) * 1e-4)
    expected_wallet = 1000.0 + ((95.0 - 100.0) * 0.1) - expected_slip

    assert ex.stop_loss_triggered is True
    assert ex.position.size == 0.0, "forced stop exit 不應被 min_notional 擋住"
    assert math.isclose(ex.total_slippage_cost, expected_slip, rel_tol=1e-9)
    assert math.isclose(ex.wallet_balance, expected_wallet, rel_tol=1e-9)


def test_invalid_adv_only_uses_base_slippage() -> None:
    ex = make_executor(
        execution_cost_mode=4,
        slip_base_bps=2.0,
        slip_vol_coeff=0.0,
        slip_size_coeff=15.0,
    )
    price = 100.0

    ex.execute(
        position_percent=0.1,
        current_price=price,
        high=price,
        low=price,
        equity=ex.initial_balance,
        atr=0.0,
        risk_base=ex.initial_balance,
        bar_volume=0.0,
        adv_notional=0.0,
    )

    expected_slip = 100.0 * 2.0e-4
    assert math.isclose(ex.total_slippage_cost, expected_slip, rel_tol=1e-9)
    assert math.isclose(ex.wallet_balance, ex.initial_balance - expected_slip, rel_tol=1e-9)


def test_vol_proxy_contributes_to_slippage() -> None:
    ex = make_executor(
        execution_cost_mode=4,
        slip_base_bps=0.0,
        slip_vol_coeff=4.0,
        slip_size_coeff=0.0,
    )
    price = 100.0

    ex.execute(
        position_percent=0.1,
        current_price=price,
        high=price,
        low=price,
        equity=ex.initial_balance,
        atr=0.0,
        risk_base=ex.initial_balance,
        bar_volume=2.0,
        bar_notional=200.0,
        adv_notional=10_000.0,
    )

    # trade_notional=100, bar_notional=200 => vol_proxy=0.5 => slip_bps=2.0
    expected_slip = 100.0 * 2.0e-4
    assert math.isclose(ex.total_slippage_cost, expected_slip, rel_tol=1e-9)
    assert math.isclose(ex.wallet_balance, ex.initial_balance - expected_slip, rel_tol=1e-9)


def test_entry_price_uses_spread_adjusted_fill_price() -> None:
    ex = make_executor(
        execution_cost_mode=1,
        spread_half_bps=10.0,
    )
    price = 100.0

    ex.execute(
        position_percent=0.1,
        current_price=price,
        high=price,
        low=price,
        equity=ex.initial_balance,
        atr=0.0,
        risk_base=ex.initial_balance,
        bar_volume=0.0,
        adv_notional=1_000.0,
    )

    expected_fill = price * (1.0 + 10.0e-4)
    assert math.isclose(ex.position.entry_price, expected_fill, rel_tol=1e-9)

