from __future__ import annotations

import math

import pytest

from Env.Executors.trade_executor import TradeExecutor


def test_trade_executor_open_long_sets_margin_fee_and_stop_loss() -> None:
    """開多單後：used_margin/fee/stop_loss 必須符合既有規則。"""
    ex = TradeExecutor(
        initial_balance=1000.0,
        fee_rate=0.04,  # 0.04%
        leverage=10.0,
        min_trade_qty=0.001,
        maintenance_margin_rate=0.005,
        margin_mode="isolated",
        stop_loss_atr=2.0,
        stop_loss_liq_buffer_pct=0.0,
        min_position_change=0.0,
    )

    price = 100.0
    atr = 2.0  # 以價格單位

    # 目標：滿倉多（以 base_amount=wallet 1000 作為 risk_base）
    ex.execute(
        position_percent=1.0,
        current_price=price,
        high=101.0,
        low=99.0,
        equity=ex.equity(price),
        atr=atr,
        risk_base=1000.0,
    )

    assert ex.position.size > 0.0
    assert ex.used_margin > 0.0
    assert ex.total_fees > 0.0
    assert ex.wallet_balance < 1000.0

    # 止損 = entry - atr*mult
    assert ex.position.entry_price == pytest.approx(price, rel=0, abs=1e-9)
    assert ex.position.stop_loss_price == pytest.approx(price - atr * 2.0, rel=0, abs=1e-6)

    # 止損必須先於強平（多單：stop > liq）
    liq = ex.get_liquidation_price(price)
    assert liq > 0.0
    assert ex.position.stop_loss_price > liq


def test_trade_executor_reduce_position_realizes_pnl_and_releases_margin() -> None:
    """減倉必須：釋放保證金、寫入 closed_trades、錢包反映 realized_pnl-fee。"""
    ex = TradeExecutor(
        initial_balance=1000.0,
        fee_rate=0.0,  # 讓檢查更乾淨
        leverage=10.0,
        min_trade_qty=0.001,
        maintenance_margin_rate=0.005,
        margin_mode="isolated",
        stop_loss_atr=0.0,
        stop_loss_liq_buffer_pct=0.0,
        min_position_change=0.0,
    )

    ex.execute(
        position_percent=1.0,
        current_price=100.0,
        high=101.0,
        low=99.0,
        equity=ex.equity(100.0),
        atr=0.0,
        risk_base=1000.0,
    )
    size0 = ex.position.size
    used0 = ex.used_margin

    # 轉為半倉 => 減倉
    ex.execute(
        position_percent=0.5,
        current_price=110.0,
        high=111.0,
        low=109.0,
        equity=ex.equity(110.0),
        atr=0.0,
        risk_base=1000.0,
    )

    assert 0.0 < ex.position.size < size0
    assert ex.used_margin < used0
    assert len(ex.closed_trades) >= 1
    last = ex.closed_trades[-1]
    assert last["type"] in ("reduce", "close")
    assert last["realized_pnl"] > 0.0


def test_trade_executor_stop_loss_triggers_before_liquidation() -> None:
    """同一根 K 線同時穿越 SL 與 LIQ 時，必須優先止損（先於強平）。"""
    ex = TradeExecutor(
        initial_balance=1000.0,
        fee_rate=0.0,
        leverage=10.0,
        min_trade_qty=0.001,
        maintenance_margin_rate=0.005,
        margin_mode="isolated",
        stop_loss_atr=2.0,
        stop_loss_liq_buffer_pct=0.0,
        min_position_change=0.0,
    )

    # 開倉：entry=100, atr=2 => stop=96（且會 clamp 在 liq 之上）
    ex.execute(
        position_percent=1.0,
        current_price=100.0,
        high=101.0,
        low=99.0,
        equity=ex.equity(100.0),
        atr=2.0,
        risk_base=1000.0,
    )
    stop = ex.position.stop_loss_price
    liq = ex.get_liquidation_price(100.0)
    assert stop > liq

    # 下一根：low 同時小於 stop 與 liq（極端），應先走 stop_loss close
    ex.execute(
        position_percent=1.0,  # 動作不重要，因為先觸發 stop
        current_price=100.0,
        high=101.0,
        low=min(stop, liq) - 1.0,
        equity=ex.equity(100.0),
        atr=2.0,
        risk_base=1000.0,
    )

    assert ex.stop_loss_triggered is True
    assert ex.liq_triggered is False
    assert ex.position.size == 0.0


def test_trade_executor_trailing_stop_uses_prev_bar_only() -> None:
    """trailing stop 必須只用「上一根已完成 K」推進，避免同根前視。"""
    ex = TradeExecutor(
        initial_balance=1000.0,
        fee_rate=0.0,
        leverage=10.0,
        min_trade_qty=0.001,
        maintenance_margin_rate=0.005,
        margin_mode="isolated",
        stop_loss_atr=1.0,
        stop_loss_liq_buffer_pct=0.0,
        min_position_change=0.0,
    )

    # bar0 開倉：entry=100, atr=2 => stop=98
    ex.execute(
        position_percent=1.0,
        current_price=100.0,
        high=101.0,
        low=99.0,
        equity=ex.equity(100.0),
        atr=2.0,
        risk_base=1000.0,
    )
    stop0 = ex.position.stop_loss_price
    assert stop0 == pytest.approx(98.0, abs=1e-6)

    # bar1：給一個超高 high=120，但 trailing 更新只能使用 prev_bar_high=101
    ex.execute(
        position_percent=1.0,
        current_price=110.0,
        high=120.0,
        low=109.0,
        equity=ex.equity(110.0),
        atr=2.0,
        risk_base=1000.0,
    )
    # 只用 prev high=101 推進 => 新止損最多到 101-2=99
    stop1 = ex.position.stop_loss_price
    assert stop1 == pytest.approx(99.0, abs=1e-6)

    # bar2：此時 prev_bar_high=120，才允許進一步推進到 120-2=118
    ex.execute(
        position_percent=1.0,
        # 注意：trailing stop 會先用 prev_bar_high 推進到 118，
        # 若本根 low <= 118 會立即觸發止損並把 stop_loss_price 清成 0。
        # 這裡刻意把 low 設在 stop 之上，純測 trailing 更新。
        current_price=120.0,
        high=121.0,
        low=119.5,
        equity=ex.equity(120.0),
        atr=2.0,
        risk_base=1000.0,
    )
    stop2 = ex.position.stop_loss_price
    assert stop2 == pytest.approx(118.0, abs=1e-6)


@pytest.mark.parametrize(
    "margin_mode",
    ["cross", "isolated"],
)
def test_trade_executor_liquidation_price_is_non_negative(margin_mode: str) -> None:
    ex = TradeExecutor(
        initial_balance=1000.0,
        fee_rate=0.0,
        leverage=10.0,
        min_trade_qty=0.001,
        maintenance_margin_rate=0.005,
        margin_mode=margin_mode,
        stop_loss_atr=0.0,
        stop_loss_liq_buffer_pct=0.0,
        min_position_change=0.0,
    )
    ex.execute(
        position_percent=1.0,
        current_price=100.0,
        high=101.0,
        low=99.0,
        equity=ex.equity(100.0),
        atr=0.0,
        risk_base=1000.0,
    )
    liq = ex.get_liquidation_price(100.0)
    assert liq >= 0.0
    assert math.isfinite(liq)


def test_trade_executor_isolated_used_margin_tracks_entry_after_adding_at_lower_price() -> None:
    """
    isolated 下：同向加倉後 used_margin 應與「平均 entry」一致。

    這是用來防止一個常見 bug：
    - 若用 current_price 估 required margin，當價格大幅下跌時，
      可能出現「加倉後 required_margin 反而變小」=> additional_margin=0，
      造成 used_margin 被低估，進而讓 liq_price 計算失真（常見為 long 的 liq 跑到 entry 之上）。
    """
    ex = TradeExecutor(
        initial_balance=1000.0,
        fee_rate=0.0,
        leverage=10.0,
        min_trade_qty=0.001,
        maintenance_margin_rate=0.005,
        margin_mode="isolated",
        stop_loss_atr=0.0,
        stop_loss_liq_buffer_pct=0.0,
        min_position_change=0.0,
    )

    # Step1: 高價少量開多
    ex.execute(
        position_percent=0.05,
        current_price=100.0,
        high=101.0,
        low=99.0,
        equity=ex.equity(100.0),
        atr=0.0,
        risk_base=1000.0,
    )
    size1 = float(ex.position.size)
    entry1 = float(ex.position.entry_price)
    assert size1 > 0.0
    assert entry1 == pytest.approx(100.0, abs=1e-9)

    # Step2: 價格下跌但仍高於強平價，再加碼（同向）
    # - 對 leverage=10 / mmr=0.005 而言，long 的 liq 約在 entry*0.9045
    # - 這裡選 current_price=91、low=90.6（> liq），確保不會在 step 開頭就被強平平倉。
    ex.execute(
        # 讓 target_size 只比原本大一點點（同向加碼）
        position_percent=0.04732,
        current_price=91.0,
        high=91.2,
        low=90.6,
        equity=ex.equity(91.0),
        atr=0.0,
        risk_base=1000.0,
    )

    assert ex.position.size > size1
    assert ex.position.entry_price > 0.0
    expected_used = abs(float(ex.position.size)) * float(ex.position.entry_price) / float(ex.leverage)
    assert ex.used_margin == pytest.approx(expected_used, rel=0, abs=1e-6)


