import os
import sys

import pytest

# Add project root to sys.path (so `Env.*` imports work when running this file alone)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from Env.trade_executor import TradeExecutor


def new_executor(
    *,
    initial_balance: float = 1000.0,
    fee_rate: float = 0.0,
    leverage: float = 1.0,
    min_trade_qty: float = 1.0,
    maintenance_margin_rate: float = 0.005,
    margin_mode: str = "cross",
    stop_loss_atr: float = 1.0,
) -> TradeExecutor:
    """建立可控參數的 TradeExecutor（避免測試被手續費與微小調倉干擾）。"""
    return TradeExecutor(
        initial_balance=initial_balance,
        fee_rate=fee_rate,
        leverage=leverage,
        min_trade_qty=min_trade_qty,
        maintenance_margin_rate=maintenance_margin_rate,
        margin_mode=margin_mode,
        stop_loss_atr=stop_loss_atr,
    )


def test_trailing_stop_uses_prev_bar_only_no_same_bar_lookahead():
    """
    驗證：trailing stop 推進只使用「上一根 K」的 high/low。

    若錯誤地用「本根」high 推進後再用「本根」low 檢查，會發生同 K 線前視：
    - 本根 high 拉高 stop
    - 再用本根 low 觸發止損（等於假設 high 在 low 之前）
    """
    ex = new_executor(leverage=1.0, min_trade_qty=1.0, stop_loss_atr=1.0)

    # Bar0：開倉，多單 SL = entry - atr*1 = 99
    p0 = 100.0
    ex.execute(
        position_percent=0.5,
        current_price=p0,
        high=p0,
        low=p0,
        equity=ex.equity(p0),
        atr=1.0,
    )
    assert ex.position.size > 0.0
    assert ex.position.stop_loss_price == pytest.approx(99.0)

    # Bar1：本根 high 很高，但 low 仍高於舊 SL(99)。
    # 若用本根 high 推進 trailing，SL 會被推到 109，接著 low=105 會觸發（前視/順序偏差）。
    ex.execute(
        position_percent=0.5,
        current_price=107.0,
        high=110.0,
        low=105.0,
        equity=ex.equity(107.0),
        atr=1.0,
    )
    assert ex.position.size > 0.0
    assert ex.stop_loss_triggered is False

    # Bar2：此時 trailing 才能用上一根（Bar1）的 high=110 推進 SL 到 109，
    # 並在本根 low=108 觸發。
    ex.execute(
        position_percent=0.5,
        current_price=109.0,
        high=109.0,
        low=108.0,
        equity=ex.equity(109.0),
        atr=1.0,
    )
    assert ex.stop_loss_triggered is True
    assert ex.position.size == 0.0


def test_liquidation_price_does_not_depend_on_close_when_using_ohlc_trigger():
    """強平價為封閉式解，不應因為傳入不同 close(current_price) 而改變。"""
    ex = new_executor(leverage=10.0, min_trade_qty=0.001, stop_loss_atr=0.0, margin_mode="cross")

    p0 = 100.0
    ex.execute(
        position_percent=1.0,
        current_price=p0,
        high=p0,
        low=p0,
        equity=ex.equity(p0),
        atr=0.0,
    )

    liq_100 = ex.get_liquidation_price(100.0)
    liq_200 = ex.get_liquidation_price(200.0)
    assert liq_100 == pytest.approx(liq_200)


def test_stop_loss_has_priority_over_liquidation_when_both_hit_same_bar():
    """同一根 K 線同時碰到 SL 與 LIQ：目前設計為 SL 優先（先 return）。"""
    ex = new_executor(leverage=10.0, min_trade_qty=0.001, stop_loss_atr=1.0, margin_mode="cross")

    p0 = 100.0
    ex.execute(
        position_percent=1.0,
        current_price=p0,
        high=p0,
        low=p0,
        equity=ex.equity(p0),
        atr=1.0,
    )
    assert ex.liq_triggered is False

    liq_price = ex.get_liquidation_price(100.0)
    assert liq_price > 0.0
    assert ex.position.stop_loss_price > 0.0

    # 低點同時跌破止損與強平價
    low = min(ex.position.stop_loss_price, liq_price) - 1.0
    ex.execute(
        position_percent=1.0,
        current_price=95.0,
        high=100.0,
        low=low,
        equity=ex.equity(95.0),
        atr=1.0,
    )

    assert ex.stop_loss_triggered is True
    assert ex.liq_triggered is False
    assert ex.position.size == 0.0


def test_stop_loss_is_clamped_to_trigger_before_liquidation_long_and_short() -> None:
    """
    驗證：若 ATR 止損距離過大導致 SL 可能落在強平價之外，執行器會自動收斂止損，
    讓 SL 一定會先於 LIQ 觸發（避免「止損看起來有設，但永遠來不及觸發就爆倉」）。
    """
    # Long case
    ex_long = new_executor(leverage=10.0, min_trade_qty=0.001, stop_loss_atr=10_000.0, margin_mode="cross")
    p0 = 100.0
    ex_long.execute(
        position_percent=1.0,
        current_price=p0,
        high=p0,
        low=p0,
        equity=ex_long.equity(p0),
        atr=1.0,
    )
    liq_long = ex_long.get_liquidation_price(p0)
    assert liq_long > 0.0
    assert ex_long.position.stop_loss_price > 0.0
    assert ex_long.position.stop_loss_price >= liq_long

    # Short case
    ex_short = new_executor(leverage=10.0, min_trade_qty=0.001, stop_loss_atr=10_000.0, margin_mode="cross")
    ex_short.execute(
        position_percent=-1.0,
        current_price=p0,
        high=p0,
        low=p0,
        equity=ex_short.equity(p0),
        atr=1.0,
    )
    liq_short = ex_short.get_liquidation_price(p0)
    assert liq_short > 0.0
    assert ex_short.position.stop_loss_price > 0.0
    assert ex_short.position.stop_loss_price <= liq_short


def test_max_stop_loss_distance_pct_tracks_entry_to_stop_distance() -> None:
    """Max StopLoss%（幅度）應該是 |entry-stop|/entry，且取最大值。"""
    # Long: entry=100, atr=1, stop_loss_atr=2 => stop=98 => 2% => 0.02
    ex_long = new_executor(leverage=10.0, min_trade_qty=0.001, stop_loss_atr=2.0, margin_mode="cross")
    p0 = 100.0
    ex_long.execute(
        position_percent=1.0,
        current_price=p0,
        high=p0,
        low=p0,
        equity=ex_long.equity(p0),
        atr=1.0,
    )
    assert ex_long.position.stop_loss_price == pytest.approx(98.0)
    assert ex_long.max_stop_loss_distance_pct == pytest.approx(0.02)

    # Short: entry=100, atr=1, stop_loss_atr=2 => stop=102 => 2% => 0.02
    ex_short = new_executor(leverage=10.0, min_trade_qty=0.001, stop_loss_atr=2.0, margin_mode="cross")
    ex_short.execute(
        position_percent=-1.0,
        current_price=p0,
        high=p0,
        low=p0,
        equity=ex_short.equity(p0),
        atr=1.0,
    )
    assert ex_short.position.stop_loss_price == pytest.approx(102.0)
    assert ex_short.max_stop_loss_distance_pct == pytest.approx(0.02)


