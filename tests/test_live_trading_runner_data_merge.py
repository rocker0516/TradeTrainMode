from __future__ import annotations

import pandas as pd

from LiveTradingRunner.binance_market_data import fetch_latest_multi_symbol_5m


class _FakeBinanceClient:
    """最小 fake client：只提供 futures_klines。"""

    def __init__(self, per_symbol_rows: dict[str, list[list[object]]]) -> None:
        self._rows = per_symbol_rows

    def futures_klines(self, *, symbol: str, interval: str, limit: int):  # noqa: ANN001
        assert interval == "5m"
        out = self._rows[str(symbol)]
        # 模擬 Binance：回傳最後 limit 筆
        return out[-int(limit) :]


def _mk_kline_row(*, open_time_ms: int, px: float) -> list[object]:
    """建立一筆 Binance futures kline row（符合 futures_klines 的欄位順序）。"""
    # open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore
    return [
        int(open_time_ms),
        str(px),
        str(px + 1.0),
        str(px - 1.0),
        str(px),
        "100.0",
        int(open_time_ms + 5 * 60 * 1000 - 1),
        "1000.0",
        "10",
        "60.0",
        "600.0",
        "0",
    ]


def test_fetch_latest_multi_symbol_5m_inner_join_and_dummy_row() -> None:
    # 兩個幣：其中 ETH 少一根 bar，inner join 會丟掉那根 timestamp
    base = pd.Timestamp("2024-01-01 00:00:00").value // 1_000_000
    ts0 = int(base)
    ts1 = int(base + 5 * 60 * 1000)
    ts2 = int(base + 10 * 60 * 1000)

    btc = [_mk_kline_row(open_time_ms=t, px=100.0 + i) for i, t in enumerate([ts0, ts1, ts2])]
    eth = [_mk_kline_row(open_time_ms=t, px=200.0 + i) for i, t in enumerate([ts0, ts2])]  # 缺 ts1

    client = _FakeBinanceClient({"BTCUSDT": btc, "ETHUSDT": eth})

    res = fetch_latest_multi_symbol_5m(
        symbols=("BTCUSDT", "ETHUSDT"),
        target_symbol="BTCUSDT",
        limit=10,
        client=client,
    )

    df = res.df_5m
    assert "timestamp" in df.columns
    assert "BTCUSDT_close" in df.columns
    assert "ETHUSDT_close" in df.columns

    # inner join 後只剩 ts0, ts2；再加 dummy => 3 rows
    assert len(df) == 3
    assert str(df["timestamp"].iloc[-2])[:19] == "2024-01-01 00:10:00"
    assert str(df["timestamp"].iloc[-1])[:19] == "2024-01-01 00:15:00"  # dummy

    # latest_closed_bar_ts 應對應倒數第二列（最後一列是 dummy）
    assert res.latest_closed_bar_ts == "2024-01-01 00:10:00"
    assert res.latest_closed_price == float(df["BTCUSDT_close"].iloc[-2])


