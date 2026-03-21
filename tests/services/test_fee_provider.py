from __future__ import annotations

from LiveTradingRunner.fee_provider import BinanceFeeRateProvider


class _FakeRawClient:
    def futures_commission_rate(self, symbol: str):  # noqa: ANN001
        _ = symbol
        return {"takerCommissionRate": "0.0004"}


class _FakeTradingClient:
    def __init__(self) -> None:
        self.client = _FakeRawClient()


def test_fee_provider_reads_binance_and_converts_to_percent(monkeypatch) -> None:
    monkeypatch.setattr(
        "LiveTradingRunner.fee_provider.build_trading_client",
        lambda testnet=False: _FakeTradingClient(),  # noqa: ARG005
    )
    provider = BinanceFeeRateProvider(default_fee_pct=0.01)
    got = provider.get_fee_rate_percent(symbol="BTCUSDT")
    assert abs(got - 0.04) < 1e-12


def test_fee_provider_fallbacks_to_default_when_api_fails(monkeypatch) -> None:
    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("LiveTradingRunner.fee_provider.build_trading_client", _raise)
    provider = BinanceFeeRateProvider(default_fee_pct=0.01)
    got = provider.get_fee_rate_percent(symbol="BTCUSDT")
    assert abs(got - 0.01) < 1e-12

