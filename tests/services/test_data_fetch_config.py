from services.data_fetch.config import load_config


def test_load_config_has_default_symbols() -> None:
    cfg = load_config()
    assert "BTCUSDT" in cfg.binance_symbols
    assert "BTCUSDT" in cfg.coinglass_symbols
    assert cfg.binance_interval_seconds > 0
    assert cfg.coinglass_interval_seconds > 0

