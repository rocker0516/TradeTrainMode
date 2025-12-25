import numpy as np
import pandas as pd
import pytest

from Env.trading_env import TradingEnvironment
from Train.config import Config


def _make_df(length: int = 400) -> pd.DataFrame:
    prices = np.linspace(100.0, 120.0, length)
    return pd.DataFrame({
        'open': prices,
        'high': prices + 1.0,
        'low': prices - 1.0,
        'close': prices,
        'volume': np.ones(length) * 1_000,
        'buy_volume': np.ones(length) * 600,
        'sell_volume': np.ones(length) * 400,
        'trades': np.ones(length) * 100,
        'quote_volume': prices * 1_000,
        'volume_ratio': np.ones(length),
        'long_short_ratio': np.ones(length),
    })


@pytest.fixture
def patched_config():
    """暫時調整 Config，讓測試使用較短窗口與寬鬆閾值。"""
    backup = {
        'WINDOW_SIZE': Config.WINDOW_SIZE,
        'MIN_EPISODE_STEPS': Config.MIN_EPISODE_STEPS,
        'MAX_EPISODE_STEPS': Config.MAX_EPISODE_STEPS,
        'INITIAL_BALANCE': Config.INITIAL_BALANCE,
        'MIN_BALANCE': Config.MIN_BALANCE,
        'LEVERAGE': Config.LEVERAGE,
        'TRANSACTION_FEE': Config.TRANSACTION_FEE,
        'MAINTENANCE_MARGIN_RATE': Config.MAINTENANCE_MARGIN_RATE,
        'LIQUIDATION_WARN_PCT': Config.LIQUIDATION_WARN_PCT,
        'STOP_LOSS_WARN_PCT': Config.STOP_LOSS_WARN_PCT,
        'STOP_LOSS_ATR': Config.STOP_LOSS_ATR,
    }
    Config.WINDOW_SIZE = 20
    Config.MIN_EPISODE_STEPS = 10
    Config.MAX_EPISODE_STEPS = 50
    Config.INITIAL_BALANCE = 1_000.0
    Config.MIN_BALANCE = 10.0
    Config.LEVERAGE = 10.0
    Config.TRANSACTION_FEE = 0.0
    Config.MAINTENANCE_MARGIN_RATE = 0.02
    Config.LIQUIDATION_WARN_PCT = 0.10
    Config.STOP_LOSS_WARN_PCT = 0.02
    Config.STOP_LOSS_ATR = 1.0
    try:
        yield
    finally:
        for key, val in backup.items():
            setattr(Config, key, val)


def test_risk_signal_computation_flags(patched_config):
    df = _make_df()
    env = TradingEnvironment(df=df, random_start=False)
    env.reset()

    # 人工設定倉位讓風險指標可控
    env.executor.position.size = 1.0
    env.executor.position.entry_price = 100.0
    env.executor.position.stop_loss_price = 99.0
    env.executor.used_margin = abs(env.executor.position.size * env.executor.position.entry_price) / env.executor.leverage
    env.executor.wallet_balance = 2.0  # 降低保證金率以觸發警告

    current_price = 100.0
    risk = env._compute_risk_signals(current_price)

    expected_liq = (env.executor.position.size * env.executor.position.entry_price - env.executor.used_margin) / (
        env.executor.position.size * (1.0 - env.executor.maintenance_margin_rate)
    )

    assert pytest.approx(risk['liq_price'], rel=1e-6) == expected_liq
    assert risk['near_liq'] is True
    assert risk['near_margin'] is True
    assert risk['near_stop'] is True
    assert risk['stop_loss_missing'] == 0.0


def test_observation_contains_risk_metrics(patched_config):
    df = _make_df()
    env = TradingEnvironment(df=df, random_start=False)
    obs, _ = env.reset()

    # cost_state 前 14 維為核心（穩定語義），後面會附加 action-conditioned proxies，因此總長度目前為 19
    assert obs['cost_state'].shape[0] == 19

    obs, _, _, _, info = env.step(np.array([0.5], dtype=np.float32))
    risk = info['risk_signals']
    expected_keys = {
        'liq_price',
        'price_gap',
        'gap_pct',
        'abs_gap_pct',
        'margin_ratio',
        'sl_gap_pct',
        'stop_loss_missing',
        'near_liq',
        'near_margin',
        'near_stop',
    }
    assert expected_keys.issubset(risk.keys())

    # 觀測中的 gap_pct 應與對應時間點的風險計算一致
    obs_price = float(env._close_arr[env.current_step])
    obs_risk = env._compute_risk_signals(obs_price)
    assert pytest.approx(obs['cost_state'][6]) == obs_risk['gap_pct']

