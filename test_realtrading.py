import types
from unittest import mock
import numpy as np


def test_build_observation_and_action_mapping():
    import RealTrading.RealTrading as rt
    # Build fake DF of window_size rows
    window = 10
    data = {
        'timestamp': [i for i in range(window)],
        'open': [1.0 + i * 0.1 for i in range(window)],
        'high': [1.1 + i * 0.1 for i in range(window)],
        'low': [0.9 + i * 0.1 for i in range(window)],
        'close': [1.05 + i * 0.1 for i in range(window)],
        'volume': [100 + i for i in range(window)],
        'buy_volume': [60 + i for i in range(window)],
        'sell_volume': [40 for _ in range(window)],
        'volume_ratio': [1.5 for _ in range(window)],
        'long_short_ratio': [1.2 for _ in range(window)],
        'trades': [10 + i for i in range(window)],
        'quote_volume': [1000 + i for i in range(window)],
    }
    import pandas as pd
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    obs = rt.build_observation(df, window)
    assert obs.shape[1] == window
    assert obs.shape[0] == 11 + 5
    assert np.isfinite(obs).all()

    mapped = rt.map_policy_action(np.array([1.2, -2.0, 0.5]), position_scale=0.5)
    assert mapped.shape == (3,)
    assert -0.5 <= mapped[0] <= 0.5
    assert 0.0 <= mapped[1] <= 50.0
    assert 0.0 <= mapped[2] <= 20.0


def test_place_delta_order_skips_small_notional():
    import RealTrading.RealTrading as rt
    fake_client = mock.MagicMock()
    res = rt.place_delta_order(
        fake_client,
        symbol='BTCUSDT',
        delta=0.00009,
        step_size=0.001,
        min_notional=10.0,
        last_price=50000.0,
        dry_run=False,
    )
    assert res is None
    fake_client.futures_create_order.assert_not_called()


def test_account_and_positions_parsing():
    import RealTrading.RealTrading as rt
    fake_client = mock.MagicMock()
    fake_client.futures_account.return_value = {
        'availableBalance': '100.1',
        'assets': [
            {'asset': 'USDT', 'walletBalance': '150.5', 'unrealizedProfit': '2.3', 'marginBalance': '152.8'}
        ]
    }
    fake_client.futures_position_information.return_value = [
        {'symbol': 'BTCUSDT', 'positionAmt': '0.01', 'entryPrice': '50000', 'unRealizedProfit': '5', 'leverage': '10'},
        {'symbol': 'ETHUSDT', 'positionAmt': '0.0', 'entryPrice': '2000', 'unRealizedProfit': '0', 'leverage': '5'},
    ]

    summary = rt.get_account_summary(fake_client)
    assert summary['wallet_balance'] == 150.5
    assert summary['available_balance'] == 100.1
    assert summary['unrealized_pnl'] == 2.3
    assert summary['margin_balance'] == 152.8

    positions = rt.get_open_positions(fake_client)
    assert len(positions) == 1
    assert positions[0]['symbol'] == 'BTCUSDT'


