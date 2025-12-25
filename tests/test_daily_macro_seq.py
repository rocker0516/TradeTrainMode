import numpy as np
import pandas as pd

from Env.trading_env import TradingEnvironment


def _write_csv(path, header, rows):
    path.write_text("\n".join([header] + rows), encoding="utf-8")


def test_daily_seq_shape_and_no_lookahead(tmp_path):
    """
    驗證：
    - env 會回傳 daily_seq
    - daily_seq shape 正確
    - daily_seq 的最後一筆對齊「昨天 (D-1)」的日線資料（避免 look-ahead）
    """
    data_dir = tmp_path

    # --- Create minimal daily macro CSVs ---
    _write_csv(
        data_dir / "altcoin_season_index_history.csv",
        "timestamp,altcoin_index,altcoin_marketcap",
        [
            "2022-12-20,50,0",
            "2022-12-21,55,0",
            "2022-12-22,60,0",
            "2022-12-23,65,0",
            "2022-12-24,70,0",
            "2022-12-25,75,0",
        ],
    )

    _write_csv(
        data_dir / "bitcoin_macro_oscillator_index_history.csv",
        "price,bmo_value,timestamp",
        [
            "10000,-0.5,2022-12-20",
            "10100,-0.4,2022-12-21",
            "10200,-0.3,2022-12-22",
            "10300,-0.2,2022-12-23",
            "10400,-0.1,2022-12-24",
            "10500,0.0,2022-12-25",
        ],
    )

    _write_csv(
        data_dir / "bitcoin_sth_sopr_index_history.csv",
        "timestamp,price,lth_sopr",
        [
            "2022-12-20,10000,1.01",
            "2022-12-21,10100,1.02",
            "2022-12-22,10200,0.99",
            "2022-12-23,10300,1.00",
            "2022-12-24,10400,1.03",
            "2022-12-25,10500,1.01",
        ],
    )

    _write_csv(
        data_dir / "fear_greed_index_history.csv",
        "time,fear_greed_index,price",
        [
            "2022-12-20 00:00:00,30,10000",
            "2022-12-21 00:00:00,35,10100",
            "2022-12-22 00:00:00,40,10200",
            "2022-12-23 00:00:00,45,10300",
            "2022-12-24 00:00:00,50,10400",
            "2022-12-25 00:00:00,55,10500",
        ],
    )

    _write_csv(
        data_dir / "BTCUSDT_futures_volume_coinglass_5years_1d.csv",
        "time,open,high,low,close,volume_usd,open_interest_close,funding_rate_close,"
        "global_long_short_account_ratio_global_account_long_short_ratio,top_long_short_position_ratio_top_position_long_short_ratio,"
        "liquidation_long_liquidation_usd,liquidation_short_liquidation_usd",
        [
            "2022-12-20,10000,10100,9900,10050,100000000,1000,0.001,1.5,1.1,10000,12000",
            "2022-12-21,10050,10200,9950,10150,110000000,1100,0.001,1.6,1.1,11000,13000",
            "2022-12-22,10150,10300,10000,10250,120000000,1200,0.001,1.7,1.1,12000,14000",
            "2022-12-23,10250,10400,10100,10350,130000000,1300,0.001,1.8,1.1,13000,15000",
            "2022-12-24,10350,10500,10200,10450,140000000,1400,0.001,1.9,1.1,14000,16000",
            "2022-12-25,10450,10600,10300,10550,150000000,1500,0.001,2.0,1.1,15000,17000",
        ],
    )

    macro_files = {
        "altcoin_season": "altcoin_season_index_history.csv",
        "bmo": "bitcoin_macro_oscillator_index_history.csv",
        "sopr": "bitcoin_sth_sopr_index_history.csv",
        "fear_greed": "fear_greed_index_history.csv",
        "coinglass_btc_1d": "BTCUSDT_futures_volume_coinglass_5years_1d.csv",
    }

    # --- Create minimal 5m OHLCV df with DatetimeIndex ---
    idx = pd.date_range(start="2022-12-25 00:00:00", periods=200, freq="5min")
    base = 10500.0 + np.linspace(0, 10, len(idx))
    df_5m = pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.5,
            "volume": np.full(len(idx), 100.0),
        },
        index=idx,
    )

    env = TradingEnvironment(
        df=df_5m,
        env_id=0,
        random_start=False,
        window_size=10,
        daily_window_size=5,
        macro_enabled=True,
        macro_data_dir=str(data_dir),
        macro_files=macro_files,
    )

    obs, _ = env.reset(seed=0)
    assert "daily_seq" in obs
    assert obs["daily_seq"].shape[0] == 5
    assert obs["daily_seq"].shape[1] == env.daily_features_dim
    assert np.isfinite(obs["daily_seq"]).all()

    # current_step = window_size when random_start=False
    step = int(env.current_step)
    step_day = pd.Timestamp(df_5m.index[step]).normalize()
    expected_day = (step_day - pd.Timedelta(days=1)).normalize()

    ptr = int(env.daily_indices[step])
    got_day = pd.Timestamp(env.macro_df.index[ptr]).normalize()
    assert got_day == expected_day

    # daily_seq last row must match macro row at ptr
    np.testing.assert_allclose(obs["daily_seq"][-1], env._macro_arr[ptr], rtol=0, atol=1e-6)
    assert np.isfinite(obs["daily_seq"]).all()


