import pandas as pd


def test_ratio_is_nan_when_sell_volume_is_zero() -> None:
    """
    與匯入 Supabase 一致：分母為 0 時比率應為 NaN（而非 inf）。
    """
    df = pd.DataFrame(
        {
            "volume": [10.0, 8.0],
            "taker_buy_base": [10.0, 2.0],
        }
    )
    df["buy_volume"] = df["taker_buy_base"]
    df["sell_volume"] = df["volume"] - df["taker_buy_base"]
    ratio_denominator = df["sell_volume"].replace(0, pd.NA)
    df["volume_ratio"] = df["buy_volume"] / ratio_denominator
    df["long_short_ratio"] = df["taker_buy_base"] / ratio_denominator

    assert pd.isna(df.loc[0, "volume_ratio"])
    assert pd.isna(df.loc[0, "long_short_ratio"])
    assert df.loc[1, "volume_ratio"] == 1 / 3
