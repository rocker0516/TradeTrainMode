from __future__ import annotations

import os
import sys
import numpy as np

# Allow running as a script on Windows (same trick as Train/run_sac_lag.py)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Env.load_file import load_data
from Env.Components.market_data import MarketData


def main() -> None:
    df_5m, df_1d = load_data()
    md = MarketData(
        df_5m=df_5m,
        df_1d=df_1d,
        window_size=64,
        window_size_1d=30,
        target_symbol="BTCUSDT",
        feature_symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT"],
    )

    cols = list(md.cols_1d)
    X = np.asarray(md.features_1d_arr, dtype=np.float64)

    def col_stats(name: str) -> None:
        if name not in cols:
            print(f"[features_1d_arr] missing col: {name}")
            return
        i = cols.index(name)
        v = X[:, i]
        print(
            f"[features_1d_arr] {name}: std={float(np.std(v)):.6g}, "
            f"zero={float((v == 0.0).mean()):.4f}, min={float(np.min(v)):.3g}, max={float(np.max(v)):.3g}"
        )

    print(f"df_5m ts range: {df_5m['timestamp'].min()} -> {df_5m['timestamp'].max()}")
    print(f"df_1d ts range: {df_1d['timestamp'].min()} -> {df_1d['timestamp'].max()}")
    print(f"features_1d_arr shape: {X.shape}, cols_1d={len(cols)}")

    for n in [
        "ret_1d_z",
        "range_1d_z",
        "close_over_ema_20_z",
        "ema_20_60_spread_z",
        "fear_greed_z",
        "altcoin_season_z",
        "oi_close_z",
        "funding_close_z",
        "sopr_z",
    ]:
        col_stats(n)

    for c in ["BTCUSDT_close", "fear_greed_fear_greed_index", "altcoin_season_altcoin_index"]:
        if c in df_1d.columns:
            nn = float(df_1d[c].notna().mean())
            print(f"[df_1d] nonnull {c}: {nn:.4f}")
        else:
            print(f"[df_1d] missing {c}")

    # 進一步：抽幾個日期看實際值（幫助判斷是否 timestamp 對不齊）
    for c in ["BTCUSDT_close", "fear_greed_fear_greed_index"]:
        if c not in df_1d.columns:
            continue
        s = df_1d[["timestamp", c]].dropna()
        print(f"[df_1d] sample {c} rows={len(s)}")
        print(s.head(3).to_string(index=False))
        print(s.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()

