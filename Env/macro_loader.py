import pandas as pd
import numpy as np
import os

def load_and_merge_daily_data(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    data_dir: str = "Data"
) -> pd.DataFrame:
    """
    載入並合併多來源的日線數據，並進行標準化處理。
    回傳的 DataFrame Index 為日期，Columns 為各類特徵。
    """
    
    # 1. 定義檔案路徑
    files = {
        "price": "BTCUSDT_futures_volume_coinglass_5years_1d.csv",
        "altcoin": "altcoin_season_index_history.csv",
        "bmo": "bitcoin_macro_oscillator_index_history.csv",
        "sopr": "bitcoin_sth_sopr_index_history.csv",
        "fng": "fear_greed_index_history.csv"
    }
    
    dfs = []
    
    # 2. 讀取 BTC 日線價格 (作為基準)
    path = os.path.join(data_dir, files["price"])
    if os.path.exists(path):
        df_price = pd.read_csv(path)
        # 統一日期欄位名稱
        date_col = 'time' if 'time' in df_price.columns else 'date'
        df_price[date_col] = pd.to_datetime(df_price[date_col])
        df_price.set_index(date_col, inplace=True)
        # 移除 index 的 timezone (如果有)
        if df_price.index.tz is not None:
            df_price.index = df_price.index.tz_localize(None)
        
        # 特徵工程: 日線 Log Returns & Normalized Volume
        # 使用 close 計算
        df_price['daily_ret'] = np.log(df_price['close'] / df_price['close'].shift(1)).fillna(0)
        df_price['daily_range'] = (df_price['high'] - df_price['low']) / df_price['close']
        df_price['daily_vol'] = np.log1p(df_price['volume_usd']) # volume_usd used in coinglass data
        
        # Z-score normalization (Rolling 30 days)
        for col in ['daily_ret', 'daily_range', 'daily_vol']:
            mean = df_price[col].rolling(30, min_periods=1).mean()
            std = df_price[col].rolling(30, min_periods=1).std().replace(0, 1)
            df_price[f'{col}_z'] = (df_price[col] - mean) / std
            
        # 只保留需要的正規化特徵
        dfs.append(df_price[['daily_ret_z', 'daily_range_z', 'daily_vol_z']])
    
    # 3. 讀取 Altcoin Season Index
    path = os.path.join(data_dir, files["altcoin"])
    if os.path.exists(path):
        df_alt = pd.read_csv(path)
        df_alt['timestamp'] = pd.to_datetime(df_alt['timestamp'])
        df_alt.set_index('timestamp', inplace=True)
        if df_alt.index.tz is not None:
            df_alt.index = df_alt.index.tz_localize(None)
        # 正規化 0-100 -> 0-1
        df_alt['altcoin_idx_norm'] = df_alt['altcoin_index'] / 100.0
        dfs.append(df_alt[['altcoin_idx_norm']])

    # 4. 讀取 BMO (Bitcoin Macro Oscillator)
    path = os.path.join(data_dir, files["bmo"])
    if os.path.exists(path):
        df_bmo = pd.read_csv(path)
        df_bmo['timestamp'] = pd.to_datetime(df_bmo['timestamp'])
        df_bmo.set_index('timestamp', inplace=True)
        if df_bmo.index.tz is not None:
            df_bmo.index = df_bmo.index.tz_localize(None)
        # BMO 通常在 -2.0 ~ 2.0 之間，可視為已正規化，或做 clip
        df_bmo['bmo_val'] = df_bmo['bmo_value'].clip(-3, 3)
        dfs.append(df_bmo[['bmo_val']])

    # 5. 讀取 SOPR
    path = os.path.join(data_dir, files["sopr"])
    if os.path.exists(path):
        df_sopr = pd.read_csv(path)
        date_col = [c for c in df_sopr.columns if 'time' in c or 'date' in c][0]
        # 找出數值欄位 (可能是 lth_sopr 或 sth_sopr)
        val_cols = [c for c in df_sopr.columns if 'sopr' in c]
        val_col = val_cols[0] if val_cols else df_sopr.columns[1]

        df_sopr[date_col] = pd.to_datetime(df_sopr[date_col])
        df_sopr.set_index(date_col, inplace=True)
        if df_sopr.index.tz is not None:
            df_sopr.index = df_sopr.index.tz_localize(None)
        # SOPR 以 1.0 為基準，做 log 處理使其對稱
        df_sopr['sopr_log'] = np.log(df_sopr[val_col].replace(0, 1))
        # 簡單標準化
        df_sopr['sopr_log'] = df_sopr['sopr_log'].clip(-0.1, 0.1) * 10 # Scale up slightly
        dfs.append(df_sopr[['sopr_log']])

    # 6. 讀取 Fear & Greed
    path = os.path.join(data_dir, files["fng"])
    if os.path.exists(path):
        df_fng = pd.read_csv(path)
        # time, fear_greed_index, price
        df_fng['time'] = pd.to_datetime(df_fng['time'])
        df_fng.set_index('time', inplace=True)
        if df_fng.index.tz is not None:
            df_fng.index = df_fng.index.tz_localize(None)
        # 正規化 0-100 -> 0-1
        df_fng['fng_norm'] = df_fng['fear_greed_index'] / 100.0
        dfs.append(df_fng[['fng_norm']])

    # 7. 合併所有數據
    if not dfs:
        return pd.DataFrame()
    
    # 使用 outer join 保留所有日期
    full_df = pd.concat(dfs, axis=1)
    
    # 8. 處理缺失值
    # 先排序索引
    full_df.sort_index(inplace=True)
    # 先 ffill (沿用昨日數據)，再 bfill (避免開頭缺失)，最後 fillna(0)
    full_df = full_df.ffill().bfill().fillna(0.0)
    
    # 9. 對齊主時間範圍
    # 確保 start_date 和 end_date 沒有時區
    if hasattr(start_date, 'tz_localize'):
        start_date = start_date.tz_localize(None)
    if hasattr(end_date, 'tz_localize'):
        end_date = end_date.tz_localize(None)
        
    idx = pd.date_range(start=start_date, end=end_date, freq='D')
    full_df = full_df.reindex(idx, method='ffill').fillna(0.0)
    
    return full_df.astype(np.float32)
