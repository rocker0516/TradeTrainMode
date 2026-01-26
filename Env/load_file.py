import pandas as pd
import numpy as np
import os
import glob


def load_data():
    """
    載入 Data/ 目錄下所有符合規則的 5min 與 1d 數據並合併。

    注意：
    - 5m：維持 inner join，確保多幣 5m 時間軸完全對齊（避免某幣缺 bar 造成 NaN 連鎖）。
    - 1d：改用 outer join，避免「任一檔案沒有重疊 timestamp」就讓整個 inner join 變成空表，
      進而被後續檔案覆蓋、最終只剩單一來源（會讓 1d 特徵全為 0，第二路 CNN 失去意義）。
    - prefix 命名：macro 檔案要去掉副檔名，確保欄位能被 FeatureTransformer 正確辨識。
    """
    #print("Loading data from Data/ directory...")
    
    data_dir = 'Data'

    def _infer_prefix(basename: str) -> str:
        """
        由檔名推導欄位前綴。

        支援兩類命名：
        - 幣種資料：{SYMBOL}_..._5min.csv / {SYMBOL}_..._1d.csv（例如 BTCUSDT_...）
        - macro/index：{name}_index_history_1d.csv（例如 fear_greed_index_history_1d.csv）
        """
        stem, _ext = os.path.splitext(str(basename))
        head = stem.split('_')[0]
        if "USDT" in head:
            return head
        if stem.endswith("_index_history_1d"):
            return stem.replace("_index_history_1d", "")
        # fallback：用整個 stem（避免 prefix 變成空字串）
        return stem
    
    # 1. 搜尋所有相關檔案
    # 假設命名規則：*_5min.csv 為 5分線， *_1d.csv 為日線
    # 這裡會讀取所有找到的檔案並合併
    files_5m = glob.glob(os.path.join(data_dir, "*_5min.csv"))
    files_1d = glob.glob(os.path.join(data_dir, "*_1d.csv"))
    
    if not files_5m:
        raise ValueError("No *_5min.csv files found in Data/")
    if not files_1d:
        raise ValueError("No *_1d.csv files found in Data/")
        
    #print(f"Found {len(files_5m)} 5m files: {[os.path.basename(f) for f in files_5m]}")
    #print(f"Found {len(files_1d)} 1d files: {[os.path.basename(f) for f in files_1d]}")

    # 2. 讀取並合併 5m 數據
    df_list_5m = pd.DataFrame()
    for f in files_5m:
        try:
            df = pd.read_csv(f)

            prefix = _infer_prefix(os.path.basename(f))

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'])
            else:
                raise ValueError(f"CSV missing timestamp/time: {os.path.basename(f)}")

            df = df.rename(columns=lambda c: f"{prefix}_{c}" if c != "timestamp" else c)

            if df_list_5m.empty:
                df_list_5m = df
            else:
                df_list_5m = pd.merge(df_list_5m, df, on='timestamp', how='inner')
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    if df_list_5m.empty:
        raise ValueError("Failed to load any 5m data.")

    df_5m = df_list_5m
    df_5m.sort_values('timestamp', inplace=True)
    df_5m.drop_duplicates(subset='timestamp', inplace=True) # 移除重複時間點
    df_5m.reset_index(drop=True, inplace=True)
    
    # 3. 讀取並合併 1d 數據
    df_list_1d = pd.DataFrame()
    for f in files_1d:
        try:
            df = pd.read_csv(f)

            prefix = _infer_prefix(os.path.basename(f))

            # 處理不同的時間欄位名稱
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'])
            else:
                raise ValueError(f"CSV missing timestamp/time: {os.path.basename(f)}")

            df = df.rename(columns=lambda c: f"{prefix}_{c}" if c != "timestamp" else c)

            if df_list_1d.empty:
                df_list_1d = df
            else:
                # 1d 允許 outer join：部分來源可能起始時間更晚，不應導致整體變空。
                df_list_1d = pd.merge(df_list_1d, df, on='timestamp', how='outer')
            
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if df_list_1d.empty:
        raise ValueError("Failed to load any 1d data.")
        
    df_1d = df_list_1d
    df_1d.sort_values('timestamp', inplace=True)
    df_1d.drop_duplicates(subset='timestamp', inplace=True)
    df_1d.reset_index(drop=True, inplace=True)
    
    
    #print(f"Merged Data loaded. 5m: {len(df_5m)} rows, 1d: {len(df_1d)} rows")
    return df_5m, df_1d