import pandas as pd
import numpy as np
import os
import glob


def load_data():
    """載入 Data/ 目錄下所有符合規則的 5min 與 1d 數據並合併"""
    print("Loading data from Data/ directory...")
    
    data_dir = 'Data'
    file_name = ''
    
    # 1. 搜尋所有相關檔案
    # 假設命名規則：*_5min.csv 為 5分線， *_1d.csv 為日線
    # 這裡會讀取所有找到的檔案並合併
    files_5m = glob.glob(os.path.join(data_dir, "*_5min.csv"))
    files_1d = glob.glob(os.path.join(data_dir, "*_1d.csv"))
    
    if not files_5m:
        raise ValueError("No *_5min.csv files found in Data/")
    if not files_1d:
        raise ValueError("No *_1d.csv files found in Data/")
        
    print(f"Found {len(files_5m)} 5m files: {[os.path.basename(f) for f in files_5m]}")
    print(f"Found {len(files_1d)} 1d files: {[os.path.basename(f) for f in files_1d]}")

    # 2. 讀取並合併 5m 數據
    df_list_5m = pd.DataFrame()
    for f in files_5m:
        try:
            df = pd.read_csv(f)

            if 'USDT' in os.path.basename(f).split('_')[0]:
                file_name = os.path.basename(f).split('_')[0]
            elif 'index_history_1d' in os.path.basename(f):
                file_name =  os.path.basename(f).replace('_index_history_1d', '')

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'])
            df = df.rename(columns=lambda c: f"{file_name}_{c}" if c != "timestamp" else c)

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

            if 'USDT' in os.path.basename(f).split('_')[0]:
                file_name = os.path.basename(f).split('_')[0]
            elif 'index_history_1d' in os.path.basename(f):
                file_name =  os.path.basename(f).replace('_index_history_1d', '')

            # 處理不同的時間欄位名稱
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'])

            df = df.rename(columns=lambda c: f"{file_name}_{c}" if c != "timestamp" else c)

            if df_list_1d.empty:
                df_list_1d = df
            else:
                df_list_1d = pd.merge(df_list_1d, df, on='timestamp', how='inner')
            
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if df_list_1d.empty:
        raise ValueError("Failed to load any 1d data.")
        
    df_1d = df_list_1d
    df_1d.sort_values('timestamp', inplace=True)
    df_1d.drop_duplicates(subset='timestamp', inplace=True)
    df_1d.reset_index(drop=True, inplace=True)
    
    
    print(f"Merged Data loaded. 5m: {len(df_5m)} rows, 1d: {len(df_1d)} rows")
    return df_5m, df_1d