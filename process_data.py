import pandas as pd

# 讀取原始數據
df = pd.read_csv('../Data/BTCUSDT_futures_volume_5years_5min.csv')
print(f"原始數據形狀: {df.shape}")
print(f"原始列名: {list(df.columns)}")

# 將timestamp設為索引
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 保存處理後的數據
df.to_csv('../Data/BTCUSDT_processed.csv')

print(f"處理後數據形狀: {df.shape}")
print(f"處理後列名: {list(df.columns)}")
print("✅ 數據已處理並保存為 BTCUSDT_processed.csv") 