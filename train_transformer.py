import pandas as pd

# 加載數據
print("加載數據...")
df = pd.read_csv(args.data_path)
print(f"數據量: {len(df)} 行")

# 將timestamp設為索引
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    print("✓ 時間戳已設為索引")

# 創建環境
print("創建環境...")
env = TransformerTradingEnvironment(df, config) 