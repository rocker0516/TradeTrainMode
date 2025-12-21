"""
tradePlot.py 使用範例
"""

from tradePlot import plot_from_csv

# 範例 1: 繪製全部數據
print("範例 1: 繪製全部數據")
plot_from_csv('Data/BTCUSDT_futures_volume_5years_5min.csv', 
              title='BTC/USDT 期貨 K線圖（全部數據）')

# 範例 2: 繪製指定日期範圍的數據
print("\n範例 2: 繪製 2024 年的數據")
plot_from_csv('Data/BTCUSDT_futures_volume_5years_5min.csv',
              start_date='2024-01-01',
              end_date='2024-12-31',
              title='BTC/USDT 期貨 K線圖（2024年）')

# 範例 3: 繪製特定時間範圍並保存圖片
print("\n範例 3: 繪製最近一個月的數據並保存")
plot_from_csv('Data/BTCUSDT_futures_volume_5years_5min.csv',
              start_date='2024-08-01',
              end_date='2024-08-31',
              save_path='btc_august_2024.png',
              title='BTC/USDT 期貨 K線圖（2024年8月）')

# 範例 4: 繪製其他交易對
print("\n範例 4: 繪製 ETH/USDT")
plot_from_csv('Data/ETHUSDT_futures_volume_5years_5min.csv',
              start_date='2024-06-01',
              end_date='2024-06-30',
              title='ETH/USDT 期貨 K線圖（2024年6月）')
