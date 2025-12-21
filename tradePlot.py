import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime
import os
import argparse


def read_csv_data(csv_file):
    """
    讀取 CSV 文件並返回 DataFrame
    
    Args:
        csv_file (str): CSV 文件路徑
        
    Returns:
        pd.DataFrame: 包含交易數據的 DataFrame
    """
    try:
        # 讀取 CSV 文件
        df = pd.read_csv(csv_file)
        
        # 將 timestamp 轉換為 datetime 格式
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 設置 timestamp 為索引
        df.set_index('timestamp', inplace=True)
        
        print(f"成功讀取 {csv_file}")
        print(f"數據範圍: {df.index[0]} 到 {df.index[-1]}")
        print(f"總共 {len(df)} 筆數據")
        
        return df
    
    except Exception as e:
        print(f"讀取 CSV 文件時發生錯誤: {e}")
        return None


def plot_candlestick(df, start_date=None, end_date=None, save_path=None, title=None):
    """
    繪製 K 線圖
    
    Args:
        df (pd.DataFrame): 包含 OHLC 數據的 DataFrame
        start_date (str): 開始日期 (格式: 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS')
        end_date (str): 結束日期 (格式: 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS')
        save_path (str): 保存圖片的路徑 (可選)
        title (str): 圖表標題 (可選)
    """
    # 複製數據以避免修改原始數據
    plot_df = df.copy()
    
    # 如果指定了日期範圍，則篩選數據
    if start_date or end_date:
        if start_date:
            start_date = pd.to_datetime(start_date)
            plot_df = plot_df[plot_df.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            plot_df = plot_df[plot_df.index <= end_date]
        
        print(f"\n繪製範圍: {plot_df.index[0]} 到 {plot_df.index[-1]}")
        print(f"共 {len(plot_df)} 筆數據")
    
    # 確保有數據可以繪製
    if len(plot_df) == 0:
        print("指定範圍內沒有數據")
        return
    
    # 根據數據量調整顯示策略
    if len(plot_df) > 1000:
        # 如果數據量太大，進行重採樣以提高性能
        print(f"數據量較大 ({len(plot_df)} 筆)，進行重採樣...")
        
        # 根據數據量決定重採樣週期
        if len(plot_df) > 10000:
            resample_period = '1H'  # 1小時
        elif len(plot_df) > 5000:
            resample_period = '30T'  # 30分鐘
        else:
            resample_period = '15T'  # 15分鐘
        
        plot_df = plot_df.resample(resample_period).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        print(f"重採樣後: {len(plot_df)} 筆數據 (週期: {resample_period})")
    
    # 設置圖表樣式
    mc = mpf.make_marketcolors(
        up='red',      # 上漲顏色
        down='green',  # 下跌顏色
        edge='inherit',
        wick={'up': 'red', 'down': 'green'},
        volume='inherit',
        alpha=0.8
    )
    
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='-',
        gridcolor='lightgray',
        figcolor='white',
        facecolor='white'
    )
    
    # 添加成交量
    add_plots = [
        mpf.make_addplot(plot_df['volume'], panel=1, color='gray', 
                        type='bar', ylabel='Volume', alpha=0.5)
    ]
    
    # 設置圖表參數
    kwargs = {
        'type': 'candle',
        'style': s,
        'title': title or 'K線圖',
        'ylabel': 'Price',
        'volume': False,  # 因為我們用 addplot 添加成交量
        'addplot': add_plots,
        'figsize': (12, 8),
        'tight_layout': True
    }
    
    # 如果數據點少於100個，顯示更多細節
    if len(plot_df) < 100:
        kwargs['show_nontrading'] = True
    
    # 繪製圖表
    if save_path:
        kwargs['savefig'] = save_path
        print(f"\n圖表已保存到: {save_path}")
    
    mpf.plot(plot_df, **kwargs)
    
    if not save_path:
        plt.show()


def main():
    """主函數，處理命令行參數"""
    parser = argparse.ArgumentParser(description='讀取CSV文件並繪製K線圖')
    parser.add_argument('csv_file', help='CSV文件路徑')
    parser.add_argument('--start', '-s', help='開始日期 (格式: YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end', '-e', help='結束日期 (格式: YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--save', help='保存圖片的路徑')
    parser.add_argument('--title', '-t', help='圖表標題')
    
    args = parser.parse_args()
    
    # 檢查文件是否存在
    if not os.path.exists(args.csv_file):
        print(f"錯誤: 找不到文件 {args.csv_file}")
        return
    
    # 讀取數據
    df = read_csv_data(args.csv_file)
    if df is None:
        return
    
    # 繪製圖表
    plot_candlestick(df, args.start, args.end, args.save, args.title)


# 提供簡單的函數接口供其他模組使用
def plot_from_csv(csv_file, start_date=None, end_date=None, save_path=None, title=None):
    """
    從 CSV 文件讀取數據並繪製 K 線圖的便捷函數
    
    Args:
        csv_file (str): CSV 文件路徑
        start_date (str): 開始日期 (可選)
        end_date (str): 結束日期 (可選)
        save_path (str): 保存路徑 (可選)
        title (str): 圖表標題 (可選)
    
    Example:
        # 繪製全部數據
        plot_from_csv('Data/BTCUSDT_futures_volume_5years_5min.csv')
        
        # 繪製指定範圍
        plot_from_csv('Data/BTCUSDT_futures_volume_5years_5min.csv', 
                      start_date='2023-01-01', 
                      end_date='2023-12-31')
    """
    df = read_csv_data(csv_file)
    if df is not None:
        plot_candlestick(df, start_date, end_date, save_path, title)


if __name__ == '__main__':
    main()
