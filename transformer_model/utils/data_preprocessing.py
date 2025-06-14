import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, List
import talib

class TradingDataPreprocessor:
    """
    交易數據預處理器
    處理原始CSV數據，提取多維特徵
    """
    
    def __init__(self, config):
        self.config = config
        self.window_size = config.WINDOW_SIZE
        
    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """
        加載並預處理數據
        Args:
            file_path: CSV文件路徑
        Returns:
            處理後的DataFrame
        """
        # 讀取數據
        df = pd.read_csv(file_path)
        
        # 確保timestamp是datetime格式
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 清理數據
        df = self._clean_data(df)
        
        # 添加技術指標
        df = self._add_technical_indicators(df)
        
        # 添加微觀結構特徵
        df = self._add_microstructure_features(df)
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理數據"""
        # 去除異常值
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume', 'quote_volume']
        
        for col in numeric_columns:
            if col in df.columns:
                # 去除負值
                df[col] = df[col].abs()
                
                # 去除極值（超過3個標準差）
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[col] = np.clip(df[col], mean_val - 3*std_val, mean_val + 3*std_val)
        
        # 前向填充缺失值
        df = df.fillna(method='ffill')
        
        # 去除剩餘的缺失值
        df = df.dropna()
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技術指標"""
        # 提取價格數據
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # RSI
        df['rsi'] = talib.RSI(close, timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # 布林帶
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # ROC (Rate of Change)
        df['roc'] = talib.ROC(close, timeperiod=10)
        df['volume_roc'] = talib.ROC(volume, timeperiod=10)
        
        # ATR (Average True Range)
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        df['atr_pct'] = df['atr'] / close
        
        # Williams %R
        df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        
        # 移動平均線
        df['sma_20'] = talib.SMA(close, timeperiod=20)
        df['ema_12'] = talib.EMA(close, timeperiod=12)
        df['ema_26'] = talib.EMA(close, timeperiod=26)
        
        # 價格變化率
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_20'] = df['close'].pct_change(20)
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加市場微觀結構特徵"""
        # 成交量特徵
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio_ma'] = df['volume_ratio'].rolling(20).mean()
        
        # 買賣壓力指標
        df['buy_pressure'] = df['buy_volume'] / df['volume']
        df['sell_pressure'] = df['sell_volume'] / df['volume']
        df['net_pressure'] = df['buy_pressure'] - df['sell_pressure']
        
        # 多空比例特徵
        df['long_short_ma'] = df['long_short_ratio'].rolling(20).mean()
        df['long_short_std'] = df['long_short_ratio'].rolling(20).std()
        
        # 交易頻率特徵
        df['trades_per_volume'] = df['trades'] / (df['volume'] + 1e-8)
        df['avg_trade_size'] = df['volume'] / (df['trades'] + 1e-8)
        
        # 大單小單比例（基於交易數量和成交量的關係）
        df['large_trade_ratio'] = np.where(
            df['avg_trade_size'] > df['avg_trade_size'].rolling(50).quantile(0.8),
            1, 0
        )
        
        return df
    
    def extract_features_for_model(self, df: pd.DataFrame, index: int) -> Dict[str, torch.Tensor]:
        """
        為模型提取特徵
        Args:
            df: 預處理後的DataFrame
            index: 當前時間步索引
        Returns:
            特徵字典
        """
        # 確保有足夠的歷史數據
        start_idx = max(0, index - self.window_size + 1)
        end_idx = index + 1
        
        window_data = df.iloc[start_idx:end_idx].copy()
        
        # 如果數據不足，進行填充
        if len(window_data) < self.window_size:
            # 複製第一行數據進行填充
            first_row = window_data.iloc[0:1]
            padding_rows = pd.concat([first_row] * (self.window_size - len(window_data)), ignore_index=True)
            window_data = pd.concat([padding_rows, window_data], ignore_index=True)
        
        features = {}
        
        # 1. 價格特徵 (OHLC)
        price_cols = ['open', 'high', 'low', 'close']
        features['price'] = torch.tensor(window_data[price_cols].values, dtype=torch.float32)
        
        # 2. 成交量特徵
        volume_cols = ['volume', 'buy_volume', 'sell_volume', 'quote_volume']
        features['volume'] = torch.tensor(window_data[volume_cols].values, dtype=torch.float32)
        
        # 3. 市場微觀結構特徵
        micro_cols = ['volume_ratio', 'long_short_ratio', 'trades']
        features['microstructure'] = torch.tensor(window_data[micro_cols].values, dtype=torch.float32)
        
        # 4. 技術指標特徵
        tech_cols = ['rsi', 'macd_hist', 'bb_position', 'roc', 'volume_roc', 'atr_pct', 'williams_r', 'stoch_k']
        tech_data = window_data[tech_cols].fillna(method='ffill').fillna(0)
        features['technical'] = torch.tensor(tech_data.values, dtype=torch.float32)
        
        # 5. 時間特徵
        timestamps = pd.to_datetime(window_data['timestamp'])
        time_features = np.zeros((len(timestamps), 5))
        
        time_features[:, 0] = timestamps.hour / 23.0
        time_features[:, 1] = timestamps.minute / 59.0
        time_features[:, 2] = timestamps.dayofweek / 6.0
        time_features[:, 3] = (timestamps.day - 1) / 30.0
        time_features[:, 4] = (timestamps.month - 1) / 11.0
        
        features['time'] = torch.tensor(time_features, dtype=torch.float32)
        
        # 添加用於技術指標計算的額外數據
        features['close'] = torch.tensor(window_data['close'].values, dtype=torch.float32)
        features['high'] = torch.tensor(window_data['high'].values, dtype=torch.float32)
        features['low'] = torch.tensor(window_data['low'].values, dtype=torch.float32)
        
        return features
    
    def create_training_data(self, df: pd.DataFrame, validation_split: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        創建訓練和驗證數據集
        Args:
            df: 預處理後的DataFrame
            validation_split: 驗證集比例
        Returns:
            (train_df, val_df)
        """
        # 按時間順序分割
        split_idx = int(len(df) * (1 - validation_split))
        
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()
        
        return train_df, val_df
    
    def normalize_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        特徵標準化
        Args:
            train_df: 訓練數據
            val_df: 驗證數據（可選）
        Returns:
            (normalized_train_df, normalized_val_df, normalization_stats)
        """
        # 需要標準化的列
        normalize_cols = [
            'open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume', 'quote_volume',
            'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower',
            'roc', 'volume_roc', 'atr', 'williams_r', 'stoch_k', 'stoch_d',
            'sma_20', 'ema_12', 'ema_26', 'volume_ma', 'volume_ratio_ma'
        ]
        
        # 計算標準化統計量（基於訓練數據）
        normalization_stats = {}
        for col in normalize_cols:
            if col in train_df.columns:
                mean_val = train_df[col].mean()
                std_val = train_df[col].std()
                normalization_stats[col] = {'mean': mean_val, 'std': std_val}
        
        # 標準化訓練數據
        normalized_train_df = train_df.copy()
        for col, stats in normalization_stats.items():
            normalized_train_df[col] = (train_df[col] - stats['mean']) / (stats['std'] + 1e-8)
        
        # 標準化驗證數據
        normalized_val_df = None
        if val_df is not None:
            normalized_val_df = val_df.copy()
            for col, stats in normalization_stats.items():
                if col in val_df.columns:
                    normalized_val_df[col] = (val_df[col] - stats['mean']) / (stats['std'] + 1e-8)
        
        return normalized_train_df, normalized_val_df, normalization_stats
    
    def get_feature_statistics(self, df: pd.DataFrame) -> Dict:
        """獲取特徵統計信息"""
        stats = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
                'missing_ratio': df[col].isnull().sum() / len(df)
            }
        
        return stats 