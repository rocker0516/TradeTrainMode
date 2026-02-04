import numpy as np
import pandas as pd
from Env.config import Config
from Env.feature_transformer import FeatureTransformer

class MarketData:
    """
    負責處理市場數據、特徵計算與緩存。
    目前支援 5min 與 1day 雙週期數據。

    重要更新（特徵工程模組化）：
    - 不再把 merge 後的「所有 numeric 欄位」直接丟進 observation（那會導致維度失控、混入其他幣種資訊、難以解析）。
    - 改為透過 `FeatureTransformer`：
      - 5m: 輸出固定 14 通道（可解釋、可控、適合 SAC）
      - 1d: 輸出固定通道（symbol-specific coinglass + global macro）
    - 1d 對齊採用策略 B：上一根已收盤日線（避免日內偷看未收盤資訊）
    """
    def __init__(self, df_5m: pd.DataFrame, df_1d: pd.DataFrame, window_size: int, window_size_1d: int, 
                 market_state_cols: list[str] = None, target_symbol: str = 'BTCUSDT', feature_symbols: list[str] | None = None):
        self.df_5m = df_5m.copy()
        self.df_1d = df_1d.copy()
        self.window_size = int(window_size)
        self.window_size_1d = int(window_size_1d)
        self.target_symbol = target_symbol
        # feature_symbols：決定 5m 跨市場摘要要納入哪些幣（固定順序、固定維度）。
        # - None 代表只使用 target_symbol（相容舊行為）
        self.feature_symbols = feature_symbols
        self.feature_lookback = int(max(288, self.window_size))
        self.feature_lookback_1d = int(max(30, self.window_size_1d))
        self._transformer = FeatureTransformer()
        
        # 1. 時間欄位處理與確保 datetime 格式
        self._ensure_datetime(self.df_5m)
        self._ensure_datetime(self.df_1d)
        
        # 2. 建立 5m 到 1d 的索引映射 (Alignment)
        # 假設: 我們在 5m 時間點 t，只能看到 t 之前（或當下已完成）的 1d 數據
        # 為了避免未來視，我們使用 searchsorted ('right') - 1 來找最近的一個過去或當下的 1d 蠟燭
        # 但通常 1d 蠟燭的 timestamp 若為 Open Time，則當天未收盤前我們不應看到當天的 High/Low/Close/Vol (除非是實時更新)
        # 這裡採取保守策略：對應到「上一個已收盤的日線」。若 df_1d index 是 Open Time，則 mapping 應指向上一個交易日。
        # 簡化起見：這裡假設 df_1d 的 timestamp 是該日「開始」時間。
        # 若 5m 時間是 2023-01-01 12:00，對應的「已完成日線」應是 2022-12-31。
        # 實作：找 df_1d.timestamp <= df_5m.timestamp 的最後一個 index，再減 1 (安全) 或視資料定義而定。
        # 暫定：mapping 指向「包含該 5m 的那一日」的日線數據（若視為實時更新）或「前一日」（若視為收盤數據）。
        # 用戶要求「直接取欄位」，通常隱含著這些欄位是可用的。
        # 這裡使用 asof merge 的概念建立索引陣列。
        
        times_5m = self.df_5m['timestamp'].values
        times_1d = self.df_1d['timestamp'].values
        
        # 找到每個 5m 時間點在 1d 時間軸上的插入位置 (left: arr[i-1] < v <= arr[i])
        # 我們希望找到 t_1d <= t_5m 的最大索引。
        # searchsorted(side='right') 返回 idx，使得 times_1d[:idx] <= t_5m (不完全正確，是 < vs <= 的差異)
        # 正確做法：searchsorted('right') - 1
        # 原本：idx_asof = last(times_1d <= t_5m)
        idx_asof = np.searchsorted(times_1d, times_5m, side='right') - 1

        # 你已選定對齊策略 B：上一根「已收盤」日線
        # 因此我們再往回退 1 根，避免在日內偷看到「今天尚未收盤」的 1d close/high/low 等資訊。
        #
        # 重要：由於 load_file.py 會 inner join 所有 1d 檔案，df_1d 的起始時間可能晚於 df_5m。
        # 當 5m 時間點早於 df_1d 的第一筆 timestamp 時，idx_asof 會是 -1，退一根後會變成 -2。
        # 若不做下界保護，get_1d_seq 的 padding 會超過 window_size_1d，導致 VecEnv stack shape mismatch。
        self.map_5m_to_1d = np.maximum(idx_asof - 1, -1)
        
        # 3. 基礎價格數據 (Numpy Access for Speed - 使用 5m 作為執行基準)
        # 根據 target_symbol 選擇價格欄位
        col_close = f"{self.target_symbol}_close"
        col_high = f"{self.target_symbol}_high"
        col_low = f"{self.target_symbol}_low"
        col_open = f"{self.target_symbol}_open"
        
        if col_close in self.df_5m.columns:
            self.close_arr = self.df_5m[col_close].values.astype(np.float64)
            self.high_arr = self.df_5m[col_high].values.astype(np.float64)
            self.low_arr = self.df_5m[col_low].values.astype(np.float64)
            # open 可能不存在（保守處理）
            self.open_arr = (
                self.df_5m[col_open].values.astype(np.float64)
                if col_open in self.df_5m.columns
                else self.close_arr.copy()
            )
        elif 'close' in self.df_5m.columns:
            # Fallback for single-pair data without prefix
            self.close_arr = self.df_5m['close'].values.astype(np.float64)
            self.high_arr = self.df_5m['high'].values.astype(np.float64)
            self.low_arr = self.df_5m['low'].values.astype(np.float64)
            self.open_arr = (
                self.df_5m['open'].values.astype(np.float64)
                if 'open' in self.df_5m.columns
                else self.close_arr.copy()
            )
        else:
            raise ValueError(f"Price columns for '{self.target_symbol}' (e.g. {col_close}) not found in df_5m columns: {self.df_5m.columns.tolist()[:10]}...")
        
        # 4. 計算 ATR Ratio (用於 ENV 內部的動態止損計算，非僅作為特徵)
        prev_close = pd.Series(self.close_arr).shift(1)
        true_range = np.maximum.reduce([
            (self.high_arr - self.low_arr),
            np.abs(self.high_arr - prev_close.fillna(0.0).values),
            np.abs(self.low_arr - prev_close.fillna(0.0).values),
        ])
        atr = pd.Series(true_range).rolling(14, min_periods=5).mean().fillna(0.0)
        self.atr_ratio_arr = (atr / np.maximum(self.close_arr, 1e-12)).astype(np.float32).values

        # 5. 計算 Rhythm Feature (RV Ratio) - 維持原邏輯供 Env 使用
        self.rv_ratio_arr = self._compute_rv_ratio()
        
        # 計算 Trend Score (簡單移動平均趨勢) - 補足缺失的屬性
        ma_50 = pd.Series(self.close_arr).rolling(window=50, min_periods=1).mean()
        ma_200 = pd.Series(self.close_arr).rolling(window=200, min_periods=1).mean()
        self.trend_score_arr = ((ma_50 - ma_200) / (ma_200 + 1e-8)).fillna(0.0).values.astype(np.float32)
        
        # 6. 透過 FeatureTransformer 建立固定特徵（可解釋、固定 shape）
        # 重要：只針對 target_symbol 產生特徵，避免把其他幣種的整套 OHLCV 混入 state。
        self.features_5m_arr, self.cols_5m = self._transformer.build_5m_features(
            self.df_5m,
            target_symbol=self.target_symbol,
            atr_ratio_arr=self.atr_ratio_arr,
            rv_ratio_arr=self.rv_ratio_arr,
            z_window=self.feature_lookback,
            feature_symbols=self.feature_symbols,
        )
        self.features_1d_arr, self.cols_1d = self._transformer.build_1d_features(
            self.df_1d,
            target_symbol=self.target_symbol,
            z_window_1d=max(60, self.feature_lookback_1d * 2),
        )

        # 7. 定義特徵維度供 Observer 使用（固定、可控）
        self.price_seq_features_dim = int(self.features_5m_arr.shape[1])
        self.features_1d_dim = int(self.features_1d_arr.shape[1])
        self.market_state_cols = self.cols_5m  # 相容既有介面：提供 5m 特徵欄位名稱

    def _ensure_datetime(self, df):
        # 確保存在 datetime 型態的「時間欄位」：若有 timestamp 或 time，統一轉成 timestamp 欄、且格式為 datetime
        time_col = None
        if 'timestamp' in df.columns:
            time_col = 'timestamp'
        elif 'time' in df.columns:
            time_col = 'time'

        if time_col is not None:
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col])
            if time_col != 'timestamp':
                df['timestamp'] = df[time_col]
        # 若皆無則略過

    # _extract_numeric_features 已不再使用（改用 FeatureTransformer 輸出固定特徵），保留舊函數會讓維度失控且難以解析。

    def _compute_rv_ratio(self):
        try:
            close_series = pd.Series(self.close_arr)
            log_close = np.log(np.clip(close_series, 1e-12, None))
            log_ret = log_close.diff().fillna(0.0)
            rv_20 = log_ret.rolling(20, min_periods=5).std().fillna(0.0)
            rv_288 = (
                log_ret.rolling(288, min_periods=20)
                .std()
                .replace(0.0, np.nan)
                .bfill()
                .fillna(1e-8)
            )
            rv_ratio = (rv_20 / rv_288).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            return rv_ratio.values.astype(np.float32)
        except Exception:
            return np.zeros(len(self.df_5m), dtype=np.float32)

    def get_market_metrics(self, step_idx: int) -> dict:
        """取得特定時間點的市場指標 (ATR, Price, etc.)"""
        idx = min(step_idx, len(self.df_5m) - 1)
        return {
            'close': float(self.close_arr[idx]),
            'high': float(self.high_arr[idx]),
            'low': float(self.low_arr[idx]),
            'atr_ratio': float(self.atr_ratio_arr[idx]),
            'rv_ratio': float(self.rv_ratio_arr[idx]),
            'trend_score': float(self.trend_score_arr[idx])
        }

    def get_price_seq(self, step_idx: int) -> np.ndarray:
        """取得 5m 序列輸入 [window_size, F_5m]"""
        start = step_idx - self.window_size
        end = step_idx
        if start < 0:
            pad = np.zeros((abs(start), self.features_5m_arr.shape[1]), dtype=np.float32)
            data = self.features_5m_arr[0:end]
            return np.vstack([pad, data])
        return self.features_5m_arr[start:end]

    def get_1d_seq(self, step_idx: int, window_size_1d: int = 30) -> np.ndarray:
        """
        取得 1d 序列輸入 [window_size_1d, F_1d]
        根據 step_idx (5m) 找到對應的 1d 索引，再往回取 window。
        """
        # 1. 找到對應的 1d index
        idx_5m = min(step_idx, len(self.df_5m) - 1)
        idx_1d_current = self.map_5m_to_1d[idx_5m]
        
        # 2. Slice 1d array
        # 注意: 這裡取到 idx_1d_current (包含)，視為當下可見的日線資訊
        # 若 idx_1d_current 指向的是「今天」(尚未收盤)，則依賴 feature extraction 傳入的是實時更新值
        # 若 idx_1d_current 指向的是「昨天」(已收盤)，則為滯後資訊
        # 由 map_5m_to_1d 的構建邏輯決定。
        
        # 注意：為了支援 VecEnv（多環境堆疊），此函式必須「無論任何邊界狀況」都回傳固定 shape：
        #   (window_size_1d, F_1d)
        out = np.zeros((int(window_size_1d), self.features_1d_arr.shape[1]), dtype=np.float32)

        idx_1d_current = int(idx_1d_current)
        if idx_1d_current < 0:
            # 沒有任何已收盤日線可用 -> 全 0
            return out

        end = idx_1d_current + 1  # slice end（不含）
        start = end - int(window_size_1d)

        # 取可用區間並放到 out 尾端（不足前面補 0）
        src_start = max(0, start)
        src_end = min(end, len(self.features_1d_arr))
        src = self.features_1d_arr[src_start:src_end]
        if len(src) == 0:
            return out

        out[-len(src) :] = src
        return out
