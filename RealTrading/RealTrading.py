from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

# 從 ApiTrading 模組匯入交易客戶端
# 若以腳本方式執行，將父層資料夾加入路徑以便匯入
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ApiTrading import ITradingClient, build_trading_client


# ========== 公開資料擷取 ==========


def build_data_client() -> Any:
    """建立僅用於公開資料的輕量 Binance 客戶端。

    公開端點不需要憑證。
    """
    from binance.client import Client

    # 公開端點不需要 API 金鑰
    return Client(api_key=None, api_secret=None)


def fetch_klines_5m(client: Any, symbol: str, limit: int) -> pd.DataFrame:
    """抓取最新 5 分鐘 K 線，並計算與訓練資料一致的衍生欄位。

    參數:
        client: python-binance 的 Client 或相容的 mock。
        symbol: 例如 "BTCUSDT" 的交易對。
        limit: 需要抓取的 K 線筆數（視窗大小）。
    回傳:
        具有以下欄位的 DataFrame：
        ['timestamp','open','high','low','close','volume','buy_volume','sell_volume',
         'volume_ratio','long_short_ratio','trades','quote_volume']
    """
    raw = client.futures_klines(symbol=symbol, interval="5m", limit=limit)
    df = pd.DataFrame(
        raw,
        columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ],
    )
    if df.empty or len(df) < limit:
        raise RuntimeError(f"Insufficient klines for symbol={symbol}, limit={limit}")

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
        df[col] = df[col].astype(float)
    # 吃單買量可能為字串，需轉為浮點數
    df['taker_buy_base'] = df['taker_buy_base'].astype(float)
    # 衍生欄位
    df['buy_volume'] = df['taker_buy_base']
    df['sell_volume'] = df['volume'] - df['taker_buy_base']
    # 避免除以零
    eps = 1e-12
    df['volume_ratio'] = df['buy_volume'] / (df['sell_volume'] + eps)
    df['long_short_ratio'] = df['taker_buy_base'] / (df['volume'] - df['taker_buy_base'] + eps)
    df['trades'] = df['trades'].astype(float)
    out = df[[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'buy_volume', 'sell_volume', 'volume_ratio', 'long_short_ratio', 'trades', 'quote_volume'
    ]].copy()
    return out


# ========== 交易客戶端（需授權） ==========
# 注意：交易客戶端相關函式已移至 ApiTrading/Trading.py


# ========== 特徵觀測構建 ==========


def build_observation(df: pd.DataFrame, window_size: int) -> np.ndarray:
    """建立包含 z-score 特徵與歸零帳戶通道的觀測張量。

    張量形狀: (n_features, window_size)，其中 n_features = 價格特徵數量 + 5。
    """
    price_cols = ['open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume',
                  'volume_ratio', 'long_short_ratio', 'trades', 'quote_volume']
    slice_df = df.tail(window_size)
    if len(slice_df) != window_size:
        raise RuntimeError(f"Not enough data to build observation: need {window_size}, got {len(slice_df)}")

    price_data = slice_df[price_cols].values.T.astype(np.float32)
    mean = price_data.mean(axis=1, keepdims=True)
    std = price_data.std(axis=1, keepdims=True) + 1e-8
    price_z = (price_data - mean) / std

    n_features = len(price_cols) + 5
    obs = np.zeros((n_features, window_size), dtype=np.float32)
    obs[:len(price_cols)] = price_z
    # 帳戶通道在即時推論基線中維持為 0
    return obs


# ========== 動作映射 ==========


def map_policy_action(raw_action: np.ndarray, position_scale: float) -> np.ndarray:
    """將模型動作映射為訓練環境所使用的語義。

    - a0 ∈ [-1, 1]，再乘上 position_scale
    - a1 ∈ [-1, 1] 轉換為 [0, 50]（止盈百分比）
    - a2 ∈ [-1, 1] 轉換為 [0, 20]（止損百分比）
    """
    a0 = float(np.clip(raw_action[0], -1.0, 1.0)) * float(max(0.0, min(position_scale, 1.0)))
    a1 = (float(np.clip(raw_action[1], -1.0, 1.0)) + 1.0) * 0.5 * 50.0
    a2 = (float(np.clip(raw_action[2], -1.0, 1.0)) + 1.0) * 0.5 * 20.0
    return np.array([a0, a1, a2], dtype=np.float32)


# ========== 交易迴圈 ==========


def determine_device() -> str:
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


# 注意：帳戶與持倉相關查詢已移至 ApiTrading/Trading.py
# - get_equity_usdt -> client.get_equity_usdt()
# - get_current_position_size -> client.get_current_position_size(symbol)
# - get_account_summary -> client.get_account_summary()
# - get_open_positions -> client.get_open_positions()
# - place_delta_order -> client.place_delta_order(...)


def trading_loop(
    symbol: str,
    model_path: str,
    window_size: int,
    leverage: int,
    position_scale: float,
    min_notional_usdt: float,
    poll_interval_sec: int,
    deterministic: bool,
    dry_run: bool,
) -> None:
    """即時交易主迴圈。

    抓取 K 線、建立觀測、進行動作預測，並調整持倉。
    """
    data_client = build_data_client()
    trade_client: ITradingClient = build_trading_client(
        testnet=(os.getenv('BINANCE_TESTNET', '0') == '1')
    )

    # 啟動時設定一次槓桿
    lev = int(max(1, min(leverage, 125)))
    trade_client.set_leverage(symbol=symbol, leverage=lev)

    # 載入模型
    from stable_baselines3 import SAC
    device = determine_device()
    model = SAC.load(model_path, device=device)

    # 取得交易對限制
    filters = trade_client.get_symbol_filters(symbol=symbol, default_min_notional=min_notional_usdt)

    last_candle_open_ms: Optional[int] = None

    while True:
        try:
            df = fetch_klines_5m(data_client, symbol, limit=window_size)
            # 以開盤時間辨識最新 K 線
            latest_open_ms = int(df['timestamp'].iloc[-1].value // 1_000_000)
            # 確保每根新 K 線只處理一次
            if last_candle_open_ms is not None and latest_open_ms == last_candle_open_ms:
                time.sleep(poll_interval_sec)
                continue
            last_candle_open_ms = latest_open_ms

            obs = build_observation(df, window_size)
            act_raw, _ = model.predict(obs, deterministic=deterministic)
            act_mapped = map_policy_action(np.asarray(act_raw).reshape(3,), position_scale=position_scale)

            # 依據權益與槓桿計算目標持倉
            price = float(df['close'].iloc[-1])
            equity = trade_client.get_equity_usdt()
            target_position = (equity * float(act_mapped[0]) * float(lev)) / max(price, 1e-12)
            current_position = trade_client.get_current_position_size(symbol)
            delta = target_position - current_position

            # 下市價單以調整持倉至目標值
            res = trade_client.place_delta_order(
                symbol=symbol,
                delta=delta,
                step_size=filters.step_size,
                min_notional=filters.min_notional,
                last_price=price,
                dry_run=dry_run,
            )

            if res is not None:
                print({"placed": res, "equity": equity, "price": price, "target_pos": target_position, "current_pos": current_position})
            else:
                print({"skipped": True, "reason": "small_delta_or_notional", "equity": equity, "price": price})

            # 休眠直到接近下一根 K 線
            time.sleep(poll_interval_sec)
        except KeyboardInterrupt:
            print("使用者中斷，結束執行。")
            break
        except Exception as exc:
            # 記錄錯誤並持續執行
            print(f"Error in loop: {exc}")
            time.sleep(poll_interval_sec)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live trading loop using SB3 model on Binance Futures (5m)")
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--model', type=str, required=True, help='Path to SB3 model .zip')
    parser.add_argument('--window_size', type=int, default=288)
    parser.add_argument('--leverage', type=int, default=5)
    parser.add_argument('--position_scale', type=float, default=0.5)
    parser.add_argument('--min_notional_usdt', type=float, default=10.0)
    parser.add_argument('--poll_interval_sec', type=int, default=10)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--dry_run', action='store_true', help='If set, do not place real orders')
    sub = parser.add_subparsers(dest='cmd')
    # 可選子命令：帳戶/持倉
    if sub is not None:
        sub.add_parser('account', help='Print futures account summary')
        sub.add_parser('positions', help='Print non-zero open positions')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if getattr(args, 'cmd', None) == 'account':
        client = build_trading_client(testnet=(os.getenv('BINANCE_TESTNET', '0') == '1'))
        summary = client.get_account_summary()
        print(summary)
        return
    if getattr(args, 'cmd', None) == 'positions':
        client = build_trading_client(testnet=(os.getenv('BINANCE_TESTNET', '0') == '1'))
        print(client.get_open_positions())
        return
    trading_loop(
        symbol=args.symbol,
        model_path=args.model,
        window_size=args.window_size,
        leverage=args.leverage,
        position_scale=args.position_scale,
        min_notional_usdt=args.min_notional_usdt,
        poll_interval_sec=args.poll_interval_sec,
        deterministic=args.deterministic,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()


