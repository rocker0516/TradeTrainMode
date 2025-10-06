from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ========== Data fetching (public) ==========


def build_data_client() -> Any:
    """Construct a lightweight Binance client for public data.

    Uses no credentials for public endpoints.
    """
    from binance.client import Client

    # Public endpoints do not require API keys
    return Client(api_key=None, api_secret=None)


def fetch_klines_5m(client: Any, symbol: str, limit: int) -> pd.DataFrame:
    """Fetch latest 5m klines and compute derived columns to match training data.

    Args:
        client: python-binance Client or compatible mock.
        symbol: Symbol like "BTCUSDT".
        limit: Number of candles to fetch (window_size).
    Returns:
        DataFrame with columns:
        ['timestamp','open','high','low','close','volume', 'buy_volume','sell_volume',
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
    # taker buy base/quote may be strings
    df['taker_buy_base'] = df['taker_buy_base'].astype(float)
    # Derived
    df['buy_volume'] = df['taker_buy_base']
    df['sell_volume'] = df['volume'] - df['taker_buy_base']
    # avoid divide by zero
    eps = 1e-12
    df['volume_ratio'] = df['buy_volume'] / (df['sell_volume'] + eps)
    df['long_short_ratio'] = df['taker_buy_base'] / (df['volume'] - df['taker_buy_base'] + eps)
    df['trades'] = df['trades'].astype(float)
    out = df[[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'buy_volume', 'sell_volume', 'volume_ratio', 'long_short_ratio', 'trades', 'quote_volume'
    ]].copy()
    return out


# ========== Trading client (authenticated) ==========


def build_trade_client() -> Any:
    """Construct an authenticated Binance UM-Futures client from env.

    Required env vars: BINANCE_TRADE_API_KEY, BINANCE_TRADE_API_SECRET
    Optional: BINANCE_TESTNET=1 to use futures testnet endpoints.
    """
    from binance.client import Client

    api_key = os.getenv('BINANCE_TRADE_API_KEY') or os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_TRADE_API_SECRET') or os.getenv('BINANCE_API_SECRET')
    if not api_key or not api_secret:
        raise RuntimeError('Missing BINANCE_TRADE_API_KEY/SECRET or BINANCE_API_KEY/SECRET')

    use_testnet = os.getenv('BINANCE_TESTNET', '0') == '1'
    try:
        client = Client(api_key, api_secret, testnet=use_testnet)  # type: ignore[arg-type]
    except TypeError:
        client = Client(api_key, api_secret)
        if use_testnet:
            try:
                client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
            except Exception:
                pass
    return client


@dataclass
class SymbolFilters:
    step_size: float
    min_qty: float
    min_notional: float


def get_symbol_filters(client: Any, symbol: str, default_min_notional: float) -> SymbolFilters:
    """Fetch quantity filters for a futures symbol."""
    info = client.futures_exchange_info()
    syms: List[Dict[str, Any]] = info.get('symbols', [])
    s = next((x for x in syms if x.get('symbol') == symbol), None)
    if not s:
        # Fallback generic
        return SymbolFilters(step_size=0.0001, min_qty=0.0001, min_notional=default_min_notional)
    step = 0.0001
    min_qty = 0.0001
    min_notional = default_min_notional
    for f in s.get('filters', []):
        if f.get('filterType') == 'LOT_SIZE':
            step = float(f.get('stepSize', step))
            min_qty = float(f.get('minQty', min_qty))
        if f.get('filterType') == 'MIN_NOTIONAL':
            # Some futures APIs use NOTIONAL
            min_notional = float(f.get('notional', default_min_notional))
        if f.get('filterType') == 'NOTIONAL':
            min_notional = float(f.get('notional', default_min_notional))
    return SymbolFilters(step_size=step, min_qty=min_qty, min_notional=min_notional)


def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step


# ========== Observation builder ==========


def build_observation(df: pd.DataFrame, window_size: int) -> np.ndarray:
    """Build observation tensor with z-score features and zeroed account channels.

    Shape: (n_features, window_size) where n_features = len(price_cols) + 5.
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
    # Account channels remain zeros for live inference baseline
    return obs


# ========== Action mapping ==========


def map_policy_action(raw_action: np.ndarray, position_scale: float) -> np.ndarray:
    """Map model action to environment semantics used in training.

    - a0 in [-1, 1] scaled by position_scale
    - a1 in [-1, 1] -> [0, 50] (take profit percent)
    - a2 in [-1, 1] -> [0, 20] (stop loss percent)
    """
    a0 = float(np.clip(raw_action[0], -1.0, 1.0)) * float(max(0.0, min(position_scale, 1.0)))
    a1 = (float(np.clip(raw_action[1], -1.0, 1.0)) + 1.0) * 0.5 * 50.0
    a2 = (float(np.clip(raw_action[2], -1.0, 1.0)) + 1.0) * 0.5 * 20.0
    return np.array([a0, a1, a2], dtype=np.float32)


# ========== Trading loop ==========


def determine_device() -> str:
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def get_equity_usdt(client: Any) -> float:
    """Fetch USDT wallet balance for UM futures."""
    balances = client.futures_account_balance()
    for b in balances:
        if b.get('asset') == 'USDT':
            return float(b.get('balance', 0.0))
    return 0.0


def get_current_position_size(client: Any, symbol: str) -> float:
    """Return current base-asset position size (>0 long, <0 short)."""
    positions = client.futures_position_information(symbol=symbol)
    if not positions:
        return 0.0
    pos_amt = float(positions[0].get('positionAmt', 0.0))
    return pos_amt


def get_account_summary(client: Any) -> Dict[str, float]:
    """Return key USDT-M futures account metrics.

    Returns:
        dict with keys: wallet_balance, available_balance, unrealized_pnl, margin_balance
    """
    acc = client.futures_account()
    total_wb = 0.0
    total_ab = 0.0
    total_upnl = 0.0
    total_mb = 0.0
    try:
        assets = acc.get('assets', [])
        for a in assets:
            if a.get('asset') == 'USDT':
                total_wb = float(a.get('walletBalance', 0.0))
                total_upnl = float(a.get('unrealizedProfit', 0.0))
                total_mb = float(a.get('marginBalance', 0.0))
        total_ab = float(acc.get('availableBalance', 0.0))
    except Exception:
        pass
    return {
        'wallet_balance': total_wb,
        'available_balance': total_ab,
        'unrealized_pnl': total_upnl,
        'margin_balance': total_mb,
    }


def get_open_positions(client: Any) -> List[Dict[str, Any]]:
    """List non-zero positions across all symbols with key fields."""
    pos = client.futures_position_information()
    out: List[Dict[str, Any]] = []
    for p in pos:
        amt = float(p.get('positionAmt', 0.0))
        if abs(amt) > 0:
            out.append({
                'symbol': p.get('symbol'),
                'position_amt': amt,
                'entry_price': float(p.get('entryPrice', 0.0)),
                'unrealized_pnl': float(p.get('unRealizedProfit', 0.0)),
                'leverage': float(p.get('leverage', 0.0)),
            })
    return out

# 調整當前持倉至目標持倉
# client: 交易客戶端
# symbol: 交易對
# delta: 目標與當前持倉的差異
# step_size: 步長
# min_notional: 最小名義價值
# last_price: 資產的最新價格
# dry_run: 是否是模擬交易
def place_delta_order(client: Any, symbol: str, delta: float, step_size: float, min_notional: float, last_price: float, dry_run: bool = False) -> Optional[Dict[str, Any]]:
    """Place a market order to adjust position by delta quantity.

    Positive delta -> BUY; Negative delta -> SELL. Quantity rounded to step_size.
    Skips if notional below min_notional.
    """
    side = 'BUY' if delta > 0 else 'SELL'
    qty = floor_to_step(abs(delta), step_size)
    if qty <= 0:
        return None
    if (qty * last_price) < min_notional:
        return None
    if dry_run:
        return {"dry_run": True, "side": side, "quantity": qty}
    return client.futures_create_order(symbol=symbol, side=side, type='MARKET', quantity=qty)


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
    """Main live trading loop.

    Fetches klines, builds observation, predicts action, and adjusts position.
    """
    data_client = build_data_client()
    trade_client = build_trade_client()

    # Set leverage once at start
    lev = int(max(1, min(leverage, 125)))
    try:
        trade_client.futures_change_leverage(symbol=symbol, leverage=lev)
    except Exception:
        pass

    # Load model
    from stable_baselines3 import SAC
    device = determine_device()
    model = SAC.load(model_path, device=device)

    # Symbol filters
    filters = get_symbol_filters(trade_client, symbol, default_min_notional=min_notional_usdt)

    last_candle_open_ms: Optional[int] = None

    while True:
        try:
            df = fetch_klines_5m(data_client, symbol, limit=window_size)
            # Identify latest candle by open timestamp
            latest_open_ms = int(df['timestamp'].iloc[-1].value // 1_000_000)
            # Ensure we process once per new candle
            if last_candle_open_ms is not None and latest_open_ms == last_candle_open_ms:
                time.sleep(poll_interval_sec)
                continue
            last_candle_open_ms = latest_open_ms

            obs = build_observation(df, window_size)
            act_raw, _ = model.predict(obs, deterministic=deterministic)
            act_mapped = map_policy_action(np.asarray(act_raw).reshape(3,), position_scale=position_scale)

            # Compute target position based on equity and leverage
            price = float(df['close'].iloc[-1])
            equity = get_equity_usdt(trade_client)
            target_position = (equity * float(act_mapped[0]) * float(lev)) / max(price, 1e-12)
            current_position = get_current_position_size(trade_client, symbol)
            delta = target_position - current_position

            #
            # 這段程式碼用於下單以調整當前持倉至目標持倉。
            # 它使用 `place_delta_order` 函數，該函數考慮了交易客戶端、
            # 交易對(symbol)、目標與當前持倉的差異(delta)、步長(step size)、
            # 最小名義價值(min notional)、資產的最新價格(last price)，
            # 以及是模擬交易(dry run)還是真實下單。
            res = place_delta_order(
                trade_client,
                symbol,
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

            # Sleep until near next candle
            time.sleep(poll_interval_sec)
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting.")
            break
        except Exception as exc:
            # Log and continue
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
    # optional subcommands for account/positions
    if sub is not None:
        sub.add_parser('account', help='Print futures account summary')
        sub.add_parser('positions', help='Print non-zero open positions')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if getattr(args, 'cmd', None) == 'account':
        client = build_trade_client()
        summary = get_account_summary(client)
        print(summary)
        return
    if getattr(args, 'cmd', None) == 'positions':
        client = build_trade_client()
        print(get_open_positions(client))
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


