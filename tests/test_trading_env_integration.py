import numpy as np
import pandas as pd

from Env.trading_env import TradingEnvironment


def make_df(length: int = 50, base: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    close = base * (1.0 + rng.normal(0.0, 0.002, size=length)).cumprod()
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.001, size=length)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.001, size=length)))
    volume = rng.integers(1000, 5000, size=length)

    # 其他必要欄位（以常數或簡單派生填充）
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'buy_volume': volume * 0.5,
        'sell_volume': volume * 0.5,
        'volume_ratio': np.clip(volume / (volume.mean() + 1e-9), 0.1, 10.0),
        'long_short_ratio': np.full(length, 1.0),
        'trades': rng.integers(100, 300, size=length),
        'quote_volume': close * volume,
    })
    return df


def test_env_reset_and_observation_shapes():
    df = make_df(60)
    env = TradingEnvironment(df=df, window_size=10, episode_length=2000, random_start=True)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))


def test_no_trade_action_keeps_position_and_gives_finite_reward():
    df = make_df(60)
    env = TradingEnvironment(df=df, window_size=10, episode_length=2000, random_start=True)
    env.reset()
    obs, reward, done, truncated, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert env.btc_held == 0
    assert np.isfinite(reward)
    assert done in (True, False)


def test_open_long_and_update_sl_tp_then_close():
    # 構造易於觸發止盈的資料（高點較高）
    df = make_df(30, base=100.0)
    df.loc[df.index[15], 'high'] = df.loc[df.index[15], 'close'] * 1.05
    env = TradingEnvironment(df=df, window_size=10, transaction_fee=0.001, leverage=10.0, min_trade_amount=1.0, episode_length=2000, random_start=True)
    env.reset()

    # 開多 50%，止盈 2%，止損 1%
    # 新動作映射：TP/SL ∈ [0,1]，此處選擇中性 0.2（TP=20%，SL=4%）
    obs, reward, done, truncated, info = env.step(np.array([0.5, 0.2, 0.2], dtype=np.float32))
    assert env.btc_held > 0
    assert env.take_profit_price > 0 and env.stop_loss_price > 0

    # 前進數步，觀察是否平倉
    for _ in range(10):
        obs, reward, done, truncated, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        if env.btc_held == 0:
            break
    assert env.btc_held == 0  # 應已觸發止盈或止損


def test_open_short_and_update_sl_tp_then_close():
    # 構造易於觸發空單止盈/止損的資料
    df = make_df(30, base=100.0)
    # 在未來步驟製造更低的 low 以利空單止盈
    df.loc[df.index[15], 'low'] = df.loc[df.index[15], 'close'] * 0.95
    # 也提高某步 high 以利空單止損
    df.loc[df.index[16], 'high'] = df.loc[df.index[16], 'close'] * 1.05

    env = TradingEnvironment(df=df, window_size=10, transaction_fee=0.001, leverage=10.0, min_trade_amount=1.0, episode_length=2000, random_start=True)
    env.reset()

    # 開空 40%，止盈 2%，止損 1%
    obs, reward, done, truncated, info = env.step(np.array([-0.4, 0.2, 0.2], dtype=np.float32))
    assert env.btc_held < 0
    assert env.take_profit_price > 0 and env.stop_loss_price > 0

    # 前進數步，觀察是否平倉
    for _ in range(10):
        obs, reward, done, truncated, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        if env.btc_held == 0:
            break
    assert env.btc_held == 0

def test_margin_fee_and_leverage_effects():
    df = make_df(40)
    env = TradingEnvironment(df=df, window_size=10, initial_balance=1000.0, transaction_fee=0.001, leverage=10.0, episode_length=2000, random_start=True)
    env.reset()
    pre_balance = env.balance
    obs, reward, done, truncated, info = env.step(np.array([0.5, 0.0, 0.0], dtype=np.float32))
    # 應該已扣保證金+手續費，餘額下降
    assert env.balance < pre_balance
    assert env.btc_held != 0


def test_episode_termination_on_balance_threshold():
    df = make_df(25)
    # 設較高的 min_balance 迫使早期終止
    env = TradingEnvironment(df=df, window_size=10, initial_balance=1000.0, min_balance=1200.0, episode_length=2000, random_start=True)
    env.reset()
    _, _, done, _, _ = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert done is True


