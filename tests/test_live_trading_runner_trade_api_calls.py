from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

from LiveTradingRunner.live_trading_loop import LiveRunnerConfig, LiveRunnerState
from LiveTradingRunner.runner_core import LiveRunner


@dataclass(frozen=True)
class _FakeLatestBars:
    df_5m: pd.DataFrame
    latest_closed_bar_ts: str
    latest_closed_price: float


class _FakeModel:
    def __init__(self, action_value: float) -> None:
        self._a = float(action_value)

    def predict(self, _obs: Any, deterministic: bool = True):  # noqa: ANN001
        _ = deterministic
        return np.array([self._a], dtype=np.float32), None


@dataclass
class _FakeObsResult:
    obs: dict
    step_idx: int
    current_price: float


def _make_minimal_obs() -> dict:
    # shape 必須能讓 fake model 收到 dict（內容不重要，因為我們 patch model.predict）
    return {
        "price_seq": np.zeros((288, 10), dtype=np.float32),
        "price_seq_1d": np.zeros((14, 5), dtype=np.float32),
        "account_state": np.zeros((31,), dtype=np.float32),
        "time_state": np.zeros((7,), dtype=np.float32),
        "rhythm_state": np.zeros((2,), dtype=np.float32),
        "cost_state": np.zeros((29,), dtype=np.float32),
    }


def _cfg(*, enable_trade_api: bool, dry_run: bool) -> LiveRunnerConfig:
    return LiveRunnerConfig(
        symbol="BTCUSDT",
        model_path="models/sac_lag_BTCUSDT/best_model/best_model.zip",
        leverage=5,
        max_position_pct=0.7,
        poll_interval_sec=10,
        deterministic=True,
        enable_trade_api=bool(enable_trade_api),
        dry_run=bool(dry_run),
        default_min_notional_usdt=10.0,
        feature_symbols=("BTCUSDT",),
        window_size_5m=288,
        window_size_1d=14,
    )


def test_trade_api_disabled_does_not_build_client(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = LiveRunner(_cfg(enable_trade_api=False, dry_run=True))
    state = LiveRunnerState()

    monkeypatch.setattr("LiveTradingRunner.runner_core.fetch_latest_multi_symbol_5m", lambda **_: _FakeLatestBars(  # type: ignore[arg-type]
        df_5m=pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")]}),
        latest_closed_bar_ts="2024-01-01 00:00:00",
        latest_closed_price=100.0,
    ))
    monkeypatch.setattr("LiveTradingRunner.runner_core.load_local_1d_data", lambda: pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")]}) )
    monkeypatch.setattr("LiveTradingRunner.runner_core.LiveObsBuilder.build", lambda *_, **__: _FakeObsResult(obs=_make_minimal_obs(), step_idx=0, current_price=100.0))
    monkeypatch.setattr("LiveTradingRunner.runner_core.LiveRunner._load_model", lambda *_a, **_kw: _FakeModel(0.1))

    build_called = {"n": 0}

    def _fake_build_trading_client(*_a, **_kw):  # noqa: ANN001
        build_called["n"] += 1
        raise AssertionError("should not be called when trade api disabled")

    monkeypatch.setattr("LiveTradingRunner.runner_core.build_trading_client", _fake_build_trading_client)

    decision = runner.run_once(state=state)
    assert decision is not None
    assert build_called["n"] == 0
    assert decision.equity_usdt is None
    # live runner 會輸出 action_raw / action_clipped / target_pos_pct / final_pos_pct
    assert isinstance(float(decision.action_raw), float)
    assert isinstance(float(decision.action_clipped), float)
    assert isinstance(float(decision.target_pos_pct), float)
    assert isinstance(float(decision.final_pos_pct), float)


def test_trade_api_enabled_calls_methods_and_respects_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = LiveRunner(_cfg(enable_trade_api=True, dry_run=True))
    state = LiveRunnerState()

    monkeypatch.setattr("LiveTradingRunner.runner_core.fetch_latest_multi_symbol_5m", lambda **_: _FakeLatestBars(  # type: ignore[arg-type]
        df_5m=pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")]}),
        latest_closed_bar_ts="2024-01-01 00:00:00",
        latest_closed_price=100.0,
    ))
    monkeypatch.setattr("LiveTradingRunner.runner_core.load_local_1d_data", lambda: pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")]}) )
    monkeypatch.setattr("LiveTradingRunner.runner_core.LiveObsBuilder.build", lambda *_, **__: _FakeObsResult(obs=_make_minimal_obs(), step_idx=0, current_price=100.0))
    monkeypatch.setattr("LiveTradingRunner.runner_core.LiveRunner._load_model", lambda *_a, **_kw: _FakeModel(0.2))

    class _Filters:
        step_size = 0.001
        min_notional = 10.0

    class _FakeTradeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Any]] = []

        def set_leverage(self, symbol: str, leverage: int) -> None:
            self.calls.append(("set_leverage", (symbol, leverage)))

        def get_equity_usdt(self) -> float:
            self.calls.append(("get_equity_usdt", None))
            return 1000.0

        def get_current_position_size(self, symbol: str) -> float:
            self.calls.append(("get_current_position_size", symbol))
            return 0.0

        def get_symbol_filters(self, symbol: str, default_min_notional: float):  # noqa: ANN001
            self.calls.append(("get_symbol_filters", (symbol, default_min_notional)))
            return _Filters()

        def place_delta_order(self, **kwargs):  # noqa: ANN001
            self.calls.append(("place_delta_order", dict(kwargs)))
            return None

    tc = _FakeTradeClient()
    monkeypatch.setattr("LiveTradingRunner.runner_core.build_trading_client", lambda **_: tc)

    decision = runner.run_once(state=state)
    assert decision is not None

    names = [c[0] for c in tc.calls]
    assert "set_leverage" in names
    assert "get_equity_usdt" in names
    assert "get_current_position_size" in names
    assert "get_symbol_filters" in names
    assert "place_delta_order" in names

    # dry_run 旗標必須傳進去
    last = [c for c in tc.calls if c[0] == "place_delta_order"][-1][1]
    assert last["symbol"] == "BTCUSDT"
    assert last["dry_run"] is True


