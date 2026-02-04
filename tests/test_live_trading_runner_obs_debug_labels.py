from __future__ import annotations

import numpy as np

from LiveTradingRunner.runner_core import _build_account_and_context_obs_named, _print_account_and_context_obs


def test_print_account_and_context_obs_accepts_expected_shapes(capsys):  # type: ignore[no-untyped-def]
    obs = {
        "account_state": np.zeros((16,), dtype=np.float32),
        "cost_state": np.zeros((1,), dtype=np.float32),
    }
    _print_account_and_context_obs(obs)
    out = capsys.readouterr().out
    assert "obs_account_context_named" in out
    assert "account_state" in out
    assert "cost_state" in out


def test_build_account_and_context_obs_named_has_expected_keys() -> None:
    obs = {
        "account_state": np.zeros((16,), dtype=np.float32),
        "cost_state": np.zeros((1,), dtype=np.float32),
    }
    named = _build_account_and_context_obs_named(obs)
    assert set(named.keys()) == {"account_state", "cost_state"}


