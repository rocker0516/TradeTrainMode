from __future__ import annotations

from LiveTradingRunner.live_trading_loop import _parse_args


def test_cli_print_obs_account_context_flag() -> None:
    args = _parse_args(["--print_obs_account_context"])
    assert bool(args.print_obs_account_context) is True


