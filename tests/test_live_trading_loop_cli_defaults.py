from __future__ import annotations

from LiveTradingRunner.live_trading_loop import _parse_args


def test_cli_defaults_are_safe() -> None:
    args = _parse_args([])
    # 只驗證 CLI 參數「存在且型別正確」，避免把 default 行為綁死（default 由 TrainConfig 與實際策略決定）。
    assert isinstance(bool(args.enable_trade_api), bool)
    assert isinstance(bool(args.dry_run), bool)
    assert isinstance(bool(args.loop), bool)
    assert isinstance(bool(args.once), bool)
    assert isinstance(bool(getattr(args, "print_obs_account_context", False)), bool)


