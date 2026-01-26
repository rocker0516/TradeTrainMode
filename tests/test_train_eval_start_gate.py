from __future__ import annotations

from Train.eval_callback import TrainEvalStartGateConfig, _TrainEpisodeWindowGate


def test_train_eval_start_gate_not_ready_until_window_full() -> None:
    gate = _TrainEpisodeWindowGate(
        config=TrainEvalStartGateConfig(enabled=True, window_size=5, min_max_steps_reached_count=4)
    )

    # 尚未滿 5 回合前，一律不 ready
    for _ in range(4):
        gate.update_from_infos([{"episode": {"r": 0.0, "l": 1}, "terminated": False, "truncated": True, "termination_reason": "max_steps_reached"}])
    assert gate.is_ready() is False


def test_train_eval_start_gate_ready_when_enough_max_steps_in_window() -> None:
    gate = _TrainEpisodeWindowGate(
        config=TrainEvalStartGateConfig(enabled=True, window_size=5, min_max_steps_reached_count=4)
    )

    # 4/5 是 max_steps_reached
    infos = [
        {"episode": {"r": 0.0, "l": 1}, "truncated": True, "termination_reason": "max_steps_reached"},
        {"episode": {"r": 0.0, "l": 1}, "truncated": True, "termination_reason": "max_steps_reached"},
        {"episode": {"r": 0.0, "l": 1}, "truncated": True, "termination_reason": "max_steps_reached"},
        {"episode": {"r": 0.0, "l": 1}, "truncated": True, "termination_reason": "balance_insufficient"},
        {"episode": {"r": 0.0, "l": 1}, "truncated": True, "termination_reason": "max_steps_reached"},
    ]
    for info in infos:
        gate.update_from_infos([info])

    assert gate.is_ready() is True


def test_train_eval_start_gate_sliding_window_updates_counts() -> None:
    gate = _TrainEpisodeWindowGate(
        config=TrainEvalStartGateConfig(enabled=True, window_size=3, min_max_steps_reached_count=2)
    )

    # 前 3 回合：2/3 max_steps => ready
    for tr in ("max_steps_reached", "balance_insufficient", "max_steps_reached"):
        gate.update_from_infos([{"episode": {"r": 0.0, "l": 1}, "truncated": True, "termination_reason": tr}])
    assert gate.is_ready() is True

    # 再加入一個 balance_insufficient，window 變成 [balance, max, balance] => 1/3 不滿足
    gate.update_from_infos([{"episode": {"r": 0.0, "l": 1}, "truncated": True, "termination_reason": "balance_insufficient"}])
    assert gate.is_ready() is False

