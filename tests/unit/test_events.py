"""Tests for inter-pipeline event data contracts."""

from __future__ import annotations

from e_heed.shared.operational.events import Turn, TurnComplete, TurnError, WakewordEvent


def test_wakeword_event_frozen() -> None:
    evt = WakewordEvent(timestamp=1.0, confidence=0.9, energy=0.05)
    assert evt.confidence == 0.9
    assert evt.energy == 0.05


def test_turn_complete_frozen() -> None:
    tc = TurnComplete(turn_id="abc", transcript="hello", response="hi", duration=1.5)
    assert tc.turn_id == "abc"
    assert tc.duration == 1.5


def test_turn_error_frozen() -> None:
    te = TurnError(turn_id="abc", reason="timeout")
    assert te.reason == "timeout"


def test_turn_to_complete() -> None:
    t = Turn(turn_id="test", transcript="hello", response="world")
    tc = t.to_complete()
    assert isinstance(tc, TurnComplete)
    assert tc.transcript == "hello"
    assert tc.response == "world"
    assert tc.duration >= 0.0


def test_turn_to_error() -> None:
    t = Turn(turn_id="test")
    te = t.to_error("failed")
    assert isinstance(te, TurnError)
    assert te.reason == "failed"


def test_turn_generates_unique_id() -> None:
    t1 = Turn()
    t2 = Turn()
    assert t1.turn_id != t2.turn_id


def test_turn_duration_increases() -> None:
    t = Turn()
    d1 = t.duration
    assert d1 >= 0.0
    d2 = t.duration
    assert d2 >= d1
