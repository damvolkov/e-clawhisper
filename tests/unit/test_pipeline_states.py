"""Tests for pipeline state machine."""

from __future__ import annotations

import time

from e_clawhisper.daemon.core.processors.turn_manager import TurnManager
from e_clawhisper.daemon.pipeline.states import ConversationMode, PipelineState

##### INITIAL STATE #####


def test_turn_manager_initial_state() -> None:
    tm = TurnManager()
    assert tm.state == PipelineState.IDLE
    assert tm.mode == ConversationMode.IDLE
    assert not tm.is_active


##### ACTIVATION #####


def test_turn_manager_activate_deactivate() -> None:
    tm = TurnManager()
    tm.activate()
    assert tm.is_active
    assert tm.mode == ConversationMode.ACTIVE

    tm.deactivate()
    assert not tm.is_active
    assert tm.mode == ConversationMode.IDLE
    assert tm.state == PipelineState.IDLE


##### BARGE-IN #####


def test_barge_in_during_speaking() -> None:
    tm = TurnManager()
    tm.state = PipelineState.SPEAKING
    assert tm.should_barge_in(is_speech=True)
    assert tm.state == PipelineState.STREAMING


def test_no_barge_in_during_idle() -> None:
    tm = TurnManager()
    tm.state = PipelineState.IDLE
    assert not tm.should_barge_in(is_speech=True)


##### TIMEOUT #####


def test_timeout_when_inactive() -> None:
    tm = TurnManager(conversation_timeout=0.0)
    assert not tm.check_timeout()


def test_timeout_deactivates() -> None:
    tm = TurnManager(conversation_timeout=0.01)
    tm.activate()
    time.sleep(0.02)

    assert tm.check_timeout()
    assert not tm.is_active
    assert tm.state == PipelineState.IDLE
