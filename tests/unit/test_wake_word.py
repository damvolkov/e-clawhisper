"""Tests for wake-word detection logic in ClawWhisperAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from e_clawhisper.sessions.voice import ClawWhisperAgent


@pytest.fixture
def agent(mock_channel: AsyncMock) -> ClawWhisperAgent:
    return ClawWhisperAgent(
        channel=mock_channel,
        wake_word="damien",
        conversation_timeout=30.0,
    )


##### STRIP WAKE WORD #####


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hey damien what's up", "hey what's up"),
        ("damien, tell me a joke", "tell me a joke"),
        ("hello damien", "hello"),
        ("Damien", ""),
        ("no wake word here", "no wake word here"),
        ("DAMIEN how are you", "how are you"),
    ],
    ids=["middle", "start-comma", "end", "only-wake", "absent", "uppercase"],
)
def test_strip_wake_word(agent: ClawWhisperAgent, text: str, expected: str) -> None:
    assert agent._strip_wake_word(text) == expected


##### ACTIVATION STATE #####


def test_agent_starts_inactive(agent: ClawWhisperAgent) -> None:
    assert not agent.is_active


def test_agent_wake_word_stored_lowercase(agent: ClawWhisperAgent) -> None:
    assert agent._wake_word == "damien"
