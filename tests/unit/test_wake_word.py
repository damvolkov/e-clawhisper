"""Tests for wake-word detection."""

from __future__ import annotations

import pytest

from e_clawhisper.daemon.core.processors.wake_word import WakeWordDetector


@pytest.fixture
def detector() -> WakeWordDetector:
    return WakeWordDetector(wake_word="damien")


##### DETECTION #####


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("hey damien what's up", True),
        ("hello world", False),
        ("DAMIEN help me", True),
        ("", False),
        ("dame", False),
    ],
    ids=["found-middle", "absent", "uppercase", "empty", "partial-no-match"],
)
def test_wake_word_check(detector: WakeWordDetector, transcript: str, expected: bool) -> None:
    assert detector.check(transcript) == expected


##### STRIP #####


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("hey damien what's up", "hey what's up"),
        ("damien, tell me a joke", "tell me a joke"),
        ("hello damien", "hello"),
        ("Damien", ""),
        ("no wake word here", "no wake word here"),
    ],
    ids=["middle", "start-comma", "end", "only-wake", "absent"],
)
def test_strip_wake_word(detector: WakeWordDetector, transcript: str, expected: str) -> None:
    assert detector.strip(transcript) == expected
