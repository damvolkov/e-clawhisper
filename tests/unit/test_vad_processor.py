"""Tests for EndOfSpeechDetector — silence tracking, should_stop logic."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import numpy as np
import pytest

from e_clawhisper.daemon.turn.vad import EndOfSpeechDetector, EndOfSpeechResult
from e_clawhisper.shared.settings import VADConfig

_WINDOW = 512
_SR = 16000


##### STUB #####


class _StubVAD:
    """Minimal SileroVAD stub — returns probabilities from iterator."""

    __slots__ = ("_threshold", "_probs")

    def __init__(self, threshold: float, probs: Iterator[float]) -> None:
        self._threshold = threshold
        self._probs = probs

    @property
    def threshold(self) -> float:
        return self._threshold

    def __call__(self, audio: np.ndarray) -> float:
        return next(self._probs, 0.0)

    def reset(self) -> None:
        pass


##### FIXTURES #####


def _make_config(
    threshold: float = 0.5,
    silence_duration: float = 1.5,
    min_recording_time: float = 1.0,
) -> VADConfig:
    return VADConfig(
        threshold=threshold,
        silence_duration=silence_duration,
        min_recording_time=min_recording_time,
    )


def _silence() -> np.ndarray:
    return np.zeros(_WINDOW, dtype=np.float32)


def _speech() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(-1, 1, _WINDOW).astype(np.float32)


def _make_detector(
    config: VADConfig | None = None,
    probabilities: list[float] | None = None,
) -> EndOfSpeechDetector:
    """Create detector with stub SileroVAD."""
    cfg = config or _make_config()
    probs = iter(probabilities) if probabilities else iter([])

    with patch("e_clawhisper.daemon.turn.vad.SileroVAD"):
        detector = EndOfSpeechDetector(cfg, sample_rate=_SR)
        detector._vad = _StubVAD(threshold=cfg.threshold, probs=probs)
        return detector


##### SPEECH DETECTION #####


def test_speech_detected_sets_has_speech() -> None:
    detector = _make_detector(probabilities=[0.9])
    result = detector.process(_speech())
    assert result.is_speech
    assert detector._has_speech


def test_silence_not_detected_as_speech() -> None:
    detector = _make_detector(probabilities=[0.1])
    result = detector.process(_silence())
    assert not result.is_speech
    assert not detector._has_speech


def test_result_type() -> None:
    detector = _make_detector(probabilities=[0.8])
    result = detector.process(_speech())
    assert isinstance(result, EndOfSpeechResult)
    assert result.probability == pytest.approx(0.8)


##### SILENCE COUNTING #####


def test_silence_frames_increment() -> None:
    detector = _make_detector(probabilities=[0.1, 0.1, 0.1])
    for _ in range(3):
        detector.process(_silence())
    assert detector._silence_frames == 3


def test_speech_resets_silence_count() -> None:
    detector = _make_detector(probabilities=[0.1, 0.1, 0.9])
    for _ in range(3):
        detector.process(_silence())
    assert detector._silence_frames == 0


##### SHOULD STOP #####


def test_should_stop_requires_speech_then_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    max_sil = 3
    cfg = _make_config(silence_duration=max_sil * _WINDOW / _SR, min_recording_time=0.001)
    probs = [0.9] + [0.1] * max_sil
    detector = _make_detector(config=cfg, probabilities=probs)

    monkeypatch.setattr("e_clawhisper.daemon.turn.vad.time.monotonic", lambda: 100.0)
    detector.process(_speech())

    monkeypatch.setattr("e_clawhisper.daemon.turn.vad.time.monotonic", lambda: 101.0)
    result = EndOfSpeechResult(is_speech=False, should_stop=False, probability=0.0)
    for _ in range(max_sil):
        result = detector.process(_silence())
    assert result.should_stop


def test_should_stop_false_without_prior_speech() -> None:
    cfg = _make_config(silence_duration=0.001, min_recording_time=0.001)
    detector = _make_detector(config=cfg, probabilities=[0.1, 0.1])
    result = EndOfSpeechResult(is_speech=False, should_stop=False, probability=0.0)
    for _ in range(2):
        result = detector.process(_silence())
    assert not result.should_stop


def test_min_recording_time_blocks_early_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config(silence_duration=0.032, min_recording_time=2.0)
    detector = _make_detector(config=cfg, probabilities=[0.9, 0.1])

    monkeypatch.setattr("e_clawhisper.daemon.turn.vad.time.monotonic", lambda: 100.0)
    detector.process(_speech())

    monkeypatch.setattr("e_clawhisper.daemon.turn.vad.time.monotonic", lambda: 100.5)
    result = detector.process(_silence())
    assert not result.should_stop


def test_min_recording_time_allows_stop_after_elapsed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config(silence_duration=0.032, min_recording_time=1.0)
    detector = _make_detector(config=cfg, probabilities=[0.9, 0.1])

    monkeypatch.setattr("e_clawhisper.daemon.turn.vad.time.monotonic", lambda: 100.0)
    detector.process(_speech())

    monkeypatch.setattr("e_clawhisper.daemon.turn.vad.time.monotonic", lambda: 101.5)
    result = detector.process(_silence())
    assert result.should_stop


##### RESET #####


def test_reset_clears_state() -> None:
    detector = _make_detector(probabilities=[0.9])
    detector.process(_speech())
    assert detector._has_speech

    detector.reset()
    assert not detector._has_speech
    assert detector._silence_frames == 0
    assert detector._recording_start == 0.0
