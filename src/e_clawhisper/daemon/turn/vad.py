"""End-of-speech detector — Silero VAD with silence-duration tracking.

Same Silero ONNX model as sentinel but configured for turn-pipeline:
tracks when user starts speaking and when they go silent long enough
to consider the utterance complete.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from e_clawhisper.daemon.sentinel.vad import SileroVAD
from e_clawhisper.shared.settings import VADConfig

_SAMPLE_RATE = 16000
_WINDOW_SIZE = 512


@dataclass(slots=True)
class EndOfSpeechResult:
    """Result of processing one audio chunk."""

    is_speech: bool
    should_stop: bool
    probability: float


class EndOfSpeechDetector:
    """Wraps SileroVAD with silence-duration tracking for end-of-utterance."""

    __slots__ = (
        "_vad",
        "_has_speech",
        "_silence_frames",
        "_max_silence_frames",
        "_min_recording_time",
        "_recording_start",
    )

    def __init__(self, config: VADConfig, sample_rate: int = _SAMPLE_RATE) -> None:
        self._vad = SileroVAD(threshold=config.threshold)
        self._has_speech = False
        self._silence_frames = 0
        self._min_recording_time = config.min_recording_time
        self._recording_start: float = 0.0
        self._max_silence_frames = int(config.silence_duration * sample_rate / _WINDOW_SIZE)

    def process(self, audio: np.ndarray) -> EndOfSpeechResult:
        """Process 512-sample chunk → (is_speech, should_stop, probability)."""
        prob = self._vad(audio)
        is_speech = prob >= self._vad.threshold

        if is_speech:
            if not self._has_speech:
                self._recording_start = time.monotonic()
            self._has_speech = True
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        elapsed = time.monotonic() - self._recording_start if self._recording_start > 0 else 0.0
        should_stop = (
            self._has_speech
            and self._silence_frames >= self._max_silence_frames
            and elapsed >= self._min_recording_time
        )

        return EndOfSpeechResult(is_speech=is_speech, should_stop=should_stop, probability=prob)

    def reset(self) -> None:
        self._vad.reset()
        self._has_speech = False
        self._silence_frames = 0
        self._recording_start = 0.0
