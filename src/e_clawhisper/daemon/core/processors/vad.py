"""TEN VAD processor — low-latency voice activity detection."""

from __future__ import annotations

import numpy as np
from ten_vad import TenVad

from e_clawhisper.daemon.core.interfaces.processor import ProcessorBase
from e_clawhisper.daemon.core.models import VADResult
from e_clawhisper.shared.logger import LogIcon, logger
from e_clawhisper.shared.settings import VADConfig


class TenVADProcessor(ProcessorBase):
    """TEN VAD wrapper with silence-duration tracking."""

    __slots__ = ("_vad", "_threshold", "_has_speech", "_silence_frames", "_max_silence_frames", "_hop_size")

    def __init__(self, config: VADConfig, sample_rate: int = 16000) -> None:
        hop_size = 256
        self._vad = TenVad(hop_size=hop_size, threshold=config.threshold)
        self._threshold = config.threshold
        self._hop_size = hop_size
        self._has_speech = False
        self._silence_frames = 0
        self._max_silence_frames = int(config.silence_duration * sample_rate / hop_size)
        logger.debug(
            "ten_vad_init threshold=%.2f silence_frames=%d",
            config.threshold,
            self._max_silence_frames,
            icon=LogIcon.VAD,
        )

    def process(self, audio_chunk: np.ndarray) -> VADResult:
        """Process one hop-sized audio frame, return speech detection result."""
        probability, flag = self._vad.process(audio_chunk)
        is_speech = flag == 1

        if is_speech:
            self._has_speech = True
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        return VADResult(is_speech=is_speech, probability=float(probability))

    @property
    def should_finalize(self) -> bool:
        """True when speech was detected followed by enough silence."""
        return self._has_speech and self._silence_frames >= self._max_silence_frames

    @property
    def has_speech(self) -> bool:
        return self._has_speech

    def reset(self) -> None:
        self._has_speech = False
        self._silence_frames = 0
