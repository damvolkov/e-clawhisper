"""TEN VAD processor — low-latency voice activity detection."""

from __future__ import annotations

import time

import numpy as np
from ten_vad import TenVad

from e_clawhisper.daemon.core.interfaces.processor import ProcessorBase
from e_clawhisper.daemon.core.models import VADResult
from e_clawhisper.shared.logger import LogIcon, logger
from e_clawhisper.shared.settings import VADConfig

_EMPTY_INT16 = np.array([], dtype=np.int16)


class TenVADProcessor(ProcessorBase[VADResult]):
    """TEN VAD wrapper with silence-duration tracking.

    Accepts arbitrary chunk sizes and internally splits into
    hop_size=256 frames required by TenVad. Tracks recording
    start time to enforce min_recording_time before allowing
    should_stop=True.
    """

    __slots__ = (
        "_vad",
        "_threshold",
        "_hop_size",
        "_has_speech",
        "_silence_frames",
        "_max_silence_frames",
        "_min_recording_time",
        "_recording_start",
        "_remainder",
    )

    def __init__(self, config: VADConfig, sample_rate: int = 16000) -> None:
        hop_size = 256
        self._vad = TenVad(hop_size=hop_size, threshold=config.threshold)
        self._threshold = config.threshold
        self._hop_size = hop_size
        self._has_speech = False
        self._silence_frames = 0
        self._min_recording_time = config.min_recording_time
        self._recording_start: float = 0.0
        self._remainder: np.ndarray = _EMPTY_INT16
        self._max_silence_frames = int(config.silence_duration * sample_rate / hop_size)
        logger.debug(
            "ten_vad_init threshold=%.2f max_silence_frames=%d min_rec=%.1fs",
            config.threshold,
            self._max_silence_frames,
            self._min_recording_time,
            icon=LogIcon.VAD,
        )

    def process(self, audio_chunk: np.ndarray) -> VADResult:
        """Process arbitrary-length audio, splitting into hop_size frames."""
        buf = np.concatenate([self._remainder, audio_chunk]) if len(self._remainder) > 0 else audio_chunk

        peak_probability = 0.0
        any_speech = False
        offset = 0

        while offset + self._hop_size <= len(buf):
            frame = buf[offset : offset + self._hop_size]
            probability, flag = self._vad.process(frame)

            if flag == 1:
                any_speech = True
                if not self._has_speech:
                    self._recording_start = time.monotonic()
                self._has_speech = True
                self._silence_frames = 0
            else:
                self._silence_frames += 1

            peak_probability = max(peak_probability, float(probability))
            offset += self._hop_size

        self._remainder = buf[offset:] if offset < len(buf) else _EMPTY_INT16

        elapsed = time.monotonic() - self._recording_start if self._recording_start > 0 else 0.0
        should_stop = (
            self._has_speech
            and self._silence_frames >= self._max_silence_frames
            and elapsed >= self._min_recording_time
        )

        return VADResult(is_speech=any_speech, should_stop=should_stop, probability=peak_probability)

    @property
    def has_speech(self) -> bool:
        return self._has_speech

    def reset(self) -> None:
        self._has_speech = False
        self._silence_frames = 0
        self._recording_start = 0.0
        self._remainder = _EMPTY_INT16
