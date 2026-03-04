"""Audio I/O via sounddevice — async-compatible mic capture and speaker playback."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    pass


class AudioDevice:
    """Wraps sounddevice for async mic input and speaker output."""

    def __init__(self, sample_rate: int, channels: int, chunk_size: int) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._input_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=100)
        self._stream: sd.InputStream | None = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        if status:
            pass
        audio_int16 = (indata[:, 0] * 32767).astype(np.int16) if indata.dtype != np.int16 else indata[:, 0].copy()
        try:
            self._input_queue.put_nowait(audio_int16)
        except asyncio.QueueFull:
            pass

    async def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            blocksize=self._chunk_size,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    async def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def read_chunk(self) -> np.ndarray:
        """Await next audio chunk from microphone."""
        return await self._input_queue.get()

    def play_audio(self, audio_data: np.ndarray, sample_rate: int | None = None) -> None:
        """Play audio through speakers (blocking)."""
        sr = sample_rate or self._sample_rate
        sd.play(audio_data, samplerate=sr)

    def stop_playback(self) -> None:
        sd.stop()
