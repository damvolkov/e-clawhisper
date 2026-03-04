"""Audio I/O via sounddevice — async-compatible mic capture and speaker playback."""

from __future__ import annotations

import asyncio
import contextlib

import numpy as np
import sounddevice as sd


class AudioDevice:
    """Wraps sounddevice for async mic input and speaker output.

    Uses call_soon_threadsafe to bridge sounddevice's audio thread
    with asyncio's event loop for correct cross-thread notification.
    """

    __slots__ = ("_sample_rate", "_channels", "_chunk_size", "_input_queue", "_stream", "_loop")

    def __init__(self, sample_rate: int, channels: int, chunk_size: int) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._input_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=100)
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    ##### CAPTURE #####

    def enqueue(self, data: np.ndarray) -> None:
        """Put audio into queue, silently drop if full (runs on event loop thread)."""
        if not self._input_queue.full():
            self._input_queue.put_nowait(data)

    def audio_callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        """Sounddevice callback — converts to int16 and schedules enqueue on event loop."""
        audio_int16 = (indata[:, 0] * 32767).astype(np.int16) if indata.dtype != np.int16 else indata[:, 0].copy()
        if self._loop is not None:
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self.enqueue, audio_int16)

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            blocksize=self._chunk_size,
            dtype="float32",
            callback=self.audio_callback,
        )
        self._stream.start()

    async def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._loop = None

    async def read_chunk(self) -> np.ndarray:
        """Await next audio chunk from microphone."""
        return await self._input_queue.get()

    ##### PLAYBACK #####

    def play_audio(self, audio_data: np.ndarray, sample_rate: int | None = None) -> None:
        sd.play(audio_data, samplerate=sample_rate or self._sample_rate)

    @staticmethod
    def stop_playback() -> None:
        sd.stop()
