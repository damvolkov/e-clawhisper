"""Audio I/O adapter — async mic capture and streaming speaker playback."""

from __future__ import annotations

import asyncio
import contextlib

import numpy as np
import sounddevice as sd

from e_clawhisper.shared.settings import AudioConfig


class AudioAdapter:
    """Bridges PortAudio thread with asyncio for mic input and PCM output."""

    __slots__ = (
        "_sample_rate",
        "_channels",
        "_chunk_size",
        "_queue",
        "_stream",
        "_loop",
        "_playback_stopped",
    )

    def __init__(self, config: AudioConfig) -> None:
        self._sample_rate = config.sample_rate
        self._channels = config.channels
        self._chunk_size = config.chunk_size
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=config.queue_size)
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._playback_stopped = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def queue(self) -> asyncio.Queue[np.ndarray]:
        return self._queue

    ##### CAPTURE #####

    def _enqueue(self, data: np.ndarray) -> None:
        if not self._queue.full():
            self._queue.put_nowait(data)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        audio_f32 = indata[:, 0].astype(np.float32) if indata.dtype != np.float32 else indata[:, 0].copy()
        if self._loop is not None:
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._enqueue, audio_f32)

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
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
        self._loop = None

    ##### PLAYBACK #####

    async def play(self, pcm_queue: asyncio.Queue[bytes | None], sample_rate: int) -> float:
        """Consume PCM int16 chunks from queue → speaker. Returns seconds played."""
        self._playback_stopped = False
        total_samples = 0

        out = sd.RawOutputStream(samplerate=sample_rate, channels=1, dtype="int16")
        out.start()
        try:
            while (pcm_chunk := await pcm_queue.get()) is not None:
                if self._playback_stopped:
                    break
                await asyncio.to_thread(out.write, pcm_chunk)
                total_samples += len(pcm_chunk) // 2
        finally:
            with contextlib.suppress(sd.PortAudioError):
                out.stop()
                out.close()

        return total_samples / sample_rate

    def drain(self) -> None:
        """Discard all queued audio frames."""
        while not self._queue.empty():
            self._queue.get_nowait()

    def stop_playback(self) -> None:
        self._playback_stopped = True
        sd.stop()
