"""Audio I/O adapter — async mic capture and callback-based speaker playback."""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque

import numpy as np
import sounddevice as sd

from e_clawhisper.shared.logger import logger
from e_clawhisper.shared.settings import AudioConfig

_MAX_RETRIES = 30
_BASE_DELAY = 2.0
_MAX_DELAY = 30.0


class AudioAdapter:
    """Bridges PortAudio thread with asyncio for mic input and PCM output."""

    __slots__ = (
        "_sample_rate",
        "_channels",
        "_chunk_size",
        "_playback_latency",
        "_queue",
        "_stream",
        "_loop",
        "_playback_stopped",
    )

    def __init__(self, config: AudioConfig) -> None:
        self._sample_rate = config.sample_rate
        self._channels = config.channels
        self._chunk_size = config.chunk_size
        self._playback_latency = config.playback_latency
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
        for attempt in range(_MAX_RETRIES):
            try:
                self._stream = sd.InputStream(
                    samplerate=self._sample_rate,
                    channels=self._channels,
                    blocksize=self._chunk_size,
                    dtype="float32",
                    callback=self._audio_callback,
                )
                self._stream.start()
                return
            except sd.PortAudioError as exc:
                delay = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
                logger.system("WARN", f"audio device unavailable ({exc}), retry {attempt + 1}/{_MAX_RETRIES} in {delay:.0f}s")
                await asyncio.sleep(delay)
        msg = f"audio device not available after {_MAX_RETRIES} retries"
        raise sd.PortAudioError(msg)

    async def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._loop = None

    ##### PLAYBACK #####

    async def play(self, pcm_queue: asyncio.Queue[bytes | None], sample_rate: int) -> float:
        """Consume PCM int16 chunks from queue → callback-based speaker. Returns seconds played.

        Uses PortAudio callback API: the callback pulls from a deque and
        writes silence when empty — eliminates xrun and thread-safety issues.
        """
        self._playback_stopped = False
        total_samples = 0

        # Thread-safe buffer: asyncio appends, PortAudio callback consumes
        chunks: deque[bytes] = deque()
        offset = [0]
        silence = bytes(sample_rate * 2)  # 1s pre-allocated int16 silence

        def _fill_output(
            outdata: memoryview,
            frames: int,
            _time: object,
            _status: sd.CallbackFlags,
        ) -> None:
            needed = frames * 2  # int16 mono = 2 bytes/frame
            pos = 0
            while pos < needed and chunks:
                chunk = chunks[0]
                available = len(chunk) - offset[0]
                n = min(available, needed - pos)
                outdata[pos : pos + n] = chunk[offset[0] : offset[0] + n]
                pos += n
                offset[0] += n
                if offset[0] >= len(chunk):
                    chunks.popleft()
                    offset[0] = 0
            if pos < needed:
                outdata[pos:needed] = silence[: needed - pos]

        out = sd.RawOutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            latency=self._playback_latency,
            callback=_fill_output,
        )
        out.start()
        try:
            while not self._playback_stopped:
                pcm_chunk = await pcm_queue.get()
                if pcm_chunk is None:
                    break
                chunks.append(pcm_chunk)
                total_samples += len(pcm_chunk) // 2
            # Drain: wait for callback to consume remaining chunks
            while chunks and not self._playback_stopped:
                await asyncio.sleep(0.05)
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
