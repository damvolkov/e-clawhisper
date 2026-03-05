"""Piper TTS adapter — Wyoming protocol over TCP."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from wyoming.audio import AudioChunk, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize

from e_clawhisper.shared.logger import logger
from e_clawhisper.shared.settings import PiperConfig


class TTSAdapter:
    """TTS via Piper Wyoming protocol."""

    __slots__ = ("_host", "_port", "_sample_rate", "_stopped")

    def __init__(self, config: PiperConfig) -> None:
        self._host = config.host
        self._port = config.port
        self._sample_rate = config.sample_rate
        self._stopped = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Connect to Piper, send text, yield PCM audio chunks."""
        self._stopped = False
        client = AsyncTcpClient(self._host, self._port)
        await client.connect()

        try:
            await client.write_event(Synthesize(text=text).event())
            logger.turn("TTS", f"synthesizing: {logger.truncate(text)}")

            while not self._stopped:
                if (event := await client.read_event()) is None:
                    break
                if AudioChunk.is_type(event.type):
                    yield AudioChunk.from_event(event).audio
                elif AudioStop.is_type(event.type):
                    break
        finally:
            with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
                await asyncio.wait_for(client.disconnect(), timeout=0.5)

    async def stop(self) -> None:
        self._stopped = True
