"""Piper TTS adapter — Wyoming protocol over TCP."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from wyoming.audio import AudioChunk, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize

from e_clawhisper.daemon.core.interfaces.tts import TTSBase
from e_clawhisper.shared.logger import LogIcon, logger
from e_clawhisper.shared.settings import PiperConfig


class PiperAdapter(TTSBase):
    """TTS via Piper Wyoming protocol."""

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
            logger.debug("tts_synthesize: %s", text[:80], icon=LogIcon.TTS)

            while not self._stopped:
                event = await client.read_event()
                if event is None:
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
