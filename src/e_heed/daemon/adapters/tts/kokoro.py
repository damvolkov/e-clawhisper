"""Kokoro TTS adapter — OpenAI-compatible /v1/audio/speech via httpx streaming."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from e_heed.daemon.adapters.tts.base import TTSAdapter
from e_heed.shared.settings import KokoroConfig

_CHUNK_SIZE = 4096


class KokoroAdapter(TTSAdapter):
    """TTS via Kokoro FastAPI (OpenAI-compatible endpoint)."""

    __slots__ = ("_base_url", "_model", "_voice", "_sample_rate", "_response_format", "_timeout", "_stopped")

    def __init__(self, config: KokoroConfig) -> None:
        self._base_url = str(config.url).rstrip("/")
        self._model = config.model
        self._voice = config.voice
        self._sample_rate = config.sample_rate
        self._response_format = config.response_format
        self._timeout = config.timeout
        self._stopped = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yield raw PCM int16 audio chunks for given text."""
        self._stopped = False

        payload = {
            "model": self._model,
            "input": text,
            "voice": self._voice,
            "response_format": self._response_format,
        }

        async with (
            httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client,
            client.stream("POST", "/v1/audio/speech", json=payload) as response,
        ):
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=_CHUNK_SIZE):
                if self._stopped:
                    break
                yield chunk

    async def stop(self) -> None:
        self._stopped = True
