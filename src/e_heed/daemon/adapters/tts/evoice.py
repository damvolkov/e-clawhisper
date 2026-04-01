"""EVoice TTS adapter — WebSocket streaming via e-voice /v1/audio/speech.

Protocol:
  1. Connect to ws://host:port/v1/audio/speech (persistent connection)
  2. Send JSON: {"input": "text", "voice": "af_heart", "speed": 1.0, "lang": "en-us"}
  3. Receive JSON events:
     - {"type": "speech.audio.delta", "audio": "<base64_pcm16_24khz>"}
     - {"type": "speech.audio.done"}
  4. Connection reusable for multiple syntheses
"""

from __future__ import annotations

import base64
import contextlib
from collections.abc import AsyncIterator

import orjson
import websockets
from websockets.asyncio.client import ClientConnection

from e_heed.daemon.adapters.tts.base import TTSAdapter
from e_heed.shared.logger import logger
from e_heed.shared.settings import EVoiceTTSConfig


class EVoiceTTSAdapter(TTSAdapter):
    """Streaming TTS via e-voice WebSocket — text in, PCM16 audio chunks out."""

    __slots__ = ("_base_url", "_voice", "_speed", "_lang", "_sample_rate", "_ws", "_stopped")

    def __init__(self, config: EVoiceTTSConfig) -> None:
        self._base_url = str(config.url).rstrip("/")
        self._voice = config.voice
        self._speed = config.speed
        self._lang = config.lang
        self._sample_rate = config.sample_rate
        self._ws: ClientConnection | None = None
        self._stopped = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    ##### CONNECTION #####

    async def connect(self) -> None:
        ws_url = self._base_url.replace("http://", "ws://").replace("https://", "wss://")
        self._ws = await websockets.connect(f"{ws_url}/v1/audio/speech", max_size=None)
        logger.system("OK", f"TTS(evoice) connected url={self._base_url}")

    async def disconnect(self) -> None:
        if self._ws:
            with contextlib.suppress(websockets.exceptions.ConnectionClosed):
                await self._ws.close()
            self._ws = None
            logger.system("STOP", "TTS(evoice) disconnected")

    ##### SYNTHESIS #####

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Send text, yield PCM16 int16 audio chunks as they arrive."""
        self._stopped = False

        if self._ws is None:
            await self.connect()

        assert self._ws is not None

        request = orjson.dumps(
            {
                "input": text,
                "voice": self._voice,
                "speed": self._speed,
                "lang": self._lang,
            }
        )
        try:
            await self._ws.send(request.decode())
        except websockets.exceptions.ConnectionClosed:
            await self.connect()
            assert self._ws is not None
            await self._ws.send(request.decode())

        try:
            async for msg in self._ws:
                if self._stopped:
                    break

                data = orjson.loads(msg)
                event_type = data.get("type", "")

                if event_type == "speech.audio.delta":
                    audio_b64 = data.get("audio", "")
                    if audio_b64:
                        yield base64.b64decode(audio_b64)
                elif event_type == "speech.audio.done":
                    break
        except websockets.exceptions.ConnectionClosed:
            logger.warning("TTS(evoice) connection lost during synthesis")
            self._ws = None

    async def stop(self) -> None:
        self._stopped = True
