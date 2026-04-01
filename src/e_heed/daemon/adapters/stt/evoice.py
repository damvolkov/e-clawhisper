"""EVoice STT adapter — WebSocket streaming via e-voice /v1/audio/transcriptions.

Protocol:
  1. Connect to ws://host:port/v1/audio/transcriptions?language=X&segmentation=true
  2. Send binary PCM16-LE frames (16kHz mono, any chunk size)
  3. Receive JSON events: transcript_update, transcript_final, segment_end, session_end
  4. Send text frame 'END_OF_AUDIO' to flush and close
"""

from __future__ import annotations

import asyncio
import contextlib

import orjson
import websockets
from websockets.asyncio.client import ClientConnection

from e_heed.daemon.adapters.stt.base import STTAdapter
from e_heed.shared.logger import logger
from e_heed.shared.settings import EVoiceSTTConfig

_FLUSH_TIMEOUT = 5.0


class EVoiceSTTAdapter(STTAdapter):
    """Streaming STT via e-voice WebSocket — binary PCM16 in, transcript events out."""

    __slots__ = (
        "_base_url",
        "_language",
        "_response_format",
        "_ws",
        "_recv_task",
        "_final_text",
        "_ready",
    )

    def __init__(self, config: EVoiceSTTConfig) -> None:
        self._base_url = str(config.url).rstrip("/")
        self._language = config.language
        self._response_format = config.response_format
        self._ws: ClientConnection | None = None
        self._recv_task: asyncio.Task[None] | None = None
        self._final_text: str = ""
        self._ready = False

    ##### CONNECTION #####

    async def connect(self) -> None:
        await self._open_session()
        logger.system("OK", f"STT(evoice) connected url={self._base_url}")

    async def disconnect(self) -> None:
        await self._close_session()
        logger.system("STOP", "STT(evoice) disconnected")

    async def start_utterance(self) -> None:
        await self._close_session()
        await self._open_session()

    ##### STREAMING #####

    async def stream(self, audio_chunk: bytes) -> None:
        if self._ws and self._ready:
            await self._ws.send(audio_chunk)

    async def finish_utterance(self) -> str:
        if self._ws:
            with contextlib.suppress(websockets.exceptions.ConnectionClosed):
                await self._ws.send("END_OF_AUDIO")
            if self._recv_task:
                with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(self._recv_task, timeout=_FLUSH_TIMEOUT)
                self._recv_task = None

        return self._final_text

    ##### RECEIVE #####

    async def _ev_receive_loop(self) -> None:
        if self._ws is None:
            return
        try:
            async for msg in self._ws:
                data = orjson.loads(msg)
                event_type = data.get("type", "")
                text = data.get("text", "")

                if event_type == "transcript_update" and text or event_type == "transcript_final" and text:
                    self._final_text = text
                elif event_type in ("segment_end", "session_end"):
                    if text:
                        self._final_text = text
                    break
        except websockets.exceptions.ConnectionClosed:
            logger.warning("STT(evoice) connection lost mid-utterance")
        except Exception as exc:
            logger.error(f"STT(evoice) receive error: {exc}")

    ##### SESSION #####

    async def _open_session(self) -> None:
        self._final_text = ""
        self._ready = False

        params = f"language={self._language}&response_format={self._response_format}&segmentation=true"
        ws_url = self._base_url.replace("http://", "ws://").replace("https://", "wss://")
        url = f"{ws_url}/v1/audio/transcriptions?{params}"

        self._ws = await websockets.connect(url, max_size=None)
        self._ready = True
        self._recv_task = asyncio.create_task(self._ev_receive_loop())

    async def _close_session(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task
            self._recv_task = None
        if self._ws:
            with contextlib.suppress(websockets.exceptions.ConnectionClosed):
                await self._ws.close()
            self._ws = None
        self._ready = False
