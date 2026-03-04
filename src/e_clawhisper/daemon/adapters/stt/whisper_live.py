"""WhisperLive STT adapter — persistent WebSocket with per-utterance sessions."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from uuid import uuid4

import numpy as np
import orjson
import websockets
from websockets.asyncio.client import ClientConnection

from e_clawhisper.daemon.core.interfaces.stt import STTBase
from e_clawhisper.shared.logger import LogIcon, logger
from e_clawhisper.shared.settings import WhisperLiveConfig


class WhisperLiveAdapter(STTBase):
    """Streaming STT via WhisperLive WebSocket.

    Connection lifecycle:
        connect()         → initial WS, triggers model load on server
        start_utterance() → reconnect WS for fresh session
        stream_audio()    → send float32 audio bytes
        finish_utterance() → END_OF_AUDIO, collect final transcript
        disconnect()      → full teardown
    """

    __slots__ = (
        "_ws_url",
        "_model",
        "_language",
        "_ws",
        "_uid",
        "_transcript_queue",
        "_recv_task",
        "_final_text",
        "_ready",
    )

    def __init__(self, config: WhisperLiveConfig) -> None:
        self._ws_url = f"ws://{config.host}:{config.port}"
        self._model = config.model
        self._language = config.language
        self._ws: ClientConnection | None = None
        self._uid: str = ""
        self._transcript_queue: asyncio.Queue[str] = asyncio.Queue()
        self._recv_task: asyncio.Task[None] | None = None
        self._final_text: str = ""
        self._ready = False

    ##### CONNECTION #####

    async def connect(self) -> None:
        """Initial connection — warms up STT model on server."""
        await self._open_session()
        logger.info("stt_connected (model warm-up) url=%s", self._ws_url, icon=LogIcon.STT)

    async def disconnect(self) -> None:
        await self._close_session()
        logger.info("stt_disconnected", icon=LogIcon.STT)

    async def start_utterance(self) -> None:
        """Reconnect for a new utterance session (fast, model already loaded)."""
        await self._close_session()
        await self._open_session()
        logger.debug("stt_utterance_started uid=%s", self._uid[:8], icon=LogIcon.STT)

    ##### STREAMING #####

    async def stream_audio(self, audio_chunk: bytes) -> None:
        if self._ws and self._ready:
            await self._ws.send(audio_chunk)

    async def finish_utterance(self) -> str:
        """Send END_OF_AUDIO and return accumulated transcript."""
        if self._ws:
            with contextlib.suppress(websockets.exceptions.ConnectionClosed):
                await self._ws.send('{"type": "END_OF_AUDIO"}')
            if self._recv_task:
                with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(self._recv_task, timeout=5.0)
                self._recv_task = None

        result = self._final_text
        logger.debug("stt_final: %s", result[:120] if result else "(empty)", icon=LogIcon.STT)
        return result

    ##### TRANSCRIPTION #####

    async def _receive_loop(self) -> None:
        """Background task: receive transcript segments from WhisperLive."""
        assert self._ws is not None
        try:
            async for msg in self._ws:
                data = orjson.loads(msg)
                if "segments" in data:
                    text = " ".join(seg_text for seg in data["segments"] if (seg_text := seg.get("text", "").strip()))
                    if text:
                        self._final_text = text
                        with contextlib.suppress(asyncio.QueueFull):
                            self._transcript_queue.put_nowait(text)
                if data.get("eos"):
                    break
        except websockets.exceptions.ConnectionClosed:
            pass

    async def transcript_stream(self) -> AsyncIterator[str]:
        while True:
            try:
                yield await asyncio.wait_for(self._transcript_queue.get(), timeout=0.1)
            except TimeoutError:
                if self._ws is None:
                    break

    ##### SESSION MANAGEMENT #####

    async def _open_session(self) -> None:
        self._uid = str(uuid4())
        self._final_text = ""
        self._ready = False

        while not self._transcript_queue.empty():
            self._transcript_queue.get_nowait()

        self._ws = await websockets.connect(self._ws_url, max_size=None)

        config_msg = orjson.dumps(
            {
                "uid": self._uid,
                "language": self._language,
                "task": "transcribe",
                "model": self._model,
                "use_vad": False,
            }
        )
        await self._ws.send(config_msg)

        ready_msg = await self._ws.recv()
        ready = orjson.loads(ready_msg)
        if ready.get("message") != "SERVER_READY":
            logger.warning("whisperlive unexpected ready: %s", ready, icon=LogIcon.STT)

        self._ready = True
        self._recv_task = asyncio.create_task(self._receive_loop())

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

    @staticmethod
    def audio_to_float32(audio_int16: np.ndarray) -> bytes:
        """Convert int16 audio to float32 bytes for WhisperLive."""
        return (audio_int16.astype(np.float32) / 32768.0).tobytes()
