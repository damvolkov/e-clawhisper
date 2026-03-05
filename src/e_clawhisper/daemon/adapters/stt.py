"""WhisperLive STT adapter — persistent WebSocket with per-utterance sessions."""

from __future__ import annotations

import asyncio
import contextlib
from uuid import uuid4

import numpy as np
import orjson
import websockets
from websockets.asyncio.client import ClientConnection

from e_clawhisper.shared.logger import logger
from e_clawhisper.shared.settings import WhisperLiveConfig


class STTAdapter:
    """Streaming STT via WhisperLive WebSocket."""

    __slots__ = (
        "_ws_url",
        "_model",
        "_language",
        "_ws",
        "_uid",
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
        self._recv_task: asyncio.Task[None] | None = None
        self._final_text: str = ""
        self._ready = False

    ##### CONNECTION #####

    async def connect(self) -> None:
        await self._open_session()
        logger.system("OK", f"STT connected url={self._ws_url}")

    async def disconnect(self) -> None:
        await self._close_session()
        logger.system("STOP", "STT disconnected")

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
                await self._ws.send('{"type": "END_OF_AUDIO"}')
            if self._recv_task:
                with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(self._recv_task, timeout=5.0)
                self._recv_task = None

        return self._final_text

    ##### RECEIVE #####

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for msg in self._ws:
                data = orjson.loads(msg)
                if "segments" in data:
                    text = " ".join(
                        seg_text
                        for seg in data["segments"]
                        if (seg_text := seg.get("text", "").strip())
                    )
                    if text:
                        self._final_text = text
                if data.get("eos"):
                    break
        except websockets.exceptions.ConnectionClosed:
            pass

    ##### SESSION #####

    async def _open_session(self) -> None:
        self._uid = str(uuid4())
        self._final_text = ""
        self._ready = False

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
            logger.warning(f"STT unexpected ready: {ready}")

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
        """Convert int16 → float32 bytes for WhisperLive."""
        return (audio_int16.astype(np.float32) / 32768.0).tobytes()
