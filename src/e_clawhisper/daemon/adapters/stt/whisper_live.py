"""WhisperLive STT adapter — streaming WebSocket transcription."""

from __future__ import annotations

import asyncio
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

    Protocol:
        1. Connect → send config JSON → wait READY
        2. Stream float32 audio chunks
        3. Receive partial transcripts (segments JSON)
        4. Send END_OF_AUDIO → collect final → close
    """

    def __init__(self, config: WhisperLiveConfig) -> None:
        self._ws_url = f"ws://{config.host}:{config.port}"
        self._model = config.model
        self._language = config.language
        self._ws: ClientConnection | None = None
        self._uid: str = ""
        self._transcript_queue: asyncio.Queue[str] = asyncio.Queue()
        self._recv_task: asyncio.Task[None] | None = None
        self._final_text: str = ""

    async def connect(self) -> None:
        self._uid = str(uuid4())
        self._ws = await websockets.connect(self._ws_url, max_size=None)

        config_msg = orjson.dumps({
            "uid": self._uid,
            "language": self._language,
            "task": "transcribe",
            "model": self._model,
            "use_vad": False,
        })
        await self._ws.send(config_msg)

        ready_msg = await self._ws.recv()
        ready = orjson.loads(ready_msg)
        if ready.get("status") != "READY":
            logger.warning("whisperlive unexpected ready: %s", ready, icon=LogIcon.STT)

        self._final_text = ""
        self._recv_task = asyncio.create_task(self._receive_loop())
        logger.debug("whisperlive_connected uid=%s", self._uid[:8], icon=LogIcon.STT)

    async def disconnect(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
            self._recv_task = None
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def stream_audio(self, audio_chunk: bytes) -> None:
        """Send raw float32 audio bytes to WhisperLive."""
        if self._ws:
            await self._ws.send(audio_chunk)

    async def finish_stream(self) -> str:
        """Signal end-of-audio and return accumulated transcript."""
        if self._ws:
            await self._ws.send('{"type": "END_OF_AUDIO"}')
            if self._recv_task:
                try:
                    await asyncio.wait_for(self._recv_task, timeout=5.0)
                except (TimeoutError, asyncio.CancelledError):
                    pass
                self._recv_task = None
            await self._ws.close()
            self._ws = None

        result = self._final_text
        logger.debug("stt_final: %s", result[:120], icon=LogIcon.STT)
        return result

    async def _receive_loop(self) -> None:
        """Background task: receive transcript segments from WhisperLive."""
        assert self._ws is not None
        try:
            async for msg in self._ws:
                data = orjson.loads(msg)
                if "segments" in data:
                    text = " ".join(
                        seg_text for seg in data["segments"] if (seg_text := seg.get("text", "").strip())
                    )
                    if text:
                        self._final_text = text
                        self._transcript_queue.put_nowait(text)
                if data.get("eos"):
                    break
        except websockets.exceptions.ConnectionClosed:
            pass

    async def transcript_stream(self) -> AsyncIterator[str]:
        """Yield partial transcripts as they arrive."""
        while True:
            try:
                text = await asyncio.wait_for(self._transcript_queue.get(), timeout=0.1)
                yield text
            except TimeoutError:
                if self._ws is None:
                    break

    @staticmethod
    def audio_to_float32(audio_int16: np.ndarray) -> bytes:
        """Convert int16 audio to float32 bytes for WhisperLive."""
        return (audio_int16.astype(np.float32) / 32768.0).tobytes()
