"""STT adapter for WhisperLive WebSocket API."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import orjson
import websockets
from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer

from e_clawhisper.core.logger import LogIcon, logger
from e_clawhisper.core.settings import settings as st


class WhisperLiveSTT(stt.STT):
    """STT via WhisperLive WebSocket — batch mode per utterance."""

    def __init__(
        self,
        *,
        ws_url: str = st.stt_ws_url,
        model: str = st.STT_MODEL,
        language: str = st.STT_LANGUAGE,
        timeout: float = st.STT_TIMEOUT,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._ws_url = ws_url
        self._model = model
        self._language = language
        self._timeout = timeout

    @property
    def model(self) -> str:
        return f"whisperlive/{self._model}"

    @property
    def provider(self) -> str:
        return "whisperlive"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Send accumulated audio to WhisperLive, receive transcript."""
        effective_lang: str = language if isinstance(language, str) else self._language

        combined = rtc.combine_audio_frames(buffer)
        audio_int16 = np.frombuffer(combined.data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        text = await self._transcribe(audio_float32, effective_lang)

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=effective_lang, text=text)],
        )

    async def _transcribe(self, audio: np.ndarray, language: str) -> str:
        """Open WS to WhisperLive, stream audio, collect final transcript."""
        uid = str(uuid4())
        config = orjson.dumps({
            "uid": uid,
            "language": language,
            "task": "transcribe",
            "model": self._model,
            "use_vad": False,
        })

        try:
            async with websockets.connect(self._ws_url, max_size=None) as ws:
                await ws.send(config)

                ready_msg = await ws.recv()
                ready = orjson.loads(ready_msg)
                if ready.get("status") != "READY":
                    logger.warning("whisperlive not ready: %s", ready, icon=LogIcon.STT)

                chunk_size = 4096
                for i in range(0, len(audio), chunk_size):
                    await ws.send(audio[i : i + chunk_size].tobytes())

                await ws.send(orjson.dumps({"type": "END_OF_AUDIO"}).decode())

                segments: list[str] = []
                try:
                    async for msg in ws:
                        data = orjson.loads(msg)
                        if "segments" in data:
                            segments = [
                                seg_text
                                for seg in data["segments"]
                                if (seg_text := seg.get("text", "").strip())
                            ]
                        if data.get("eos"):
                            break
                except websockets.exceptions.ConnectionClosed:
                    pass

                text = " ".join(segments)
                logger.debug("stt_result: %s", text[:120], icon=LogIcon.STT)
                return text

        except Exception as exc:
            logger.error("stt_error: %s", exc, icon=LogIcon.ERROR)
            return ""

    async def aclose(self) -> None:
        pass
