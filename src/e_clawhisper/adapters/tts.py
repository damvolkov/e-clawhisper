"""TTS adapter for Piper via Wyoming protocol."""

from __future__ import annotations

import asyncio
import contextlib

from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid
from wyoming.audio import AudioChunk, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize

from e_clawhisper.core.settings import settings as st


class PiperChunkedStream(tts.ChunkedStream):
    """Chunked TTS stream from Piper via Wyoming protocol."""

    def __init__(
        self,
        *,
        tts_instance: PiperTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts_instance, input_text=input_text, conn_options=conn_options)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        tts_instance: PiperTTS = self._tts  # type: ignore[assignment]
        client = AsyncTcpClient(tts_instance._host, tts_instance._port)
        await client.connect()

        try:
            await client.write_event(Synthesize(text=self._input_text).event())

            output_emitter.initialize(
                request_id=shortuuid(),
                sample_rate=tts_instance.sample_rate,
                num_channels=tts_instance.num_channels,
                mime_type="audio/pcm",
            )

            while True:
                event = await client.read_event()
                if event is None:
                    break

                if AudioChunk.is_type(event.type):
                    output_emitter.push(AudioChunk.from_event(event).audio)
                elif AudioStop.is_type(event.type):
                    break

            silence_samples = int(tts_instance.sample_rate * st.TTS_SILENCE_PADDING)
            silence_bytes = b"\x00\x00" * silence_samples * tts_instance.num_channels
            output_emitter.push(silence_bytes)

        finally:
            with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
                await asyncio.wait_for(client.disconnect(), timeout=0.5)


class PiperTTS(tts.TTS):
    """TTS via Piper Wyoming protocol."""

    def __init__(
        self,
        *,
        host: str = st.TTS_HOST,
        port: int = st.TTS_PORT,
        sample_rate: int = st.TTS_SAMPLE_RATE,
        num_channels: int = st.AUDIO_CHANNELS,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self._host = host
        self._port = port

    @property
    def model(self) -> str:
        return f"piper/{st.PIPER_VOICE}"

    @property
    def provider(self) -> str:
        return "piper-wyoming"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return PiperChunkedStream(
            tts_instance=self,
            input_text=text,
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        pass
