"""Pipeline runner — the core async audio processing loop.

Flow:
    Mic → VAD → [speech?] → STT stream → [transcript] → wake-word/agent → TTS → speaker
"""

from __future__ import annotations

import asyncio

import numpy as np

from e_clawhisper.daemon.adapters.stt.whisper_live import WhisperLiveAdapter
from e_clawhisper.daemon.core.interfaces.agent import AgentBase
from e_clawhisper.daemon.core.interfaces.stt import STTBase
from e_clawhisper.daemon.core.interfaces.tts import TTSBase
from e_clawhisper.daemon.core.processors.turn_manager import TurnManager
from e_clawhisper.daemon.core.processors.vad import TenVADProcessor
from e_clawhisper.daemon.core.processors.wake_word import WakeWordDetector
from e_clawhisper.daemon.pipeline.states import ConversationMode, PipelineState
from e_clawhisper.shared.logger import LogIcon, logger
from e_clawhisper.shared.operational.audio_device import AudioDevice


class PipelineRunner:
    """Async loop: capture audio → VAD → STT → agent → TTS → speaker."""

    def __init__(
        self,
        *,
        audio_device: AudioDevice,
        vad: TenVADProcessor,
        stt: STTBase,
        tts: TTSBase,
        agent: AgentBase,
        wake_word: WakeWordDetector,
        turn_manager: TurnManager,
    ) -> None:
        self._audio = audio_device
        self._vad = vad
        self._stt = stt
        self._tts = tts
        self._agent = agent
        self._wake = wake_word
        self._turn = turn_manager
        self._running = False
        self._stt_streaming = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def state(self) -> PipelineState:
        return self._turn.state

    @property
    def conversation_mode(self) -> ConversationMode:
        return self._turn.mode

    async def start(self) -> None:
        """Start the pipeline loop."""
        self._running = True
        await self._audio.start()
        logger.info("pipeline_started", icon=LogIcon.START)

        try:
            await self._run_loop()
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        self._running = False

    async def _run_loop(self) -> None:
        while self._running:
            audio_chunk = await self._audio.read_chunk()

            self._turn.check_timeout()

            vad_result = self._vad.process(audio_chunk)

            if self._turn.should_barge_in(vad_result.is_speech):
                await self._tts.stop()
                self._audio.stop_playback()

            if vad_result.is_speech and self._turn.state == PipelineState.LISTENING:
                if not self._stt_streaming:
                    await self._start_stt_stream()

                float32_bytes = WhisperLiveAdapter.audio_to_float32(audio_chunk)
                await self._stt.stream_audio(float32_bytes)

            if self._vad.should_finalize and self._stt_streaming:
                transcript = await self._finish_stt_stream()
                if transcript.strip():
                    await self._handle_transcript(transcript)

    async def _start_stt_stream(self) -> None:
        await self._stt.connect()
        self._stt_streaming = True
        logger.debug("stt_stream_opened", icon=LogIcon.STT)

    async def _finish_stt_stream(self) -> str:
        transcript = await self._stt.finish_stream()
        self._stt_streaming = False
        self._vad.reset()
        logger.debug("stt_stream_closed transcript=%s", transcript[:80], icon=LogIcon.STT)
        return transcript

    async def _handle_transcript(self, transcript: str) -> None:
        if not self._turn.is_active:
            if self._wake.check(transcript):
                self._turn.activate()
                query = self._wake.strip(transcript)
                if query:
                    await self._send_to_agent(query)
                else:
                    await self._speak("I'm listening. How can I help you?")
        else:
            self._turn.touch()
            await self._send_to_agent(transcript)

    async def _send_to_agent(self, text: str) -> None:
        self._turn.state = PipelineState.THINKING
        logger.debug("agent_query: %s", text[:100], icon=LogIcon.AGENT)

        response_parts: list[str] = []
        async for chunk in self._agent.send_message(text):
            response_parts.append(chunk)

        full_response = "".join(response_parts)
        if full_response.strip():
            await self._speak(full_response)

        self._turn.state = PipelineState.LISTENING

    async def _speak(self, text: str) -> None:
        self._turn.state = PipelineState.SPEAKING
        logger.debug("tts_speak: %s", text[:80], icon=LogIcon.TTS)

        audio_chunks: list[bytes] = []
        async for chunk in self._tts.synthesize(text):
            audio_chunks.append(chunk)

        if audio_chunks:
            pcm = b"".join(audio_chunks)
            audio_array = np.frombuffer(pcm, dtype=np.int16)
            self._audio.play_audio(audio_array, sample_rate=22050)
            duration = len(audio_array) / 22050
            await asyncio.sleep(duration)

    async def _cleanup(self) -> None:
        await self._audio.stop()
        if self._stt_streaming:
            await self._stt.disconnect()
        await self._agent.disconnect()
        logger.info("pipeline_stopped", icon=LogIcon.STOP)
