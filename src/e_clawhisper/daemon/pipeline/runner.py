"""Pipeline runner — core async audio loop.

Flow:
    IDLE:      Mic → VAD → speech? → STT → check wake word
    STREAMING: Mic → VAD → STT → silence? → Agent → TTS
    SPEAKING:  TTS playback → barge-in monitoring
"""

from __future__ import annotations

import time

import numpy as np

from e_clawhisper.daemon.adapters.stt.whisper_live import WhisperLiveAdapter
from e_clawhisper.daemon.core.interfaces.agent import AgentBase
from e_clawhisper.daemon.core.interfaces.stt import STTBase
from e_clawhisper.daemon.core.interfaces.tts import TTSBase
from e_clawhisper.daemon.core.models import VADResult
from e_clawhisper.daemon.core.processors.turn_manager import TurnManager
from e_clawhisper.daemon.core.processors.vad import TenVADProcessor
from e_clawhisper.daemon.core.processors.wake_word import WakeWordDetector
from e_clawhisper.daemon.pipeline.states import ConversationMode, PipelineState
from e_clawhisper.shared.logger import LogIcon, logger
from e_clawhisper.shared.operational.audio_device import AudioDevice
from e_clawhisper.shared.operational.buffer import RingBuffer


class PipelineRunner:
    """Async loop: Mic → VAD → STT → Agent → TTS → Speaker."""

    __slots__ = (
        "_audio",
        "_vad",
        "_stt",
        "_tts",
        "_agent",
        "_wake_detector",
        "_turn_manager",
        "_pre_roll",
        "_tts_sample_rate",
        "_running",
        "_stt_streaming",
        "_speaking_until",
    )

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
        pre_roll_capacity: int,
        tts_sample_rate: int = 22050,
    ) -> None:
        self._audio = audio_device
        self._vad = vad
        self._stt = stt
        self._tts = tts
        self._agent = agent
        self._wake_detector = wake_word
        self._turn_manager = turn_manager
        self._pre_roll = RingBuffer(pre_roll_capacity)
        self._tts_sample_rate = tts_sample_rate
        self._running = False
        self._stt_streaming = False
        self._speaking_until: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def state(self) -> PipelineState:
        return self._turn_manager.state

    @property
    def conversation_mode(self) -> ConversationMode:
        return self._turn_manager.mode

    ##### LIFECYCLE #####

    async def start(self) -> None:
        self._running = True
        await self._audio.start()
        logger.info(
            "pipeline_started — state=IDLE, listening for '%s'", self._wake_detector.wake_word, icon=LogIcon.START
        )

        try:
            await self._run_loop()
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        self._running = False

    async def _run_loop(self) -> None:
        while self._running:
            chunk = await self._audio.read_chunk()
            vad_result = self._vad.process(chunk)
            self._pre_roll.write(chunk)

            if self._turn_manager.should_barge_in(vad_result.is_speech):
                await self._tts.stop()
                self._audio.stop_playback()
                self._speaking_until = 0.0
                self._vad.reset()
                if not self._stt_streaming:
                    await self._open_stt()
                continue

            match self._turn_manager.state:
                case PipelineState.IDLE:
                    await self._handle_idle(chunk, vad_result)
                case PipelineState.STREAMING:
                    await self._handle_streaming(chunk, vad_result)
                case PipelineState.SPEAKING:
                    self._handle_speaking()

    ##### IDLE — WAKE WORD DETECTION #####

    async def _handle_idle(self, chunk: np.ndarray, vad_result: VADResult) -> None:
        self._turn_manager.check_timeout()

        if vad_result.is_speech:
            if not self._stt_streaming:
                await self._open_stt()
            await self._stt.stream_audio(WhisperLiveAdapter.audio_to_float32(chunk))

        if vad_result.should_stop and self._stt_streaming:  # noqa: SIM102
            if (transcript := await self._close_stt()).strip() and self._wake_detector.check(transcript):
                self._turn_manager.activate()
                logger.info("wake_word — entering conversation", icon=LogIcon.WAKE)
                if query := self._wake_detector.strip(transcript):
                    await self.process_query(query)
                else:
                    await self.speak("I'm listening.")

    ##### STREAMING — ACTIVE CONVERSATION #####

    async def _handle_streaming(self, chunk: np.ndarray, vad_result: VADResult) -> None:
        self._turn_manager.touch()

        if vad_result.is_speech:
            if not self._stt_streaming:
                await self._open_stt()
            await self._stt.stream_audio(WhisperLiveAdapter.audio_to_float32(chunk))

        if vad_result.should_stop and self._stt_streaming:  # noqa: SIM102
            if (transcript := await self._close_stt()).strip():
                await self.process_query(transcript)

        if self._turn_manager.check_timeout() and self._stt_streaming:
            await self._close_stt()

    ##### SPEAKING — TTS PLAYBACK + BARGE-IN #####

    def _handle_speaking(self) -> None:
        if self._speaking_until > 0 and time.monotonic() >= self._speaking_until:
            self._speaking_until = 0.0
            self._turn_manager.state = PipelineState.STREAMING if self._turn_manager.is_active else PipelineState.IDLE
            logger.debug("tts_done — back to %s", self._turn_manager.state, icon=LogIcon.TTS)

    ##### STT SESSION #####

    async def _open_stt(self) -> None:
        await self._stt.start_utterance()
        self._stt_streaming = True
        logger.debug("stt_stream_opened", icon=LogIcon.STT)

    async def _close_stt(self) -> str:
        transcript = await self._stt.finish_utterance()
        self._stt_streaming = False
        self._vad.reset()
        if transcript.strip():
            logger.debug("stt_transcript: %s", transcript[:80], icon=LogIcon.STT)
        return transcript

    ##### AGENT + TTS #####

    async def process_query(self, text: str) -> None:
        """Send text to agent, speak the response. Callable from IPC."""
        logger.info("agent_query: %s", text[:100], icon=LogIcon.AGENT)

        if (response := "".join([chunk async for chunk in self._agent.send_message(text)])).strip():
            logger.info("agent_response: %s", response[:100], icon=LogIcon.AGENT)
            await self.speak(response)
        elif self._turn_manager.is_active:
            self._turn_manager.state = PipelineState.STREAMING

    async def speak(self, text: str) -> None:
        """Synthesize and play text via TTS. Callable from IPC."""
        self._turn_manager.state = PipelineState.SPEAKING

        if audio_chunks := [chunk async for chunk in self._tts.synthesize(text)]:
            pcm = b"".join(audio_chunks)
            audio_array = np.frombuffer(pcm, dtype=np.int16)
            self._audio.play_audio(audio_array, sample_rate=self._tts_sample_rate)
            self._speaking_until = time.monotonic() + len(audio_array) / self._tts_sample_rate
        else:
            self._turn_manager.state = PipelineState.STREAMING if self._turn_manager.is_active else PipelineState.IDLE

    ##### CLEANUP #####

    async def _cleanup(self) -> None:
        await self._audio.stop()
        if self._stt_streaming:
            await self._stt.finish_utterance()
        await self._stt.disconnect()
        await self._agent.disconnect()
        logger.info("pipeline_stopped", icon=LogIcon.STOP)
