"""Orchestrator — state machine SENTINEL → TURN ⇄ LOOP.

┌──────────┐  WakewordEvent   ┌──────┐  TurnComplete   ┌──────┐
│ SENTINEL │ ───────────────► │ TURN │ ───────────────► │ LOOP │
│          │ ◄─────────────── │      │ ◄─────────────── │      │
└──────────┘  silence timeout └──────┘  voice detected  └──────┘

LOOP reuses SentinelPipeline layers 1-2 (energy + VAD) to detect
follow-up speech without requiring a wakeword.
"""

from __future__ import annotations

import asyncio
import contextlib
from enum import StrEnum, auto

from e_clawhisper.daemon.adapters.agent.generic import GenericAdapter
from e_clawhisper.daemon.adapters.agent.openfang import OpenfangAdapter
from e_clawhisper.daemon.adapters.audio import AudioAdapter
from e_clawhisper.daemon.adapters.base import AgentPort, STTPort, TTSPort
from e_clawhisper.daemon.adapters.stt.whisperlive import WhisperliveAdapter
from e_clawhisper.daemon.adapters.tts.kokoro import KokoroAdapter
from e_clawhisper.daemon.adapters.tts.piper import PiperAdapter
from e_clawhisper.daemon.sentinel.pipeline import SentinelPipeline
from e_clawhisper.daemon.turn.pipeline import TurnPipeline
from e_clawhisper.shared.logger import logger
from e_clawhisper.shared.operational.events import TurnComplete, TurnError
from e_clawhisper.shared.settings import AgentBackend, AppConfig, STTBackend, TTSBackend


class PipelinePhase(StrEnum):
    SENTINEL = auto()
    TURN = auto()
    LOOP = auto()


class Orchestrator:
    """Assembles adapters, manages SENTINEL → TURN ⇄ LOOP transitions."""

    __slots__ = (
        "_config",
        "_audio",
        "_stt",
        "_tts",
        "_agent",
        "_sentinel",
        "_turn",
        "_phase",
        "_running",
        "_conversation_turns",
    )

    def __init__(self, config: AppConfig) -> None:
        self._config = config

        # Adapters — shared across pipelines
        self._audio = AudioAdapter(config.audio)
        self._stt = self._create_stt()
        self._tts = self._create_tts()
        self._agent = self._create_agent()

        # Pipelines
        self._sentinel = SentinelPipeline(config.sentinel)
        self._turn = TurnPipeline(
            stt=self._stt,
            agent=self._agent,
            tts=self._tts,
            vad_config=config.vad,
            tts_sample_rate=config.tts_sample_rate,
            pcm_queue_size=config.audio.pcm_queue_size,
        )

        self._phase = PipelinePhase.SENTINEL
        self._running = False
        self._conversation_turns = 0

    @property
    def phase(self) -> PipelinePhase:
        return self._phase

    ##### LIFECYCLE #####

    async def start(self) -> None:
        self._running = True

        logger.system("START", f"resolving agent '{self._config.agent.name}'...")
        agent_id = await self._agent.resolve_agent_id(self._config.agent.name)
        await self._agent.connect(agent_id)

        logger.system("START", "connecting STT (warm-up)...")
        await self._stt.connect()

        logger.system("START", "starting audio device...")
        await self._audio.start()

        logger.system(
            "START",
            f"ready — agent='{self._config.agent.name}' wakeword='{self._sentinel.wakeword_name}'",
        )

        try:
            await self._run_loop()
        finally:
            await self._shutdown()

    async def stop(self) -> None:
        self._running = False
        await self._sentinel.stop()
        await self._turn.stop()

    ##### STATE MACHINE #####

    async def _run_loop(self) -> None:
        while self._running:
            match self._phase:
                case PipelinePhase.SENTINEL:
                    await self._run_sentinel()
                case PipelinePhase.TURN:
                    await self._run_turn()
                case PipelinePhase.LOOP:
                    await self._run_conversation_loop()

    async def _run_sentinel(self) -> None:
        logger.set_pipeline("SENTINEL")
        await self._sentinel.run(self._audio.queue)

        if self._sentinel.last_event and self._running:
            self._phase = PipelinePhase.TURN

    async def _run_turn(self) -> None:
        logger.set_pipeline("TURN")

        if not await self._ensure_agent():
            self._end_conversation()
            return

        try:
            async with asyncio.timeout(self._config.turn_timeout):
                result = await self._turn.run(audio=self._audio)
        except TimeoutError:
            logger.warning(f"turn timeout after {self._config.turn_timeout:.0f}s")
            await self._turn.stop()
            result = TurnError(turn_id="timeout", reason="turn_timeout")

        self._audio.drain()

        match result:
            case TurnComplete() if self._should_loop():
                self._conversation_turns += 1
                logger.turn(
                    "DEFAULT",
                    f"turn {self._conversation_turns} complete ({result.duration:.1f}s)",
                    transcript=logger.truncate(result.transcript, 60),
                )
                self._phase = PipelinePhase.LOOP
            case TurnComplete():
                logger.turn(
                    "DEFAULT",
                    f"turn complete ({result.duration:.1f}s)",
                    transcript=logger.truncate(result.transcript, 60),
                )
                self._end_conversation()
            case TurnError():
                logger.warning(f"turn failed: {result.reason}")
                self._end_conversation()

    async def _run_conversation_loop(self) -> None:
        logger.set_pipeline("LOOP")

        voice_detected = await self._sentinel.wait_for_voice(
            self._audio.queue,
            timeout=self._config.conversation.loop_timeout,
        )

        self._audio.drain()

        if voice_detected and self._running:
            self._phase = PipelinePhase.TURN
        else:
            self._end_conversation()

    ##### CONVERSATION #####

    def _should_loop(self) -> bool:
        conv = self._config.conversation
        return conv.enabled and self._conversation_turns < conv.max_turns

    def _end_conversation(self) -> None:
        if self._conversation_turns > 0:
            logger.loop("DEFAULT", f"conversation ended ({self._conversation_turns} turns)")
            if isinstance(self._agent, GenericAdapter):
                self._agent.clear_history()
        self._conversation_turns = 0
        self._phase = PipelinePhase.SENTINEL

    ##### SERVICE HEALTH #####

    async def _ensure_agent(self) -> bool:
        """Verify agent connection, attempt reconnect if lost."""
        if await self._agent.is_connected():
            return True
        try:
            logger.system("WARN", "agent reconnecting...")
            with contextlib.suppress(Exception):
                await self._agent.disconnect()
            await self._agent.connect(self._agent.agent_id)
            return True
        except Exception as exc:
            logger.error(f"agent reconnect failed: {exc}")
            return False

    ##### SHUTDOWN #####

    async def _shutdown(self) -> None:
        for _name, coro in [
            ("audio", self._audio.stop()),
            ("stt", self._stt.disconnect()),
            ("agent", self._agent.disconnect()),
        ]:
            with contextlib.suppress(Exception):
                await coro
        logger.system("STOP", "orchestrator stopped")

    ##### FACTORIES #####

    def _create_stt(self) -> STTPort:
        match self._config.stt.backend:
            case STTBackend.WHISPERLIVE:
                return WhisperliveAdapter(config=self._config.stt.whisperlive)
            case _:
                raise ValueError(f"Unsupported STT: {self._config.stt.backend}")

    def _create_tts(self) -> TTSPort:
        match self._config.tts.backend:
            case TTSBackend.PIPER:
                return PiperAdapter(config=self._config.tts.piper)
            case TTSBackend.KOKORO:
                return KokoroAdapter(config=self._config.tts.kokoro)
            case _:
                raise ValueError(f"Unsupported TTS: {self._config.tts.backend}")

    def _create_agent(self) -> AgentPort:
        match self._config.agent.backend:
            case AgentBackend.OPENFANG:
                return OpenfangAdapter(config=self._config.backends.openfang)
            case AgentBackend.GENERIC:
                return GenericAdapter(config=self._config.backends.generic)
            case _:
                raise ValueError(f"Unsupported agent: {self._config.agent.backend}")
