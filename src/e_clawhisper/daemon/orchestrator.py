"""Orchestrator — state machine SENTINEL ↔ TURN.

┌──────────┐  WakewordEvent   ┌──────┐
│ SENTINEL │ ───────────────► │ TURN │
│          │ ◄─────────────── │      │
└──────────┘  TurnComplete    └──────┘

Does NOT process audio — only manages pipeline transitions.
"""

from __future__ import annotations

from enum import StrEnum, auto

from e_clawhisper.daemon.adapters.audio import AudioAdapter
from e_clawhisper.daemon.adapters.llm import LLMAdapter
from e_clawhisper.daemon.adapters.stt import STTAdapter
from e_clawhisper.daemon.adapters.tts import TTSAdapter
from e_clawhisper.daemon.sentinel.pipeline import SentinelPipeline
from e_clawhisper.daemon.turn.pipeline import TurnPipeline
from e_clawhisper.shared.logger import logger
from e_clawhisper.shared.operational.events import TurnComplete, TurnError
from e_clawhisper.shared.settings import AgentBackend, AppConfig, STTBackend, TTSBackend


class PipelinePhase(StrEnum):
    SENTINEL = auto()
    TURN = auto()


class Orchestrator:
    """Assembles adapters, manages SENTINEL ↔ TURN transitions."""

    __slots__ = (
        "_config",
        "_audio",
        "_stt",
        "_tts",
        "_llm",
        "_sentinel",
        "_turn",
        "_phase",
        "_running",
    )

    def __init__(self, config: AppConfig) -> None:
        self._config = config

        # Adapters — shared across pipelines
        self._audio = AudioAdapter(config.audio)
        self._stt = self._create_stt()
        self._tts = self._create_tts()
        self._llm = self._create_llm()

        # Pipelines
        self._sentinel = SentinelPipeline(config.sentinel)
        self._turn = TurnPipeline(
            stt=self._stt,
            llm=self._llm,
            tts=self._tts,
            vad_config=config.vad,
            tts_sample_rate=config.tts.piper.sample_rate,
        )

        self._phase = PipelinePhase.SENTINEL
        self._running = False

    @property
    def phase(self) -> PipelinePhase:
        return self._phase

    ##### LIFECYCLE #####

    async def start(self) -> None:
        self._running = True

        logger.system("START", f"resolving agent '{self._config.agent.name}'...")
        agent_id = await self._llm.resolve_agent_id(self._config.agent.name)
        await self._llm.connect(agent_id)

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

    async def _run_sentinel(self) -> None:
        logger.set_pipeline("SENTINEL")
        await self._sentinel.run(self._audio.queue)

        if self._sentinel.last_event and self._running:
            self._phase = PipelinePhase.TURN

    async def _run_turn(self) -> None:
        logger.set_pipeline("TURN")

        result = await self._turn.run(audio=self._audio)

        match result:
            case TurnComplete():
                logger.turn(
                    "DEFAULT",
                    f"turn complete ({result.duration:.1f}s)",
                    transcript=logger.truncate(result.transcript, 60),
                )
            case TurnError():
                logger.warning(f"turn failed: {result.reason}")

        self._phase = PipelinePhase.SENTINEL

    ##### SHUTDOWN #####

    async def _shutdown(self) -> None:
        await self._audio.stop()
        await self._stt.disconnect()
        await self._llm.disconnect()
        self._sentinel.shutdown()
        self._turn.shutdown()
        logger.system("STOP", "orchestrator stopped")

    ##### FACTORIES #####

    def _create_stt(self) -> STTAdapter:
        match self._config.stt.backend:
            case STTBackend.WHISPERLIVE:
                return STTAdapter(config=self._config.stt.whisperlive)
            case _:
                raise ValueError(f"Unsupported STT: {self._config.stt.backend}")

    def _create_tts(self) -> TTSAdapter:
        match self._config.tts.backend:
            case TTSBackend.PIPER:
                return TTSAdapter(config=self._config.tts.piper)
            case _:
                raise ValueError(f"Unsupported TTS: {self._config.tts.backend}")

    def _create_llm(self) -> LLMAdapter:
        match self._config.agent.backend:
            case AgentBackend.OPENFANG:
                return LLMAdapter(config=self._config.backends.openfang)
            case _:
                raise ValueError(f"Unsupported agent: {self._config.agent.backend}")
