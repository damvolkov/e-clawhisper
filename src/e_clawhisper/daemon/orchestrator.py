"""Orchestrator — high-level daemon lifecycle management."""

from __future__ import annotations

from e_clawhisper.daemon.pipeline.manager import PipelineManager
from e_clawhisper.daemon.pipeline.runner import PipelineRunner
from e_clawhisper.shared.logger import LogIcon, logger
from e_clawhisper.shared.settings import AppConfig


class Orchestrator:
    """Assembles the pipeline from config and manages its lifecycle."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._runner: PipelineRunner | None = None
        self._manager = PipelineManager(config)

    @property
    def runner(self) -> PipelineRunner | None:
        return self._runner

    async def start(self) -> None:
        """Resolve agent, assemble pipeline, start the loop."""
        agent = self._manager.create_agent()
        agent_name = self._config.agent.name

        logger.info("resolving agent '%s'...", agent_name, icon=LogIcon.AGENT)
        agent_id = await agent.resolve_agent_id(agent_name)
        await agent.connect(agent_id)

        self._runner = PipelineRunner(
            audio_device=self._manager.create_audio_device(),
            vad=self._manager.create_vad(),
            stt=self._manager.create_stt(),
            tts=self._manager.create_tts(),
            agent=agent,
            wake_word=self._manager.create_wake_word(),
            turn_manager=self._manager.create_turn_manager(),
        )

        logger.info(
            "orchestrator_ready agent=%s backend=%s wake_word=%s",
            agent_name,
            self._config.agent.backend,
            agent_name,
            icon=LogIcon.START,
        )

        await self._runner.start()

    async def stop(self) -> None:
        if self._runner:
            await self._runner.stop()
            self._runner = None
