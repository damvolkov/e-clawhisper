"""Daemon server — Unix socket IPC + pipeline lifecycle."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import orjson

from e_clawhisper.daemon.core.models import DaemonStatus
from e_clawhisper.daemon.orchestrator import Orchestrator
from e_clawhisper.shared.logger import LogIcon, configure_logging, logger
from e_clawhisper.shared.settings import AppConfig, load_config, settings


class DaemonServer:
    """Runs the audio pipeline and an IPC socket for CLI control."""

    __slots__ = ("_config", "_orchestrator", "_socket_path", "_running")

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or load_config()
        self._orchestrator = Orchestrator(self._config)
        self._socket_path = settings.SOCKET_PATH
        self._running = False

    ##### LIFECYCLE #####

    async def run(self) -> None:
        """Start IPC server and pipeline concurrently."""
        configure_logging(settings.LOG_LEVEL)

        logger.info(
            "daemon_starting agent=%s backend=%s socket=%s",
            self._config.agent.name,
            self._config.agent.backend,
            self._socket_path,
            icon=LogIcon.START,
        )

        self._running = True
        self._cleanup_socket()

        ipc_server = await asyncio.start_unix_server(self._handle_ipc, path=self._socket_path)
        os.chmod(self._socket_path, 0o600)
        logger.info("ipc_server_ready path=%s", self._socket_path, icon=LogIcon.IPC)

        try:
            async with ipc_server:
                pipeline_task = asyncio.create_task(self._orchestrator.start())
                ipc_task = asyncio.create_task(ipc_server.serve_forever())

                done, pending = await asyncio.wait(
                    [pipeline_task, ipc_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending:
                    task.cancel()
                for task in done:
                    if exc := task.exception():
                        logger.error("daemon_error: %s", exc, icon=LogIcon.ERROR)
        finally:
            await self._orchestrator.stop()
            self._cleanup_socket()
            logger.info("daemon_stopped", icon=LogIcon.STOP)

    ##### IPC #####

    async def _handle_ipc(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle a single IPC client connection."""
        try:
            if not (data := await reader.read(4096)):
                return

            request = orjson.loads(data)
            match request.get("command", ""):
                case "status":
                    response = self.get_status()
                case "stop":
                    self._running = False
                    await self._orchestrator.stop()
                    response = {"status": "ok", "message": "stopping"}
                case unknown:
                    response = {"status": "error", "message": f"unknown command: {unknown}"}

            writer.write(orjson.dumps(response))
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    def get_status(self) -> dict[str, object]:
        """Build daemon status response."""
        runner = self._orchestrator.runner
        status = DaemonStatus(
            running=self._running,
            pipeline_state=runner.state if runner else "stopped",
            conversation_active=runner.conversation_mode == "active" if runner else False,
            agent_name=self._config.agent.name,
            agent_backend=self._config.agent.backend,
        )
        return {"status": "ok", "data": status.model_dump()}

    def _cleanup_socket(self) -> None:
        if (path := Path(self._socket_path)).exists():
            path.unlink()
