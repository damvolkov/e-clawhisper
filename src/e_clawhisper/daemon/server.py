"""Daemon server — Unix socket IPC + orchestrator lifecycle."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import orjson

from e_clawhisper.daemon.orchestrator import Orchestrator
from e_clawhisper.shared.logger import configure_logging, logger
from e_clawhisper.shared.settings import AppConfig, load_config, settings


class DaemonServer:
    """Runs orchestrator + IPC socket for CLI control."""

    __slots__ = ("_config", "_orchestrator", "_socket_path", "_running")

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or load_config()
        self._orchestrator = Orchestrator(self._config)
        self._socket_path = settings.SOCKET_PATH
        self._running = False

    ##### LIFECYCLE #####

    async def run(self) -> None:
        """Start IPC server and orchestrator concurrently."""
        configure_logging(
            settings.LOG_LEVEL,
            idle_interval=self._config.logging.idle_interval,
            turn_interval=self._config.logging.turn_interval,
        )

        logger.system(
            "START",
            f"daemon starting agent={self._config.agent.name} socket={self._socket_path}",
        )

        self._running = True
        self._cleanup_socket()

        ipc_server = await asyncio.start_unix_server(self._handle_ipc, path=self._socket_path)
        os.chmod(self._socket_path, 0o600)
        logger.system("IPC", f"ready path={self._socket_path}")

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
                        logger.error(f"daemon_error: {exc}")
        finally:
            await self._orchestrator.stop()
            self._cleanup_socket()
            logger.system("STOP", "daemon stopped")

    ##### IPC #####

    async def _handle_ipc(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            if not (data := await reader.read(4096)):
                return

            request = orjson.loads(data)
            match request.get("command", ""):
                case "status":
                    response = self._build_status()
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

    def _build_status(self) -> dict[str, object]:
        return {
            "status": "ok",
            "data": {
                "running": self._running,
                "phase": str(self._orchestrator.phase),
                "agent_name": self._config.agent.name,
                "agent_backend": str(self._config.agent.backend),
                "wakeword": self._config.sentinel.wakeword.model,
            },
        }

    def _cleanup_socket(self) -> None:
        if (path := Path(self._socket_path)).exists():
            path.unlink()
