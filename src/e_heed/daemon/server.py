"""Daemon server — Unix socket IPC + orchestrator lifecycle + health monitoring."""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path

import orjson

from e_heed.daemon.orchestrator import Orchestrator
from e_heed.health import HealthChecker, check_services
from e_heed.shared.logger import configure_logging, logger
from e_heed.shared.operational.exceptions import HealthCheckError
from e_heed.shared.settings import AppConfig, settings


class DaemonServer:
    """Runs orchestrator + IPC socket + health monitor."""

    __slots__ = ("_config", "_orchestrator", "_health", "_socket_path", "_running")

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or AppConfig.load()
        self._orchestrator = Orchestrator(self._config)
        self._health = HealthChecker(self._config)
        self._socket_path = settings.SOCKET_PATH
        self._running = False

    ##### LIFECYCLE #####

    async def run(self) -> None:
        """Start IPC server, orchestrator, and health monitor concurrently."""
        configure_logging(
            settings.LOG_LEVEL,
            idle_interval=self._config.logging.idle_interval,
            turn_interval=self._config.logging.turn_interval,
        )

        logger.system(
            "START",
            f"daemon starting agent={self._config.agent.name} socket={self._socket_path}",
        )

        # Pre-flight health check
        result = await check_services(self._config)
        for svc in result.services:
            status = "OK" if svc.healthy else "FAIL"
            logger.system("HEALTH", f"{status} {svc.name} {svc.detail}")
        if not result.healthy:
            logger.error(f"startup_health_check_failed\n{result}")
            msg = f"Required services unreachable:\n{result}"
            raise HealthCheckError(msg)

        self._running = True
        self._cleanup_socket()

        ipc_server = await asyncio.start_unix_server(self._handle_ipc, path=self._socket_path)
        os.chmod(self._socket_path, 0o600)
        logger.system("IPC", f"ready path={self._socket_path}")

        try:
            async with ipc_server:
                pipeline_task = asyncio.create_task(self._orchestrator.start())
                ipc_task = asyncio.create_task(ipc_server.serve_forever())
                health_task = asyncio.create_task(self._health.run(self._shutdown))

                done, pending = await asyncio.wait(
                    [pipeline_task, ipc_task, health_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending:
                    task.cancel()
                for task in done:
                    with contextlib.suppress(asyncio.CancelledError, HealthCheckError):
                        if exc := task.exception():
                            logger.error(f"daemon_error: {exc}")
        finally:
            self._health.stop()
            await self._orchestrator.stop()
            self._cleanup_socket()
            logger.system("STOP", "daemon stopped")

    ##### SHUTDOWN #####

    async def _shutdown(self) -> None:
        self._running = False
        await self._orchestrator.stop()

    ##### IPC #####

    async def _handle_ipc(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            if not (data := await reader.read(4096)):
                return

            request = orjson.loads(data)
            match request.get("command", ""):
                case "status":
                    response = self._build_status()
                case "health":
                    result = await check_services(self._config)
                    response = {
                        "status": "ok" if result.healthy else "degraded",
                        "data": {s.name: {"healthy": s.healthy, "detail": s.detail} for s in result.services},
                    }
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
