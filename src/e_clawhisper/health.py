"""Health checker — periodic async pings to dependent services."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from e_clawhisper.shared.logger import logger
from e_clawhisper.shared.operational.exceptions import HealthCheckError
from e_clawhisper.shared.settings import AppConfig, STTBackend, TTSBackend

_PROBE_TIMEOUT = 5.0

##### RESULT #####


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    """Health status for a single service."""

    name: str
    healthy: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class HealthResult:
    """Aggregated health check result."""

    services: tuple[ServiceStatus, ...] = field(default_factory=tuple)

    @property
    def healthy(self) -> bool:
        return all(s.healthy for s in self.services)

    def __str__(self) -> str:
        lines = [f"  {'OK' if s.healthy else 'FAIL'} {s.name}: {s.detail}" for s in self.services]
        return "\n".join(lines)


##### PROBES #####


async def _probe_http(url: str, name: str) -> ServiceStatus:
    """HTTP GET to service base URL."""
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
            resp = await client.get(url)
            return ServiceStatus(name=name, healthy=resp.status_code < 500, detail=f"{resp.status_code}")
    except httpx.HTTPError as exc:
        return ServiceStatus(name=name, healthy=False, detail=str(exc))


async def _probe_tcp(host: str, port: int, name: str) -> ServiceStatus:
    """Raw TCP connect probe."""
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=_PROBE_TIMEOUT)
        writer.close()
        await writer.wait_closed()
        return ServiceStatus(name=name, healthy=True, detail=f"tcp://{host}:{port}")
    except (OSError, TimeoutError) as exc:
        return ServiceStatus(name=name, healthy=False, detail=str(exc))


async def _probe_ws(url: str, name: str) -> ServiceStatus:
    """WebSocket TCP-level reachability (connect to underlying TCP port)."""
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 80
    return await _probe_tcp(host, port, name)


##### CHECKER #####


async def check_services(config: AppConfig) -> HealthResult:
    """Run all health probes concurrently."""
    tasks: list[Coroutine[Any, Any, ServiceStatus]] = []

    match config.stt.backend:
        case STTBackend.WHISPERLIVE:
            tasks.append(_probe_ws(str(config.stt.whisperlive.url), "stt:whisperlive"))

    match config.tts.backend:
        case TTSBackend.KOKORO:
            tasks.append(_probe_http(str(config.tts.kokoro.url), "tts:kokoro"))
        case TTSBackend.PIPER:
            tasks.append(_probe_tcp(config.tts.piper.host, config.tts.piper.port, "tts:piper"))

    statuses = await asyncio.gather(*tasks)
    return HealthResult(services=tuple(statuses))


class HealthChecker:
    """Periodic health monitor — shuts down daemon on failure."""

    __slots__ = ("_config", "_interval", "_running")

    def __init__(self, config: AppConfig, interval: float = 30.0) -> None:
        self._config = config
        self._interval = interval
        self._running = False

    async def run(self, shutdown: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Periodic check loop. Calls shutdown on first failure."""
        self._running = True
        while self._running:
            await asyncio.sleep(self._interval)
            if not self._running:
                break
            result = await check_services(self._config)
            if not result.healthy:
                logger.error(f"health_check_failed\n{result}")
                self._running = False
                raise HealthCheckError(f"Service(s) down:\n{result}")

    async def check_once(self) -> HealthResult:
        """Single health check — used at startup."""
        return await check_services(self._config)

    def stop(self) -> None:
        self._running = False
