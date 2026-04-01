"""Tests for health checker — probes, results, service checking."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from e_clawhisper.health import (
    HealthChecker,
    HealthResult,
    ServiceStatus,
    _probe_http,
    _probe_tcp,
    _probe_ws,
    check_services,
)
from e_clawhisper.shared.operational.exceptions import HealthCheckError
from e_clawhisper.shared.settings import AppConfig, TTSBackend

##### SERVICE STATUS #####


def test_service_status_healthy() -> None:
    s = ServiceStatus(name="test", healthy=True, detail="200")
    assert s.healthy is True
    assert s.name == "test"


def test_service_status_unhealthy() -> None:
    s = ServiceStatus(name="test", healthy=False, detail="connection refused")
    assert s.healthy is False


##### HEALTH RESULT #####


def test_health_result_all_healthy() -> None:
    result = HealthResult(
        services=(
            ServiceStatus(name="stt", healthy=True, detail="ok"),
            ServiceStatus(name="tts", healthy=True, detail="ok"),
        )
    )
    assert result.healthy is True


def test_health_result_one_unhealthy() -> None:
    result = HealthResult(
        services=(
            ServiceStatus(name="stt", healthy=True, detail="ok"),
            ServiceStatus(name="tts", healthy=False, detail="down"),
        )
    )
    assert result.healthy is False


def test_health_result_empty_is_healthy() -> None:
    result = HealthResult()
    assert result.healthy is True


def test_health_result_str_formatting() -> None:
    result = HealthResult(
        services=(ServiceStatus(name="stt", healthy=True, detail="200"),)
    )
    assert "OK" in str(result)
    assert "stt" in str(result)


##### TCP PROBE #####


async def test_probe_tcp_unreachable_host() -> None:
    status = await _probe_tcp("127.0.0.1", 1, "test")
    assert status.healthy is False
    assert status.name == "test"


##### WS PROBE #####


async def test_probe_ws_unreachable() -> None:
    status = await _probe_ws("ws://127.0.0.1:1", "test")
    assert status.healthy is False


##### HTTP PROBE #####


async def test_probe_http_unreachable() -> None:
    status = await _probe_http("http://127.0.0.1:1", "test")
    assert status.healthy is False
    assert status.name == "test"


##### CHECK SERVICES #####


async def test_check_services_default_config_fails_gracefully() -> None:
    """Default config points to localhost services that aren't running in tests."""
    cfg = AppConfig()
    result = await check_services(cfg)
    assert isinstance(result, HealthResult)
    assert len(result.services) > 0


##### HEALTH CHECKER #####


def test_health_checker_stop() -> None:
    checker = HealthChecker(AppConfig(), interval=1.0)
    checker.stop()
    assert checker._running is False


async def test_health_checker_check_once() -> None:
    checker = HealthChecker(AppConfig())
    result = await checker.check_once()
    assert isinstance(result, HealthResult)


async def test_check_services_piper_backend() -> None:
    cfg = AppConfig()
    cfg.tts.backend = TTSBackend.PIPER
    result = await check_services(cfg)
    assert isinstance(result, HealthResult)
    tts_svc = [s for s in result.services if "tts" in s.name]
    assert len(tts_svc) == 1


##### HEALTH CHECKER RUN #####


@patch("e_clawhisper.health.check_services")
async def test_health_checker_run_raises_on_failure(mock_check: AsyncMock) -> None:
    mock_check.return_value = HealthResult(
        services=(ServiceStatus(name="stt", healthy=False, detail="down"),)
    )

    checker = HealthChecker(AppConfig(), interval=0.01)

    with pytest.raises(HealthCheckError, match="down"):
        await checker.run(shutdown=AsyncMock())


@patch("e_clawhisper.health.check_services")
async def test_health_checker_run_stops_cleanly(mock_check: AsyncMock) -> None:
    mock_check.return_value = HealthResult(
        services=(ServiceStatus(name="stt", healthy=True, detail="ok"),)
    )

    checker = HealthChecker(AppConfig(), interval=0.01)

    async def _stop_later() -> None:
        await asyncio.sleep(0.03)
        checker.stop()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_stop_later())
        tg.create_task(checker.run(shutdown=AsyncMock()))

    assert checker._running is False
