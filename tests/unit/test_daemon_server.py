"""Tests for DaemonServer — IPC, status, socket cleanup."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from e_heed.daemon.server import DaemonServer
from e_heed.health import HealthResult, ServiceStatus
from e_heed.shared.operational.exceptions import HealthCheckError
from e_heed.shared.settings import AppConfig

##### FIXTURES #####


@patch("e_heed.daemon.server.Orchestrator")
@patch("e_heed.daemon.server.HealthChecker")
def _make_server(mock_health: MagicMock, mock_orch: MagicMock, config: AppConfig | None = None) -> DaemonServer:
    cfg = config or AppConfig()
    server = DaemonServer(cfg)
    # Orchestrator.stop() is awaited — must be AsyncMock
    server._orchestrator.stop = AsyncMock()
    return server


##### BUILD STATUS #####


def test_build_status_includes_keys() -> None:
    server = _make_server()
    status = server._build_status()
    assert status["status"] == "ok"
    data = status["data"]
    assert "running" in data
    assert "phase" in data
    assert "agent_name" in data
    assert "agent_backend" in data
    assert "wakeword" in data


def test_build_status_reflects_config() -> None:
    cfg = AppConfig()
    cfg.agent.name = "test-agent"
    server = _make_server(config=cfg)
    status = server._build_status()
    assert status["data"]["agent_name"] == "test-agent"


##### CLEANUP SOCKET #####


def test_cleanup_socket_removes_file(tmp_path: Path) -> None:
    sock = tmp_path / "test.sock"
    sock.write_text("")
    server = _make_server()
    server._socket_path = str(sock)
    server._cleanup_socket()
    assert not sock.exists()


def test_cleanup_socket_noop_when_missing(tmp_path: Path) -> None:
    server = _make_server()
    server._socket_path = str(tmp_path / "nonexistent.sock")
    server._cleanup_socket()  # should not raise


##### HANDLE IPC #####


def _make_writer() -> MagicMock:
    """StreamWriter mock — write() is sync, drain()/close()/wait_closed() are async."""
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return writer


async def test_handle_ipc_status() -> None:
    server = _make_server()
    reader = AsyncMock()
    reader.read.return_value = orjson.dumps({"command": "status"})
    writer = _make_writer()

    await server._handle_ipc(reader, writer)

    data = orjson.loads(writer.write.call_args[0][0])
    assert data["status"] == "ok"


async def test_handle_ipc_stop() -> None:
    server = _make_server()
    server._running = True
    reader = AsyncMock()
    reader.read.return_value = orjson.dumps({"command": "stop"})
    writer = _make_writer()

    await server._handle_ipc(reader, writer)

    assert server._running is False
    data = orjson.loads(writer.write.call_args[0][0])
    assert data["status"] == "ok"
    assert data["message"] == "stopping"


async def test_handle_ipc_unknown_command() -> None:
    server = _make_server()
    reader = AsyncMock()
    reader.read.return_value = orjson.dumps({"command": "bad"})
    writer = _make_writer()

    await server._handle_ipc(reader, writer)

    data = orjson.loads(writer.write.call_args[0][0])
    assert data["status"] == "error"
    assert "bad" in data["message"]


async def test_handle_ipc_empty_data() -> None:
    server = _make_server()
    reader = AsyncMock()
    reader.read.return_value = b""
    writer = _make_writer()

    await server._handle_ipc(reader, writer)

    writer.write.assert_not_called()


async def test_handle_ipc_health() -> None:
    server = _make_server()
    reader = AsyncMock()
    reader.read.return_value = orjson.dumps({"command": "health"})
    writer = _make_writer()

    with patch("e_heed.daemon.server.check_services") as mock_check:
        mock_result = MagicMock()
        mock_result.healthy = True
        mock_result.services = []
        mock_check.return_value = mock_result

        await server._handle_ipc(reader, writer)

    data = orjson.loads(writer.write.call_args[0][0])
    assert data["status"] == "ok"


##### SHUTDOWN #####


async def test_shutdown_stops_orchestrator() -> None:
    server = _make_server()
    server._running = True
    await server._shutdown()
    assert server._running is False
    server._orchestrator.stop.assert_called_once()


##### RUN — HEALTH CHECK FAILURE #####


@patch("e_heed.daemon.server.configure_logging")
@patch("e_heed.daemon.server.check_services")
async def test_run_raises_on_unhealthy_services(mock_check: AsyncMock, mock_log: MagicMock) -> None:
    server = _make_server()
    mock_check.return_value = HealthResult(
        services=(ServiceStatus(name="stt", healthy=False, detail="connection refused"),)
    )

    with pytest.raises(HealthCheckError, match="unreachable"):
        await server.run()
