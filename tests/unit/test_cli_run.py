"""Tests for eheed run CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from e_heed.shared.settings import AgentBackend, AppConfig

##### RUN COMMAND — CONFIG LOADING #####


@patch("e_heed.cli.commands.run.asyncio.run")
@patch("e_heed.cli.commands.run.DaemonServer")
@patch("e_heed.cli.commands.run.AppConfig.load")
async def test_run_default_backend_is_generic(
    mock_load: MagicMock, mock_server_cls: MagicMock, mock_asyncio_run: MagicMock
) -> None:
    config = AppConfig()
    mock_load.return_value = config
    mock_server_cls.return_value = MagicMock()

    from e_heed.cli.commands.run import run

    run()

    assert config.agent.backend == AgentBackend.GENERIC
    mock_server_cls.assert_called_once_with(config)
    mock_asyncio_run.assert_called_once()


@patch("e_heed.cli.commands.run.asyncio.run")
@patch("e_heed.cli.commands.run.DaemonServer")
@patch("e_heed.cli.commands.run.AppConfig.load")
async def test_run_openfang_backend(
    mock_load: MagicMock, mock_server_cls: MagicMock, mock_asyncio_run: MagicMock
) -> None:
    config = AppConfig()
    mock_load.return_value = config
    mock_server_cls.return_value = MagicMock()

    from e_heed.cli.commands.run import run

    run(backend=AgentBackend.OPENFANG)

    assert config.agent.backend == AgentBackend.OPENFANG
    mock_server_cls.assert_called_once_with(config)


@patch("e_heed.cli.commands.run.AppConfig.load")
async def test_run_invalid_config_exits(mock_load: MagicMock) -> None:
    from pydantic import ValidationError

    mock_load.side_effect = ValidationError.from_exception_data("AppConfig", [])

    from e_heed.cli.commands.run import run

    with pytest.raises(SystemExit) as exc_info:
        run()
    assert exc_info.value.code == 1


@patch("e_heed.cli.commands.run.asyncio.run")
@patch("e_heed.cli.commands.run.DaemonServer")
@patch("e_heed.cli.commands.run.AppConfig.load")
async def test_run_health_check_failure_exits(
    mock_load: MagicMock, mock_server_cls: MagicMock, mock_asyncio_run: MagicMock
) -> None:
    from e_heed.shared.operational.exceptions import HealthCheckError

    mock_load.return_value = AppConfig()
    mock_asyncio_run.side_effect = HealthCheckError("stt down")

    from e_heed.cli.commands.run import run

    with pytest.raises(SystemExit) as exc_info:
        run()
    assert exc_info.value.code == 2
