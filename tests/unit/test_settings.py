"""Tests for settings."""

from __future__ import annotations

import pytest

from e_clawhisper.core.settings import Settings


##### DEFAULTS #####


def test_settings_defaults() -> None:
    s = Settings(AGENT_NAME="test", AGENT_ID="id-1", _env_file=None)
    assert s.AGENT_NAME == "test"
    assert s.CHANNEL_BACKEND == "openfang"
    assert s.STT_PORT == 9090
    assert s.TTS_PORT == 10200
    assert s.CONVERSATION_TIMEOUT == 30.0


##### COMPUTED URLS #####


@pytest.mark.parametrize(
    ("host", "port", "expected"),
    [
        ("localhost", 4200, "http://localhost:4200"),
        ("10.0.0.5", 8080, "http://10.0.0.5:8080"),
    ],
    ids=["localhost", "custom-ip"],
)
def test_settings_openfang_base_url(host: str, port: int, expected: str) -> None:
    s = Settings(OPENFANG_HOST=host, OPENFANG_PORT=port, _env_file=None)
    assert s.openfang_base_url == expected


def test_settings_stt_ws_url() -> None:
    s = Settings(STT_HOST="stt-host", STT_PORT=9999, _env_file=None)
    assert s.stt_ws_url == "ws://stt-host:9999"


def test_settings_dev_defaults() -> None:
    s = Settings(ENVIRONMENT="DEV", _env_file=None)
    assert s.is_dev is True
    assert s.log_level == "debug"


def test_settings_prod() -> None:
    s = Settings(ENVIRONMENT="PROD", _env_file=None)
    assert s.is_dev is False
    assert s.log_level == "info"
